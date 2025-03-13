#!/usr/bin/env python3
"""
cgMLST distance calculator - Optimized version
Calculates pairwise Hamming distances for cgMLST data with GPU/multicore acceleration
and optimized I/O performance for large datasets.
"""

import os
import argparse
import sys
from typing import Optional, Tuple, List
import pandas as pd
import numpy as np
import time
import math
from numba import jit, prange, set_num_threads, cuda
from tqdm import tqdm
import gzip
import mmap
import io
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor

DEFAULT_THREADS = max(1, os.cpu_count() // 2)
VERSION = "0.1.0"

def filter_loci_by_completeness(data: pd.DataFrame, missing_char: str, min_completeness: float) -> tuple[list, dict]:
    """Filter loci based on completeness threshold."""
    loci = data.columns.tolist()
    loci_stats = {}
    filtered_loci = []
    
    print("\nLoci filtering details:")
    print("-" * 80)
    print(f"{'Locus':<30} {'Completeness %':<15} {'Status':<10} {'Missing/Total'}")
    print("-" * 80)
    
    for locus in loci:
        total = len(data)
        missing = sum(data[locus].isin([missing_char, np.nan]))
        completeness = ((total - missing) / total) * 100
        
        loci_stats[locus] = {
            'total_samples': total,
            'missing_data': missing,
            'completeness': completeness
        }
        
        status = "INCLUDED" if completeness >= min_completeness else "EXCLUDED"
        print(f"{locus:<30} {completeness:>11.2f}%  {status:<10} {missing}/{total}")
        
        if completeness >= min_completeness:
            filtered_loci.append(locus)
    
    print("-" * 80)
    print(f"Total loci included: {len(filtered_loci)}/{len(loci)} ({len(filtered_loci)/len(loci)*100:.2f}%)")
    print(f"Total loci excluded: {len(loci)-len(filtered_loci)}/{len(loci)} ({(len(loci)-len(filtered_loci))/len(loci)*100:.2f}%)")
    print()
            
    return filtered_loci, loci_stats

def filter_samples_by_completeness(data: pd.DataFrame, missing_char: str, min_completeness: float) -> tuple[pd.DataFrame, dict]:
    """Filter samples based on completeness threshold."""
    sample_stats = {}
    filtered_samples = []
    
    print("\nSample filtering details:")
    print("-" * 80)
    print(f"{'Sample ID':<30} {'Completeness %':<15} {'Status':<10} {'Missing/Total'}")
    print("-" * 80)
    
    for idx, row in data.iterrows():
        total = len(data.columns)
        missing = sum(row.isin([missing_char, np.nan]))
        completeness = ((total - missing) / total) * 100
        
        sample_stats[idx] = {
            'total_loci': total,
            'missing_data': missing,
            'completeness': completeness
        }
        
        status = "INCLUDED" if completeness >= min_completeness else "EXCLUDED"
        print(f"{str(idx)[:30]:<30} {completeness:>11.2f}%  {status:<10} {missing}/{total}")
        
        if completeness >= min_completeness:
            filtered_samples.append(idx)
    
    print("-" * 80)
    print(f"Total samples included: {len(filtered_samples)}/{len(data)} ({len(filtered_samples)/len(data)*100:.2f}%)")
    print(f"Total samples excluded: {len(data)-len(filtered_samples)}/{len(data)} ({(len(data)-len(filtered_samples))/len(data)*100:.2f}%)")
    print()
    
    filtered_df = data.loc[filtered_samples]
    return filtered_df, sample_stats

def process_chunk(chunk_data, skip_input_replacements, missing_char):
    """Process a data chunk for parallel loading."""
    if not skip_input_replacements:
        chunk_data.replace(r'^INF-', '', regex=True, inplace=True)
        chunk_data = pd.to_numeric(chunk_data.stack(), errors='coerce').unstack().fillna(0)
    else:
        chunk_data = chunk_data.apply(pd.to_numeric, errors='coerce')
    
    return chunk_data

def estimate_file_size(file_path):
    """Get file size in bytes."""
    try:
        return os.path.getsize(file_path)
    except:
        return 0

def count_lines(file_path):
    """Count number of lines in file efficiently."""
    with open(file_path, 'rb') as f:
        # Memory-map the file for efficient line counting
        try:
            mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
            return mm.count(b'\n')
        except:
            # Fallback if memory-mapping fails
            return sum(1 for _ in f)

def load_data_optimized(file_path: str, input_sep: str = "\t", skip_input_replacements: bool = False, 
              min_locus_completeness: float = None, min_sample_completeness: float = None,
              missing_char: str = "-", chunk_size: int = 10000, io_threads: int = 4) -> tuple[Optional[pd.DataFrame], Optional[dict], Optional[dict]]:
    """Load data with optimized I/O performance."""
    try:
        load_start = time.time()
        file_size = estimate_file_size(file_path)
        print(f"\nLoading data from {file_path} ({file_size/1024/1024:.1f} MB)...")
        
        # Get metadata for the file
        print("Reading file metadata...")
        with open(file_path, 'r') as f:
            header = f.readline().strip().split(input_sep)
            num_columns = len(header) - 1  # Subtract 1 for index column
        
        # Count lines efficiently
        print("Estimating file size...")
        num_rows = count_lines(file_path) - 1  # Subtract 1 for header
        
        print(f"File contains approximately {num_rows:,} rows and {num_columns:,} columns")
        
        # For very large files, adjust chunk size
        if num_rows > 100000:
            adjusted_chunk_size = min(num_rows // 20, 10000)  # Aim for ~20 chunks
            if adjusted_chunk_size != chunk_size:
                print(f"Adjusting chunk size to {adjusted_chunk_size:,} for better performance")
                chunk_size = adjusted_chunk_size
        
        print(f"Reading file in chunks of {chunk_size:,} rows using {io_threads} I/O threads...")
        
        # Read the data in chunks to conserve memory and use parallel processing
        chunk_reader = pd.read_csv(
            file_path, 
            sep=input_sep, 
            index_col=0, 
            chunksize=chunk_size,
            low_memory=False
        )
        
        # Process each chunk, potentially in parallel
        chunks = []
        with ThreadPoolExecutor(max_workers=io_threads) as executor:
            futures = []
            
            # Submit chunks for processing
            for i, chunk in enumerate(tqdm(chunk_reader, desc="Reading chunks")):
                futures.append(executor.submit(process_chunk, chunk, skip_input_replacements, missing_char))
            
            # Collect processed chunks
            for future in tqdm(futures, desc="Processing chunks"):
                chunks.append(future.result())
        
        # Combine all chunks
        print("Combining chunks...")
        data = pd.concat(chunks)
        
        print(f"Initial data shape: {data.shape[0]} samples × {data.shape[1]} loci")
        
        # Apply completeness filtering if thresholds are provided
        loci_stats = sample_stats = None
        
        if min_locus_completeness is not None:
            print(f"\nApplying locus completeness filter (threshold: {min_locus_completeness}%)...")
            filtered_loci, loci_stats = filter_loci_by_completeness(data, missing_char, min_locus_completeness)
            data = data[filtered_loci]
        
        if min_sample_completeness is not None:
            print(f"\nApplying sample completeness filter (threshold: {min_sample_completeness}%)...")
            data, sample_stats = filter_samples_by_completeness(data, missing_char, min_sample_completeness)
        
        # Convert to numeric data type if needed
        if not skip_input_replacements:
            print("Converting data to numeric format...")
            data = data.astype(np.int32)
        
        print(f"\nFinal data shape after filtering: {data.shape[0]} samples × {data.shape[1]} loci")
        
        load_end = time.time()
        load_time = load_end - load_start
        print(f"Data loading time: {load_time:.2f} seconds")
            
        return data, loci_stats, sample_stats
        
    except Exception as e:
        print(f"Error loading data: {e}")
        return None, None, None

@jit(nopython=True, parallel=True, fastmath=True)
def calculate_hamming_distances_numba(values):
    n_samples = values.shape[0]
    distances = np.zeros((n_samples, n_samples), dtype=np.int32)
    for i in prange(n_samples):
        for j in prange(i + 1, n_samples):
            dist = 0
            for k in range(values.shape[1]):
                if (values[i, k] != values[j, k]) and (values[i, k] != 0) and (values[j, k] != 0):
                    dist += 1
            distances[i, j] = dist
            distances[j, i] = dist
    return distances

@jit(nopython=True, parallel=True, fastmath=True)
def calculate_hamming_distances_numba_batch(values, start_i, end_i, start_j, end_j):
    """Calculate distances for a batch of samples."""
    batch_i_size = end_i - start_i
    batch_j_size = end_j - start_j
    distances = np.zeros((batch_i_size, batch_j_size), dtype=np.int32)
    
    for i in prange(batch_i_size):
        for j in prange(batch_j_size):
            real_i = start_i + i
            real_j = start_j + j
            
            if real_i < real_j:  # Only compute upper triangle
                dist = 0
                for k in range(values.shape[1]):
                    if (values[real_i, k] != values[real_j, k]) and (values[real_i, k] != 0) and (values[real_j, k] != 0):
                        dist += 1
                distances[i, j] = dist
    
    return distances

@cuda.jit
def calculate_hamming_distances_cuda_kernel(values, distances, start_i, start_j):
    """CUDA kernel function for calculating Hamming distances."""
    # Get thread indices
    i, j = cuda.grid(2)
    
    # Get dimensions of the distance matrix
    dist_i_size = distances.shape[0]
    dist_j_size = distances.shape[1]
    
    # Check if indices are within bounds
    if i < dist_i_size and j < dist_j_size:
        # Map to real indices in the original data
        real_i = start_i + i
        real_j = start_j + j
        
        # Only compute upper triangle (avoid duplicating work)
        if real_i < real_j:
            dist = 0
            for k in range(values.shape[1]):
                if (values[real_i, k] != values[real_j, k]) and (values[real_i, k] != 0) and (values[real_j, k] != 0):
                    dist += 1
            distances[i, j] = dist

def calculate_hamming_distances_cuda_batch(values, start_i, end_i, start_j, end_j):
    """Calculate distances for a batch of samples using CUDA."""
    try:
        batch_i_size = end_i - start_i
        batch_j_size = end_j - start_j
        
        # Transfer batch data to GPU
        values_device = cuda.to_device(values)
        distances_device = cuda.to_device(np.zeros((batch_i_size, batch_j_size), dtype=np.int32))
        
        # Configure the grid and blocks
        threads_per_block = (16, 16)
        blocks_per_grid_x = (batch_i_size + threads_per_block[0] - 1) // threads_per_block[0]
        blocks_per_grid_y = (batch_j_size + threads_per_block[1] - 1) // threads_per_block[1]
        blocks_per_grid = (blocks_per_grid_x, blocks_per_grid_y)
        
        # Launch the kernel
        calculate_hamming_distances_cuda_kernel[blocks_per_grid, threads_per_block](
            values_device, distances_device, start_i, start_j
        )
        
        # Ensure all GPU operations are complete
        cuda.synchronize()
        
        # Copy back the results from GPU
        batch_distances = distances_device.copy_to_host()
        
        return batch_distances
    except Exception as e:
        print(f"CUDA batch error: {e}")
        return calculate_hamming_distances_numba_batch(values, start_i, end_i, start_j, end_j)

def calculate_distances_batched(data, use_gpu=False, max_memory_gb=8):
    """Calculate distances in batches to manage memory usage."""
    try:
        values = data.values
        n_samples = values.shape[0]
        
        # Determine maximum batch size based on memory constraint
        # For an int32 matrix, each element is 4 bytes
        bytes_per_element = 4  # int32
        
        # Calculate maximum elements based on memory constraint
        max_elements = int((max_memory_gb * 1024 * 1024 * 1024) / bytes_per_element)
        
        # Calculate max batch size (square root of max elements)
        max_batch_size = int(math.sqrt(max_elements))
        
        # Limit batch size to be no larger than n_samples
        batch_size = min(max_batch_size, n_samples)
        
        print(f"Processing distances in batches of ~{batch_size:,} samples")
        print(f"Total batches: {math.ceil(n_samples/batch_size):,}")
        
        # Initialize the full distance matrix (use int32 to save memory)
        # We'll only populate the upper triangle and then mirror it later
        distances = np.zeros((n_samples, n_samples), dtype=np.int32)
        
        # Calculate the distances in batches
        calc_start = time.time()
        
        # Process upper triangle in batches
        total_batches = sum(1 for i in range(0, n_samples, batch_size) 
                           for j in range(i, n_samples, batch_size))
        
        with tqdm(total=total_batches, desc="Processing batches") as pbar:
            for i_start in range(0, n_samples, batch_size):
                i_end = min(i_start + batch_size, n_samples)
                
                for j_start in range(i_start, n_samples, batch_size):
                    j_end = min(j_start + batch_size, n_samples)
                    
                    batch_start = time.time()
                    
                    if use_gpu:
                        # Use GPU for this batch
                        batch_distances = calculate_hamming_distances_cuda_batch(
                            values, i_start, i_end, j_start, j_end
                        )
                    else:
                        # Use CPU for this batch
                        batch_distances = calculate_hamming_distances_numba_batch(
                            values, i_start, i_end, j_start, j_end
                        )
                    
                    # Copy the batch results to the full distance matrix (upper triangle)
                    distances[i_start:i_end, j_start:j_end] = batch_distances
                    
                    batch_end = time.time()
                    pbar.update(1)
        
        # Mirror the upper triangle to the lower triangle
        print("Mirroring distance matrix...")
        for i in range(n_samples):
            for j in range(i+1, n_samples):
                distances[j, i] = distances[i, j]
        
        calc_end = time.time()
        calc_time = calc_end - calc_start
        print(f"Total distance calculation time: {calc_time:.2f} seconds")
        
        return distances
    except Exception as e:
        print(f"Error calculating distances: {e}")
        return None

def process_save_chunk(chunk_idx, row_data, output_file, output_sep, append=True):
    """Write a chunk of data to disk in a separate process."""
    mode = 'a' if append else 'w'
    with open(output_file, mode) as f:
        f.write('\n'.join(row_data))
        if row_data:  # Add newline if there's data
            f.write('\n')
    return chunk_idx

def save_distances_optimized(distances, file_path, index, output_sep="\t", index_name="cgmlst-dists", 
                          matrix_format="full", chunk_size=1000, io_threads=4, use_binary=False):
    """Save pairwise distances with optimized I/O."""
    try:
        save_start = time.time()
        
        if distances is not None:
            n_samples = distances.shape[0]
            matrix_size_mb = (n_samples * n_samples * 4) / (1024 * 1024)  # 4 bytes per int32
            print(f"Saving distance matrix ({n_samples} x {n_samples}, ~{matrix_size_mb:.1f} MB) to {file_path}")
            
            # Use binary format for very large matrices if requested
            if use_binary and matrix_size_mb > 1000:  # Over 1GB
                # Save as numpy binary file which is much faster
                binary_file = file_path + ".npy"
                print(f"Using binary format for large matrix (saving to {binary_file})")
                np.save(binary_file, distances)
                
                # Also save sample index separately
                index_file = file_path + ".index.txt"
                with open(index_file, 'w') as f:
                    f.write('\n'.join(map(str, index)))
                
                print(f"Matrix saved in binary format. To load:")
                print(f"    import numpy as np")
                print(f"    distances = np.load('{binary_file}')")
                print(f"    with open('{index_file}', 'r') as f:")
                print(f"        index = [line.strip() for line in f]")
                
                binary_save_time = time.time() - save_start
                print(f"Binary save completed in {binary_save_time:.2f} seconds")
                
                # Continue with normal TSV save
                print("Now saving in TSV format as requested...")
            
            # Write the header row with sample names in the column headers
            sample_names = list(map(str, index))
            header = output_sep.join([""] + sample_names)
            
            with open(file_path, 'w') as f:
                f.write(header + '\n')
            
            # Determine optimal chunk size based on matrix size
            if n_samples > 50000:
                chunk_size = min(chunk_size, 100)  # Use smaller chunks for very large matrices
            
            # Write data in chunks to control memory usage
            total_chunks = math.ceil(n_samples / chunk_size)
            
            # Prepare all chunks before writing to avoid GIL contention
            print("Preparing chunks for writing...")
            all_chunks = {}
            
            with tqdm(total=total_chunks, desc="Preparing chunks") as pbar:
                for i in range(0, n_samples, chunk_size):
                    chunk_end = min(i + chunk_size, n_samples)
                    
                    # Create a chunk of the distances dataframe
                    if matrix_format == "lower-tri":
                        # Only include lower triangle
                        distances_chunk = np.zeros((chunk_end - i, n_samples), dtype=np.int32)
                        for r in range(i, chunk_end):
                            for c in range(0, r + 1):  # Lower triangle including diagonal
                                distances_chunk[r - i, c] = distances[r, c]
                    elif matrix_format == "upper-tri":
                        # Only include upper triangle
                        distances_chunk = np.zeros((chunk_end - i, n_samples), dtype=np.int32)
                        for r in range(i, chunk_end):
                            for c in range(r, n_samples):  # Upper triangle including diagonal
                                distances_chunk[r - i, c] = distances[r, c]
                    else:
                        # Full matrix
                        distances_chunk = distances[i:chunk_end, :]
                    
                    # Convert chunk to strings with sample IDs in the first column
                    chunk_rows = []
                    for r in range(chunk_end - i):
                        row_values = [str(distances_chunk[r, c]) for c in range(n_samples)]
                        chunk_rows.append(output_sep.join([sample_names[i + r]] + row_values))
                    
                    all_chunks[i // chunk_size] = chunk_rows
                    pbar.update(1)
            
            # Write chunks to file using multiple workers
            print(f"Writing {total_chunks} chunks to file using {io_threads} I/O threads...")
            with ProcessPoolExecutor(max_workers=io_threads) as executor:
                futures = []
                
                # Submit first chunk without append
                if 0 in all_chunks:
                    futures.append(executor.submit(
                        process_save_chunk, 0, all_chunks[0], file_path, output_sep, False
                    ))
                
                # Submit remaining chunks with append
                for chunk_idx in range(1, total_chunks):
                    if chunk_idx in all_chunks:
                        futures.append(executor.submit(
                            process_save_chunk, chunk_idx, all_chunks[chunk_idx], 
                            file_path, output_sep, True
                        ))
                
                # Wait for all chunks to be written
                for future in tqdm(futures, desc="Writing chunks"):
                    chunk_idx = future.result()
            
            save_end = time.time()
            save_time = save_end - save_start
            print(f"Total save process time: {save_time:.2f} seconds")
        else:
            print("No distances to save.")
    except Exception as e:
        print(f"Error saving distances: {e}")

def check_gpu_availability():
    """Check if CUDA GPU is available."""
    try:
        cuda_available = cuda.is_available()
        if cuda_available:
            # Get device information
            device = cuda.get_current_device()
            print(f"CUDA GPU is available: {device.name}")
            print(f"Compute capability: {device.compute_capability}")
            
            # Try to get some basic device properties without accessing total_memory
            # which might not be available in some environments
            try:
                # This might fail on some installations, but we'll catch it
                mem_info = cuda.current_context().get_memory_info()
                print(f"GPU Memory: Free {mem_info[0]/1024**3:.2f} GB, Total {mem_info[1]/1024**3:.2f} GB")
            except Exception as mem_e:
                # If we can't get memory info, just log that we're still using the GPU
                print(f"GPU memory information not available: {mem_e}")
                print("GPU will still be used")
            
            return cuda_available
        else:
            print("CUDA GPU is not available.")
            return False
    except Exception as e:
        print(f"Error checking GPU availability: {e}")
        # Still try to continue with GPU if cuda.is_available() was True
        return cuda.is_available()

def detect_system_capabilities():
    """Detect system capabilities and return recommended settings."""
    # Detect number of available CPU cores
    cpu_count = os.cpu_count() or 1
    
    # For Numba, it's often best to use all available cores
    recommended_threads = cpu_count
    
    # Check if GPU is available
    gpu_available = cuda.is_available()
    
    # Estimate available memory (this is approximate)
    try:
        import psutil
        available_memory_gb = psutil.virtual_memory().available / (1024 * 1024 * 1024)
    except ImportError:
        # If psutil is not available, make a conservative estimate
        available_memory_gb = 8.0  # Default to 8GB if we can't detect
    
    # Recommend using at most 75% of available memory
    recommended_memory_gb = max(1.0, available_memory_gb * 0.75)
    
    # Determine recommended I/O threads (typically 4-8 is good)
    io_threads = min(8, max(4, cpu_count // 4))
    
    return {
        'cpu_count': cpu_count,
        'recommended_threads': recommended_threads,
        'gpu_available': gpu_available,
        'available_memory_gb': available_memory_gb,
        'recommended_memory_gb': recommended_memory_gb,
        'io_threads': io_threads
    }

def detect_io_capabilities():
    """Detect I/O capabilities of the system."""
    try:
        import tempfile
        import time
        
        # Create a temporary file for testing
        temp_file = tempfile.NamedTemporaryFile(delete=False)
        temp_path = temp_file.name
        temp_file.close()
        
        # Write test
        data_size = 50 * 1024 * 1024  # 50 MB of data
        data = b'0' * data_size
        
        write_start = time.time()
        with open(temp_path, 'wb') as f:
            f.write(data)
        write_time = time.time() - write_start
        write_speed = data_size / write_time / (1024 * 1024)  # MB/s
        
        # Read test
        read_start = time.time()
        with open(temp_path, 'rb') as f:
            _ = f.read()
        read_time = time.time() - read_start
        read_speed = data_size / read_time / (1024 * 1024)  # MB/s
        
        # Clean up
        os.unlink(temp_path)
        
        return {
            'read_speed_mb_per_sec': read_speed,
            'write_speed_mb_per_sec': write_speed
        }
    except Exception as e:
        print(f"Error testing I/O capabilities: {e}")
        return {
            'read_speed_mb_per_sec': 100,  # Default conservative estimate
            'write_speed_mb_per_sec': 50
        }

def print_performance_summary(load_time, calc_time, save_time, total_time):
    """Print a summary of performance timing."""
    print("\n" + "=" * 50)
    print("PERFORMANCE SUMMARY")
    print("=" * 50)
    print(f"{'Operation':<25} {'Time (s)':<10} {'Percentage':<10}")
    print("-" * 50)
    print(f"{'Data loading':<25} {load_time:<10.2f} {(load_time/total_time)*100:<9.2f}%")
    print(f"{'Distance calculation':<25} {calc_time:<10.2f} {(calc_time/total_time)*100:<9.2f}%")
    print(f"{'Saving results':<25} {save_time:<10.2f} {(save_time/total_time)*100:<9.2f}%")
    print(f"{'Other operations':<25} {total_time-load_time-calc_time-save_time:<10.2f} {((total_time-load_time-calc_time-save_time)/total_time)*100:<9.2f}%")
    print("-" * 50)
    print(f"{'Total':<25} {total_time:<10.2f} {100:<9.2f}%")
    print("=" * 50)

def main():
    try:
        # Detect system capabilities for default settings
        capabilities = detect_system_capabilities()
        default_threads = capabilities['recommended_threads']
        default_memory_gb = round(capabilities['recommended_memory_gb'])
        default_io_threads = capabilities['io_threads']
        
        # Check I/O capabilities
        io_capabilities = detect_io_capabilities()
        print(f"\nSystem capabilities detected:")
        print(f"- CPU cores: {capabilities['cpu_count']}")
        print(f"- GPU available: {'Yes' if capabilities['gpu_available'] else 'No'}")
        print(f"- Available memory: ~{capabilities['available_memory_gb']:.1f} GB")
        print(f"- Disk read speed: ~{io_capabilities['read_speed_mb_per_sec']:.1f} MB/s")
        print(f"- Disk write speed: ~{io_capabilities['write_speed_mb_per_sec']:.1f} MB/s")
        print(f"- Recommended settings: {default_threads} compute threads, {default_io_threads} I/O threads, {default_memory_gb} GB max memory")
        
        parser = argparse.ArgumentParser(description=f"Calculate pairwise Hamming distances. Version: {VERSION}")
        parser.add_argument("--input", help="Path to the input TSV file")
        parser.add_argument("--output", help="Path to save the output TSV file")
        parser.add_argument("--skip_input_replacements", action="store_true", help="Skip input replacements when there are no strings in the input")
        parser.add_argument("--input_sep", default="\t", help="Input file separator (default: '\t')")
        parser.add_argument("--output_sep", default="\t", help="Output file separator (default: '\t')")
        parser.add_argument("--index_name", default="cgmlst-dists", help="Name for the index column (default: 'cgmlst-dists')")
        parser.add_argument("--matrix-format", choices=["full", "lower-tri", "upper-tri"], default="full", help="Format for the output matrix")
        parser.add_argument("--num_threads", type=int, default=default_threads, help=f"Number of threads for parallel execution (default: auto-detected {default_threads})")
        parser.add_argument("--io_threads", type=int, default=default_io_threads, help=f"Number of I/O threads for file operations (default: {default_io_threads})")
        parser.add_argument("--max_memory_gb", type=float, default=default_memory_gb, help=f"Maximum memory to use in GB for distance calculation (default: {default_memory_gb})")
        parser.add_argument("--chunk_size", type=int, default=1000, help="Size of chunks for reading/writing files")
        parser.add_argument("--missing_char", default="-", help="Character used for missing data (default: '-')")
        parser.add_argument("--locus-completeness", type=float, default=None, 
                          help="Minimum percentage of non-missing data required for a locus (0-100)")
        parser.add_argument("--sample-completeness", type=float, default=None,
                          help="Minimum percentage of non-missing data required for a sample (0-100)")
        parser.add_argument("--gpu", action="store_true", help="Use GPU acceleration when available")
        parser.add_argument("--binary-output", action="store_true", help="Also save results in binary format for large matrices")
        parser.add_argument("--version", action="version", version=VERSION)

        args = parser.parse_args()
        
        # Set number of threads for Numba
        num_threads = args.num_threads
        print(f"Using {num_threads} threads for parallel processing")
        set_num_threads(num_threads)

        if not args.input or not args.output:
            parser.print_help()
            return
        
        # Start total timing
        total_start_time = time.time()
        
        # Check GPU availability if requested
        use_gpu = False
        if args.gpu:
            use_gpu = check_gpu_availability()
            
        # Load data and measure time
        load_start_time = time.time()
        data, loci_stats, sample_stats = load_data_optimized(
            args.input, args.input_sep, args.skip_input_replacements,
            args.locus_completeness, args.sample_completeness,
            args.missing_char, args.chunk_size, args.io_threads
        )
        load_end_time = time.time()
        load_time = load_end_time - load_start_time
        
        if data is None:
            return

        n_samples = data.shape[0]
        print(f"\nCalculating distances for {n_samples} samples and {data.shape[1]} allele calls")
        print(f"The final matrix will have {n_samples*n_samples} distances")
        
        # Calculate distances and measure time
        calc_start_time = time.time()
        distances = calculate_distances_batched(data, use_gpu, args.max_memory_gb)
        calc_end_time = time.time()
        calc_time = calc_end_time - calc_start_time
        
        print("\nCalculations completed. Saving distances...")

        # Save distances and measure time
        save_start_time = time.time()
        if distances is not None:
            save_distances_optimized(
                distances, args.output, data.index, args.output_sep, 
                args.index_name, args.matrix_format, args.chunk_size, 
                args.io_threads, args.binary_output
            )
        save_end_time = time.time()
        save_time = save_end_time - save_start_time
        
        print("\nProcess completed successfully")

        # Calculate total time
        total_end_time = time.time()
        total_time = total_end_time - total_start_time
        
        # Print performance summary
        print_performance_summary(load_time, calc_time, save_time, total_time)
        
        print(f"Total time taken: {total_time:.2f} seconds")

    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()