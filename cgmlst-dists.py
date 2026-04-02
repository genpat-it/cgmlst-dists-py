#!/usr/bin/env python3
"""
cgMLST distance calculator - Highly optimized version
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
import warnings

# Suppress pandas warnings
warnings.filterwarnings("ignore", category=FutureWarning)

DEFAULT_THREADS = max(1, os.cpu_count() // 2)
VERSION = "0.1.2"

def filter_loci_by_completeness(data: pd.DataFrame, missing_char: str, min_completeness: float, silent: bool = False) -> tuple[list, dict]:
    """Filter loci based on completeness threshold."""
    total = len(data)
    # Vectorized missing count: count missing_char and NaN per column
    missing_counts = data.isin([missing_char]).sum() + data.isna().sum()
    completeness_pct = ((total - missing_counts) / total) * 100

    loci_stats = {
        locus: {
            'total_samples': total,
            'missing_data': int(missing_counts[locus]),
            'completeness': completeness_pct[locus]
        }
        for locus in data.columns
    }

    mask = completeness_pct >= min_completeness
    filtered_loci = data.columns[mask].tolist()

    if not silent:
        print("\nLoci filtering details:")
        print("-" * 80)
        print(f"{'Locus':<30} {'Completeness %':<15} {'Status':<10} {'Missing/Total'}")
        print("-" * 80)
        for locus in data.columns:
            status = "INCLUDED" if mask[locus] else "EXCLUDED"
            print(f"{locus:<30} {completeness_pct[locus]:>11.2f}%  {status:<10} {int(missing_counts[locus])}/{total}")
        print("-" * 80)
        n_loci = len(data.columns)
        print(f"Total loci included: {len(filtered_loci)}/{n_loci} ({len(filtered_loci)/n_loci*100:.2f}%)")
        print(f"Total loci excluded: {n_loci-len(filtered_loci)}/{n_loci} ({(n_loci-len(filtered_loci))/n_loci*100:.2f}%)")
        print()

    return filtered_loci, loci_stats

def filter_samples_by_completeness(data: pd.DataFrame, missing_char: str, min_completeness: float, silent: bool = False) -> tuple[pd.DataFrame, dict]:
    """Filter samples based on completeness threshold."""
    total = len(data.columns)
    # Vectorized missing count per row
    missing_counts = data.isin([missing_char]).sum(axis=1) + data.isna().sum(axis=1)
    completeness_pct = ((total - missing_counts) / total) * 100

    sample_stats = {
        idx: {
            'total_loci': total,
            'missing_data': int(missing_counts[idx]),
            'completeness': completeness_pct[idx]
        }
        for idx in data.index
    }

    mask = completeness_pct >= min_completeness
    filtered_samples = data.index[mask].tolist()

    if not silent:
        print("\nSample filtering details:")
        print("-" * 80)
        print(f"{'Sample ID':<30} {'Completeness %':<15} {'Status':<10} {'Missing/Total'}")
        print("-" * 80)
        for idx in data.index:
            status = "INCLUDED" if mask[idx] else "EXCLUDED"
            print(f"{str(idx)[:30]:<30} {completeness_pct[idx]:>11.2f}%  {status:<10} {int(missing_counts[idx])}/{total}")
        print("-" * 80)
        n_samples = len(data)
        print(f"Total samples included: {len(filtered_samples)}/{n_samples} ({len(filtered_samples)/n_samples*100:.2f}%)")
        print(f"Total samples excluded: {n_samples-len(filtered_samples)}/{n_samples} ({(n_samples-len(filtered_samples))/n_samples*100:.2f}%)")
        print()

    filtered_df = data.loc[filtered_samples]
    return filtered_df, sample_stats

def process_chunk(chunk_data, skip_input_replacements, missing_char):
    """Process a data chunk for parallel loading."""
    if not skip_input_replacements:
        # Remove INF- prefix before replacing missing char, so dtype stays consistent
        chunk_data.replace(r'^INF-', '', regex=True, inplace=True)
        chunk_data.replace(missing_char, 0, inplace=True)
        chunk_data = chunk_data.apply(pd.to_numeric, errors='coerce').fillna(0)
    else:
        chunk_data.replace(missing_char, 0, inplace=True)
        chunk_data = chunk_data.apply(pd.to_numeric, errors='coerce')
        chunk_data.fillna(0, inplace=True)

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
              missing_char: str = "-", chunk_size: int = 10000, io_threads: int = 4, 
              silent: bool = False) -> tuple[Optional[pd.DataFrame], Optional[dict], Optional[dict]]:
    """Load data with optimized I/O performance."""
    try:
        load_start = time.time()
        file_size = estimate_file_size(file_path)
        
        if not silent:
            print(f"\nLoading data from {file_path} ({file_size/1024/1024:.1f} MB)...")
        
        # For small files, load directly for better performance
        if file_size < 50 * 1024 * 1024:  # Less than 50MB
            if not silent:
                print("Small file detected, loading directly...")
            
            data = pd.read_csv(file_path, sep=input_sep, index_col=0, low_memory=False)
            data = process_chunk(data, skip_input_replacements, missing_char)
            
            if not silent:
                print(f"Data loaded: {data.shape[0]} samples × {data.shape[1]} loci")
        else:
            # For larger files, use chunked reading
            if not silent:
                print("Reading file metadata...")
                
            with open(file_path, 'r') as f:
                header = f.readline().strip().split(input_sep)
                num_columns = len(header) - 1  # Subtract 1 for index column
            
            # Count lines efficiently
            if not silent:
                print("Estimating file size...")
                
            num_rows = count_lines(file_path) - 1  # Subtract 1 for header
            
            if not silent:
                print(f"File contains approximately {num_rows:,} rows and {num_columns:,} columns")
            
            # For very large files, adjust chunk size
            if num_rows > 100000:
                adjusted_chunk_size = min(num_rows // 20, 10000)  # Aim for ~20 chunks
                if adjusted_chunk_size != chunk_size and not silent:
                    print(f"Adjusting chunk size to {adjusted_chunk_size:,} for better performance")
                chunk_size = adjusted_chunk_size
            
            if not silent:
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
                for i, chunk in enumerate(tqdm(chunk_reader, desc="Reading chunks", disable=silent)):
                    futures.append(executor.submit(process_chunk, chunk, skip_input_replacements, missing_char))
                
                # Collect processed chunks
                for future in tqdm(futures, desc="Processing chunks", disable=silent):
                    chunks.append(future.result())
            
            if not silent:
                print("Combining chunks...")
                
            data = pd.concat(chunks)
            
            if not silent:
                print(f"Initial data shape: {data.shape[0]} samples × {data.shape[1]} loci")
        
        # Apply completeness filtering if thresholds are provided
        loci_stats = sample_stats = None
        
        if min_locus_completeness is not None:
            if not silent:
                print(f"\nApplying locus completeness filter (threshold: {min_locus_completeness}%)...")
            filtered_loci, loci_stats = filter_loci_by_completeness(data, missing_char, min_locus_completeness, silent)
            data = data[filtered_loci]
        
        if min_sample_completeness is not None:
            if not silent:
                print(f"\nApplying sample completeness filter (threshold: {min_sample_completeness}%)...")
            data, sample_stats = filter_samples_by_completeness(data, missing_char, min_sample_completeness, silent)
        
        # Convert to numeric data type if needed
        if not skip_input_replacements and not silent:
            print("Converting data to numeric format...")
        
        data = data.astype(np.int32)
        
        if not silent:
            print(f"\nFinal data shape after filtering: {data.shape[0]} samples × {data.shape[1]} loci")
        
        load_end = time.time()
        load_time = load_end - load_start
        
        if not silent:
            print(f"Data loading time: {load_time:.2f} seconds")
            
        return data, loci_stats, sample_stats
        
    except Exception as e:
        if not silent:
            print(f"Error loading data: {e}")
        return None, None, None

def calculate_hamming_distances_numpy(values, num_threads=None, silent=False):
    """Calculate Hamming distances using numpy vectorized operations with threading."""
    n_samples = values.shape[0]
    mask = values != 0  # (N, L) boolean - precompute once
    distances = np.zeros((n_samples, n_samples), dtype=np.int32)

    if num_threads is None:
        num_threads = os.cpu_count() or 1

    def compute_row(i):
        if i + 1 >= n_samples:
            return
        rest_values = values[i + 1:]  # (N-i-1, L)
        rest_mask = mask[i + 1:]  # (N-i-1, L)
        both_valid = mask[i] & rest_mask  # broadcast (L,) & (N-i-1, L)
        different = values[i] != rest_values  # broadcast
        distances[i, i + 1:] = (both_valid & different).sum(axis=1)

    # numpy releases GIL so threads parallelize well
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        list(tqdm(executor.map(compute_row, range(n_samples)), total=n_samples,
                  desc="Computing distances", disable=silent))

    return distances

@jit(nopython=True, parallel=True, fastmath=True, cache=True)
def calculate_hamming_distances_numba(values):
    """Calculate Hamming distances between all pairs of samples (fallback)."""
    n_samples = values.shape[0]
    distances = np.zeros((n_samples, n_samples), dtype=np.int32)

    for i in prange(n_samples):
        for j in range(i + 1, n_samples):
            dist = 0
            for k in range(values.shape[1]):
                vi = values[i, k]
                vj = values[j, k]
                if vi != 0 and vj != 0 and vi != vj:
                    dist += 1

            distances[i, j] = dist

    return distances

@cuda.jit
def calculate_hamming_distances_cuda_kernel(values, distances, start_i, start_j):
    """CUDA kernel function for calculating Hamming distances."""
    i, j = cuda.grid(2)

    dist_i_size = distances.shape[0]
    dist_j_size = distances.shape[1]

    if i < dist_i_size and j < dist_j_size:
        real_i = start_i + i
        real_j = start_j + j

        if real_i < real_j:
            dist = 0
            n_loci = values.shape[1]

            for k in range(n_loci):
                vi = values[real_i, k]
                vj = values[real_j, k]
                if vi != 0 and vj != 0 and vi != vj:
                    dist += 1

            distances[i, j] = dist

def estimate_gpu_batch_size(n_samples, n_loci, silent=False):
    """Estimate optimal batch size based on available GPU memory."""
    try:
        mem_info = cuda.current_context().get_memory_info()
        free_mem = mem_info[0]
    except Exception:
        free_mem = 2 * 1024**3  # Default 2GB

    # Memory needed: values array (n_samples * n_loci * 4 bytes) + distance batch (batch^2 * 4 bytes)
    values_mem = n_samples * n_loci * 4
    available_for_distances = max(free_mem - values_mem - 256 * 1024**2, 256 * 1024**2)  # Reserve 256MB
    max_batch = int(math.sqrt(available_for_distances / 4))
    batch_size = min(max_batch, n_samples)

    if not silent:
        print(f"GPU memory: {free_mem/1024**3:.2f} GB free, batch size: {batch_size}")

    return batch_size

def calculate_hamming_distances_cuda_batch(values, start_i, end_i, start_j, end_j, silent=False):
    """Calculate distances for a batch of samples using CUDA."""
    try:
        batch_i_size = end_i - start_i
        batch_j_size = end_j - start_j

        # Transfer data to GPU
        values_device = cuda.to_device(values)
        distances_device = cuda.device_array((batch_i_size, batch_j_size), dtype=np.int32)
        # Zero-initialize
        distances_device[:] = 0

        # Use 32x32 thread blocks for better occupancy
        threads_per_block = (32, 32)
        blocks_per_grid_x = (batch_i_size + threads_per_block[0] - 1) // threads_per_block[0]
        blocks_per_grid_y = (batch_j_size + threads_per_block[1] - 1) // threads_per_block[1]
        blocks_per_grid = (blocks_per_grid_x, blocks_per_grid_y)

        calculate_hamming_distances_cuda_kernel[blocks_per_grid, threads_per_block](
            values_device, distances_device, start_i, start_j
        )

        cuda.synchronize()
        batch_distances = distances_device.copy_to_host()

        return batch_distances
    except Exception as e:
        if not silent:
            print(f"CUDA batch error: {e}, falling back to CPU")
        return calculate_hamming_distances_numba_batch(values, start_i, end_i, start_j, end_j)

def calculate_distances_batched(data, use_gpu=False, max_memory_gb=8, silent=False, **kwargs):
    """Calculate distances in batches to manage memory usage."""
    try:
        values = data.values
        n_samples = values.shape[0]
        num_threads = kwargs.get('num_threads', os.cpu_count() or 1)

        # CPU path: use numpy vectorized approach (no batching needed)
        if not use_gpu:
            if not silent:
                print(f"Using numpy vectorized calculation with {num_threads} threads")
            distances = calculate_hamming_distances_numpy(values, num_threads, silent)
            distances += distances.T
            return distances

        # GPU path
        # For small datasets, skip batching
        if n_samples < 100:
            if not silent:
                print("Small dataset detected, using direct calculation without batching")
            try:
                result = calculate_hamming_distances_cuda_batch(values, 0, n_samples, 0, n_samples, silent)
            except:
                if not silent:
                    print("GPU calculation failed, falling back to CPU")
                result = calculate_hamming_distances_numpy(values, num_threads, silent)
            result += result.T
            return result

        # GPU batching
        batch_size = estimate_gpu_batch_size(n_samples, values.shape[1], silent)

        if not silent:
            print(f"Processing distances in batches of ~{batch_size:,} samples")
            print(f"Total batches: {math.ceil(n_samples/batch_size):,}")

        # Initialize the full distance matrix
        distances = np.zeros((n_samples, n_samples), dtype=np.int32)

        # Calculate the distances in batches
        total_batches = sum(1 for i in range(0, n_samples, batch_size)
                          for j in range(i, n_samples, batch_size))

        with tqdm(total=total_batches, desc="Processing batches", disable=silent) as pbar:
            for i_start in range(0, n_samples, batch_size):
                i_end = min(i_start + batch_size, n_samples)

                for j_start in range(i_start, n_samples, batch_size):
                    j_end = min(j_start + batch_size, n_samples)

                    try:
                        batch_distances = calculate_hamming_distances_cuda_batch(
                            values, i_start, i_end, j_start, j_end, silent
                        )
                    except:
                        if not silent:
                            print("GPU batch failed, falling back to CPU for this batch")
                        # Fallback: compute this batch with numpy
                        batch_i = end_i - start_i
                        batch_j = end_j - start_j
                        sub_distances = np.zeros((batch_i, batch_j), dtype=np.int32)
                        for ii in range(batch_i):
                            ri = i_start + ii
                            for jj in range(batch_j):
                                rj = j_start + jj
                                if ri < rj:
                                    both = (values[ri] != 0) & (values[rj] != 0)
                                    sub_distances[ii, jj] = ((values[ri] != values[rj]) & both).sum()
                        batch_distances = sub_distances
                    
                    # Copy the batch results to the full distance matrix (upper triangle)
                    distances[i_start:i_end, j_start:j_end] = batch_distances
                    pbar.update(1)
        
        # Mirror the upper triangle to the lower triangle (vectorized)
        if not silent:
            print("Mirroring distance matrix...")

        distances += distances.T

        return distances
        
    except Exception as e:
        if not silent:
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
                          matrix_format="full", chunk_size=1000, io_threads=4, use_binary=False, silent=False):
    """Save pairwise distances with optimized I/O."""
    try:
        save_start = time.time()
        
        if distances is not None:
            n_samples = distances.shape[0]
            matrix_size_mb = (n_samples * n_samples * 4) / (1024 * 1024)  # 4 bytes per int32
            
            if not silent:
                print(f"Saving distance matrix ({n_samples} x {n_samples}, ~{matrix_size_mb:.1f} MB) to {file_path}")
            
            # Use binary format for very large matrices if requested
            if use_binary and matrix_size_mb > 1000:  # Over 1GB
                # Save as numpy binary file which is much faster
                binary_file = file_path + ".npy"
                if not silent:
                    print(f"Using binary format for large matrix (saving to {binary_file})")
                    
                np.save(binary_file, distances)
                
                # Also save sample index separately
                index_file = file_path + ".index.txt"
                with open(index_file, 'w') as f:
                    f.write('\n'.join(map(str, index)))
                
                if not silent:
                    print(f"Matrix saved in binary format. To load:")
                    print(f"    import numpy as np")
                    print(f"    distances = np.load('{binary_file}')")
                    print(f"    with open('{index_file}', 'r') as f:")
                    print(f"        index = [line.strip() for line in f]")
                    
                    binary_save_time = time.time() - save_start
                    print(f"Binary save completed in {binary_save_time:.2f} seconds")
                    print("Now saving in TSV format as requested...")
            
            sample_names = list(map(str, index))

            if matrix_format == "full":
                # Use pandas to_csv for fast full-matrix output
                if not silent:
                    print("Writing full matrix using pandas...")
                df_out = pd.DataFrame(distances, index=sample_names, columns=sample_names)
                df_out.index.name = index_name
                df_out.to_csv(file_path, sep=output_sep)
            else:
                # For triangular formats, write row by row
                if not silent:
                    print("Converting distance matrix to strings...")
                dist_str = distances.astype(str)
                header = output_sep.join([index_name] + sample_names)

                with open(file_path, 'w', buffering=8*1024*1024) as f:
                    f.write(header + '\n')

                    for i in tqdm(range(n_samples), desc="Writing rows", disable=silent):
                        if matrix_format == "lower-tri":
                            row_values = list(dist_str[i, :i+1]) + ['0'] * (n_samples - i - 1)
                        else:
                            row_values = ['0'] * i + list(dist_str[i, i:])

                        f.write(output_sep.join([sample_names[i]] + row_values) + '\n')
            
            save_end = time.time()
            save_time = save_end - save_start
            
            if not silent:
                print(f"Total save process time: {save_time:.2f} seconds")
        else:
            if not silent:
                print("No distances to save.")
                
    except Exception as e:
        if not silent:
            print(f"Error saving distances: {e}")

def write_to_stdout(distances, index, output_sep="\t", index_name="cgmlst-dists", matrix_format="full"):
    """Write distance matrix directly to stdout for maximum performance."""
    try:
        if distances is not None:
            n_samples = distances.shape[0]
            sample_names = list(map(str, index))
            dist_str = distances.astype(str)

            # Write the header row with sample names
            header = output_sep.join([index_name] + sample_names)
            sys.stdout.write(header + '\n')

            # Write data rows
            for i in range(n_samples):
                if matrix_format == "lower-tri":
                    row_values = list(dist_str[i, :i+1]) + ['0'] * (n_samples - i - 1)
                elif matrix_format == "upper-tri":
                    row_values = ['0'] * i + list(dist_str[i, i:])
                else:
                    row_values = list(dist_str[i])

                sys.stdout.write(output_sep.join([sample_names[i]] + row_values) + '\n')

    except Exception as e:
        sys.stderr.write(f"Error writing to stdout: {e}\n")

def check_gpu_availability(silent=False):
    """Check if CUDA GPU is available."""
    try:
        cuda_available = cuda.is_available()
        if cuda_available and not silent:
            # Get device information
            device = cuda.get_current_device()
            print(f"CUDA GPU is available: {device.name}")
            print(f"Compute capability: {device.compute_capability}")
            
            # Try to get some basic device properties
            try:
                mem_info = cuda.current_context().get_memory_info()
                print(f"GPU Memory: Free {mem_info[0]/1024**3:.2f} GB, Total {mem_info[1]/1024**3:.2f} GB")
            except Exception:
                # If we can't get memory info, just log that we're still using the GPU
                print("GPU will still be used")
            
        return cuda_available
    except Exception:
        return False

def detect_system_capabilities(silent=False):
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

def detect_io_capabilities(silent=False):
    """Detect I/O capabilities of the system."""
    try:
        import tempfile
        
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
        if not silent:
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
        
        # Set up argument parser
        parser = argparse.ArgumentParser(description=f"Calculate pairwise Hamming distances. Version: {VERSION}")
        
        # Input/output options
        parser.add_argument("--input", help="Path to the input TSV file")
        parser.add_argument("--output", help="Path to save the output TSV file")
        parser.add_argument("--skip_input_replacements", action="store_true", help="Skip input replacements when there are no strings in the input")
        parser.add_argument("--input_sep", default="\t", help="Input file separator (default: '\\t')")
        parser.add_argument("--output_sep", default="\t", help="Output file separator (default: '\\t')")
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
        parser.add_argument("--silent", action="store_true", help="Disable all console output for maximum performance")
        parser.add_argument("--stdout", action="store_true", help="Write results to stdout instead of a file")
        parser.add_argument("--version", action="version", version=VERSION)

        args = parser.parse_args()
        
        # Check if input/output parameters are provided
        if not args.input:
            parser.print_help()
            return
            
        if not args.output and not args.stdout:
            if not args.silent:
                print("No output specified. Please provide --output or use --stdout.")
            return
        
        # Set number of threads for Numba
        num_threads = args.num_threads
        if not args.silent:
            print(f"Using {num_threads} threads for parallel processing")
        set_num_threads(num_threads)
        
        # Start total timing
        total_start_time = time.time()
        
        # Check GPU availability if requested
        use_gpu = False
        if args.gpu:
            use_gpu = check_gpu_availability(args.silent)
        
        # Print system capabilities if not silent
        if not args.silent:
            # Check I/O capabilities
            io_capabilities = detect_io_capabilities(args.silent)
            print(f"\nSystem capabilities detected:")
            print(f"- CPU cores: {capabilities['cpu_count']}")
            print(f"- GPU available: {'Yes' if capabilities['gpu_available'] else 'No'}")
            print(f"- Available memory: ~{capabilities['available_memory_gb']:.1f} GB")
            print(f"- Disk read speed: ~{io_capabilities['read_speed_mb_per_sec']:.1f} MB/s")
            print(f"- Disk write speed: ~{io_capabilities['write_speed_mb_per_sec']:.1f} MB/s")
            print(f"- Recommended settings: {default_threads} compute threads, {default_io_threads} I/O threads, {default_memory_gb} GB max memory")
            
        # Load data and measure time
        load_start_time = time.time()
        data, loci_stats, sample_stats = load_data_optimized(
            args.input, args.input_sep, args.skip_input_replacements,
            args.locus_completeness, args.sample_completeness,
            args.missing_char, args.chunk_size, args.io_threads, args.silent
        )
        load_end_time = time.time()
        load_time = load_end_time - load_start_time
        
        if data is None:
            return

        n_samples = data.shape[0]
        if not args.silent:
            print(f"\nCalculating distances for {n_samples} samples and {data.shape[1]} allele calls")
            print(f"The final matrix will have {n_samples*n_samples} distances")
        
        # Calculate distances and measure time
        calc_start_time = time.time()
        distances = calculate_distances_batched(data, use_gpu, args.max_memory_gb, args.silent, num_threads=num_threads)
        calc_end_time = time.time()
        calc_time = calc_end_time - calc_start_time
        
        if not args.silent:
            print("\nCalculations completed.")

        # Output results
        save_start_time = time.time()
        if distances is not None:
            if args.stdout:
                # Write directly to stdout for maximum performance
                write_to_stdout(distances, data.index, args.output_sep, args.index_name, args.matrix_format)
            else:
                # Save to file
                if not args.silent:
                    print(f"Saving distances to {args.output}...")
                    
                save_distances_optimized(
                    distances, args.output, data.index, args.output_sep, 
                    args.index_name, args.matrix_format, args.chunk_size, 
                    args.io_threads, args.binary_output, args.silent
                )
        save_end_time = time.time()
        save_time = save_end_time - save_start_time
        
        if not args.silent:
            print("\nProcess completed successfully")

        # Calculate total time
        total_end_time = time.time()
        total_time = total_end_time - total_start_time
        
        # Print performance summary if not silent
        if not args.silent:
            print_performance_summary(load_time, calc_time, save_time, total_time)
            print(f"Total time taken: {total_time:.2f} seconds")

    except Exception as e:
        if not args.silent:
            print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()