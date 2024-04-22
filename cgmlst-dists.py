import os

import argparse
import pandas as pd
import numpy as np
import time
from numba import jit, prange, set_num_threads

DEFAULT_THREADS = max(1, os.cpu_count() // 2 + 1)
VERSION = "0.0.2"

def load_data(file_path, input_sep="\t", crc32=False):
    """Load data from a TSV file."""
    try:
        data = pd.read_csv(file_path, sep=input_sep, index_col=0)
        if not crc32:
            data = data.replace(r'INF-(\d+)', r'\1', regex=True)
            data = data.apply(pd.to_numeric, errors='coerce').fillna(0).astype(int)
        else:
            data = data.apply(pd.to_numeric, errors='coerce').astype(int)
        return data
    except Exception as e:
        print(f"Error loading data: {e}")
        return None

@jit(nopython=True, parallel=True)
def calculate_hamming_distances_numba(values):
    n_samples = values.shape[0]
    distances = np.zeros((n_samples, n_samples), dtype=np.int64)
    for i in prange(n_samples):
        for j in prange(i + 1, n_samples):
            dist = 0
            for k in range(values.shape[1]):
                if (values[i, k] != values[j, k]) and (values[i, k] != 0) and (values[j, k] != 0):
                    dist += 1
            distances[i, j] = dist
            distances[j, i] = dist
    return distances

def calculate_hamming_distances(data):
    """Wrapper function to use Numba-optimized distance calculation."""
    try:
        values = data.values
        diff_count = calculate_hamming_distances_numba(values)
        if diff_count is not None:
            np.fill_diagonal(diff_count, 0)
        return diff_count
    except Exception as e:
        print(f"Error calculating Hamming distances: {e}")
        return None

def save_distances(distances, file_path, index, output_sep="\t", index_name="cgmlst-dists", matrix_format="full", chunk_size=1000):
    """Save pairwise distances to a TSV file."""
    try:
        if distances is not None:
            distance_df = pd.DataFrame(distances, index=index, columns=index)
            if matrix_format == "lower-tri":
                np.fill_diagonal(distance_df.values, 0)
                distance_df = np.tril(distance_df)
            elif matrix_format == "upper-tri":
                np.fill_diagonal(distance_df.values, 0)
                distance_df = np.triu(distance_df)
            distance_df.index.name = index_name
            
            with open(file_path, 'w') as f:
                for i in range(0, len(distance_df), chunk_size):
                    distance_df.iloc[i:i+chunk_size].to_csv(f, sep=output_sep, mode='a', header=(i==0))
        else:
            print("No distances to save.")
    except Exception as e:
        print(f"Error saving distances: {e}")

def main():
    try:
        parser = argparse.ArgumentParser(description=f"Calculate pairwise Hamming distances. Version: {VERSION}")
        parser.add_argument("--input", help="Path to the input TSV file")
        parser.add_argument("--output", help="Path to save the output TSV file")
        parser.add_argument("--crc32", action="store_true", help="Skip input transformations")
        parser.add_argument("--input_sep", default="\t", help="Input file separator (default: '\t')")
        parser.add_argument("--output_sep", default="\t", help="Output file separator (default: '\t')")
        parser.add_argument("--index_name", default="cgmlst-dists", help="Name for the index column (default: 'cgmlst-dists')")
        parser.add_argument("--matrix-format", choices=["full", "lower-tri", "upper-tri"], default="full", help="Format for the output matrix (default: full)")
        parser.add_argument("--num_threads", type=int, default=DEFAULT_THREADS, help=f"Number of threads for parallel execution (default: {DEFAULT_THREADS} cpus)")
        parser.add_argument("--chunk_size", type=int, default=1000, help="Size of chunks to save the output file (default: 1000)")
        parser.add_argument("--version", action="version", version=VERSION)

        args = parser.parse_args()
        
        set_num_threads(int(args.num_threads))

        if not args.input or not args.output:
            parser.print_help()
            return
            
        start_time = time.time()

        data = load_data(args.input, args.input_sep, args.crc32)
        if data is None:
            return

        n_samples = data.shape[0]
        print(f"Loaded matrix of {n_samples} samples and {data.shape[1]} allele calls.")
        print(f"The final matrix will have {n_samples*n_samples} distances.")
        
        distances = calculate_hamming_distances(data)
        print("Calculations completed. Saving distances...")

        if distances is not None:
            save_distances(distances, args.output, data.index, args.output_sep, args.index_name, args.matrix_format, args.chunk_size)
        
        print("Process completed successfully.")

        total_time = time.time() - start_time
        print(f"\nTotal time taken: {total_time:.2f} seconds")

    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
