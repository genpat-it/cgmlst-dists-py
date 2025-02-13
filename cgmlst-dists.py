import os
import argparse
from typing import Optional
import pandas as pd
import numpy as np
import time
from numba import jit, prange, set_num_threads

DEFAULT_THREADS = max(1, os.cpu_count() // 2 + 1)
VERSION = "0.0.3"

def filter_loci_by_completeness(data: pd.DataFrame, missing_char: str, min_completeness: float) -> tuple[list, dict]:
    """Filter loci based on completeness threshold.
    
    Args:
        data: DataFrame with allelic profiles
        missing_char: Character used for missing data
        min_completeness: Minimum percentage of non-missing data required (0-100)
        
    Returns:
        Tuple of (filtered_loci_list, loci_statistics)
    """
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
    """Filter samples based on completeness threshold.
    
    Args:
        data: DataFrame with allelic profiles
        missing_char: Character used for missing data
        min_completeness: Minimum percentage of non-missing data required (0-100)
        
    Returns:
        Tuple of (filtered_dataframe, sample_statistics)
    """
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

def load_data(file_path: str, input_sep: str = "\t", skip_input_replacements: bool = False, 
              min_locus_completeness: float = None, min_sample_completeness: float = None,
              missing_char: str = "-") -> tuple[Optional[pd.DataFrame], Optional[dict], Optional[dict]]:
    """Load data from a TSV file with optimized performance while maintaining exact output.
    
    Args:
        file_path: Path to input file
        input_sep: Input file separator (default: tab)
        skip_input_replacements: Skip string replacements for numeric-only data
        min_locus_completeness: Minimum percentage for locus inclusion (0-100)
        min_sample_completeness: Minimum percentage for sample inclusion (0-100)
        missing_char: Character used for missing data
    
    Returns:
        Tuple of (DataFrame or None if error occurs, loci_statistics, sample_statistics)
    """
    try:
        print(f"\nLoading data from {file_path}...")
        data = pd.read_csv(file_path, sep=input_sep, index_col=0, low_memory=False)
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
        
        print(f"\nFinal data shape after filtering: {data.shape[0]} samples × {data.shape[1]} loci")
        
        # Process data based on skip_input_replacements flag
        if not skip_input_replacements:
            data.replace(r'^INF-', '', regex=True, inplace=True)
            data = pd.to_numeric(data.stack(), errors='coerce').unstack().fillna(0)
        else:
            data = data.apply(pd.to_numeric, errors='coerce').astype(int)
            
        return data, loci_stats, sample_stats
        
    except Exception as e:
        print(f"Error loading data: {e}")
        return None, None, None

@jit(nopython=True, parallel=True, fastmath=True)
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
        parser.add_argument("--skip_input_replacements", action="store_true", help="Skip input replacements when there are no strings in the input")
        parser.add_argument("--input_sep", default="\t", help="Input file separator (default: '\t')")
        parser.add_argument("--output_sep", default="\t", help="Output file separator (default: '\t')")
        parser.add_argument("--index_name", default="cgmlst-dists", help="Name for the index column (default: 'cgmlst-dists')")
        parser.add_argument("--matrix-format", choices=["full", "lower-tri", "upper-tri"], default="full", help="Format for the output matrix")
        parser.add_argument("--num_threads", type=int, default=DEFAULT_THREADS, help=f"Number of threads for parallel execution")
        parser.add_argument("--chunk_size", type=int, default=1000, help="Size of chunks to save the output file")
        parser.add_argument("--missing_char", default="-", help="Character used for missing data (default: '-')")
        parser.add_argument("--locus-completeness", type=float, default=None, 
                          help="Minimum percentage of non-missing data required for a locus (0-100)")
        parser.add_argument("--sample-completeness", type=float, default=None,
                          help="Minimum percentage of non-missing data required for a sample (0-100)")
        parser.add_argument("--version", action="version", version=VERSION)

        args = parser.parse_args()
        
        set_num_threads(int(args.num_threads))

        if not args.input or not args.output:
            parser.print_help()
            return
            
        start_time = time.time()

        data, loci_stats, sample_stats = load_data(
            args.input, args.input_sep, args.skip_input_replacements,
            args.locus_completeness, args.sample_completeness,
            args.missing_char
        )
        
        if data is None:
            return

        n_samples = data.shape[0]
        print(f"\nCalculating distances for {n_samples} samples and {data.shape[1]} allele calls")
        print(f"The final matrix will have {n_samples*n_samples} distances")
        
        distances = calculate_hamming_distances(data)
        print("\nCalculations completed. Saving distances...")

        if distances is not None:
            save_distances(distances, args.output, data.index, args.output_sep, 
                         args.index_name, args.matrix_format, args.chunk_size)
        
        print("\nProcess completed successfully")

        total_time = time.time() - start_time
        print(f"Total time taken: {total_time:.2f} seconds")

    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()