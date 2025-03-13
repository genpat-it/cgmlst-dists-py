#!/usr/bin/env python3
"""
Generate synthetic cgMLST allelic profiles for testing purposes.
This script creates a tab-separated values (TSV) file with a 
configurable number of samples and loci, using multithreading for performance.
"""

import argparse
import numpy as np
import pandas as pd
import random
import string
import time
import concurrent.futures
import os
from tqdm import tqdm

def generate_sample_id(prefix="Sample_", length=8):
    """Generate a random sample ID."""
    suffix = ''.join(random.choices(string.ascii_uppercase + string.digits, k=length))
    return f"{prefix}{suffix}"

def generate_column_data(locus_name, n_samples, missing_percentage):
    """Generate data for a single locus column."""
    # Generate random allele numbers (1-100)
    data = np.random.randint(1, 100, size=n_samples).astype(object)
    
    # Calculate how many values should be missing in this column
    n_missing = int((missing_percentage / 100) * n_samples)
    
    if n_missing > 0:
        # Randomly select which entries to mark as missing
        missing_indices = np.random.choice(n_samples, n_missing, replace=False)
        data[missing_indices] = "-"
    
    return locus_name, data

def generate_test_data(n_samples, n_loci, missing_percentage=5, output_file="test_data.tsv", num_threads=None):
    """
    Generate synthetic cgMLST allelic profile data using multithreading.
    
    Args:
        n_samples: Number of samples to generate
        n_loci: Number of loci to generate
        missing_percentage: Percentage of missing data (0-100)
        output_file: Output file path
        num_threads: Number of threads to use (default: CPU count)
    """
    start_time = time.time()
    
    if num_threads is None:
        num_threads = os.cpu_count() or 4
    
    print(f"Generating data with {num_threads} threads...")
    
    # Generate sample IDs
    print("Generating sample IDs...")
    sample_ids = [generate_sample_id() for _ in range(n_samples)]
    
    # Generate locus names
    locus_names = [f"locus_{i+1}" for i in range(n_loci)]
    
    # Create an empty DataFrame with object dtype
    df = pd.DataFrame(index=sample_ids, columns=locus_names, dtype=object)
    
    # Generate column data in parallel
    print(f"Generating data for {n_loci} loci using {num_threads} threads...")
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
        # Submit all column generation tasks
        future_to_locus = {
            executor.submit(generate_column_data, locus, n_samples, missing_percentage): locus
            for locus in locus_names
        }
        
        # Process completed tasks with a progress bar
        with tqdm(total=n_loci, desc="Generating loci") as pbar:
            for future in concurrent.futures.as_completed(future_to_locus):
                locus_name, data = future.result()
                df[locus_name] = data
                pbar.update(1)
    
    # Save to file
    print(f"Saving data to {output_file}...")
    chunk_size = 1000  # Save in chunks to avoid memory issues with large datasets
    
    for i in range(0, len(df), chunk_size):
        if i == 0:
            # First chunk, create file
            df.iloc[i:i+chunk_size].to_csv(output_file, sep='\t')
        else:
            # Append to existing file
            df.iloc[i:i+chunk_size].to_csv(output_file, sep='\t', mode='a', header=False)
    
    end_time = time.time()
    elapsed_time = end_time - start_time
    
    # Print summary
    print(f"\nGenerated test data with:")
    print(f"- {n_samples} samples")
    print(f"- {n_loci} loci")
    print(f"- {missing_percentage}% missing data")
    print(f"- Total cells: {n_samples * n_loci:,}")
    print(f"- File saved to: {output_file}")
    print(f"- Generation time: {elapsed_time:.2f} seconds")
    
    # Calculate estimated file size
    avg_sample_id_len = len("Sample_XXXXXXXX")
    avg_locus_name_len = len("locus_1000")
    avg_allele_len = 2  # 2 characters per allele on average
    
    estimated_size_bytes = (
        (avg_sample_id_len * n_samples) +  # Sample IDs
        (avg_locus_name_len * n_loci) +    # Locus names
        (avg_allele_len * n_samples * n_loci)  # Allele data
    )
    
    estimated_size_mb = estimated_size_bytes / (1024 * 1024)
    print(f"- Estimated file size: {estimated_size_mb:.2f} MB")
    
    # Print example command to process this data
    print("\nExample command to process this data:")
    print(f"python cgmlst_distance.py --input {output_file} --output distances.tsv --gpu")

def main():
    parser = argparse.ArgumentParser(description="Generate synthetic cgMLST data for testing")
    parser.add_argument("--samples", type=int, default=100, help="Number of samples to generate")
    parser.add_argument("--loci", type=int, default=1000, help="Number of loci to generate")
    parser.add_argument("--missing", type=float, default=5.0, help="Percentage of missing data (0-100)")
    parser.add_argument("--output", default="test_data.tsv", help="Output file path")
    parser.add_argument("--threads", type=int, default=None, help="Number of threads to use (default: CPU count)")
    
    args = parser.parse_args()
    
    generate_test_data(args.samples, args.loci, args.missing, args.output, args.threads)

if __name__ == "__main__":
    main()