# cgmlst-data-generator

A multithreaded utility for generating synthetic cgMLST allelic profiles for testing and benchmarking purposes.

## Overview

This Python tool creates tab-separated values (TSV) files containing synthetic core genome multilocus sequence typing (cgMLST) data with customizable parameters. It's designed to generate test datasets of varying sizes and characteristics to benchmark distance calculation tools like `cgmlst-dists`.

## Features

- **Configurable Dataset Size**: Generate any number of samples and loci
- **Adjustable Missing Data**: Control the percentage of missing alleles
- **Multithreaded Generation**: Utilizes parallel processing for faster generation of large datasets
- **Memory-Efficient**: Chunk-based file writing to manage memory usage with very large datasets
- **Progress Tracking**: Visual progress bar during generation
- **Realistic Format**: Creates TSV files compatible with real cgMLST analysis tools

## Requirements

- Python 3.9+
- NumPy (2.2.3)
- Pandas (2.2.3)
- tqdm (4.67.1)

## Installation

```bash
pip install -r requirements.txt
```

Create a `requirements.txt` file containing:
```
numpy==2.2.3
pandas==2.2.3
tqdm==4.67.1
```

## Usage

```console
$ python cgmlst-data-generator.py --help
usage: cgmlst-data-generator.py [-h] [--samples SAMPLES] [--loci LOCI] [--missing MISSING]
                               [--output OUTPUT] [--threads THREADS]

Generate synthetic cgMLST data for testing

options:
  -h, --help           show this help message and exit
  --samples SAMPLES    Number of samples to generate (default: 100)
  --loci LOCI          Number of loci to generate (default: 1000)
  --missing MISSING    Percentage of missing data (0-100) (default: 5.0)
  --output OUTPUT      Output file path (default: test_data.tsv)
  --threads THREADS    Number of threads to use (default: CPU count)
```

## Examples

### Generate a Small Test Dataset

```bash
python cgmlst-data-generator.py --samples 50 --loci 500 --output small_test.tsv
```

### Generate a Large Dataset with More Missing Data

```bash
python cgmlst-data-generator.py --samples 5000 --loci 2000 --missing 10 --output large_test.tsv
```

### Specify Thread Count

```bash
python cgmlst-data-generator.py --samples 1000 --loci 3000 --threads 8 --output medium_test.tsv
```

## Output Format

The generator creates a TSV file with the following structure:

- **First column**: Sample IDs (randomly generated with prefix "Sample_")
- **First row**: Locus names (formatted as "locus_1", "locus_2", etc.)
- **Data cells**: Random allele numbers (1-99) or "-" for missing data
- **File extension**: .tsv (tab-separated values)

Example output:
```
          locus_1 locus_2 locus_3 ...
Sample_A2B3C4D5   45      -       12  ...
Sample_F6G7H8I9   23      67      -   ...
...
```

## Performance Considerations

- **Memory Usage**: For very large datasets (>1000 samples × >5000 loci), ensure your system has sufficient RAM
- **Thread Count**: Using more threads speeds up generation but with diminishing returns beyond your CPU's core count
- **Disk Space**: Large datasets can create sizeable files; check the estimated file size in the output summary

## Use with cgmlst-dists

The generated data is compatible with cgmlst-dists and cgmlst-dists-py. After generating a dataset, you can use it for benchmarking:

```bash
# With original cgmlst-dists
cgmlst-dists test_data.tsv > distances.tsv

# With Python implementation
python cgmlst-dists.py --input test_data.tsv --output distances.tsv --gpu
```

## Benchmark Example

### Generating a Large Test Dataset

The following benchmark demonstrates generating a very large dataset (50,000 samples × 5,000 loci):

```bash
$ python cgmlst-data-generator.py --samples 50000 --loci 5000 --output 50k5k.tsv --threads 16
```

### Benchmark Results

The large dataset was then processed using both CPU and GPU implementations:

```bash
# GPU processing
$ time python cgmlst-dists.py --input /mnt/disk2/a.deruvo/50k5k.tsv --output /mnt/disk2/a.deruvo/50k5k_distance_gpu.tsv --gpu

System capabilities detected:
- CPU cores: 16
- GPU available: Yes
- Available memory: ~8.0 GB
- Disk read speed: ~2256.6 MB/s
- Disk write speed: ~1430.6 MB/s
- Recommended settings: 16 compute threads, 4 I/O threads, 6 GB max memory
Using 16 threads for parallel processing
CUDA GPU is available: b'NVIDIA L4'
Compute capability: (8, 9)
GPU Memory: Free 21.86 GB, Total 22.05 GB

Loading data from /mnt/disk2/a.deruvo/50k5k.tsv (683.6 MB)...
Reading file metadata...
Estimating file size...
File contains approximately 50,000 rows and 5,000 columns
Reading file in chunks of 1,000 rows using 4 I/O threads...
Combining chunks...
Initial data shape: 50000 samples × 5000 loci
Converting data to numeric format...

Final data shape after filtering: 50000 samples × 5000 loci
Data loading time: 142.78 seconds

Calculating distances for 50000 samples and 5000 allele calls
The final matrix will have 2500000000 distances
Processing distances in batches of ~40,132 samples
Total batches: 2
Mirroring distance matrix...
Total distance calculation time: 215.59 seconds

Calculations completed. Saving distances...
Saving distance matrix (50000 x 50000, ~9536.7 MB) to /mnt/disk2/a.deruvo/50k5k_distance_gpu.tsv
Preparing chunks for writing...
Writing 50 chunks to file using 4 I/O threads...
Total save process time: 364.02 seconds

Process completed successfully

==================================================
PERFORMANCE SUMMARY
==================================================
Operation                 Time (s)   Percentage
--------------------------------------------------
Data loading              142.80     19.76    %
Distance calculation      215.59     29.84    %
Saving results            364.04     50.39    %
Other operations          0.08       0.01     %
--------------------------------------------------
Total                     722.52     100.00   %
==================================================
Total time taken: 722.52 seconds
```

### Original C Implementation Attempt

```bash
$ time ./cgmlst-dists /mnt/disk2/a.deruvo/50k5k.tsv > 50k5k_distance_tseeman
This is cgmlst-dists 0.4.0
Loaded 50000 samples x 5000 allele calls
ERROR: could not allocate 18014398502470393 kb RAM
real    0m5.985s
user    0m4.900s
sys     0m0.929s
```

### Output Validation

Validating that both CPU and GPU implementations produce identical results:

```bash
$ md5sum /mnt/disk2/a.deruvo/50k5k_distance_cpu.tsv
ff0ba122315ffc22f56023cb03c0c04f  /mnt/disk2/a.deruvo/50k5k_distance_cpu.tsv
$ md5sum /mnt/disk2/a.deruvo/50k5k_distance_gpu.tsv
ff0ba122315ffc22f56023cb03c0c04f  /mnt/disk2/a.deruvo/50k5k_distance_gpu.tsv
```

## Example Output (Small Dataset)

```
Generating data with 8 threads...
Generating sample IDs...
Generating data for 1000 loci using 8 threads...
Generating loci: 100%|██████████| 1000/1000 [00:02<00:00, 448.22it/s]
Saving data to test_data.tsv...

Generated test data with:
- 100 samples
- 1000 loci
- 5% missing data
- Total cells: 100,000
- File saved to: test_data.tsv
- Generation time: 3.74 seconds
- Estimated file size: 0.31 MB

Example command to process this data:
python cgmlst-dists.py --input test_data.tsv --output distances.tsv --gpu
```