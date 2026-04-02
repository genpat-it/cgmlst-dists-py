# cgmlst-dists-py

A high-performance Python implementation of `cgmlst-dists` for calculating pairwise Hamming distances in cgMLST data.

## Overview

This is an enhanced Python implementation of `cgmlst-dists` originally developed by Torsten Seemann. It's designed for calculating pairwise Hamming distances for genome profiles in core genome multilocus sequence typing (cgMLST) schemas.

Key features in this version (0.1.2):

- **GPU Acceleration**: Optional CUDA GPU support for dramatically faster calculations (up to 123x speedup)
- **Vectorized CPU Computation**: NumPy-based vectorized distance calculation with multi-threaded parallelism
- **Optimized Memory Management**: Batch processing to handle large datasets efficiently
- **Multithreaded Processing**: Parallelized calculations across CPU cores (numpy releases the GIL)
- **Intelligent I/O**: Chunked file operations for better performance with large files
- **Advanced Filtering**: Quality control via loci and sample completeness thresholds
- **Automatic System Detection**: Optimizes settings based on available hardware
- **Binary Output Option**: For extremely large matrices

## Installation

### Requirements

- Python 3.9+
- NumPy
- Pandas
- Numba
- tqdm
- psutil

### Optional Requirements

- CUDA Toolkit (for GPU acceleration)

### Install Dependencies

```bash
pip install -r requirements.txt
```

For GPU support, make sure you have a compatible CUDA Toolkit installed.

## Usage

```console
$ python cgmlst-dists.py --help
usage: cgmlst-dists.py [-h] [--input INPUT] [--output OUTPUT] [--skip_input_replacements] 
                       [--input_sep INPUT_SEP] [--output_sep OUTPUT_SEP] [--index_name INDEX_NAME]
                       [--matrix-format {full,lower-tri,upper-tri}] [--num_threads NUM_THREADS] 
                       [--io_threads IO_THREADS] [--max_memory_gb MAX_MEMORY_GB] [--chunk_size CHUNK_SIZE]
                       [--missing_char MISSING_CHAR] [--locus-completeness LOCUS_COMPLETENESS]
                       [--sample-completeness SAMPLE_COMPLETENESS] [--gpu] [--binary-output] [--version]

Calculate pairwise Hamming distances. Version: 0.1.2

options:
  -h, --help            show this help message and exit
  --input INPUT         Path to the input TSV file
  --output OUTPUT       Path to save the output TSV file
  --skip_input_replacements
                        Skip input replacements when there are no strings in the input
  --input_sep INPUT_SEP
                        Input file separator (default: '\t')
  --output_sep OUTPUT_SEP
                        Output file separator (default: '\t')
  --index_name INDEX_NAME
                        Name for the index column (default: 'cgmlst-dists')
  --matrix-format {full,lower-tri,upper-tri}
                        Format for the output matrix (default: full)
  --num_threads NUM_THREADS
                        Number of threads for parallel execution (default: auto-detected)
  --io_threads IO_THREADS
                        Number of I/O threads for file operations
  --max_memory_gb MAX_MEMORY_GB
                        Maximum memory to use in GB for distance calculation
  --chunk_size CHUNK_SIZE
                        Size of chunks for reading/writing files (default: 1000)
  --missing_char MISSING_CHAR
                        Character used for missing data (default: '-')
  --locus-completeness LOCUS_COMPLETENESS
                        Minimum percentage of non-missing data required for a locus (0-100)
  --sample-completeness SAMPLE_COMPLETENESS
                        Minimum percentage of non-missing data required for a sample (0-100)
  --gpu                 Use GPU acceleration when available
  --binary-output       Also save results in binary format for large matrices
  --version            show program's version number and exit
```

## Examples

### Basic Usage

```bash
python cgmlst-dists.py --input input.tsv --output output.tsv
```

### With GPU Acceleration (if available)

```bash
python cgmlst-dists.py --input input.tsv --output output.tsv --gpu
```

### Data Filtering

Filter both loci and samples to include only those with ≥90% data completeness:

```bash
python cgmlst-dists.py --input input.tsv --output output.tsv --locus-completeness 90 --sample-completeness 90
```

### Handling Large Datasets

For very large datasets, optimize memory and I/O:

```bash
python cgmlst-dists.py --input large_data.tsv --output large_output.tsv --max_memory_gb 16 --chunk_size 500 --binary-output
```

## Performance Considerations

- **GPU Acceleration**: Provides dramatic speedup for the distance calculation kernel (up to 123x on NVIDIA L4), requires CUDA-capable NVIDIA GPU
- **CPU Vectorization**: The numpy-based CPU kernel is significantly faster than the previous numba triple-loop approach, scaling well with thread count
- **Memory Usage**: Adjust `--max_memory_gb` based on your system's available RAM to prevent out-of-memory errors
- **I/O Performance**: For large files, increase `--io_threads` on systems with fast storage
- **Binary Output**: Useful for very large matrices (>5000 samples) as it provides faster saving/loading for future analysis

### Performance Benchmarks

#### Test System Specifications

```
CPU: INTEL(R) XEON(R) GOLD 6542Y
CPU Cores: 80
Memory: 480 GB
GPU: NVIDIA L4
GPU Memory: 22 GB
OS: AlmaLinux 10
```

#### Distance Calculation Benchmarks (5,000 samples × 3,000 loci)

| Method | Calc Time | Total Time | Speedup (calc) |
|--------|-----------|------------|----------------|
| v0.1.1 CPU (8 threads, numba) | 55.5s | 64.2s | 1x |
| **v0.1.2 CPU (8 threads, numpy)** | **8.5s** | **17.1s** | **6.5x** |
| v0.1.2 CPU (16 threads, numpy) | 5.1s | 13.9s | 10.9x |
| **v0.1.2 GPU (NVIDIA L4)** | **0.45s** | **9.6s** | **123x** |

#### Distance Calculation Benchmarks (10,000 samples × 3,000 loci)

| Method | Calc Time | Total Time |
|--------|-----------|------------|
| v0.1.1 CPU (8 threads) | 50.8s | 84.9s |
| **v0.1.2 CPU (8 threads)** | **33.9s** | **68.7s** |
| **v0.1.2 GPU (NVIDIA L4)** | **1.3s** | **35.7s** |

#### Large-Scale Test (50,000 samples × 5,000 loci)

| Implementation | Hardware | Runtime | Notes |
|----------------|----------|---------|-------|
| Original C version | 16-core CPU | Failed | Out of memory error |
| Python CPU version | 16-core CPU | ~32 minutes | Full processing time |
| Python GPU version | NVIDIA L4 GPU | ~12 minutes | Full processing time |

Both CPU and GPU implementations produce identical output (verified via MD5 checksum).

## Docker

### Build

```bash
docker build -t cgmlst-dists-py .
```

### Run

```bash
docker run --rm -v "$(pwd):/app/data" cgmlst-dists-py --input data/input.tab --output data/output.tab
```

### With GPU Support

```bash
docker run --rm --gpus all -v "$(pwd):/app/data" cgmlst-dists-py --input data/input.tab --output data/output.tab --gpu
```

## Advantages Over Original Implementation

1. **Scalability**: Efficiently handles much larger datasets through batch processing and memory optimization
2. **Speed**: Significantly faster for large matrices through multithreading and optional GPU acceleration
3. **Data Quality**: Advanced filtering options for more accurate analysis
4. **Hardware Optimization**: Auto-detects and adapts to available system resources
5. **More Output Options**: Supports binary format for very large matrices

## Limitations

- Requires more dependencies than the C implementation
- More complex configuration options (though with sensible defaults)
- GPU acceleration requires CUDA-capable NVIDIA graphics card

## Citation

If you use this tool in your research, please cite the original cgmlst-dists tool:

Seemann T, cgmlst-dists: https://github.com/tseemann/cgmlst-dists/

## License

This project is licensed under the same terms as the original cgmlst-dists.

## Contact

Please submit issues and feature requests through the GitHub repository.