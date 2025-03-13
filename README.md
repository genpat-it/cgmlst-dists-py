# cgmlst-dists-py

A high-performance Python implementation of `cgmlst-dists` for calculating pairwise Hamming distances in cgMLST data.

## Overview

This is an enhanced Python implementation of `cgmlst-dists` originally developed by Torsten Seemann. It's designed for calculating pairwise Hamming distances for genome profiles in core genome multilocus sequence typing (cgMLST) schemas.

Key features in this version (0.1.0):

- **GPU Acceleration**: Optional CUDA GPU support for faster calculations
- **Optimized Memory Management**: Batch processing to handle large datasets efficiently
- **Multithreaded Processing**: Parallelized calculations across CPU cores
- **Intelligent I/O**: Chunked file operations for better performance with large files
- **Advanced Filtering**: Quality control via loci and sample completeness thresholds
- **Automatic System Detection**: Optimizes settings based on available hardware
- **Binary Output Option**: For extremely large matrices

## Installation

### Requirements

- Python 3.9+
- NumPy (2.2.3)
- Pandas (2.2.3)
- Numba (0.61.0)
- tqdm (4.67.1)
- psutil (7.0.0)

### Optional Requirements

- CUDA Toolkit (for GPU acceleration)

### Install Dependencies

```bash
pip install -r requirements.txt
```

Create a `requirements.txt` file containing:
```
numba==0.61.0
numpy==2.2.3
pandas==2.2.3
psutil==7.0.0
tqdm==4.67.1
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

Calculate pairwise Hamming distances. Version: 0.1.0

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

- **GPU Acceleration**: Can provide significant speedup (5-20x) for large matrices compared to single-threaded CPU, but requires CUDA-capable NVIDIA GPU
- **Memory Usage**: Adjust `--max_memory_gb` based on your system's available RAM to prevent out-of-memory errors
- **I/O Performance**: For large files, increase `--io_threads` on systems with fast storage
- **Binary Output**: Useful for very large matrices (>5000 samples) as it provides faster saving/loading for future analysis

### Performance Benchmarks

#### Test System Specifications

```
CPU: INTEL(R) XEON(R) GOLD 6542Y
CPU Cores: 16
Memory: 62Gi
GPU: NVIDIA L4
GPU Memory: 23034 MiB
OS: AlmaLinux 9.5 (Teal Serval)
Kernel: 5.14.0-503.23.2.el9_5.x86_64
Storage: 1.5T total, 1.4T available
```

#### Large-Scale Test (50,000 samples × 5,000 loci)

| Implementation | Hardware | Runtime | Notes |
|----------------|----------|---------|-------|
| Original C version | 16-core CPU | Failed | Out of memory error |
| Python CPU version | 16-core CPU | ~32 minutes | Full processing time |
| Python GPU version | NVIDIA L4 GPU | ~12 minutes | Full processing time |

The test dataset was generated using the companion `cgmlst-data-generator.py` tool:
```bash
python cgmlst-data-generator.py --samples 50000 --loci 5000 --output 50k5k.tsv --threads 16
```

Key performance metrics from GPU run:
- Data loading: ~2.4 minutes
- Distance calculation: ~3.6 minutes
- Saving results: ~6 minutes
- Total runtime: ~12 minutes

The original C implementation couldn't handle this dataset size, reporting:
```
ERROR: could not allocate 18014398502470393 kb RAM
```

Both CPU and GPU implementations produced identical output files (verified via MD5 checksum), demonstrating accuracy alongside performance gains.

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