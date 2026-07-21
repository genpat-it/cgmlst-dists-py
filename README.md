# cgmlst-dists-py

[![Release](https://img.shields.io/github/v/release/genpat-it/cgmlst-dists-py)](https://github.com/genpat-it/cgmlst-dists-py/releases)
[![Test and Publish Docker](https://github.com/genpat-it/cgmlst-dists-py/actions/workflows/release.yml/badge.svg)](https://github.com/genpat-it/cgmlst-dists-py/actions/workflows/release.yml)
[![Docker](https://img.shields.io/badge/docker-ghcr.io-blue)](https://ghcr.io/genpat-it/cgmlst-dists-py)
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![Bioconda](https://img.shields.io/conda/vn/bioconda/cgmlst-dists-py.svg)](https://bioconda.github.io/recipes/cgmlst-dists-py/README.html)

A high-performance Python implementation of `cgmlst-dists` for calculating pairwise Hamming distances in cgMLST data.

## Installation

### Bioconda (recommended)

```bash
conda install -c bioconda cgmlst-dists-py
```

### Docker

```bash
docker pull ghcr.io/genpat-it/cgmlst-dists-py
```

### From source

```bash
git clone https://github.com/genpat-it/cgmlst-dists-py.git
cd cgmlst-dists-py
pip install -r requirements.txt
```

For GPU support, make sure you have a compatible CUDA Toolkit installed.

## Overview

This is an enhanced Python implementation of `cgmlst-dists` originally developed by Torsten Seemann. It's designed for calculating pairwise Hamming distances for genome profiles in core genome multilocus sequence typing (cgMLST) schemas.

Key features in this version (0.1.3):

- **GPU Acceleration**: Optional CUDA GPU support for dramatically faster calculations (up to 123x speedup)
- **Vectorized CPU Computation**: NumPy-based vectorized distance calculation with multi-threaded parallelism
- **Optimized Memory Management**: Batch processing to handle large datasets efficiently
- **Multithreaded Processing**: Parallelized calculations across CPU cores (numpy releases the GIL)
- **Intelligent I/O**: Chunked file operations for better performance with large files
- **Advanced Filtering**: Quality control via loci and sample completeness thresholds
- **Automatic System Detection**: Optimizes settings based on available hardware
- **Binary Output Option**: For extremely large matrices

## Usage

```console
$ python cgmlst-dists.py --help
usage: cgmlst-dists.py [-h] [--input INPUT] [--output OUTPUT] [--skip_input_replacements] 
                       [--input_sep INPUT_SEP] [--output_sep OUTPUT_SEP] [--index_name INDEX_NAME]
                       [--matrix-format {full,lower-tri,upper-tri}] [--num_threads NUM_THREADS] 
                       [--io_threads IO_THREADS] [--max_memory_gb MAX_MEMORY_GB] [--chunk_size CHUNK_SIZE]
                       [--missing_char MISSING_CHAR] [--locus-completeness LOCUS_COMPLETENESS]
                       [--sample-completeness SAMPLE_COMPLETENESS] [--gpu] [--binary-output] [--version]

Calculate pairwise Hamming distances. Version: 0.1.3

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

> **Memory requirements (important).** The full distance matrix is held in RAM
> as `int32`, so it needs roughly **N² × 4 bytes**. For example ~1 GiB for
> 16k samples, ~16 GiB for 63k samples, ~63 GiB for 126k samples. Since 0.1.5
> the tool estimates this **before** the distance computation and aborts
> immediately with a clear message if it will not fit in available RAM,
> instead of crashing after minutes of work. Run on a machine with more RAM,
> reduce the number of samples, or pass `--force` to attempt it anyway.

### Writing to stdout

`--stdout` streams the matrix to standard output (all logs and progress go to
`stderr`), so you can safely redirect it to a file or pipe it to another tool:

```bash
python cgmlst-dists.py --input data.tsv --stdout > matrix.tsv
```

Do not run `--stdout` without redirecting on large datasets, or the terminal
will be flooded with the full N×N matrix.

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
| **v0.1.3 CPU (8 threads, numpy)** | **8.5s** | **17.1s** | **6.5x** |
| v0.1.3 CPU (16 threads, numpy) | 5.1s | 13.9s | 10.9x |
| **v0.1.3 GPU (NVIDIA L4)** | **0.45s** | **9.6s** | **123x** |

#### Distance Calculation Benchmarks (10,000 samples × 3,000 loci)

| Method | Calc Time | Total Time |
|--------|-----------|------------|
| v0.1.1 CPU (8 threads) | 50.8s | 84.9s |
| **v0.1.3 CPU (8 threads)** | **33.9s** | **68.7s** |
| **v0.1.3 GPU (NVIDIA L4)** | **1.3s** | **35.7s** |

#### Large-Scale Test (50,000 samples × 5,000 loci)

| Implementation | Hardware | Runtime | Notes |
|----------------|----------|---------|-------|
| Original C version | 16-core CPU | Failed | Out of memory error |
| Python CPU version | 16-core CPU | ~32 minutes | Full processing time |
| Python GPU version | NVIDIA L4 GPU | ~12 minutes | Full processing time |

Both CPU and GPU implementations produce identical output (verified via MD5 checksum).

## Docker Usage

Basic run — mount your working directory into the container and point `--input`/`--output` at it:

```bash
docker run --rm -v "$(pwd):/app/data" ghcr.io/genpat-it/cgmlst-dists-py --input data/input.tab --output data/output.tab
```

### Running as the current user

By default the container runs as `root`, so any output file it writes will be owned by `root`.
To keep the output owned by you, run the container as your own user:

```bash
docker run --rm -u "$(id -u):$(id -g)" -v "$(pwd):/app/data" \
  ghcr.io/genpat-it/cgmlst-dists-py --input data/input.tab --output data/output.tab
```

> **Note (fixed in 0.1.4):** In versions **≤ 0.1.3** running with `-u` crashed at startup with
> `cannot cache function ...: no locator available for file '/app/cgmlst-dists.py'`. This was a
> [numba](https://numba.pydata.org/) JIT cache issue: as a non-root user neither `/app` nor `$HOME`
> are writable, so numba had nowhere to store its compiled cache.
> If you are stuck on an older image, work around it by pointing the cache at a writable dir:
> `docker run -u "$(id -u):$(id -g)" -e NUMBA_CACHE_DIR=/tmp ...` (or simply omit `-u` to run as root).
> From 0.1.4 the entrypoint sets a writable `NUMBA_CACHE_DIR` automatically, so no extra flags are needed.

### Numba cache directory

The tool uses [numba](https://numba.pydata.org/) JIT compilation, which stores its compiled cache
on disk in the directory given by the `NUMBA_CACHE_DIR` environment variable.

- **Default (nothing to do):** if you don't set `NUMBA_CACHE_DIR`, the container creates a fresh,
  writable temporary directory on each run. This "just works" for any user, including
  `-u $UID` and UIDs with no `/etc/passwd` entry.
- **Override it yourself:** set `NUMBA_CACHE_DIR` to any path the container user can write to.
  This is useful to keep a **persistent** cache across runs (so numba doesn't recompile every time)
  by mounting a host directory:

  ```bash
  mkdir -p ./numba-cache
  docker run --rm -u "$(id -u):$(id -g)" \
    -e NUMBA_CACHE_DIR=/cache -v "$(pwd)/numba-cache:/cache" \
    -v "$(pwd):/app/data" ghcr.io/genpat-it/cgmlst-dists-py \
    --input data/input.tab --output data/output.tab
  ```

### GPU support

> **The published Docker image runs on CPU only.** It is based on
> `python:3.13-slim`, which does not ship the CUDA runtime that numba needs, so
> even with `--gpus all` the tool reports `GPU available: No` and falls back to
> the (fast) multi-threaded CPU kernel. GPU acceleration is available when
> running **from source / conda** on a host with a CUDA-capable NVIDIA GPU and
> the CUDA Toolkit installed. Running a GPU-enabled container would require an
> image built on a CUDA base (e.g. `nvidia/cuda:*-runtime`), which is not
> currently provided.

Note: in `docker run`, `--gpus all` is a **Docker** flag (it exposes host GPUs
to the container) while `--gpu` is the **tool** flag (it asks the program to use
a GPU). They are different flags and both would be needed for GPU runs:

```bash
# Requires a CUDA-based image AND the NVIDIA Container Toolkit on the host.
docker run --rm --gpus all -v "$(pwd):/app/data" ghcr.io/genpat-it/cgmlst-dists-py --input data/input.tab --output data/output.tab --gpu
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