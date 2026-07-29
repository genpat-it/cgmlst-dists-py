# cgmlst-dists-py

[![Release](https://img.shields.io/github/v/release/genpat-it/cgmlst-dists-py)](https://github.com/genpat-it/cgmlst-dists-py/releases)
[![CI](https://github.com/genpat-it/cgmlst-dists-py/actions/workflows/ci.yml/badge.svg)](https://github.com/genpat-it/cgmlst-dists-py/actions/workflows/ci.yml)
[![Publish](https://github.com/genpat-it/cgmlst-dists-py/actions/workflows/release.yml/badge.svg)](https://github.com/genpat-it/cgmlst-dists-py/actions/workflows/release.yml)
[![Docker](https://img.shields.io/badge/docker-ghcr.io-blue)](https://ghcr.io/genpat-it/cgmlst-dists-py)
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![Bioconda](https://img.shields.io/conda/vn/bioconda/cgmlst-dists-py.svg)](https://bioconda.github.io/recipes/cgmlst-dists-py/README.html)
[![Bioconda downloads](https://img.shields.io/conda/dn/bioconda/cgmlst-dists-py.svg)](https://anaconda.org/bioconda/cgmlst-dists-py)

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

`pyarrow` is included in `requirements.txt` and enables an Arrow-based loader
(multithreaded C++) that is 3-4x faster at reading and parsing large inputs. It
is detected at runtime, so the tool still works without it, falling back to the
pandas loader with identical results: if pyarrow cannot be installed on your
platform, drop it from `requirements.txt` and everything keeps working, just
more slowly.

```bash
# minimal install, without the Arrow loader
pip install numpy pandas tqdm psutil
```

**GPU users only:** `numba` is not in `requirements.txt`, because the CPU path
computes with numpy and never imports it, while numba plus llvmlite weigh ~199 MB
and constrain which Python versions are supported. Install it when you want
`--gpu`:

```bash
pip install -r requirements-gpu.txt
```

### GPU support

GPU acceleration needs `numba` (see above, it is not installed by default) and an
NVIDIA driver on the host. The CUDA components themselves come with numba, so no
separate toolkit install is required. Without numba, or without a usable device,
`--gpu` warns on stderr and computes on the CPU with identical results.

| Install route | `--gpu` |
|---|---|
| Bioconda | supported — needs only an NVIDIA driver on the host |
| From source | supported — install `requirements-gpu.txt`, plus an NVIDIA driver |
| Docker, published image | **not supported**: CPU-only, see [GPU support in Docker](#gpu-support-in-docker) |
| Docker, `Dockerfile.gpu` | supported — build it yourself, it is not published |

Passing `--gpu` without a usable device is not an error: the tool warns on stderr
and computes on the CPU, producing identical results. Verified on an NVIDIA L4
(driver 590.48, compute capability 8.9), where the GPU and CPU matrices are
byte-identical. The benchmark figures below were measured on the same card.

## Overview

This is an enhanced Python implementation of `cgmlst-dists` originally developed by Torsten Seemann. It's designed for calculating pairwise Hamming distances for genome profiles in core genome multilocus sequence typing (cgMLST) schemas.

Key features in this version (0.1.7):

- **GPU Acceleration**: Optional CUDA support for the distance kernel; requires numba and an NVIDIA driver, see [GPU support](#gpu-support)
- **Vectorized CPU Computation**: NumPy-based vectorized distance calculation with multi-threaded parallelism
- **Optimized Memory Management**: Batch processing to handle large datasets efficiently
- **Multithreaded Processing**: Parallelized calculations across CPU cores (numpy releases the GIL)
- **Intelligent I/O**: Chunked file operations for better performance with large files
- **Advanced Filtering**: Quality control via loci and sample completeness thresholds
- **Missing-Data Handlers**: Four policies for absent allele calls (`-y`), numbered as in GrapeTree
- **Automatic System Detection**: Optimizes settings based on available hardware
- **Binary Output Option**: For extremely large matrices

## Usage

Every option has a short alias (e.g. `-i`/`--input`); the long forms are shown below.

```console
$ python cgmlst-dists.py --help
usage: cgmlst-dists.py [-h] [-i INPUT] [-o OUTPUT] [-r] [-d INPUT_SEP]
                       [-D OUTPUT_SEP] [-x INDEX_NAME]
                       [-m {full,lower-tri,upper-tri}] [-t NUM_THREADS]
                       [-j IO_THREADS] [-M MAX_MEMORY_GB] [-k CHUNK_SIZE]
                       [-n MISSING_CHAR] [-u] [-y {0,1,2,3}]
                       [-L LOCUS_COMPLETENESS] [-S SAMPLE_COMPLETENESS] [-g]
                       [-b] [-s] [-c] [-f] [-V]

Calculate pairwise Hamming distances.

options:
  -h, --help            show this help message and exit
  -i, --input INPUT     Path to the input TSV file
  -o, --output OUTPUT   Path to save the output TSV file
  -r, --skip_input_replacements
                        Skip input replacements when there are no strings in the input
  -d, --input_sep INPUT_SEP
                        Input file separator (default: '\t')
  -D, --output_sep OUTPUT_SEP
                        Output file separator (default: '\t')
  -x, --index_name INDEX_NAME
                        Name for the index column (default: 'cgmlst-dists')
  -m, --matrix-format {full,lower-tri,upper-tri}
                        Format for the output matrix (default: full)
  -t, --num_threads NUM_THREADS
                        Number of threads for parallel execution (default: auto-detected)
  -j, --io_threads IO_THREADS
                        Number of I/O threads for file operations (default: auto-detected)
  -M, --max_memory_gb MAX_MEMORY_GB
                        Maximum memory to use in GB for distance calculation (default: auto-detected)
  -k, --chunk_size CHUNK_SIZE
                        Size of chunks for reading/writing files (default: 1000)
  -n, --missing_char MISSING_CHAR
                        Character used for missing data (default: '-')
  -u, --dedup           Collapse identical profiles, compute distances over
                        the unique ones and expand the result. Exact, and much
                        faster on clonal data; costs a few percent when there
                        are no duplicates. Not compatible with -y 0
  -y, --missing-handler {0,1,2,3}
                        How to treat missing calls, numbered as in GrapeTree's -y flag.
                        0: pair_delete, 1: complete_delete, 2: as_allele,
                        3: absolute_distance (default: 3, matching the original C cgmlst-dists)
  -L, --locus-completeness LOCUS_COMPLETENESS
                        Minimum percentage of non-missing data required for a locus (0-100)
  -S, --sample-completeness SAMPLE_COMPLETENESS
                        Minimum percentage of non-missing data required for a sample (0-100)
  -g, --gpu             Use GPU acceleration when available
  -b, --binary-output   Also save results in binary format for large matrices
  -s, --silent          Disable all console output for maximum performance
  -c, --stdout          Write results to stdout instead of a file
  -f, --force           Skip the up-front memory feasibility check and run even if the matrix may not fit in RAM
  -V, --version         show program's version number and exit
```

## Examples

### Basic Usage

```bash
python cgmlst-dists.py --input input.tsv --output output.tsv

# Same command using short options
python cgmlst-dists.py -i input.tsv -o output.tsv

# Stream a lower-triangular matrix to stdout with 8 threads
python cgmlst-dists.py -i input.tsv -c -m lower-tri -t 8 > distances.tsv
```

### With GPU Acceleration (if available)

```bash
python cgmlst-dists.py --input input.tsv --output output.tsv --gpu
```

Without a usable CUDA device this falls back to the CPU with a warning on
stderr; see [GPU support](#gpu-support) for what each install route provides.

### Missing Data

A locus with no call (`-` by default, see `-n/--missing_char`) is stored
internally as `0`, so an allele literally named `0` cannot be represented. Note
that any value that is not a positive integer is also treated as missing.

`-y/--missing-handler` selects how those absent calls affect the distance. The
numbering follows GrapeTree's `-y` flag so the two tools can be compared:

| `-y` | Name | Behaviour |
|------|------|-----------|
| `0` | `pair_delete` | Ignore missing loci per pair, then rescale the count to the full locus set: `(diff + 0.01) * n_loci / (comparable + 0.01)`, rounded |
| `1` | `complete_delete` | Drop every locus not called in all samples, then count |
| `2` | `as_allele` | Treat "missing" as a regular allele: missing vs. called counts as a difference, missing vs. missing does not |
| `3` | `absolute_distance` | Ignore missing loci per pair, report the absolute count — **default** |

The default is `3`, which is the semantics of the original
[`cgmlst-dists`](https://github.com/tseemann/cgmlst-dists) and keeps the output
byte-identical to it. **GrapeTree defaults to `0` instead**, so the two tools do
not agree out of the box: `pair_delete` inflates distances by
`n_loci / n_comparable`, which grows with the amount of missing data (roughly
×1.2 at 10% missing per sample, ×1.6 at 20%). Clustering thresholds tuned on
GrapeTree defaults therefore do not transfer unchanged.

```bash
# Absolute allelic differences (default)
python cgmlst-dists.py -i input.tsv -o output.tsv

# GrapeTree's default policy instead
python cgmlst-dists.py -i input.tsv -o output.tsv -y 0

# Only loci called in every sample
python cgmlst-dists.py -i input.tsv -o output.tsv -y 1
```

Two deliberate differences from GrapeTree, both consequences of this tool
emitting an integer matrix of allele differences:

- GrapeTree's `--method distance` divides the result by the locus count for
  handlers `0`, `1` and `2`, so it prints fractions there while we always print
  counts. To compare, multiply its values by the number of loci.
- Handler `1` implements GrapeTree's *documented* behaviour ("remove column with
  missing data"). GrapeTree's own implementation inverts the test and keeps only
  the columns that **do** contain a missing call, which collapses its `-y 1`
  distances to nearly zero; we do not reproduce that bug.

If you need the handlers exactly as GrapeTree computes them, including its tree
methods, see [`grapetree-rs`](https://github.com/genpat-it/grapetree-rs).

### Deduplication

Distances depend only on the pair of profiles, so identical profiles can be
collapsed: `-u/--dedup` computes the matrix over the unique profiles and expands
it back. The result is **exact**, not approximate.

```bash
python cgmlst-dists.py -i input.tsv -o output.tsv --dedup
```

Worth it on clonal data, which is common in surveillance: 2000 samples with 200
distinct profiles compute 4.5x faster, 4000 with 400 distinct 5x. With no
duplicates at all it costs about 8%, since the profiles still have to be compared
once; detection itself is 13-51 ms on those inputs.

Two caveats:

- It saves computation, not memory. The expansion still materializes the full
  N x N matrix, so the up-front feasibility check applies exactly as before.
- It cannot be combined with `-y 0` (`pair_delete`), and the tool refuses the
  combination rather than returning different numbers. Under that handler two
  identical profiles are not necessarily at distance 0: the rescaling is
  `round(0.01 * n_loci / (comparable + 0.01))`, which exceeds 0 when very few
  loci are called, and collapsing such profiles onto the diagonal would report 0
  instead.

### Data Filtering

Filter both loci and samples to include only those with ≥90% data completeness:

```bash
python cgmlst-dists.py --input input.tsv --output output.tsv --locus-completeness 90 --sample-completeness 90
```

`--locus-completeness 100` keeps exactly the loci that `-y 1` keeps, so the two
routes produce the same matrix; the filter also accepts looser thresholds, and
`--sample-completeness` has no GrapeTree equivalent.

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

- **Fast loading (pyarrow)**: bundled by default; the Arrow-based loader reads and parses large inputs ~1.5x faster end to end than the pandas one. Auto-detected, with an identical-result fallback.
- **Deduplication**: `--dedup` removes nearly all of the distance computation on clonal data, which is the common case in surveillance.
- **CPU vectorization**: the numpy kernel scales with thread count; on 80 cores it is 7.6x faster than the single-threaded C implementation (see below).
- **GPU acceleration**: helps the distance kernel, but the kernel is only part of the total runtime, and the gain depends heavily on free GPU memory, since batches are sized to fit it.
- **Memory**: the full N x N int32 matrix is held in RAM (1.5 GiB at 20,000 samples, 9.3 GiB at 50,000). The tool checks this up front and refuses to start rather than crashing halfway; `--force` overrides.
- **I/O**: on large matrices, writing the output is a substantial share of the runtime — 38% in the benchmark below — so `--matrix-format lower-tri` or `--binary-output` are worth considering.

### Benchmarks

Measured on 0.1.7. The dataset is synthetic and reproducible with the generator in
this repository, so anyone can repeat these runs:

```bash
python benchmark/cgmlst-data-generator.py --samples 20000 --loci 3000 --missing 5 --output bench20k.tsv
```

Test system: Intel Xeon Gold 6542Y, 80 cores, 480 GB RAM, NVIDIA L4, AlmaLinux 10.

#### 20,000 samples x 3,000 loci (164 MB input, 2.0 GB output matrix)

| Implementation | Runtime | vs C |
|---|---|---|
| C `cgmlst-dists` 0.4.0 (single-threaded) | 963.8 s | 1x |
| this tool, CPU (80 cores) | **127.0 s** | **7.6x** |
| this tool, CPU + `--dedup`, clonal data* | **63.8 s** | **15.1x** |

\* same 20,000 samples but only 2,000 distinct profiles, as is typical of
surveillance data. `--dedup` is exact: the output is byte-identical.

**The C matrix and ours are identical (same md5 over 400 million distances).**
That is what makes the comparison meaningful: it is the same computation, not a
different one that happens to be faster.

Where the 127 s go:

| Phase | Time | Share |
|---|---|---|
| Loading and parsing | 12.7 s | 11% |
| Distance calculation | 61.3 s | 51% |
| Writing the matrix | 46.2 s | 38% |

#### 5,000 samples x 2,000 loci

| Configuration | Runtime |
|---|---|
| CPU, pandas loader | 10.1 s |
| CPU, Arrow loader | 7.0 s |
| CPU, Arrow loader, 1 thread | 19.8 s |

So the Arrow loader is worth ~1.5x end to end, and multithreading ~2.9x at this
size (the matrix is small enough that loading and writing dominate).

#### On the GPU numbers

Earlier versions of this README advertised "up to 123x" for the GPU. That figure
described the distance **kernel** in isolation, not a run: even when the kernel is
10x faster, loading and writing do not change, so the end-to-end gain is far
smaller. At 5,000 samples the kernel went from 2.05 s to 0.69 s while total
runtime moved from 7.0 s to 5.9 s.

The 20,000-sample GPU figure is deliberately **not** published here: the only card
available was 92% occupied by another process, leaving 1.2 GB, and since batch size
is derived from free GPU memory the result (308 s, slower than the CPU) says more
about the contention than about the tool. GPU numbers need a card to yourself.

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

### GPU support in Docker

The published image is **CPU-only**, and `--gpus all` is not enough to change
that. The NVIDIA container runtime exposes the device and the driver, so numba
can see the card, but it cannot compile its kernels: `libnvvm.so`, the NVVM
compiler from the CUDA toolkit, is not in the image, and the kernels are
compiled at runtime inside the container. The failure is explicit:

```
NvvmSupportError: libNVVM cannot be found
```

The tool detects this, warns on stderr and computes on the CPU, so results are
still correct.

For GPU runs in a container, build `Dockerfile.gpu`, which is provided but
deliberately **not published**: it is built on a conda base so that conda-forge's
numba supplies the CUDA compiler, which makes it 3.4 GB against 877 MB for the
CPU-only image. Most users do not need it, so it is not worth pushing to the
registry on every release.

```bash
docker build -f Dockerfile.gpu -t cgmlst-dists-py:gpu .

docker run --rm --gpus all -v "$PWD":/data cgmlst-dists-py:gpu \
    -i /data/input.tsv -o /data/output.tsv --gpu
```

This needs an NVIDIA driver and the NVIDIA Container Toolkit on the host, plus
`--gpus all` on the run command. Verified on an NVIDIA L4: the image reports the
device and produces a matrix byte-identical to the CPU path. Note that
`cuda-version` is pinned to 12 in the file, because a host driver cannot run code
built for a newer CUDA major version than it supports.

## Advantages Over Original Implementation

1. **Scalability**: Efficiently handles much larger datasets through batch processing and memory optimization
2. **Speed**: 7.6x faster than the C implementation on 20,000 samples with 80 cores, producing an identical matrix; up to 15x on clonal data with `--dedup`
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