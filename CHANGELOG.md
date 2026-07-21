# Changelog

## [Unreleased] - 0.1.6

### Performance
- Vectorized input parsing: `process_chunk` converts the whole block with a
  single `pd.to_numeric` (plus a vectorized `INF-` strip) instead of a per-cell
  regex replace and a per-column `apply(pd.to_numeric)`. Added a fast path that
  skips all string handling when every column is already numeric. (~15-20%
  faster loading.)
- Integer→string lookup table for output writers (full, lower-tri, upper-tri,
  stdout): rows are stringified by vectorized indexing instead of a per-element
  `astype(str)` (~60% faster row-by-row save, ~13% faster full-matrix write).
- Distance kernel fast path when there is no missing data: skips the both-valid
  masking (~25% faster distance calculation on complete datasets).
- Downcast alleles to int16 when they fit (halves memory bandwidth in the
  distance kernel).

### Fixed
- Triangular file output (`--matrix-format lower-tri/upper-tri`) no longer
  materializes the whole matrix as strings (`distances.astype(str)`), matching
  the row-by-row streaming already used for stdout — avoids the large-matrix
  memory blow-up.

_All changes above are output-preserving: results are byte-identical to 0.1.5._

## [0.1.5] - 2026-07-21

### Added
- Up-front memory feasibility check: the tool now estimates the RAM needed for
  the N×N distance matrix and **aborts before the distance computation** if it
  will not fit in available memory, instead of crashing after minutes of work.
  Override with the new `--force` flag.
- `--force` flag to skip the feasibility check and run anyway.
- Short option aliases for every flag (e.g. `-i/--input`, `-o/--output`,
  `-c/--stdout`, `-t/--num_threads`, `-m/--matrix-format`, `-g/--gpu`,
  `-s/--silent`, `-f/--force`, `-V/--version`). The long options are unchanged.

### Fixed
- `--stdout` no longer allocates the entire matrix as strings at once
  (`distances.astype(str)`), which could try to reserve ~163 GiB for a
  63k-sample dataset and abort with a `MemoryError` *after* the computation.
  Rows are now stringified one at a time, bounding peak memory to a single row.
- When `--stdout` is used, all informational logging and progress now go to
  **stderr**, so `cgmlst-dists ... --stdout > matrix.tsv` produces a clean,
  uncorrupted TSV (previously log lines were interleaved into stdout).

## [0.1.4] - 2026-07-21

### Fixed
- Fixed crash at startup when running the Docker image as a non-root user
  (e.g. `docker run -u $UID ...`). Numba's `@jit(cache=True)` aborted with
  `cannot cache function ...: no locator available for file '/app/cgmlst-dists.py'`
  because neither `/app` nor `$HOME` were writable. The entrypoint now sets a
  writable `NUMBA_CACHE_DIR` when the caller hasn't provided one.

### Changed
- README: clearer Docker usage section, including how to run as the current
  user and how to override the numba cache directory.

## [0.1.3] - 2026-04-02

### Performance
- Replaced numba triple-loop distance kernel with numpy vectorized computation (~6.5x faster on CPU with 8 threads)
- Optimized GPU CUDA kernel with 32x32 thread blocks and GPU-aware memory batching
- Vectorized loci and sample completeness filtering (was row-by-row Python loops)
- Replaced Python mirroring loop with vectorized `distances += distances.T`
- Optimized matrix output using pandas `to_csv` for full matrices
- Added buffered I/O (8MB) for triangular matrix output
- Added `low_memory=False` for more reliable CSV loading

### Added
- GPU-aware batch size estimation based on available GPU memory
- LICENSE file (GPL-3.0)

### Changed
- Updated Dockerfile to Python 3.13-slim
- Unpinned dependency versions in requirements.txt for broader compatibility

### Fixed
- Fixed mixed-dtype issue when processing files with INF- prefixed allele calls

## [0.1.2] - 2026-04-02

- Added LICENSE file (GPL-3.0) for Bioconda packaging

## [0.1.1] - 2025-03-17

- Initial optimized release with GPU and multithreaded CPU support

## [0.1.0] - 2025-03-17

- First release
