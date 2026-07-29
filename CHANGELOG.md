# Changelog

## [0.1.6] - 2026-07-29

### Added
- `-y/--missing-handler`: four policies for missing allele calls, numbered as in
  GrapeTree's `-y` flag so the two tools can be compared.
  - `0` `pair_delete`: ignore missing loci per pair, then rescale the count to
    the full locus set (`(diff + 0.01) * n_loci / (comparable + 0.01)`, rounded).
  - `1` `complete_delete`: drop every locus not called in all samples.
  - `2` `as_allele`: treat "missing" as a regular allele value.
  - `3` `absolute_distance`: ignore missing loci per pair, absolute count.

  **The default is `3`, which is the previous (and only) behaviour**, so output
  is unchanged unless `-y` is passed explicitly. Note that GrapeTree defaults to
  `0`, which inflates distances by `n_loci / n_comparable`; the two tools
  therefore do not agree out of the box, and thresholds tuned on GrapeTree
  defaults do not transfer. Supported on the CPU, GPU and block-fallback paths.

  Two deliberate departures from GrapeTree, both because this tool emits an
  integer matrix: its `--method distance` divides by the locus count for
  handlers `0`, `1` and `2` (we always report counts), and its `complete_delete`
  implementation inverts the column test, keeping only the columns that *do*
  contain a missing call — we implement the documented behaviour instead.
  Requested in #3.
- `test/test_missing_handlers.py`: pytest suite covering the handlers (known
  values, cross-agreement of the numpy/numba/block-fallback/GPU paths,
  invariants, CLI behaviour) plus regression tests pinning the default output
  against both the stored `validation/` matrices and the C implementation's.
- Continuous integration (`.github/workflows/ci.yml`): the test suite now runs on
  every push and pull request, not only at release time, across Python 3.11/3.13
  and with and without `pyarrow` (the optional loader replaces the whole input
  path, so both configurations must agree). The ~333 MiB of Git LFS datasets are
  not fetched there, to stay within the LFS bandwidth quota; the full-size
  `crc32` dataset is validated at release time, where the workflow also fails if
  those regression tests skip because the LFS objects did not arrive. It can be
  run on demand from the CI workflow via the `full_datasets` input.
- The Docker image is now built and executed in CI and, at release time, is
  smoke-tested **before** being pushed: reported version against the tag, output
  byte-identical to the C reference, a run under an arbitrary UID (numba's cache
  directory has broken this before), all four handlers, and confirmation that
  the bundled `pyarrow` loader is the one in use.

### Performance
- Optional Arrow-based loader: when `pyarrow` is installed, the read + numeric
  conversion is done with Arrow compute kernels (multithreaded C++), ~3-4x
  faster loading on realistic files with `INF-`/missing values (end-to-end
  ~2x on large inputs). Auto-detected and used only when no completeness
  filtering is requested; falls back transparently to the pandas loader when
  pyarrow is absent. Bundled in the Docker image; kept out of `requirements.txt`
  so pip/conda installs stay lightweight (opt-in via `pip install pyarrow`).
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
- Completeness filtering (`--locus-completeness` / `--sample-completeness`) was
  a no-op: it counted the missing *character* in data that had already been
  converted to numeric (missing → 0), so it never matched and nothing was
  filtered. It now counts zeros (the post-conversion missing sentinel), so the
  thresholds actually take effect.
- Triangular file output (`--matrix-format lower-tri/upper-tri`) no longer
  materializes the whole matrix as strings (`distances.astype(str)`), matching
  the row-by-row streaming already used for stdout — avoids the large-matrix
  memory blow-up.
- GPU batch fallback: when a CUDA batch failed, the CPU fallback raised
  `NameError` instead of computing the batch (an undefined
  `calculate_hamming_distances_numba_batch`, and undefined `start_i`/`end_i` in
  the batched loop). Both now call a shared `calculate_distances_cpu_batch`.

_Output is unchanged for a default invocation: results are byte-identical to
0.1.5, and to the C `cgmlst-dists`, on the `validation/` datasets. The two
exceptions are deliberate: `--locus-completeness`/`--sample-completeness` now
actually filter (they were a no-op), and `-y` other than the default `3` selects
a different metric._

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
