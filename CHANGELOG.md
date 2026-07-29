# Changelog

## [Unreleased] - 0.1.7

### Added
- `-u/--dedup`: collapse identical profiles, compute the matrix over the unique
  ones and expand it back. Exact, not approximate. On clonal data — common in
  surveillance — 2000 samples with 200 distinct profiles compute 4.5x faster and
  4000 with 400 distinct 5x; with no duplicates it costs about 8%. Detection is
  done in numpy (rows compared as opaque records) rather than with a compiled
  numba kernel, which is what makes the no-duplicate case cheap: 13-51 ms instead
  of ~500 ms of fixed overhead.

  It refuses to run with `-y 0` (`pair_delete`) instead of silently changing the
  result: under that handler two identical profiles can be at non-zero distance,
  because `round(0.01 * n_loci / (comparable + 0.01))` exceeds 0 when very few
  loci are called, and collapsing them onto the diagonal would report 0. Measured
  boundary: duplicates of profiles with <= 2% of loci called.

  It saves computation, not memory: the expansion still materializes the full
  N x N matrix, so the feasibility check from 0.1.5 is as binding as before.

### Changed
- **numba is no longer a required dependency.** It is only needed for `--gpu`, and
  with llvmlite it is ~199 MB, the heaviest dependency in the package and the one
  that gates which Python versions can be supported. It moved to
  `requirements-gpu.txt`; the CPU path computes with numpy and never imports it.
  Without numba, `--gpu` warns and computes on the CPU with identical results,
  rather than crashing at import as it did before the lazy import. The CPU Docker
  image drops from 877 MB to 592 MB as a result. CI gained a matrix dimension so
  the numba-dependent tests keep running on the legs that install it.
- numba is now imported lazily, and CUDA is only probed when `--gpu` is passed.
  Startup drops from 458 ms to 284 ms. The default CPU path computes with numpy
  and never needed numba, and the tool now runs correctly with numba absent
  altogether (`--gpu` then warns and computes on the CPU); previously it crashed
  at import. The four tests using the numba kernel as an oracle skip when it is
  missing, as the GPU tests already did.
- `pyarrow` is now part of `requirements.txt`, and of the Bioconda run
  dependencies, instead of being an opt-in extra. It is still probed at runtime
  and the pandas fallback is unchanged, so an environment without it behaves
  exactly as before, only slower; the three install routes (pip, conda, Docker)
  now behave alike rather than two out of three shipping the accelerator. CI
  keeps one matrix leg that uninstalls pyarrow so the fallback stays covered.
  The Docker image no longer installs it separately, since requirements.txt
  covers it.

### Documentation
- Benchmarks re-measured on 0.1.7 and made reproducible: the dataset is generated
  by the script in `benchmark/`, and the exact command is published. On 20,000
  samples x 3,000 loci this tool takes 127.0 s against 963.8 s for the
  single-threaded C `cgmlst-dists` 0.4.0 (7.6x), or 63.8 s with `--dedup` on
  clonal data (15.1x), producing a matrix with the same md5 as the C one over 400
  million distances.

  The previous figures were from 0.1.3, measured on a dataset that is not in the
  repository, and the headline "up to 123x" GPU claim described the distance
  kernel in isolation rather than a run: loading and writing do not speed up, and
  at 5,000 samples the kernel going 2.05 s -> 0.69 s moved the total only from
  7.0 s to 5.9 s. The runtime breakdown is now published instead (at 20,000
  samples: 11% loading, 51% computing, 38% writing), and GPU figures at scale are
  omitted rather than reported from a card that was 92% occupied by another
  process.
- `benchmark/cgmlst-data-generator.py` wrote an unnamed index column, so pandas
  emitted a leading empty field and the C `cgmlst-dists` rejected the generated
  file outright ("row 2 had N+1 cols, expected N-1"). The index column is now
  named, which is what allowed the cross-implementation comparison above.
- Removed the hardcoded local paths from `benchmark/README.md`.

### Removed
- Dead code: the never-called `process_save_chunk`, the unused `Tuple`, `List`,
  `gzip`, `io`, `multiprocessing` and `ProcessPoolExecutor` imports, and the
  unread `DEFAULT_THREADS` constant (thread defaults come from
  `detect_system_capabilities()`). The numba kernel is kept deliberately, with a
  docstring saying why: it is an independent implementation of the same
  semantics, used by the test suite as an oracle for the numpy kernel and as the
  fallback for a failed CUDA batch.

### Performance
- The Arrow loader is now used with `--locus-completeness`/`--sample-completeness`
  as well. It had been gated on both filters being absent, on the grounds that
  they "need the raw string data"; they do not, they count zeros on the numeric
  frame, so anyone passing `-L`/`-S` was silently getting the 3-4x slower pandas
  loader. The Arrow result was in fact computed and then discarded in that case,
  because the pandas loader ran straight afterwards and overwrote it.

### Fixed
- The tool exited 0 when it produced no output. A missing input file reported
  nothing under `--silent` and exited successfully, so a pipeline could not
  detect the failure. Loading errors now always reach stderr and exit 1, missing
  arguments exit 2, and a failed calculation exits 1.
- `Dockerfile.gpu`: an optional GPU-capable image recipe, not published to the
  registry. Built on a conda base so conda-forge's numba supplies the CUDA
  compiler that the slim Python image lacks; verified on an NVIDIA L4, where it
  reports the device and produces a matrix byte-identical to the CPU path. It is
  3.4 GB against 877 MB for the CPU-only image, which is why it is left for users
  who actually need it to build themselves.
- Documented what GPU acceleration actually requires per install route, after
  verifying each one: conda and from-source work with only an NVIDIA driver on the
  host (numba supplies the CUDA pieces), while the published Docker image cannot
  use the GPU even with `--gpus all`, because `libnvvm.so` (the NVVM compiler
  numba needs to build kernels at runtime) is not in the image. The README
  previously advertised a `--gpus all` Docker command that cannot work.
- `--gpu` on a machine without a usable CUDA device fell back to the CPU
  silently, so a user who believed they were running on the GPU had no way to
  tell (the only clue, `GPU available: No`, was hidden by `--silent`). It now
  warns on stderr, always, and names what is missing.
- Input that parses but contains no samples or no loci is rejected with a message
  pointing at the likely cause (wrong `--input_sep`) instead of producing an empty
  matrix. This also fixes a cryptic "zero-size array to reduction operation
  maximum" on header-only input, and it now applies to the Arrow loader too,
  which previously returned before the validation could run.

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
