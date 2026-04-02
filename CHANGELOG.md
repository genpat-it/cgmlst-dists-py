# Changelog

## [0.1.2] - 2026-04-02

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

## [0.1.1] - 2025-03-17

- Initial optimized release with GPU and multithreaded CPU support

## [0.1.0] - 2025-03-17

- First release
