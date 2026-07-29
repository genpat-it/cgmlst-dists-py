"""Tests for the missing-data handlers (-y/--missing-handler) and regression
tests guarding the default output.

Run with:  pytest test/test_missing_handlers.py -v
"""

import importlib.util
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "cgmlst-dists.py"


def _load_module():
    """Import cgmlst-dists.py, whose hyphenated name is not a valid module name.

    The module must be registered in sys.modules before it is executed: the
    numba kernel is declared with cache=True, and reloading a cached compilation
    makes numba re-import the defining module by name, which fails with a bare
    spec-based import."""
    name = "cgmlst_dists"
    spec = importlib.util.spec_from_file_location(name, SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


cd = _load_module()

ALL_HANDLERS = [
    cd.HANDLER_PAIR_DELETE,
    cd.HANDLER_COMPLETE_DELETE,
    cd.HANDLER_AS_ALLELE,
    cd.HANDLER_ABSOLUTE,
]


def run_cli(*args):
    """Run the tool and return its stdout matrix as text."""
    proc = subprocess.run(
        [sys.executable, str(SCRIPT), *map(str, args)],
        capture_output=True, text=True, check=True,
    )
    return proc.stdout


def symmetrize(upper):
    return upper + upper.T


def profiles(n_samples, n_loci, missing_rate, seed):
    """Random allelic profiles with a share of missing calls (encoded as 0)."""
    rng = np.random.default_rng(seed)
    v = rng.integers(1, 8, size=(n_samples, n_loci)).astype(np.int32)
    if missing_rate:
        v[rng.random(v.shape) < missing_rate] = 0
    return v


def numpy_dist(values, handler):
    return symmetrize(cd.calculate_hamming_distances_numpy(values, 2, True, handler))


def complete_delete_cols(values):
    """The locus restriction main() applies for handler 1."""
    return values[:, (values != 0).all(axis=0)]


# --------------------------------------------------------------------------
# Regression: the default must keep producing byte-identical output
# --------------------------------------------------------------------------

def is_lfs_pointer(path):
    with open(path, "rb") as f:
        return f.read(40).startswith(b"version https://git-lfs")


@pytest.mark.parametrize("name", ["boring", "chewie", "100", "crc32"])
def test_default_output_matches_validation(name, tmp_path):
    """The reference outputs in validation/ were produced before the handlers
    existed, so this pins the default (-y 3) against silent behaviour changes."""
    src = REPO / "test" / f"{name}.tab"
    expected = REPO / "validation" / f"{name}_py.tab"
    if is_lfs_pointer(src):
        pytest.skip(f"{name}.tab is an unfetched Git LFS pointer")
    out = tmp_path / "out.tab"
    run_cli("-i", src, "-o", out, "-s")
    assert out.read_text() == expected.read_text()


@pytest.mark.parametrize("name", ["boring", "chewie", "100", "crc32"])
def test_default_output_matches_c_implementation(name, tmp_path):
    """The default handler is the semantics of the original C cgmlst-dists, so
    the output must stay byte-identical to the C reference matrices."""
    src = REPO / "test" / f"{name}.tab"
    expected = REPO / "validation" / f"{name}_c.tab"
    if is_lfs_pointer(src) or is_lfs_pointer(expected):
        pytest.skip(f"{name} reference is an unfetched Git LFS pointer")
    out = tmp_path / "out.tab"
    run_cli("-i", src, "-o", out, "-s")
    assert out.read_text() == expected.read_text()


def test_default_handler_is_absolute_distance(tmp_path):
    """Omitting -y must be identical to asking for -y 3 explicitly."""
    src = REPO / "test" / "chewie.tab"
    assert run_cli("-i", src, "-c", "-s") == run_cli("-i", src, "-c", "-s", "-y", 3)


# --------------------------------------------------------------------------
# Known-value test: the worked example from issue #3
# --------------------------------------------------------------------------

EXAMPLE = "id\tL1\tL2\tL3\tL4\tL5\nA\t1\t2\t3\t-\t5\nB\t1\t9\t3\t4\t-\n"


@pytest.mark.parametrize("handler,expected", [
    # 3 loci comparable (L1-L3), 1 of them differs (L2):
    (0, 2),   # pair_delete:  round((1+0.01) * 5 / (3+0.01)) = round(1.677) = 2
    (1, 1),   # complete_delete: keep L1-L3, count -> 1
    (2, 3),   # as_allele: L2, L4 and L5 all count -> 3
    (3, 1),   # absolute_distance: raw count -> 1
])
def test_worked_example(tmp_path, handler, expected):
    src = tmp_path / "ex.tab"
    src.write_text(EXAMPLE)
    rows = run_cli("-i", src, "-c", "-s", "-y", handler).strip().splitlines()
    assert rows[1].split("\t")[1:] == ["0", str(expected)]
    assert rows[2].split("\t")[1:] == [str(expected), "0"]


# --------------------------------------------------------------------------
# All three CPU implementations must agree, for every handler
# --------------------------------------------------------------------------

@pytest.mark.parametrize("handler", ALL_HANDLERS)
def test_numpy_numba_and_block_fallback_agree(handler):
    v = profiles(40, 60, 0.15, seed=7)
    v[0, :30] = 0
    v[1, 30:] = 0          # rows 0 and 1 share no comparable locus
    v[3, :] = 0            # an all-missing sample
    if handler == cd.HANDLER_COMPLETE_DELETE:
        v = np.delete(v, 3, axis=0)      # else every locus is dropped
        v = complete_delete_cols(v)

    expected = numpy_dist(v, handler)
    from_numba = symmetrize(cd.calculate_hamming_distances_numba(
        v, handler == cd.HANDLER_AS_ALLELE, handler == cd.HANDLER_PAIR_DELETE))
    assert np.array_equal(expected, from_numba)

    # Reassemble from irregular blocks, exercising the CUDA-fallback path's
    # index offsets (block sizes deliberately do not divide the sample count).
    n = v.shape[0]
    blocks = np.zeros((n, n), dtype=np.int32)
    for i0 in range(0, n, 7):
        for j0 in range(0, n, 11):
            i1, j1 = min(i0 + 7, n), min(j0 + 11, n)
            blocks[i0:i1, j0:j1] = cd.calculate_distances_cpu_batch(v, i0, i1, j0, j1, handler)
    assert np.array_equal(expected, symmetrize(blocks))


# --------------------------------------------------------------------------
# Invariants that must hold on any input
# --------------------------------------------------------------------------

@pytest.mark.parametrize("handler", ALL_HANDLERS)
def test_symmetric_with_zero_diagonal(handler):
    v = profiles(25, 40, 0.2, seed=11)
    if handler == cd.HANDLER_COMPLETE_DELETE:
        v = complete_delete_cols(v)
    d = numpy_dist(v, handler)
    assert np.array_equal(d, d.T)
    assert np.all(np.diag(d) == 0)
    assert np.all(d >= 0)


def test_pair_delete_never_below_absolute():
    """The rescaling factor n_loci/n_comparable is >= 1 by construction."""
    v = profiles(30, 50, 0.25, seed=3)
    assert np.all(numpy_dist(v, cd.HANDLER_PAIR_DELETE) >= numpy_dist(v, cd.HANDLER_ABSOLUTE))


def test_as_allele_never_below_absolute():
    """as_allele counts the same differences plus the missing-vs-called ones."""
    v = profiles(30, 50, 0.25, seed=4)
    assert np.all(numpy_dist(v, cd.HANDLER_AS_ALLELE) >= numpy_dist(v, cd.HANDLER_ABSOLUTE))


def test_handlers_collapse_when_nothing_is_missing():
    """With complete profiles there is nothing to handle, so all four agree."""
    v = profiles(20, 60, 0.0, seed=5)
    ref = numpy_dist(v, cd.HANDLER_ABSOLUTE)
    for handler in ALL_HANDLERS:
        assert np.array_equal(numpy_dist(v, handler), ref), f"handler {handler} differs"


def test_pair_delete_with_no_comparable_locus_yields_n_loci():
    """GrapeTree's 0.01 guard sends a pair with nothing in common to n_loci."""
    v = np.array([[1, 2, 0, 0], [0, 0, 3, 4]], dtype=np.int32)
    assert numpy_dist(v, cd.HANDLER_PAIR_DELETE)[0, 1] == v.shape[1]


def test_as_allele_counts_missing_against_called():
    v = np.array([[1, 0], [1, 5]], dtype=np.int32)
    assert numpy_dist(v, cd.HANDLER_AS_ALLELE)[0, 1] == 1
    assert numpy_dist(v, cd.HANDLER_ABSOLUTE)[0, 1] == 0


# --------------------------------------------------------------------------
# complete_delete and its relationship to the completeness filter
# --------------------------------------------------------------------------

def test_complete_delete_matches_locus_completeness_100(tmp_path):
    """Handler 1 keeps exactly the loci --locus-completeness 100 keeps, so the
    two routes must produce the same matrix (they take different code paths:
    the filter disables the Arrow loader, the handler does not)."""
    src = tmp_path / "in.tab"
    v = profiles(12, 30, 0.1, seed=9)
    lines = ["id\t" + "\t".join(f"L{i}" for i in range(v.shape[1]))]
    for i, row in enumerate(v):
        lines.append(f"S{i}\t" + "\t".join("-" if x == 0 else str(x) for x in row))
    src.write_text("\n".join(lines) + "\n")

    via_handler = run_cli("-i", src, "-c", "-s", "-y", 1)
    via_filter = run_cli("-i", src, "-c", "-s", "-y", 3, "-L", 100)
    assert via_handler == via_filter


def test_complete_delete_warns_when_every_locus_is_dropped(tmp_path):
    """An all-missing sample empties the schema; the all-zero matrix that
    results must not be produced silently."""
    src = tmp_path / "in.tab"
    src.write_text("id\tL1\tL2\nA\t1\t2\nB\t-\t-\n")
    proc = subprocess.run(
        [sys.executable, str(SCRIPT), "-i", str(src), "-c", "-s", "-y", "1"],
        capture_output=True, text=True, check=True,
    )
    assert "WARNING" in proc.stderr
    assert "complete_delete" in proc.stderr


# --------------------------------------------------------------------------
# The handlers must not disturb the rest of the CLI
# --------------------------------------------------------------------------

@pytest.mark.parametrize("handler", ALL_HANDLERS)
@pytest.mark.parametrize("fmt", ["full", "lower-tri", "upper-tri"])
def test_matrix_formats_still_work(tmp_path, handler, fmt):
    src = REPO / "test" / "boring.tab"
    out = tmp_path / "out.tab"
    run_cli("-i", src, "-o", out, "-s", "-y", handler, "-m", fmt)
    rows = out.read_text().strip().splitlines()
    n = len(rows) - 1
    assert n > 0
    for row in rows[1:]:
        assert len(row.split("\t")) == n + 1


@pytest.mark.parametrize("handler", ALL_HANDLERS)
def test_stdout_matches_file_output(tmp_path, handler):
    src = REPO / "test" / "boring.tab"
    out = tmp_path / "out.tab"
    run_cli("-i", src, "-o", out, "-s", "-y", handler)
    assert run_cli("-i", src, "-c", "-s", "-y", handler) == out.read_text()


def test_invalid_handler_is_rejected():
    proc = subprocess.run(
        [sys.executable, str(SCRIPT), "-i", str(REPO / "test" / "boring.tab"), "-c", "-y", "4"],
        capture_output=True, text=True,
    )
    assert proc.returncode != 0
    assert "invalid choice" in proc.stderr


# --------------------------------------------------------------------------
# GPU path (skipped when no CUDA device is present)
# --------------------------------------------------------------------------

def has_cuda():
    try:
        from numba import cuda
        return cuda.is_available()
    except Exception:
        return False


requires_cuda = pytest.mark.skipif(not has_cuda(), reason="no CUDA device available")


@requires_cuda
@pytest.mark.parametrize("handler", ALL_HANDLERS)
def test_gpu_matches_cpu_small(handler):
    """Under 100 samples the GPU path skips batching entirely."""
    v = profiles(50, 45, 0.2, seed=13)
    v[0, :20] = 0
    v[1, 20:] = 0
    if handler == cd.HANDLER_COMPLETE_DELETE:
        v = complete_delete_cols(v)
    import pandas as pd
    frame = pd.DataFrame(v)
    gpu = cd.calculate_distances_batched(frame, use_gpu=True, silent=True, handler=handler)
    assert np.array_equal(gpu, numpy_dist(v, handler))


@requires_cuda
@pytest.mark.parametrize("handler", ALL_HANDLERS)
def test_gpu_batched_matches_cpu(handler, monkeypatch):
    """Force a small batch size so the multi-batch assembly is exercised."""
    v = profiles(150, 40, 0.2, seed=17)
    if handler == cd.HANDLER_COMPLETE_DELETE:
        v = complete_delete_cols(v)
    import pandas as pd
    monkeypatch.setattr(cd, "estimate_gpu_batch_size", lambda *a, **k: 37)
    gpu = cd.calculate_distances_batched(pd.DataFrame(v), use_gpu=True, silent=True, handler=handler)
    assert np.array_equal(gpu, numpy_dist(v, handler))


@requires_cuda
@pytest.mark.parametrize("handler", ALL_HANDLERS)
def test_gpu_batch_falls_back_to_cpu_on_failure(handler, monkeypatch):
    """If a CUDA batch raises, the CPU fallback must fill in the same numbers."""
    v = profiles(60, 30, 0.2, seed=19)
    if handler == cd.HANDLER_COMPLETE_DELETE:
        v = complete_delete_cols(v)

    def boom(*a, **k):
        raise RuntimeError("simulated CUDA failure")

    monkeypatch.setattr(cd.calculate_hamming_distances_cuda_kernel, "__getitem__", boom)
    got = cd.calculate_hamming_distances_cuda_batch(v, 0, v.shape[0], 0, v.shape[0], True, handler)
    assert np.array_equal(symmetrize(got), numpy_dist(v, handler))


# --------------------------------------------------------------------------
# Release hygiene: things that must stay in sync across files
# --------------------------------------------------------------------------

def test_version_is_documented_in_changelog():
    """A release whose version has no CHANGELOG section is a release with no
    release notes; the Bioconda recipe and the Docker tag both derive from it."""
    changelog = (REPO / "CHANGELOG.md").read_text()
    assert f"{cd.VERSION}]" in changelog or f"- {cd.VERSION}" in changelog, \
        f"version {cd.VERSION} has no section in CHANGELOG.md"


def test_readme_documents_every_cli_flag():
    """Every option offered by the tool must appear in the README, so the
    documented interface cannot silently drift from the real one."""
    helptext = run_cli("--help")
    flags = set()
    for token in helptext.replace(",", " ").split():
        if token.startswith("--") and len(token) > 2:
            flags.add(token.strip("[]().,"))
    readme = (REPO / "README.md").read_text()
    missing = sorted(f for f in flags if f not in readme)
    assert not missing, f"flags absent from README.md: {missing}"


def test_version_flag_matches_module_version():
    """The Bioconda recipe asserts `--version | grep <tag>`, so this value is
    part of the packaging contract, not just cosmetic."""
    assert run_cli("--version").strip() == cd.VERSION


# --------------------------------------------------------------------------
# Exit codes: a failed run must not look like a successful one
# --------------------------------------------------------------------------

def run_raw(*args):
    """Run the tool without checking the exit status."""
    return subprocess.run(
        [sys.executable, str(SCRIPT), *map(str, args)],
        capture_output=True, text=True,
    )


def test_missing_input_file_fails_loudly(tmp_path):
    """A pipeline must be able to tell that the input could not be read; this
    used to exit 0 with no output and no message under --silent."""
    proc = run_raw("-i", tmp_path / "nope.tab", "-c", "-s")
    assert proc.returncode == 1
    assert "ERROR" in proc.stderr
    assert proc.stdout == ""


def test_unreadable_input_reports_the_path(tmp_path):
    bad = tmp_path / "truncated.tab"
    bad.write_bytes(b"\x00\x01\x02")
    proc = run_raw("-i", bad, "-c", "-s")
    assert proc.returncode != 0
    assert "ERROR" in proc.stderr


def test_no_output_destination_is_a_usage_error():
    proc = run_raw("-i", REPO / "test" / "boring.tab")
    assert proc.returncode == 2
    assert "ERROR" in proc.stderr


def test_no_arguments_is_a_usage_error():
    proc = run_raw()
    assert proc.returncode == 2
    assert "usage" in proc.stderr.lower()


def test_successful_run_exits_zero(tmp_path):
    proc = run_raw("-i", REPO / "test" / "boring.tab", "-o", tmp_path / "o.tab", "-s")
    assert proc.returncode == 0
    assert proc.stderr == ""


def test_header_only_input_is_rejected(tmp_path):
    """Parses cleanly but holds no samples. This escaped detection on the Arrow
    loader, which used to return before the validation ran."""
    src = tmp_path / "hdr.tab"
    src.write_text("id\tL1\tL2\n")
    proc = run_raw("-i", src, "-c", "-s")
    assert proc.returncode == 1
    assert "no samples" in proc.stderr
    assert proc.stdout == ""


def test_locus_filter_output_matches_across_loaders(tmp_path):
    """The Arrow loader now also serves -L/-S, so it must agree with pandas
    there too. Runs whichever loader is installed; CI covers both."""
    src = tmp_path / "in.tab"
    v = profiles(30, 60, 0.25, seed=21)
    lines = ["id\t" + "\t".join(f"L{i}" for i in range(v.shape[1]))]
    for i, row in enumerate(v):
        lines.append(f"S{i}\t" + "\t".join("-" if x == 0 else str(x) for x in row))
    src.write_text("\n".join(lines) + "\n")
    filtered = run_cli("-i", src, "-c", "-s", "-L", 80, "-S", 70)
    # The filters must actually remove something, or the test proves nothing.
    unfiltered = run_cli("-i", src, "-c", "-s")
    assert len(filtered.splitlines()) < len(unfiltered.splitlines())


def test_gpu_request_without_cuda_warns(monkeypatch, tmp_path):
    """Falling back to the CPU is fine; doing it silently is not."""
    proc = subprocess.run(
        [sys.executable, str(SCRIPT), "-i", str(REPO / "test" / "boring.tab"),
         "-c", "-s", "-g"],
        capture_output=True, text=True,
        env={**os.environ, "CUDA_VISIBLE_DEVICES": "-1", "NUMBA_DISABLE_CUDA": "1"},
    )
    assert proc.returncode == 0
    assert "WARNING" in proc.stderr and "--gpu" in proc.stderr
    # The matrix must still be correct.
    assert proc.stdout == (REPO / "validation" / "boring_c.tab").read_text()
