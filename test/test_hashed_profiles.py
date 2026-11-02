"""chewBBACA hashed profiles: allele ids are CRC32 hashes up to 2^32-1.

Up to 0.1.7 every hash above 2^31-1 in a column that also held text (`*N`,
chewBBACA classes) went through float64 -> int32 and became INT32_MIN, so all
those hashes compared equal and distances collapsed (857 instead of 1196 on the
Listeria pair below). Both loaders must keep every hash distinct.
"""
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "cgmlst-dists.py"

spec = importlib.util.spec_from_file_location("cgmlst_dists", SCRIPT)
cd = sys.modules.get("cgmlst_dists")
if cd is None:
    cd = importlib.util.module_from_spec(spec)
    sys.modules["cgmlst_dists"] = cd
    spec.loader.exec_module(cd)

LOADERS = ["arrow", "pandas"]


def load(path, loader, monkeypatch, novel_as_allele=False):
    if loader == "pandas":
        monkeypatch.setattr(cd, "pyarrow_load", lambda *a, **k: None)
    data, _, _ = cd.load_data_optimized(str(path), silent=True, novel_as_allele=novel_as_allele)
    assert data is not None
    return data


def hamming(v):
    present = v != 0
    return np.array([((v[i] != v) & present[i] & present).sum(1) for i in range(len(v))])


@pytest.mark.parametrize("loader", LOADERS)
def test_large_hashes_stay_distinct(loader, tmp_path, monkeypatch):
    # locus a: equal near 2^32-1; b: differ just above 2^31; c: text forces a
    # string column (the case that used to collapse); d: missing on s1
    src = tmp_path / "h.tab"
    src.write_text(
        "FILE\ta\tb\tc\td\n"
        "s1\t4294967295\t2147483648\t3000000000\t-\n"
        "s2\t4294967295\t2147483649\t3000000001\t4000000000\n"
        "s3\t4294967294\t2147483648\t*12\tLNF\n"
    )
    v = load(src, loader, monkeypatch).to_numpy()
    assert hamming(v).tolist() == [[0, 2, 1], [2, 0, 2], [1, 2, 0]]


def test_compact_allele_ids_preserves_equality():
    big = np.iinfo(np.int32).max + 1
    df = pd.DataFrame({"a": [big, big, 0, big + 5], "b": [1, 2, 3, 4]}, dtype=np.int64)
    out = cd.compact_allele_ids(df)
    assert out.dtypes.eq(np.int32).all()
    a = out["a"].tolist()
    assert a[0] == a[1] and a[2] == 0 and a[3] not in (0, a[0])
    assert out["b"].tolist() == [1, 2, 3, 4]  # small ids untouched


@pytest.mark.parametrize("loader", LOADERS)
def test_listeria_hashed_profiles(loader, monkeypatch):
    """Real chewBBACA hashed calls (Pasteur cgMLST, 1748 loci). Expected values
    from an independent per-locus comparison, `*N` novel alleles as missing;
    the original C cgmlst-dists gives the same."""
    data = load(REPO / "test" / "listeria_hashed_3.tab", loader, monkeypatch)
    assert list(data.index) == ["ERR10109288", "ERR10109289", "ERR10109291"]
    assert hamming(data.to_numpy()).tolist() == [[0, 1196, 1199], [1196, 0, 35], [1199, 35, 0]]


@pytest.mark.parametrize("loader", LOADERS)
def test_listeria_hashed_profiles_novel_as_allele(loader, monkeypatch):
    """-N/--novel-as-allele: `*N` novel alleles compared as alleles (same
    independent reference, matches cgmlst-dists-rs -N)."""
    data = load(REPO / "test" / "listeria_hashed_3.tab", loader, monkeypatch, novel_as_allele=True)
    assert hamming(data.to_numpy()).tolist() == [[0, 1234, 1233], [1234, 0, 59], [1233, 59, 0]]


@pytest.mark.parametrize("loader", LOADERS)
def test_novel_alleles(loader, tmp_path, monkeypatch):
    # a: same *5; b: *5 vs *6; c: *5 vs integer 5 (different); d: *x is not novel
    src = tmp_path / "n.tab"
    src.write_text("FILE\ta\tb\tc\td\n" "s1\t*5\t*5\t*5\t*x\n" "s2\t*5\t*6\t5\t7\n")
    on = load(src, loader, monkeypatch, novel_as_allele=True).to_numpy()
    assert hamming(on).tolist() == [[0, 2], [2, 0]]
    off = load(src, loader, monkeypatch).to_numpy()
    assert hamming(off).tolist() == [[0, 0], [0, 0]]  # default: *N missing


def generated_hashed_table(path, n=200, n_loci=400, seed=7):
    """chewBBACA-like hashed table generated on the fly (replaces the former
    Git LFS `crc32` dataset): clonal CRC32 profiles with INF-, *N, LNF and '-'."""
    import random
    import zlib
    rng = random.Random(seed)
    founders = [[rng.randint(1, 40) for _ in range(n_loci)] for _ in range(10)]
    rows = []
    for i in range(n):
        cells = []
        for k, v in enumerate(founders[rng.randrange(10)]):
            r = rng.random()
            if r < 0.05:
                v += rng.randint(1, 20)
            h = str(zlib.crc32(f"L{k}:{v}".encode()))
            r = rng.random()
            cells.append("-" if r < 0.02 else "LNF" if r < 0.03 else f"*{rng.randint(1, 3)}"
                         if r < 0.05 else "INF-" + h if r < 0.08 else h)
        rows.append(cells)
    with open(path, "w") as f:
        f.write("FILE\t" + "\t".join(f"L{k}" for k in range(n_loci)) + "\n")
        for i, cells in enumerate(rows):
            f.write(f"S{i}\t" + "\t".join(cells) + "\n")
    return rows


def reference_matrix(rows, novel_as_allele):
    """Independent reference: per-locus text comparison of the raw cells."""
    def key(c):
        c = c[4:] if c.startswith("INF-") else c
        if c.isdigit() and int(c) != 0:
            return ("i", int(c))
        if novel_as_allele and c.startswith("*") and c[1:].isdigit():
            return ("n", c)
        return None
    keys = [[key(c) for c in r] for r in rows]
    n = len(rows)
    out = np.zeros((n, n), dtype=np.int64)
    for i in range(n):
        for j in range(i + 1, n):
            out[i, j] = out[j, i] = sum(1 for a, b in zip(keys[i], keys[j])
                                        if a is not None and b is not None and a != b)
    return out


@pytest.mark.parametrize("flags", [[], ["-N"]])
def test_generated_hashed_table_cli(flags, tmp_path):
    import subprocess
    src = tmp_path / "gen.tab"
    rows = generated_hashed_table(src)
    out = subprocess.run([sys.executable, str(SCRIPT), "-s", "-c", "-i", str(src), *flags],
                         capture_output=True, text=True, check=True).stdout
    got = np.array([l.split("\t")[1:] for l in out.splitlines()[1:]], dtype=np.int64)
    assert (got == reference_matrix(rows, novel_as_allele=bool(flags))).all()
