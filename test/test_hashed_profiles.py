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


def load(path, loader, monkeypatch):
    if loader == "pandas":
        monkeypatch.setattr(cd, "pyarrow_load", lambda *a, **k: None)
    data, _, _ = cd.load_data_optimized(str(path), silent=True)
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
