"""Input contract validation and the input report."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from ddc_audit.io import (
    InputError,
    check,
    load_all,
    load_manifest,
    load_matrix,
    validate,
)


def _write(tmp_path, pred=None, true=None, meta=None, n=27, g=300,
           drug_col="drug_id", cell_col="cell_line"):
    rng = np.random.default_rng(0)
    pred = pred if pred is not None else rng.normal(size=(n, g))
    true = true if true is not None else rng.normal(size=(n, g))
    meta = meta if meta is not None else pd.DataFrame(
        {drug_col: [f"d{i // 3}" for i in range(n)],
         cell_col: ["A549", "K562", "MCF7"] * (n // 3)}
    )
    np.save(tmp_path / "pred.npy", pred)
    np.save(tmp_path / "true.npy", true)
    meta.to_csv(tmp_path / "meta.csv", index=False)
    return tmp_path / "pred.npy", tmp_path / "true.npy", tmp_path / "meta.csv"


def test_valid_input_passes(tmp_path):
    paths = _write(tmp_path)
    pred, true, drugs, cells, _, _ = load_all(*paths)
    assert pred.shape == (27, 300)
    assert len(set(drugs.tolist())) == 9


def test_drug_and_cell_column_aliases_are_accepted(tmp_path):
    """The released demo metadata uses ``drug`` / ``cell_line``."""
    paths = _write(tmp_path, drug_col="drug", cell_col="cell_line")
    _, _, drugs, _, _, _ = load_all(*paths)
    assert len(set(drugs.tolist())) == 9


def test_shape_mismatch_rejected(tmp_path):
    paths = _write(tmp_path, true=np.random.default_rng(0).normal(size=(26, 300)))
    with pytest.raises(InputError, match="shape mismatch"):
        load_all(*paths)


def test_nan_rejected(tmp_path):
    bad = np.random.default_rng(0).normal(size=(27, 300))
    bad[0, 0] = np.nan
    paths = _write(tmp_path, pred=bad)
    with pytest.raises(InputError, match="NaN"):
        load_all(*paths)


def test_meta_row_count_rejected(tmp_path):
    meta = pd.DataFrame({"drug_id": [f"d{i}" for i in range(26)],
                         "cell_line": ["A549"] * 26})
    paths = _write(tmp_path, meta=meta)
    with pytest.raises(InputError, match="meta has"):
        load_all(*paths)


def test_missing_cell_line_column_rejected(tmp_path):
    meta = pd.DataFrame({"drug_id": [f"d{i}" for i in range(27)]})
    paths = _write(tmp_path, meta=meta)
    with pytest.raises(InputError, match="cell-line"):
        load_all(*paths)


def test_single_drug_per_line_rejected(tmp_path):
    n = 27
    meta = pd.DataFrame({"drug_id": [f"d{i}" for i in range(n)],
                         "cell_line": [f"c{i}" for i in range(n)]})
    rng = np.random.default_rng(0)
    pred = rng.normal(size=(n, 50))
    with pytest.raises(InputError, match="at least 2"):
        validate(pred, pred.copy(), meta)


def test_unsupported_suffix_rejected(tmp_path):
    path = tmp_path / "pred.parquet"
    path.write_bytes(b"x")
    with pytest.raises(InputError, match="unsupported suffix"):
        load_matrix(path, what="predictions")


def test_vehicle_shape_mismatch_rejected(tmp_path):
    paths = _write(tmp_path)
    meta = pd.read_csv(tmp_path / "meta.csv")
    bad = np.ones((3, 300))
    with pytest.raises(InputError, match="vehicle_profiles shape"):
        validate(np.zeros((27, 300)), np.zeros((27, 300)), meta, bad, bad)


def test_vehicle_negative_counts_rejected(tmp_path):
    paths = _write(tmp_path)
    meta = pd.read_csv(tmp_path / "meta.csv")
    bad = -np.ones((27, 300))
    with pytest.raises(InputError, match="negative"):
        validate(np.zeros((27, 300)), np.zeros((27, 300)), meta, bad, bad)


def test_check_reports_warnings(tmp_path):
    pred = np.ones((27, 300))          # every row constant
    paths = _write(tmp_path, pred=pred)
    p, t, _, _, _, _ = load_all(*paths)
    report = check(p, t, pd.read_csv(tmp_path / "meta.csv"))
    assert report["n_pairs"] == 27
    assert report["pairs_per_cell_line"] == {"A549": 9, "K562": 9, "MCF7": 9}
    assert report["vehicle_supplied"] is False
    assert any("constant" in w for w in report["warnings"])


def test_single_cell_line_rejected_before_any_scoring():
    """One cell line cannot host an off-diagonal control at all."""
    n = 27
    meta = pd.DataFrame({"drug_id": [f"d{i % 3}" for i in range(n)],
                         "cell_line": ["A549"] * n})
    rng = np.random.default_rng(0)
    with pytest.raises(InputError, match="only one cell line"):
        validate(rng.normal(size=(n, 50)), rng.normal(size=(n, 50)), meta)


def test_drug_scored_in_one_context_only_rejected():
    """Every drug must appear in at least two cell lines, or its anchors
    cannot be scored against anything."""
    n = 27
    cells = ["A549", "K562", "MCF7"] * 9
    drug = [f"d{i // 3}" for i in range(n)]
    drug[0] = "orphan"                      # appears once, in A549 only
    meta = pd.DataFrame({"drug_id": drug, "cell_line": cells})
    rng = np.random.default_rng(0)
    with pytest.raises(InputError, match="orphan"):
        validate(rng.normal(size=(n, 50)), rng.normal(size=(n, 50)), meta)


def test_repeated_condition_key_is_reported_not_rejected(tmp_path):
    """Two rows with the same (drug, cell_line) key are ambiguous; the loader
    accepts them so the audit can still run, and ``check`` warns."""
    n = 27
    meta = pd.DataFrame({
        "drug_id": [f"d{i // 3}" for i in range(n)],
        "cell_line": ["A549", "K562", "MCF7"] * 9,
    })
    # positional assignment: .loc on a default RangeIndex would collapse rows
    meta.iat[1, meta.columns.get_loc("drug_id")] = meta.iat[0, meta.columns.get_loc("drug_id")]
    meta.iat[1, meta.columns.get_loc("cell_line")] = meta.iat[0, meta.columns.get_loc("cell_line")]
    paths = _write(tmp_path, meta=meta)
    p, t, _, _, _, _ = load_all(*paths)
    report = check(p, t, pd.read_csv(tmp_path / "meta.csv"))
    assert any("duplicate (drug, cell_line)" in w for w in report["warnings"])


def test_manifest_round_trip(tmp_path):
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps({
        "response_space": "logFC vs pooled vehicle",
        "vehicle_construction": "one vehicle per cell line",
        "row_alignment": "positional, same order as targets.npy",
    }), encoding="utf-8")
    manifest = load_manifest(path)
    assert manifest["response_space"].startswith("logFC")


def test_manifest_rejects_unknown_keys(tmp_path):
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps({"answer": 42}), encoding="utf-8")
    with pytest.raises(InputError, match="unknown key"):
        load_manifest(path)
