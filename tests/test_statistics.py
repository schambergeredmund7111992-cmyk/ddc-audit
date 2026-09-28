"""Uncertainty quantification: the two bootstrap estimands and the permutation null.

The point of these tests is the *estimand*, not the seed: a bootstrap that
duplicates an anchor into its own negative pool is measurably biased low, and
the suite pins that difference down instead of tolerating it.
"""
from __future__ import annotations

import numpy as np
import pytest

from ddc_audit.statistics import (
    between_drug_sd,
    delete_one_drug_sd,
    drug_clustered_anchor_bootstrap,
    permutation_null,
    row_resample_bootstrap,
)


def _synthetic(n_pairs=27, n_genes=300, seed=0, signal=0.6):
    rng = np.random.default_rng(seed)
    block = n_pairs // 3
    cl = np.array(
        [cell for cell in ("A549", "K562", "MCF7") for _ in range(block)]
    )
    drugs = np.array([f"drug_{i // 3}" for i in range(n_pairs)])
    base = rng.normal(size=(3, n_genes))
    true = np.stack(
        [base[i // block] + rng.normal(0, 0.05, n_genes) for i in range(n_pairs)]
    )
    collapsed = np.stack(
        [base[i // block] + rng.normal(0, 0.02, n_genes) for i in range(n_pairs)]
    )
    pred = signal * true + (1 - signal) * collapsed
    return pred, true, cl, drugs


def _exact_collapsed(n_pairs=27, n_genes=300, seed=0):
    """Exactly constant per cell line: DDC is exactly 0.5."""
    rng = np.random.default_rng(seed)
    block = n_pairs // 3
    cl = np.array(
        [cell for cell in ("A549", "K562", "MCF7") for _ in range(block)]
    )
    drugs = np.array([f"drug_{i // 3}" for i in range(n_pairs)])
    base = rng.normal(size=(3, n_genes))
    pred = np.stack([base[i // block] for i in range(n_pairs)])
    true = pred + rng.normal(0, 0.05, (n_pairs, n_genes))
    return pred, true, cl, drugs


# --------------------------------------------------------------------------
# the corrected estimand
# --------------------------------------------------------------------------

def test_anchor_bootstrap_is_centered_on_the_point_estimate():
    pred, true, cl, drugs = _synthetic()
    result = drug_clustered_anchor_bootstrap(pred, true, cl, drugs, n_boot=400, seed=11)
    # a resample of the per-anchor scores cannot shift the mean
    assert result["bias"] == pytest.approx(0.0, abs=0.02)
    assert result["ci_lo"] <= result["observed"] <= result["ci_hi"]


def test_anchor_bootstrap_reproducible_with_seed():
    pred, true, cl, drugs = _synthetic()
    a = drug_clustered_anchor_bootstrap(pred, true, cl, drugs, n_boot=60, seed=7)
    b = drug_clustered_anchor_bootstrap(pred, true, cl, drugs, n_boot=60, seed=7)
    np.testing.assert_array_equal(a["draws"], b["draws"])
    assert 0.0 <= a["ci_lo"] <= a["ci_hi"] <= 1.0


def test_anchor_bootstrap_draw_count_scales_with_drugs():
    pred, true, cl, drugs = _synthetic()
    result = drug_clustered_anchor_bootstrap(pred, true, cl, drugs, n_boot=50, seed=2)
    assert result["n_drugs"] == 9
    assert result["draws"].shape == (50,)


def test_anchor_bootstrap_rejects_unscorable_anchors():
    """A cell line with a single drug cannot score any anchor."""
    pred, true, cl, drugs = _synthetic()
    cl_bad = cl.copy()
    cl_bad[0] = "lonely"  # orphan cell line, one row
    with pytest.raises(Exception):
        drug_clustered_anchor_bootstrap(pred, true, cl_bad, drugs, n_boot=10, seed=1)


# --------------------------------------------------------------------------
# the superseded estimand, pinned so the difference stays visible
# --------------------------------------------------------------------------

def test_row_resample_bootstrap_is_biased_low_when_duplicates_appear():
    """Duplicating an anchor puts a copy in its own negative pool, which the
    strict comparison scores as a loss, so the mean sits below the estimate."""
    pred, true, cl, drugs = _synthetic(signal=0.35, seed=5)
    corrected = drug_clustered_anchor_bootstrap(pred, true, cl, drugs, n_boot=400, seed=11)
    legacy = row_resample_bootstrap(pred, true, cl, drugs, n_boot=400, seed=11)
    assert legacy["observed"] == pytest.approx(corrected["observed"], abs=1e-12)
    assert legacy["bias"] < -0.005
    assert abs(legacy["bias"]) > abs(corrected["bias"])


def test_bootstraps_require_at_least_two_drugs():
    """One drug means one cluster: neither bootstrap can resample anything."""
    pred, true, cl, _ = _synthetic()
    single = np.array(["drug_a"] * 27)
    with pytest.raises(ValueError, match="at least 2 drugs"):
        row_resample_bootstrap(pred, true, cl, single, n_boot=10, seed=1)
    with pytest.raises(ValueError, match="at least 2 drugs"):
        drug_clustered_anchor_bootstrap(pred, true, cl, single, n_boot=10, seed=1)


def test_bootstraps_tolerate_a_duplicated_condition_key():
    """Two rows sharing a (drug, cell_line) key are treated as two anchors;
    the tool warns about them in ``check`` but must not crash."""
    pred, true, cl, drugs = _synthetic()
    drugs = drugs.copy()
    drugs[1] = drugs[0]                     # duplicate a condition
    anchors = drug_clustered_anchor_bootstrap(pred, true, cl, drugs, n_boot=20, seed=1)
    legacy = row_resample_bootstrap(pred, true, cl, drugs, n_boot=20, seed=1)
    assert 0.0 <= anchors["ci_lo"] <= anchors["ci_hi"] <= 1.0
    assert 0.0 <= legacy["ci_lo"] <= legacy["ci_hi"] <= 1.0


# --------------------------------------------------------------------------
# permutation null
# --------------------------------------------------------------------------

def test_permutation_null_of_collapsed_predictor():
    pred, true, cl, drugs = _exact_collapsed()
    result = permutation_null(pred, true, cl, n_perm=120, seed=3)
    assert result["observed"] == 0.5
    assert abs(result["null_mean"] - 0.5) < 0.02
    assert result["p_value"] == 1.0


def test_permutation_null_of_drug_aware_predictor():
    pred, true, cl, drugs = _synthetic(signal=0.8)
    result = permutation_null(pred, true, cl, n_perm=120, seed=3)
    assert result["p_value"] <= 1 / (120 + 1)  # no shuffled draw beats it


def test_permutation_null_never_duplicates_rows():
    """The null permutes within a cell line, so a null draw cannot contain a
    duplicated row and carries no duplicate-copy artifact."""
    pred, true, cl, drugs = _synthetic(signal=0.5)
    result = permutation_null(pred, true, cl, n_perm=50, seed=4)
    assert result["null_mean"] == pytest.approx(0.5, abs=0.08)
    assert result["test"].startswith("within-cell-line")


# --------------------------------------------------------------------------
# dispersion measures
# --------------------------------------------------------------------------

def test_loo_and_between_drug_sd_bounds():
    pred, true, cl, drugs = _synthetic()
    assert 0.0 <= delete_one_drug_sd(pred, true, cl, drugs) <= 1.0
    assert 0.0 <= between_drug_sd(pred, true, cl, drugs) <= 1.0


def test_delete_one_drug_sd_needs_two_drugs():
    pred, true, cl, drugs = _synthetic()
    assert np.isnan(delete_one_drug_sd(pred, true, cl, np.array(["only"] * 27)))
