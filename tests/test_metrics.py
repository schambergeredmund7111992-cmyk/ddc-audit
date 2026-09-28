"""Core metric semantics.

The reference-implementation equivalence test locates the research repository
through ``DDC_AUDIT_REFERENCE_REPO`` or a sibling checkout. When the repository
is absent the test is *skipped*, never silently passed.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pytest

from ddc_audit.metrics import (
    drug_discrimination_score,
    inter_drug_pearson,
    per_pair_spearman,
)

CANDIDATE_REPOS = [
    os.environ.get("DDC_AUDIT_REFERENCE_REPO"),
    str(Path.home() / "cytobridge-benchmark"),
    str(Path(__file__).resolve().parents[1].parent / "cytobridge-benchmark"),
]


def _reference_repo() -> Path | None:
    for candidate in CANDIDATE_REPOS:
        if not candidate:
            continue
        path = Path(candidate)
        if (path / "eval" / "metrics.py").is_file():
            return path
    return None


REFERENCE_REPO = _reference_repo()

requires_reference = pytest.mark.skipif(
    REFERENCE_REPO is None,
    reason="reference repository not found; set DDC_AUDIT_REFERENCE_REPO to enable",
)


def _blocks(n_pairs: int = 27) -> np.ndarray:
    """Cell-line labels in blocks of 9 (A549 x9, K562 x9, MCF7 x9)."""
    return np.array(
        [cell for cell in ("A549", "K562", "MCF7") for _ in range(n_pairs // 3)]
    )


def _synthetic(n_pairs=27, n_genes=300, seed=0, signal=0.6):
    rng = np.random.default_rng(seed)
    cl = _blocks(n_pairs)
    base = rng.normal(size=(3, n_genes))
    true = np.stack(
        [base[i // (n_pairs // 3)] + rng.normal(0, 0.05, n_genes)
         for i in range(n_pairs)]
    )
    collapsed = np.stack(
        [base[i // (n_pairs // 3)] + rng.normal(0, 0.02, n_genes)
         for i in range(n_pairs)]
    )
    pred = signal * true + (1 - signal) * collapsed
    return pred, true, cl


def _exact_collapsed(n_pairs=27, n_genes=300, seed=0):
    """A predictor that is EXACTLY constant per cell line: every on/off
    similarity is a tie and the DDC is exactly 0.5."""
    rng = np.random.default_rng(seed)
    base = rng.normal(size=(3, n_genes))
    pred = np.stack([base[i // (n_pairs // 3)] for i in range(n_pairs)])
    true = pred + rng.normal(0, 0.05, (n_pairs, n_genes))
    return pred, true, _blocks(n_pairs)


def test_oracle_scores_one():
    _, true, cl = _synthetic()
    score = drug_discrimination_score(true, true, cl)
    assert score["specificity_auc"] == pytest.approx(1.0, abs=1e-9)
    assert score["gap"] > 0


def test_collapsed_predictor_scores_chance():
    # Exactly constant per cell line -> every comparison is a tie -> ties
    # score as non-wins -> AUC exactly 0.5 (the paper's no-drug-info anchor).
    pred, true, cl = _exact_collapsed()
    score = drug_discrimination_score(pred, true, cl)
    assert score["specificity_auc"] == pytest.approx(0.5, abs=1e-9)


def test_drug_aware_scores_above_chance():
    pred, true, cl = _synthetic(signal=0.6)
    score = drug_discrimination_score(pred, true, cl)
    assert score["specificity_auc"] > 0.7


def test_auc_and_gap_are_consistent():
    pred, true, cl = _synthetic()
    score = drug_discrimination_score(pred, true, cl)
    assert 0.0 <= score["specificity_auc"] <= 1.0
    assert abs(score["gap"] - (score["on_diag_mean"] - score["off_diag_mean"])) < 1e-12
    assert score["n_pairs_scored"] == 27


def test_spearman_and_interdrug_shapes():
    pred, true, cl = _synthetic()
    rho = per_pair_spearman(true, pred, top_k=50)
    assert rho.shape == (27,)
    assert np.isfinite(rho).all()
    inter = inter_drug_pearson(pred, cl)
    assert 0.9 <= inter <= 1.0  # near-collapsed synthetic


def test_near_collapsed_predictor_stays_near_chance():
    """A near-constant predictor (inter-drug r ~ 1.0, tiny per-pair noise)
    stays near chance under the DDC: residual noise is not credited as drug
    discrimination. The exactly-constant case is the exact 0.5 anchor."""
    pred, true, cl = _synthetic(signal=0.0)  # base + noise(0.02)
    score = drug_discrimination_score(pred, true, cl)
    assert abs(score["specificity_auc"] - 0.5) < 0.08
    assert inter_drug_pearson(pred, cl) > 0.9  # yet the collapse meter is ~1


def test_exact_ties_are_scored_as_losses_not_half_wins():
    """Documented behaviour: a tie is a miss, not half a win.

    With identical prediction and truth rows every comparison ties; under the
    strict rule the anchor wins nothing.
    """
    rng = np.random.default_rng(3)
    base = rng.normal(size=(1, 60))
    pred = np.repeat(base, 6, axis=0)
    true = np.repeat(base, 6, axis=0)
    cl = np.array(["A549"] * 6)
    assert drug_discrimination_score(pred, true, cl)["specificity_auc"] == pytest.approx(
        0.0, abs=1e-12
    )


def test_constant_prediction_row_does_not_crash():
    """A degenerate prediction row still scores; it simply wins nothing."""
    pred, true, cl = _synthetic(signal=0.6)
    pred = pred.copy()
    pred[4] = 7.0
    score = drug_discrimination_score(pred, true, cl)
    assert 0.0 <= score["specificity_auc"] <= 1.0
    assert score["n_pairs_scored"] == 27


def test_matches_the_frozen_reference_outputs():
    """Equivalence against the author's implementation, without needing it here.

    ``tests/data/reference_metric_outputs.json`` records the inputs and the
    exact outputs that ``eval/metrics.py`` at commit 87189db produced for them
    (regenerate with ``tools/freeze_reference_metric.py``). This test runs on
    every machine; the live comparison below it can only run where the research
    repository is checked out.
    """
    import json
    from pathlib import Path

    payload = json.loads(
        (Path(__file__).parent / "data" / "reference_metric_outputs.json")
        .read_text(encoding="utf-8")
    )
    cl = np.array(payload["cell_lines"])
    lines = list(dict.fromkeys(cl.tolist()))
    rng = np.random.default_rng(payload["seed"])
    true = rng.normal(size=(payload["n_pairs"], payload["n_genes"]))
    # per-cell-line mean of `true`, broadcast back to the rows of that line
    base = np.stack([true[cl == c].mean(0) for c in lines])
    row_of_line = np.array([lines.index(c) for c in cl.tolist()])
    pred = (0.03 * true + 0.97 * base[row_of_line]
            + rng.normal(0, 0.35, true.shape))

    assert payload["cases"], "the frozen artifact records no cases"
    for case in payload["cases"]:
        ours = drug_discrimination_score(
            pred, true, cl, top_k=case["top_k"], metric=case["metric"]
        )
        assert ours["specificity_auc"] == pytest.approx(
            case["specificity_auc"], abs=1e-12
        ), f"case {case['top_k']}/{case['metric']}"
        assert ours["gap"] == pytest.approx(case["gap"], abs=1e-12)
        assert ours["on_diag_mean"] == pytest.approx(case["on_diag_mean"], abs=1e-12)
        assert ours["off_diag_mean"] == pytest.approx(case["off_diag_mean"], abs=1e-12)
        assert ours["wilcoxon_p_on_gt_off"] == pytest.approx(
            case["wilcoxon_p_on_gt_off"], abs=1e-12
        )
    # the recorded inputs must not be degenerate, or the check proves nothing
    assert 0.5 < payload["cases"][0]["specificity_auc"] < 1.0


@requires_reference
def test_numerically_identical_to_reference_implementation():
    """Same random input must give the same DDC output as the paper repo."""
    sys.path.insert(0, str(REFERENCE_REPO))
    from eval.metrics import drug_discrimination_score as reference

    rng = np.random.default_rng(1)
    cl = np.array([cell for cell in ("A549", "K562", "MCF7") for _ in range(9)])
    true = rng.normal(size=(27, 500))
    pred = 0.3 * true + rng.normal(0, 0.3, (27, 500))
    for top_k, metric in ((50, "pearson"), (50, "spearman"), (None, "pearson")):
        ours = drug_discrimination_score(pred, true, cl, top_k=top_k, metric=metric)
        theirs = reference(pred, true, cl, top_k=top_k, metric=metric)
        assert ours["specificity_auc"] == pytest.approx(
            theirs["specificity_auc"], abs=1e-12
        )
        assert ours["gap"] == pytest.approx(theirs["gap"], abs=1e-12)
        assert ours["wilcoxon_p_on_gt_off"] == pytest.approx(
            theirs["wilcoxon_p_on_gt_off"], abs=1e-12
        )
