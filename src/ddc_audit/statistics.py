"""Uncertainty quantification for the drug-discrimination control.

Estimand
--------
The reported endpoint is the *per-anchor* mean discrimination score:
``AUC = mean over the n anchors of (1/(m-1)) * sum_{j != i} 1[C_ii > C_ij]``,
where an anchor is one held-out (drug, cell line) pair (paper Eq. 2, Section IV).
Uncertainty is therefore a statement about *that mean*, and the resampling unit
is the anchor's score, not the raw matrix row.

Two resampling schemes are implemented, and the difference matters:

``drug_clustered_anchor_bootstrap`` (default; correct for this estimand)
    Resamples the held-out drugs with replacement, keeping each drug's cell
    lines together, and averages the per-anchor scores of the drawn drugs.
    Every drug carries the same number of anchors in every draw, so the
    resampled mean stays centered on the observed mean.

``row_resample_bootstrap`` (superseded; kept for traceability)
    Resamples matrix *rows* with replacement and rebuilds the cross-correlation
    matrix on the resampled rows. When a drug is drawn twice, its copies land
    in each other's off-diagonal sets carrying an on-diagonal-valued
    correlation. The strict ``diag > off`` rule counts those as losses, the
    negative pool grows faster than the positive set, and the distribution is
    biased low: on the shipped loss-only matrices the observed 0.5694 sits at
    the 97th percentile of the draws. Report it, never verdict on it.

The author's release reaches the same conclusion independently in
``release_asoc/code/compute_valid_bootstrap.py``; this module implements the
same estimand with the drug clustering retained.
"""
from __future__ import annotations

import numpy as np

from ddc_audit.metrics import drug_discrimination_score, per_anchor_scores

TOP_K = 50
METRIC = "pearson"


def _auc(pred, true, cl, top_k=TOP_K, metric=METRIC) -> float:
    return float(
        drug_discrimination_score(
            np.asarray(pred, dtype=float),
            np.asarray(true, dtype=float),
            np.asarray(cl),
            top_k=top_k,
            metric=metric,
        )["specificity_auc"]
    )


def _unique(values) -> np.ndarray:
    return np.asarray(sorted(set(np.asarray(values).tolist()), key=repr))


def drug_clustered_anchor_bootstrap(
    pred, true, cl, drugs, *, n_boot: int = 1000, seed: int = 7301,
    top_k=TOP_K, metric=METRIC,
) -> dict:
    """Bootstrap the per-anchor mean by resampling the held-out drugs.

    The unit of resampling is the drug, because the anchors of one drug share a
    compound and are not independent. A drawn anchor is scored against the drug
    panel actually present in that draw.
    """
    cl = np.asarray(cl)
    drugs = np.asarray(drugs)
    anchors = per_anchor_scores(pred, true, cl, top_k=top_k, metric=metric)
    finite = np.isfinite(anchors)
    if not finite.all():
        raise ValueError(
            f"{int((~finite).sum())} anchor(s) could not be scored; every anchor "
            "needs at least one scorable partner in its cell line."
        )
    unique = _unique(drugs)
    if unique.size < 2:
        raise ValueError("drug-clustered bootstrap needs at least 2 drugs.")
    per_drug = np.array([anchors[drugs == drug].mean() for drug in unique], dtype=float)
    observed = float(anchors.mean())
    rng = np.random.default_rng(seed)
    draws = per_drug[rng.integers(0, unique.size, size=(n_boot, unique.size))].mean(axis=1)
    return {
        "estimand": "mean of the per-anchor discrimination scores",
        "unit": "drug (all cell lines of a drawn drug kept together)",
        "observed": observed,
        "ci_lo": float(np.percentile(draws, 2.5)),
        "ci_hi": float(np.percentile(draws, 97.5)),
        "bootstrap_mean": float(draws.mean()),
        "bias": float(draws.mean() - observed),
        "draws": draws,
        "seed": seed,
        "n_boot": n_boot,
        "n_drugs": int(unique.size),
    }


def row_resample_bootstrap(
    pred, true, cl, drugs, *, n_boot: int = 1000, seed: int = 7301,
    top_k=TOP_K, metric=METRIC,
) -> dict:
    """Superseded row-level bootstrap, kept so earlier runs stay auditable.

    Drawing the same drug twice duplicates its rows, so a duplicated anchor
    meets a copy of itself as an off-diagonal neighbour and the strict
    comparison scores that copy as a loss. The estimator is therefore biased
    toward 0.5; use :func:`drug_clustered_anchor_bootstrap` for verdicts.

    It is also **numerically unstable by construction**: every draw takes
    ``diag > off`` on correlations that are frequently near-ties, so an input
    differing in the last float32 bit can move a quantile by ~1e-3. Two callers
    that compute their targets by different routes (one loading the shipped
    matrix, one recomputing it as ``log1p(treated) - log1p(control)``) will
    therefore disagree in the third decimal even though they agree on the
    observed value to 1e-7. Report it with that caveat and never verdict on it.
    """
    pred = np.asarray(pred, dtype=float)
    true = np.asarray(true, dtype=float)
    cl = np.asarray(cl)
    drugs = np.asarray(drugs)
    unique = _unique(drugs)
    if unique.size < 2:
        raise ValueError("drug-clustered bootstrap needs at least 2 drugs.")
    rng = np.random.default_rng(seed)
    observed = _auc(pred, true, cl, top_k, metric)
    draws = np.empty(n_boot, dtype=float)
    for i in range(n_boot):
        sampled = rng.choice(unique, size=unique.size, replace=True)
        rows = [int(k) for drug in sampled for k in np.flatnonzero(drugs == drug)]
        draws[i] = _auc(pred[rows], true[rows], cl[rows], top_k, metric)
    return {
        "estimand": "row-level resample of the discrimination AUC (biased low)",
        "unit": "matrix row (duplicates the anchor's own drug)",
        "observed": observed,
        "ci_lo": float(np.percentile(draws, 2.5)),
        "ci_hi": float(np.percentile(draws, 97.5)),
        "bootstrap_mean": float(draws.mean()),
        "bias": float(draws.mean() - observed),
        "draws": draws,
        "seed": seed,
        "n_boot": n_boot,
        "n_drugs": int(unique.size),
    }


def permutation_null(
    pred, true, cl, *, n_perm: int = 1000, seed: int = 7301,
    top_k=TOP_K, metric=METRIC,
) -> dict:
    """Shuffle predicted rows within each cell line, destroying the drug pairing.

    Permuting prediction rows among the drugs of one cell line (rather than
    resampling anchors) never duplicates a row, so the null carries no
    duplicate-copy artifact. The p-value is the plus-one estimate
    ``(r + 1) / (n + 1)``. This is a *permutation* p-value and answers a
    different question from the paired Wilcoxon p on the on-versus-off gap,
    which is directional only.
    """
    pred = np.asarray(pred, dtype=float)
    true = np.asarray(true, dtype=float)
    cl = np.asarray(cl)
    rng = np.random.default_rng(seed)
    observed = _auc(pred, true, cl, top_k, metric)
    null = np.empty(n_perm, dtype=float)
    shuffled = pred.copy()
    for i in range(n_perm):
        for cell in np.unique(cl):
            mask = cl == cell
            shuffled[mask] = rng.permutation(shuffled[mask])
        null[i] = _auc(shuffled, true, cl, top_k, metric)
    p_value = float((int(np.sum(null >= observed)) + 1) / (n_perm + 1))
    return {
        "test": "within-cell-line permutation of the predicted rows",
        "observed": observed,
        "null_mean": float(null.mean()),
        "null_sd": float(null.std(ddof=1)),
        "p_value": p_value,
        "null": null,
        "seed": seed,
        "n_perm": n_perm,
    }


def delete_one_drug_sd(pred, true, cl, drugs, *, top_k=TOP_K, metric=METRIC) -> float:
    """Standard deviation of the AUC across delete-one-drug subsets."""
    pred = np.asarray(pred, dtype=float)
    true = np.asarray(true, dtype=float)
    cl = np.asarray(cl)
    drugs = np.asarray(drugs)
    values = [
        _auc(pred[drugs != drug], true[drugs != drug], cl[drugs != drug], top_k, metric)
        for drug in _unique(drugs)
    ]
    return float(np.std(values, ddof=1)) if len(values) > 1 else float("nan")


def between_drug_sd(pred, true, cl, drugs, *, top_k=TOP_K, metric=METRIC) -> float:
    """SD across drugs of the per-drug mean anchor score."""
    anchors = per_anchor_scores(pred, true, cl, top_k=top_k, metric=metric)
    per_drug = [
        float(anchors[np.asarray(drugs) == drug].mean()) for drug in _unique(drugs)
    ]
    return float(np.std(per_drug, ddof=1)) if len(per_drug) > 1 else float("nan")
