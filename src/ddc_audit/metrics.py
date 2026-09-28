"""Core scoring functions.

Numerically identical to the paper's reference implementation
(github.com/schambergeredmund7111992-cmyk/cytobridge-benchmark,
eval/metrics.py); adapted here so the audit tool is self-contained.
"""
from __future__ import annotations

import numpy as np
from scipy import stats

TOP_K = 50
METRIC = "pearson"

#: Two similarities count as tied when they differ by no more than this.
#: Chosen to sit far above float64 rounding at these magnitudes (about
#: 1e-16) and far below any difference a reader would call real (the
#: shipped panels separate by ~1e-2). See the note in
#: drug_discrimination_score for why a strict comparison is not portable.
TIE_ATOL = 1e-12


def _corr_vec(a: np.ndarray, b: np.ndarray, metric: str = "pearson") -> float:
    if metric == "pearson":
        a = np.asarray(a, dtype=float)
        b = np.asarray(b, dtype=float)
        a = a - a.mean()
        b = b - b.mean()
        denom = np.sqrt((a * a).sum() * (b * b).sum())
        if denom < 1e-12:
            return 0.0
        return float((a * b).sum() / denom)
    if metric == "spearman":
        value = float(stats.spearmanr(np.asarray(a), np.asarray(b)).statistic)
        return value if np.isfinite(value) else 0.0
    raise ValueError(f"unknown metric {metric!r}; use 'pearson' or 'spearman'")


def _corr_matrix(P: np.ndarray, T: np.ndarray, metric: str) -> np.ndarray:
    """Cross-correlation C[i, j] = corr(P[i], T[j]) for one cell line.

    Vectorised; identical to the per-pair ``_corr_vec`` loop it replaces
    (Pearson on centred vectors, a 0.0 guard for degenerate rows).
    """
    P = np.asarray(P, dtype=float)
    T = np.asarray(T, dtype=float)
    if metric == "pearson":
        Pc = P - P.mean(axis=1, keepdims=True)
        Tc = T - T.mean(axis=1, keepdims=True)
        denom = np.sqrt((Pc * Pc).sum(1))[:, None] * np.sqrt((Tc * Tc).sum(1))[None, :]
        with np.errstate(divide="ignore", invalid="ignore"):
            C = np.where(denom > 1e-12, (Pc @ Tc.T) / denom, 0.0)
        return np.where(np.isfinite(C), C, 0.0)
    if metric == "spearman":
        from scipy import stats

        out = np.empty((P.shape[0], T.shape[0]), dtype=float)
        for i in range(P.shape[0]):
            for j in range(T.shape[0]):
                value = float(stats.spearmanr(P[i], T[j]).statistic)
                out[i, j] = value if np.isfinite(value) else 0.0
        return out
    raise ValueError(f"unknown metric {metric!r}; use 'pearson' or 'spearman'")


def drug_discrimination_score(
    pred: np.ndarray,  # [n_pairs, D] predicted vectors (logFC or pathway)
    true: np.ndarray,  # [n_pairs, D] ground-truth vectors
    cell_lines,        # [n_pairs] cell-line label per pair
    top_k: int | None = TOP_K,
    metric: str = METRIC,
) -> dict:
    """Off-diagonal drug-discrimination control (paper Eq. 2).

    Within each cell line, build the cross-correlation C[i, j] =
    corr(pred_i, true_j) over the union of the per-drug top-k |true| genes,
    then:  on_diag_mean = mean_i C[i, i];
           off_diag_mean = mean_{i != j} C[i, j];
           gap = on_diag_mean - off_diag_mean;
           specificity_auc = mean_i #{j != i : C[i, i] > C[i, j]} / (m - 1).

    A collapsed model sits near AUC 0.5; a drug-aware model approaches 1.0.

    Ties. Two similarities count as tied when they differ by no more than
    ``TIE_ATOL`` (1e-12, i.e. thousands of times the float64 rounding error at
    these magnitudes), and a tie is scored as non-discriminating (0) rather
    than as a half win.

    The tolerance is not cosmetic. A strict ``>`` on mathematically equal
    inputs is decided by BLAS reduction order: ``C[i, i]`` and ``C[i, j]``
    accumulate the same products in different orders, so a backend is free to
    return values that differ in the last bit. On the maintainers' Windows
    build an exactly collapsed panel produces bit-identical entries and scores
    0.0; the same code on macOS Accelerate produces last-bit differences and
    scored 16 of 30 comparisons as wins. A tool whose whole purpose is to
    recognise an exactly collapsed predictor cannot have its answer depend on
    which BLAS is installed.

    On the shipped loss-only matrices nothing changes: the 27 anchors score in
    increments of 1/8 and none of the 216 off-diagonal similarities comes near
    its diagonal, so this returns 0.5694 exactly as the strict rule did.
    """
    cl = np.asarray(cell_lines)
    pred = np.asarray(pred, float)
    true = np.asarray(true, float)
    D = pred.shape[1]
    use_all = (top_k is None) or (top_k >= D)
    on_all, off_all, auc_all = [], [], []
    for c in np.unique(cl):
        m = np.flatnonzero(cl == c)
        if m.size < 2:
            continue
        P, T = pred[m], true[m]
        if use_all:
            panel = np.arange(D)
        else:
            panel = sorted(
                set().union(
                    *[
                        set(np.argsort(-np.abs(T[i]))[:top_k].tolist())
                        for i in range(m.size)
                    ]
                )
            )
        Ps, Ts = P[:, panel], T[:, panel]
        C = _corr_matrix(Ps, Ts, metric)
        for i in range(m.size):
            diag = C[i, i]
            offs = np.array([C[i, j] for j in range(m.size) if j != i], dtype=float)
            offs = offs[~np.isnan(offs)]
            if np.isnan(diag) or offs.size == 0:
                continue
            on_all.append(float(diag))
            off_all.append(float(np.mean(offs)))
            auc_all.append(float(np.mean(diag - offs > TIE_ATOL)))
    on_arr = np.array(on_all)
    off_arr = np.array(off_all)
    out = {
        "on_diag_mean": float(np.mean(on_arr)) if on_arr.size else float("nan"),
        "off_diag_mean": float(np.mean(off_arr)) if off_arr.size else float("nan"),
        "gap": float(np.mean(on_arr - off_arr)) if on_arr.size else float("nan"),
        "specificity_auc": float(np.mean(auc_all)) if auc_all else float("nan"),
        "n_pairs_scored": int(on_arr.size),
    }
    if on_arr.size >= 2 and np.any(on_arr != off_arr):
        try:
            out["wilcoxon_p_on_gt_off"] = float(
                stats.wilcoxon(on_arr, off_arr, alternative="greater").pvalue
            )
        except Exception:  # noqa: BLE001
            out["wilcoxon_p_on_gt_off"] = float("nan")
    else:
        out["wilcoxon_p_on_gt_off"] = float("nan")
    return out


def per_pair_spearman(
    true: np.ndarray, pred: np.ndarray, top_k: int = TOP_K
) -> np.ndarray:
    """Per-pair Spearman on each pair's own truth-ranked top-k genes."""
    true_arr = np.asarray(true, dtype=float)
    pred_arr = np.asarray(pred, dtype=float)
    if true_arr.ndim != 2 or pred_arr.ndim != 2 or true_arr.shape != pred_arr.shape:
        raise ValueError("true and pred must have the same [pairs, genes] shape.")
    if not np.isfinite(true_arr).all() or not np.isfinite(pred_arr).all():
        raise ValueError("true and pred must contain only finite values.")
    if top_k <= 1:
        raise ValueError("top_k must be at least 2 for a rank correlation.")
    n_select = min(int(top_k), true_arr.shape[1])
    gene_index = np.arange(true_arr.shape[1])
    scores = []
    for truth_row, pred_row in zip(true_arr, pred_arr):
        panel = np.lexsort((gene_index, -np.abs(truth_row)))[:n_select]
        truth_panel = truth_row[panel]
        pred_panel = pred_row[panel]
        if np.std(truth_panel) < 1e-12:
            raise ValueError("a pair row has constant truth on its top-k panel.")
        if np.std(pred_panel) < 1e-12:
            scores.append(0.0)
        else:
            value = float(stats.spearmanr(truth_panel, pred_panel).statistic)
            if not np.isfinite(value):
                raise ValueError("a pair row produced a non-finite Spearman.")
            scores.append(value)
    return np.asarray(scores, dtype=float)


def per_pair_pearson(
    true: np.ndarray, pred: np.ndarray, top_k: int = TOP_K
) -> np.ndarray:
    """Per-pair Pearson on each pair's own truth-ranked top-k genes."""
    true_arr = np.asarray(true, dtype=float)
    pred_arr = np.asarray(pred, dtype=float)
    gene_index = np.arange(true_arr.shape[1])
    n_select = min(int(top_k), true_arr.shape[1])
    scores = []
    for truth_row, pred_row in zip(true_arr, pred_arr):
        panel = np.lexsort((gene_index, -np.abs(truth_row)))[:n_select]
        scores.append(_corr_vec(pred_row[panel], truth_row[panel], "pearson"))
    return np.asarray(scores, dtype=float)


def inter_drug_pearson(pred_logfc: np.ndarray, cell_lines) -> float:
    """Mean pairwise Pearson of predicted profiles between different drugs
    within each cell line (the collapse meter)."""
    cl = np.asarray(cell_lines)
    vals = []
    for c in np.unique(cl):
        m = np.flatnonzero(cl == c)
        if m.size < 2:
            continue
        P = pred_logfc[m]
        if np.std(P) < 1e-12:
            vals.append(1.0)  # exactly collapsed
            continue
        C = np.corrcoef(P)
        iu = np.triu_indices(m.size, 1)
        vals.append(float(np.nanmean(C[iu])))
    return float(np.mean(vals)) if vals else float("nan")


def per_anchor_scores(
    pred: np.ndarray, true: np.ndarray, cell_lines, top_k=TOP_K, metric=METRIC
) -> np.ndarray:
    """Per-anchor discrimination scores, aligned with the input rows."""
    pred = np.asarray(pred, dtype=float)
    true = np.asarray(true, dtype=float)
    cl = np.asarray(cell_lines)
    out = np.full(len(pred), np.nan, dtype=float)
    for cell in np.unique(cl):
        rows = np.flatnonzero(cl == cell)
        if rows.size < 2:
            continue
        P, T = pred[rows], true[rows]
        if top_k is None or top_k >= P.shape[1]:
            panel = np.arange(P.shape[1])
        else:
            panel = sorted(
                set().union(
                    *[
                        set(np.argsort(-np.abs(T[i]))[:top_k].tolist())
                        for i in range(len(rows))
                    ]
                )
            )
        Ps, Ts = P[:, panel], T[:, panel]
        C = _corr_matrix(Ps, Ts, metric)
        for position, i in enumerate(rows):
            offs = np.delete(C[position], position)
            offs = offs[np.isfinite(offs)]
            if offs.size:
                out[i] = float(np.mean(C[position, position] - offs > TIE_ATOL))
    return out
