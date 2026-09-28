"""Figures for the audit report: on/off similarity distributions, the
calibration ladder, and the bootstrap distribution."""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from ddc_audit.metrics import _corr_vec


def on_off_dataframe(pred, true, cl, top_k=50, metric="pearson"):
    """on/off similarity values as a DataFrame with a 'kind' column."""
    import pandas as pd

    pred = np.asarray(pred, dtype=float)
    true = np.asarray(true, dtype=float)
    rows = []
    for cell in np.unique(cl):
        idx = np.flatnonzero(cl == cell)
        if idx.size < 2:
            continue
        P, T = pred[idx], true[idx]
        if top_k is None or top_k >= P.shape[1]:
            panel = np.arange(P.shape[1])
        else:
            panel = sorted(
                set().union(
                    *[
                        set(np.argsort(-np.abs(T[i]))[:top_k].tolist())
                        for i in range(len(idx))
                    ]
                )
            )
        Ps, Ts = P[:, panel], T[:, panel]
        for i in range(len(idx)):
            rows.append({"score": _corr_vec(Ps[i], Ts[i], metric), "kind": "on"})
            for j in range(len(idx)):
                if j != i:
                    rows.append(
                        {"score": _corr_vec(Ps[i], Ts[j], metric), "kind": "off"}
                    )
    return pd.DataFrame(rows)


def write_figures(
    out_dir: Path,
    *,
    pred, true, cl,
    bootstrap: dict | None,
    ladder: dict | None,
    legacy: dict | None = None,
    top_k=50,
    metric="pearson",
) -> list[Path]:
    """Write the audit figures into out_dir.

    ``onoff_distribution.pdf``, ``calibration_ladder.pdf`` and
    ``bootstrap_distribution.pdf``, plus ``bootstrap_comparison.pdf`` when a
    legacy row-resample bootstrap is supplied, so the two estimands can be
    compared side by side rather than one silently replacing the other.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []

    frame = on_off_dataframe(pred, true, cl, top_k, metric)
    fig, ax = plt.subplots(figsize=(3.4, 2.6), dpi=300)
    violin = ax.violinplot(
        [frame[frame.kind == "on"].score, frame[frame.kind == "off"].score],
        showmeans=True,
    )
    for body, color in zip(violin["bodies"], ["#2A9D8F", "#E76F51"]):
        body.set_facecolor(color)
        body.set_alpha(0.6)
    ax.set_xticks([1, 2])
    ax.set_xticklabels(["matched (on)", "mismatched (off)"], fontsize=7)
    ax.set_ylabel("similarity (correlation)")
    ax.set_title("drug-discrimination control", fontsize=8, loc="left")
    fig.tight_layout()
    path = out_dir / "onoff_distribution.pdf"
    fig.savefig(path)
    plt.close(fig)
    written.append(path)

    if ladder is not None:
        cal = ladder["calibration"]
        labels = ["random", "cell-line\nmean", "oracle",
                  "anchor\nper-pair", "anchor\npooled"]
        values = [
            ladder["random_auc"],
            ladder["cell_line_mean_auc"],
            ladder["oracle_auc"],
            cal["anchor_perpair_response_auc"],
            cal["anchor_pooled_response_auc"],
        ]
        fig, ax = plt.subplots(figsize=(3.9, 2.7), dpi=300)
        colors = ["#6C7A89", "#6C7A89", "#1F2937", "#E76F51", "#2A9D8F"]
        plotted = [0.0 if value is None else value for value in values]
        ax.bar(range(len(labels)), plotted, color=colors, width=0.66,
               edgecolor="white", lw=0.5)
        ax.axhline(0.5, color="#1F2937", ls=":", lw=0.9)
        lo = 0.5 - cal["tolerance"]
        hi = 0.5 + cal["tolerance"]
        ax.axhspan(lo, hi, color="#9CA3AF", alpha=0.18, zorder=0)
        model = cal["model_target_response_auc"]
        if model is not None:
            ax.axhline(model, color="#B45309", lw=1.2, ls="--")
            ax.text(len(labels) - 0.45, model + 0.02,
                    f"model {model:.3f}", color="#B45309", fontsize=6, ha="right")
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, fontsize=6.2)
        ax.set_ylim(0, 1.12)
        ax.set_ylabel("DDC AUC", fontsize=7)
        ax.set_title(
            f"construction calibration: {cal['calibration_status']}",
            fontsize=8, loc="left",
        )
        for i, value in enumerate(values):
            ax.text(i, (0.0 if value is None else value) + 0.03,
                    "n/a" if value is None else f"{value:.3f}",
                    ha="center", fontsize=6)
        fig.tight_layout()
        path = out_dir / "calibration_ladder.pdf"
        fig.savefig(path)
        plt.close(fig)
        written.append(path)

    if bootstrap is not None:
        fig, ax = plt.subplots(figsize=(3.4, 2.6), dpi=300)
        ax.hist(bootstrap["draws"], bins=30, color="#2A9D8F", alpha=0.75)
        ax.axvline(0.5, color="#1F2937", ls=":", lw=1)
        ax.axvline(bootstrap["observed"], color="#111827", lw=1.1)
        ax.axvspan(bootstrap["ci_lo"], bootstrap["ci_hi"], color="#E76F51", alpha=0.15)
        ax.set_xlabel("bootstrap DDC AUC")
        ax.set_ylabel("count")
        ax.set_title(
            f"drug-clustered, 95% CI "
            f"[{bootstrap['ci_lo']:.2f}, {bootstrap['ci_hi']:.2f}]",
            fontsize=8, loc="left",
        )
        fig.tight_layout()
        path = out_dir / "bootstrap_distribution.pdf"
        fig.savefig(path)
        plt.close(fig)
        written.append(path)

    if bootstrap is not None and legacy is not None:
        fig, ax = plt.subplots(figsize=(3.4, 2.6), dpi=300)
        ax.hist(legacy["draws"], bins=30, color="#9CA3AF", alpha=0.8,
                label="row resample (superseded)")
        ax.hist(bootstrap["draws"], bins=30, color="#2A9D8F", alpha=0.55,
                label="drug-clustered (used)")
        ax.axvline(bootstrap["observed"], color="#111827", lw=1.1)
        ax.set_xlabel("bootstrap DDC AUC")
        ax.set_ylabel("count")
        ax.set_title("bootstrap estimands differ", fontsize=8, loc="left")
        ax.legend(fontsize=5.5, frameon=False, loc="upper left")
        fig.tight_layout()
        path = out_dir / "bootstrap_comparison.pdf"
        fig.savefig(path)
        plt.close(fig)
        written.append(path)

    return written
