"""Freeze the author's reference implementation's output into a small artifact.

The research repository is not shipped with this package, so the live
equivalence test can only skip on a judging machine. This artifact replaces the
live comparison with a stored one: the exact inputs, and the exact outputs the
author's own ``eval/metrics.py`` produced for them where the reference was
available. The test that reads it then runs everywhere.

The inputs are chosen to be non-degenerate -- a partially collapsed predictor,
so the AUC lands strictly between chance and 1.

    python tools/freeze_reference_metric.py

Set CYTOBRIDGE_REPO to the research checkout; it defaults to
``~/cytobridge-benchmark``. The reference implementation lives in the
`cytobridge-benchmark` research repository, not here. Re-run only if the packaged metric changes.
"""
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

S = Path(__file__).resolve().parents[1]
REF = Path(os.environ.get("CYTOBRIDGE_REPO", Path.home() / "cytobridge-benchmark"))
OUT = S / "tests" / "data" / "reference_metric_outputs.json"
OUT.parent.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(REF))
from eval.metrics import drug_discrimination_score as reference  # noqa: E402

sys.path.insert(0, str(S / "src"))
from ddc_audit.metrics import drug_discrimination_score as ours  # noqa: E402


def build_inputs(seed: int = 1, n_genes: int = 500):
    """Documented generator: any reader can regenerate these exact arrays."""
    rng = np.random.default_rng(seed)
    cl = np.array([c for c in ("A549", "K562", "MCF7") for _ in range(9)])
    true = rng.normal(size=(27, n_genes))
    base = np.stack([true[cl == c].mean(0) for c in cl])
    pred = 0.03 * true + 0.97 * base + rng.normal(0, 0.35, (27, n_genes))
    return pred, true, cl


pred, true, cl = build_inputs()
cases = []
for top_k, metric in ((50, "pearson"), (50, "spearman"), (None, "pearson"),
                      (20, "pearson")):
    theirs = reference(pred, true, cl, top_k=top_k, metric=metric)
    mine = ours(pred, true, cl, top_k=top_k, metric=metric)
    diff = abs(theirs["specificity_auc"] - mine["specificity_auc"])
    cases.append({
        "top_k": top_k,
        "metric": metric,
        "specificity_auc": float(theirs["specificity_auc"]),
        "gap": float(theirs["gap"]),
        "on_diag_mean": float(theirs["on_diag_mean"]),
        "off_diag_mean": float(theirs["off_diag_mean"]),
        "wilcoxon_p_on_gt_off": float(theirs["wilcoxon_p_on_gt_off"]),
        "our_specificity_auc": float(mine["specificity_auc"]),
    })
    print(f"top_k={str(top_k):5s} {metric:8s} reference "
          f"{theirs['specificity_auc']:.12f}  ours {mine['specificity_auc']:.12f}"
          f"  |diff| {diff:.1e}")

payload = {
    "source": "eval/metrics.py::drug_discrimination_score",
    "source_repository": "github.com/schambergeredmund7111992-cmyk/cytobridge-benchmark",
    "source_commit": subprocess.run(
        ["git", "-C", str(REF), "rev-parse", "HEAD"],
        capture_output=True, text=True, check=True,
    ).stdout.strip(),
    "source_sha256": hashlib.sha256((REF / "eval" / "metrics.py").read_bytes()).hexdigest(),
    "generator": "tools/freeze_reference_metric.py",
    "input_recipe": (
        "rng = numpy.random.default_rng(1); cl = 3 cell lines x 9; "
        "true = rng.normal(size=(27, 500)); base = per-cell-line mean of true; "
        "pred = 0.03*true + 0.97*base + rng.normal(0, 0.35, (27, 500))"
    ),
    "seed": 1,
    "n_pairs": 27,
    "n_genes": 500,
    "cell_lines": cl.tolist(),
    "cases": cases,
}
OUT.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
print(f"\nwrote {OUT} ({OUT.stat().st_size} bytes)")
print("source commit", payload["source_commit"][:12],
      "| metrics.py sha256", payload["source_sha256"][:16])
