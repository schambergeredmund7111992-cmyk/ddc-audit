"""Generate the small example dataset for the quickstart.

Three cell lines x nine drugs x 300 genes. Three predictors are produced, and
every file is labelled ``synthetic`` in its name so it can never be mistaken
for an experimental result:

``predictions_collapsed``      near-identical profiles per drug: high per-pair
                               correlation, DDC near chance. This is the
                               illusion the control exists to expose.
``predictions_drug_aware``     the same field plus 60% of each drug's own
                               signal: DDC clears chance but not the ceiling.
``predictions_oracle``         the measured targets themselves: the upper
                               bound, DDC 1.0.

All three are stored as ``.npy`` beside ``targets.npy`` and ``meta.csv``, so
the quickstart exercises the exact contract documented in the README.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

N_DRUGS = 9
N_CELLS = 3
N_GENES = 300
DRUG_NAMES = [f"drug_{i:02d}" for i in range(N_DRUGS)]
CELL_LINES = ["A549", "K562", "MCF7"]


def build(out: Path, seed: int = 0) -> None:
    rng = np.random.default_rng(seed)
    meta = pd.DataFrame(
        {
            "drug_id": [drug for _ in CELL_LINES for drug in DRUG_NAMES],
            "cell_line": [cell for cell in CELL_LINES for _ in DRUG_NAMES],
        }
    )
    base = rng.normal(size=(N_CELLS, N_GENES))          # cell-line shared structure
    true = np.stack(
        [base[i // N_DRUGS] + rng.normal(0, 0.05, N_GENES) for i in range(len(meta))]
    )
    collapsed = np.stack(
        [base[i // N_DRUGS] + rng.normal(0, 0.02, N_GENES) for i in range(len(meta))]
    )
    drug_aware = 0.6 * true + 0.4 * collapsed
    oracle = true.copy()

    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / "predictions_collapsed.npy", collapsed.astype(np.float32))
    np.save(out / "predictions_drug_aware.npy", drug_aware.astype(np.float32))
    np.save(out / "predictions_oracle.npy", oracle.astype(np.float32))
    np.save(out / "targets.npy", true.astype(np.float32))
    meta.to_csv(out / "meta.csv", index=False)
    print(f"SYNTHETIC example dataset written to {out} "
          f"({N_CELLS} cell lines x {N_DRUGS} drugs x {N_GENES} genes)")
    print("these are simulated profiles, not experimental measurements")
    print("quickstart:")
    print(
        "  ddc-audit run --predictions examples/synthetic/predictions_collapsed.npy "
        "--targets examples/synthetic/targets.npy "
        "--meta examples/synthetic/meta.csv --out /tmp/audit_collapsed"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("example"))
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    build(args.out, args.seed)
