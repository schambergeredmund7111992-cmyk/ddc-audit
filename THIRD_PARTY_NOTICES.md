# Third-party notices

Licences and provenance, recorded per input rather than applied in bulk. No
blanket licence is asserted over third-party data: each upstream source keeps
its own terms, and the files redistributed here are either derived summaries
that those terms permit or files the author's own MIT-licensed release already
distributes.

## This package

**MIT** — see `LICENSE`. Covers the audit tool (`code/ddc_audit/`), the
regeneration pipeline (`code/regeneration/regenerate_paper_numbers.py`), and the
derived matrices the pipeline computes
(`target_perpair.npy`, `target_pooled.npy`, `prediction_loss_only_pooled.npy`,
and the pseudobulk matrices).

## Redistributed from the author's release

The release `cytobridge-benchmark` (commit `87189db`, `release_asoc/`) is
**MIT**-licensed by its author; its `LICENSE` file is reproduced in spirit here
and the original is at `release_asoc/LICENSE` in that repository.

| file here | from there | licence |
|---|---|---|
| `code/regeneration/bundle/prediction_loss_only_perpair.npy` | `release_asoc/predictions/cytobridge/logfc_pred_t7_sub_loss_only.npy` | MIT |
| `code/regeneration/bundle/pseudobulk_control.npy` | derived from `release_asoc/split/sciplex_test_control_counts.npy` | derived; MIT for the derivation, upstream data under its own terms (below) |
| `code/regeneration/bundle/pseudobulk_treated.npy` | derived from `release_asoc/split/sciplex_test_treated_counts.npy` | as above |
| `data/pair_order.csv`, `data/internal_splits.json` | the release's pair metadata and frozen split | MIT |

## Upstream data

| source | terms | how it is used here |
|---|---|---|
| **sci-Plex** — Srivatsan et al., *Massively multiplex chemical transcriptomics at single-cell resolution*, Science 367:45–51, 2020. GEO **GSE139944**. | Distributed by GEO under NCBI's terms of use; the authors' processed matrices are provided for research use. | Only derived pseudobulk summaries are redistributed (means over cells). The raw h5ad is **not** included. |
| **MSigDB Hallmark** — Liberzon et al., Cell Systems 1:417–425, 2015. | MSigDB terms of use (registration required upstream). | Used as a gene-set reference in the paper's pathway analysis. No MSigDB content is redistributed here. |
| **PubChem** | public domain | Not redistributed; the fingerprint-based oracle rungs that use SMILES are not recomputed in this package. |

## Audited methods, cited but not redistributed

scGPT (Cui et al., Nature Methods 21:1470–1480, 2024), MolFormer / MoLFormer
(Ross et al., Nature Machine Intelligence 4:1256–1264, 2022), chemCPA (Hetzel
et al., NeurIPS 35, 2022) and biolord (Piran et al., Nature Biotechnology
42:1678–1683, 2024) are the model families discussed in the accompanying paper.
**No code, weights, checkpoints or outputs of these methods are redistributed in
this package.** The audit operates on stored prediction matrices only.

## Python dependencies

| package | licence |
|---|---|
| numpy | BSD-3-Clause |
| scipy | BSD-3-Clause |
| pandas | BSD-3-Clause |
| matplotlib | PSF-based (matplotlib licence) |
| pytest (test-only) | MIT |
| rdkit, scikit-learn (optional, for the fingerprint oracle rungs) | BSD-3-Clause |

## Data protection

No human-subject data, clinical records, or personally identifiable information
is used. All inputs are public cell-line perturbation screens.
