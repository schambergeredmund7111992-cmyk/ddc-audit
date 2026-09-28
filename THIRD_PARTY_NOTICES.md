# Third-party notices

Licences and provenance, recorded per input rather than applied in bulk. **No
blanket licence is asserted over third-party data**: each upstream source keeps
its own terms, and the files redistributed here are either derived summaries
those terms permit or files the author's own MIT-licensed release already
distributes.

## This repository

**MIT** — see `LICENSE`. That licence covers:

- the audit tool (`src/ddc_audit/`);
- the evidence-reproduction pipeline (`scripts/`);
- the derived matrices this repository computes — `target_perpair.npy`,
  `target_pooled.npy`, `prediction_loss_only_pooled.npy`, and the pseudobulk
  matrices under `evidence/`.

It does **not** apply to upstream data redistributed from other sources
(sci-Plex, MSigDB Hallmark) or to the audited methods' code and weights, none of
which is redistributed here. The sections below say what applies to what.

## Redistributed from the research release

The research release
[cytobridge-benchmark](https://github.com/schambergeredmund7111992-cmyk/cytobridge-benchmark)
(commit `87189db`) is **MIT**-licensed by its author.

| file here | from there |
|---|---|
| `evidence/prediction_loss_only_perpair.npy` | `release_asoc/predictions/cytobridge/logfc_pred_t7_sub_loss_only.npy` |
| `evidence/pseudobulk_control.npy` | derived from `release_asoc/split/sciplex_test_control_counts.npy` |
| `evidence/pseudobulk_treated.npy` | derived from `release_asoc/split/sciplex_test_treated_counts.npy` |
| `evidence/pair_order.csv`, `evidence/manifests/*.json` | the release's pair metadata and frozen split definition |
| `tests/data/reference_metric_outputs.json` | output of the release's `eval/metrics.py`, frozen as a comparison artefact |

## Upstream data

| source | terms | how it is used here |
|---|---|---|
| **sci-Plex** — Srivatsan et al., *Massively multiplex chemical transcriptomics at single-cell resolution*, Science 367:45–51, 2020. GEO **GSE139944**. | Distributed by GEO under NCBI's terms of use; the authors' processed matrices are provided for research use. | Only derived pseudobulk summaries are redistributed (means over cells). The raw h5ad is **not** included. |
| **MSigDB Hallmark** — Liberzon et al., Cell Systems 1:417–425, 2015. | MSigDB terms of use (registration upstream). | Used as a gene-set reference in the paper's pathway analysis. No MSigDB content is redistributed here. |
| **PubChem** | public domain | Not redistributed; the fingerprint-based oracle rungs that use SMILES are not recomputed here. |

## Audited methods — cited, not redistributed

scGPT (Cui et al., Nature Methods 21:1470–1480, 2024), MoLFormer (Ross et al.,
Nature Machine Intelligence 4:1256–1264, 2022), chemCPA (Hetzel et al., NeurIPS
35, 2022) and biolord (Piran et al., Nature Biotechnology 42:1678–1683, 2024)
are the model families discussed in the accompanying paper. **No code, weights,
checkpoints or outputs of these methods are redistributed here.** The audit
operates on stored prediction matrices only.

## Python dependencies

| package | licence |
|---|---|
| numpy | BSD-3-Clause |
| scipy | BSD-3-Clause |
| pandas | BSD-3-Clause |
| matplotlib | PSF-based (matplotlib licence) |
| pytest (test-only) | MIT |
| build (packaging, optional) | MIT |
| rdkit, scikit-learn (optional, only for the fingerprint oracle rungs that this repository does not run) | BSD-3-Clause |

## Data protection

No human-subject data, clinical records, or personally identifiable information
is used. All inputs are public cell-line perturbation screens.
