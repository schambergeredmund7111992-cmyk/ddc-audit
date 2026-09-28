# evidence/

The minimum numerical evidence needed to reproduce the two real audits in the
README, and nothing else. The full research pipeline, the model experiments and
the historical analysis live in
[cytobridge-benchmark](https://github.com/schambergeredmund7111992-cmyk/cytobridge-benchmark).

## What is here, and where it comes from

Everything below derives from the author's frozen split published in the research
repository at commit `87189db29653d0ee760714c84f012e494e5e761a` — specifically
`release_asoc/split/sciplex_test.parquet`,
`release_asoc/split/sciplex_test_control_counts.npy`,
`release_asoc/split/sciplex_test_treated_counts.npy` and
`release_asoc/predictions/cytobridge/logfc_pred_t7_sub_loss_only.npy`.

| file | shape | what it is | how it was made |
|---|---|---|---|
| `pseudobulk_control.npy` | 27 × 3000 | mean of the 150 matched control cells of each (drug, cell line) pair, counts space | mean over the rows of the frozen split's control counts belonging to that pair |
| `pseudobulk_treated.npy` | 27 × 3000 | mean of the treated cells of each (drug, cell line) condition | mean over that condition's treated rows |
| `target_perpair.npy` | 27 × 3000 | `log1p(treated) − log1p(ctrl_pair)` — the release's own construction | fixed formula |
| `target_pooled.npy` | 27 × 3000 | `log1p(treated) − log1p(ctrl_line)` — one shared vehicle per cell line | fixed formula; `ctrl_line` is the mean of that line's per-pair controls |
| `prediction_loss_only_perpair.npy` | 27 × 3000 | the audited model's stored loss-only predictions | copied unchanged from the release |
| `prediction_loss_only_pooled.npy` | 27 × 3000 | the same predictions re-expressed in the pooled space | `per_pair + log1p(ctrl_pair) − log1p(ctrl_line)` |
| `vehicle_profiles_perpair_{control,treated}.npy` | 27 × 3000 | the per-pair pseudobulks, named for `--vehicle-profiles` / `--vehicle-treated` | copies of the two files above |
| `vehicle_profiles_pooled_{control,treated}.npy` | 27 × 3000 | the same counts with one row per cell line | required by the pooled manifest |
| `vehicle_profiles.npz` | — | both per-pair pseudobulks in one archive | convenience |
| `pair_order.csv` | 27 rows | `drug`, `cell_line` in positional order | copied from the release |
| `bundle_manifest.json` | — | SHA-256 and byte size of every file above | generated |
| `manifests/input_manifest.json` | — | declares the **pooled** response space (`"per cell line"`) and the alignment convention | written for this repository |
| `manifests/input_manifest_perpair.json` | — | declares the **per-pair** response space (`"per pair"`) | written for this repository |

## The gate

The pseudobulks rebuild the shipped per-pair targets to `max |diff| = 2.20e-07`.
`scripts/reproduce_evidence.py` refuses to report any number above `1e-5`, so a
wrong construction cannot produce a green report.

## Why the bundle is here and the rest is not

These matrices are redistributed because the audit is meaningless without them
and because they are derived summaries the upstream licences permit. The full
160-compound training-response library, the cross-plate replicate matrices and
the pathway-level matrices are **not** included; every quantity that needs them
is labelled *needs an additional input* in the reproduction report rather than
approximated. `../docs/evidence_and_limitations.md` lists each one and the file
that would close the gap.

No input here was derived from an expected value. An earlier version of the
research pipeline selected its pooled-truth candidate by proximity to the paper's
printed AUC; that procedure was removed and is recorded in `../CHANGELOG.md`.

## Licences

Code and the derived matrices this repository computes: MIT (`../LICENSE`).
Upstream terms: sci-Plex (GEO GSE139944) and MSigDB Hallmark keep their own;
the release supplement is MIT. Per-file provenance is in
`../THIRD_PARTY_NOTICES.md`. No blanket licence is applied to upstream data.
