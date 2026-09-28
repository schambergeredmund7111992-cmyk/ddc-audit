# Input contract

## Files

| flag | required | format | shape / columns |
|---|---|---|---|
| `--predictions` | yes | `.npy` or headerless `.csv` | `[n_pairs, n_genes]` predicted profiles |
| `--targets` | yes | `.npy` or headerless `.csv` | `[n_pairs, n_genes]` measured profiles, **same shape and same response space** |
| `--meta` | yes | `.csv` | one row per pair: `drug_id` (alias `drug`) and `cell_line` (aliases `cellline`, `cell`). Extra columns ignored. |
| `--vehicle-profiles` | no | `.npy` or headerless `.csv` | `[n_pairs, n_genes]` per-pair **control** pseudobulk, **counts space** |
| `--vehicle-treated` | no | `.npy` or headerless `.csv` | `[n_pairs, n_genes]` per-pair **treated** pseudobulk, counts space |
| `--manifest` | no | `.json` | declarative: response space, vehicle definition, row alignment |

`--vehicle-profiles` and `--vehicle-treated` must be supplied together: the
drug-blind predictor is built from both. Supplying one without the other is an
input error.

## Alignment

**Positional.** Row *i* of every matrix corresponds to row *i* of `--meta` and, if
supplied, row *i* of both pseudobulk matrices. Nothing is joined by key. A
reordered meta file silently mislabels every pair, so `check-input` prints the
column names it used and the pairs per cell line, which makes a misalignment
visible before anything is scored.

Feature alignment between predictions and targets is also positional. Feature
names are not required and are not read; the shipped evidence follows the
research release's 3000-gene HVG order.

## Response space

The audit scores whatever space it is handed. It does **not** guess which one
that is. When vehicle pseudobulks are supplied, the manifest's
`vehicle_construction` field declares it:

```json
{
  "response_space": "logFC against ONE vehicle per cell line ...",
  "vehicle_construction": "per cell line",
  "vehicle_construction_detail": "one shared vehicle per cell line ...",
  "row_alignment": "positional; row i of every matrix matches row i of pair_order.csv",
  "feature_names": "none supplied; ..."
}
```

Accepted declarations: `"per pair"` (also written `per-(drug, cell line)`) and
`"per cell line"` (also `pooled`, `shared`). Anything else raises rather than
being guessed at. A manifest that declares `per cell line` while the supplied
controls differ within a cell line is rejected as a contradiction.

When no manifest is supplied the calibration is evaluated conservatively in both
spaces (`both_space_worst_case`) and fails if either is biased.

## Validation performed by `check-input`

Rejected with a friendly error:

- missing file, unreadable file, unsupported suffix
- non-2-D matrix, non-numeric matrix
- NaN or infinite values anywhere
- predictions and targets of different shape
- meta row count different from the matrix row count
- fewer than 4 pairs, fewer than 2 genes
- a cell line with fewer than 2 held-out drugs (the off-diagonal control is
  undefined), or a panel with only one cell line
- a drug scored in fewer than 2 cell lines
- pseudobulk matrices of the wrong shape, non-finite, or containing negative counts
- the two vehicle flags supplied without each other

Reported as warnings, not rejected:

- rows constant across genes (in the predictions or the targets)
- duplicate `(drug, cell_line)` keys
- a drug scored in more than one cell line — which is the **intended** design of a
  within-cell-line control, reported so a reader can confirm the panel is the
  expected one

No heuristic is applied to guess the response space. Two were tried and removed;
the reasoning is recorded in `src/ddc_audit/io.py` so nobody re-adds them.

## Exit codes

`0` the audit completed, `2` the input contract was violated. **The verdict is in
the report, not in the exit code** — an `INDETERMINATE` audit still exits 0.

## Example

```bash
ddc-audit check-input \
  --predictions evidence/prediction_loss_only_pooled.npy \
  --targets     evidence/target_pooled.npy \
  --meta        evidence/pair_order.csv \
  --vehicle-profiles evidence/vehicle_profiles_pooled_control.npy \
  --vehicle-treated  evidence/vehicle_profiles_pooled_treated.npy
```
