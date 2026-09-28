# Methodology

## The endpoint

Within each cell line, build the cross-correlation over the union of the per-drug
top-50 `|target|` genes:

```
C[i, j] = corr(prediction_i, target_j)
```

and report the mean over anchors of the within-cell-line win rate

```
DDC = (1/n) * sum_i  (1/(m-1)) * sum_{j != i}  1[ C[i,i] > C[i,j] ]
```

where an **anchor** is one held-out (drug, cell line) pair, `m` is the number of
held-out drugs in that cell line, and `n` is the number of anchors (27 here).
Chance is a fixed 0.5 by construction: for one anchor the win rate is a fraction
of `m-1` comparisons, and the mean over anchors of a coin flip is 0.5.

The estimator is a **conditional-on-anchor mean**, not a pooled ROC over all
on- and off-diagonal pairs, and that choice fixes the resampling unit (below).

The metric itself is a within-context retrieval rank already used elsewhere in
the field. This tool does not claim it as new. What it adds is the calibration
around it.

## Ties

`C[i,i] == C[i,j]` is scored as non-discriminating (0), not as a half win. On the
shipped evidence this is exact rather than approximate: the 27 anchors move in
increments of 1/8 and **none** of the 216 off-diagonal similarities ties its
diagonal, so half-counting ties returns 0.5694 as well. That is checked, not
assumed — `tests/test_metrics.py` pins the strict rule on a panel where ties do
exist.

## The three calibration checks

Three different things are called a "baseline", and conflating them is the defect
this tool exists to avoid.

### 1. Collapsed-prediction check (a property of the *metric*)

Replace the prediction with the cell line's mean measured profile. The endpoint
must be exactly 0.500, because a predictor that is constant within a cell line
cannot rank one drug above another. Reported as `metric_checks.cell_line_mean_auc`.

**This says nothing about the data.** A 0.500 here is not evidence that the
response construction is clean.

### 2. Construction calibration (a property of the *data*)

A predictor that receives no drug information at all — it knows only the cell
line — emits, for each pair:

```
drug-blind[pair] = log1p(mean_treated[cell_line]) - log1p(ctrl[pair])
```

where the first term is the mean of that line's **counts** and the second is that
pair's own control pseudobulk.

> `log1p(mean(counts))`, never `mean(log1p(counts))`. The two differ, they are not
> interchangeable, and the pipeline that produced the shipped matrices uses the
> former. A test asserts they differ on synthetic data with non-degenerate counts.

The same predictor in the pooled response space replaces `ctrl[pair]` with the
shared per-cell-line vehicle and is then constant within a cell line, so it scores
exactly 0.500.

The reported `calibration_status` is:

| status | condition |
|---|---|
| `VERIFIED` | the anchor used for the verdict does not depart from 0.5 by more than the tolerance |
| `FAILED` | it does depart. The construction is biased **whatever the model scores** |
| `NOT_ASSESSED` | the control and treated pseudobulks were not supplied |

`calibration_scope` records which response spaces were considered — `per_pair` or
`pooled` when a manifest declares one, `both_space_worst_case` (conservative) when
none does. A manifest that declares a pooled vehicle while the supplied controls
vary within a cell line is rejected as a contradiction rather than silently
resolved; the response space is **declared, never guessed from a filename**.

### 3. Model-versus-anchor comparison (a *performance* statement)

`model_minus_anchor` is reported separately. It answers "how does the model
compare with a drug-blind predictor in the same space", which is a different
question from "is the construction sound". **It never rescues a failed
calibration** — a model scored in a biased response space cannot be read against
0.5 regardless of how it compares with the anchor.

## The tolerance

Default **0.06**, configurable with `--anchor-tolerance`, and printed in every
report so a reader can disagree with it explicitly.

Justification, stated rather than hidden: the Monte Carlo standard error of the
endpoint at 1000 resamples is about 0.015 on this panel, so 0.06 is four standard
errors — a departure that large is not sampling noise. The paper's own per-pair
anchor effect is +0.088 (printed 0.588) and the value measured here is +0.458
(0.958), so any boundary in `[0.02, 0.10]` separates the two regimes on every
dataset in this repository. It is a documented convention, not a derived constant.

## Uncertainty: the estimand fixes the resampling unit

The endpoint is a mean over per-anchor scores, so the resampling unit is the
**anchor's score**.

**Drug-clustered anchor bootstrap** (used for verdicts). Resample the nine
held-out drugs with replacement, keeping each drug's cell lines together, and
average the drawn drugs' per-anchor scores. Every drug carries the same number of
anchors in every draw, so the resampled mean stays centered on the observed mean.
Measured bias on the shipped evidence: **+0.002**.

**Row-resample bootstrap** (superseded, retained for traceability). Resample
matrix *rows* with replacement and rebuild the cross-correlation matrix. When a
drug is drawn twice, its copies land in each other's off-diagonal sets carrying an
on-diagonal-valued correlation; the strict comparison scores those copies as
losses, so the distribution is biased low. Measured bias: **−0.060**, with the
observed 0.5694 near the top of its own distribution.

It is additionally **unstable by construction**: every draw applies `diag > off`
to correlations that are frequently near-ties, so an input differing in the last
float32 bit moves its quantiles by about `1e-3`. Two callers that reach the same
targets by different routes — one loading the shipped matrix, one recomputing it
as `log1p(treated) - log1p(control)` — therefore disagree in the third decimal
while agreeing on the observed value to `1e-7`. It is reported with that caveat
and never used for a verdict. Both distributions are drawn side by side in
`bootstrap_comparison.pdf`.

**Permutation null.** Shuffle the predicted rows within each cell line, which
destroys the drug pairing while preserving cell-line structure and never
duplicates a row. The p-value is the plus-one estimate `(r + 1) / (n + 1)`. This
is a *permutation* p and answers a different question from the paired one-sided
Wilcoxon p on the on-versus-off gap, which is directional only; both are reported
and labelled.

## Input contract

Three files are required and two are optional; all matrices are `[n_pairs,
n_genes]` and align **positionally** with the metadata. The meta file is never
joined by key. See `input_contract.md`.

## Verdict structure

The report keeps four questions apart — data and calibration, statistical
evidence, effect size, threshold — and the verdict is one of `ABOVE THRESHOLD`,
`BELOW THRESHOLD`, `NO DETECTABLE DISCRIMINATION`, or `INDETERMINATE`. The last
means the calibration was `NOT_ASSESSED` or `FAILED`: no clean pass or fail can be
issued. "Below threshold", "no signal detected" and "no signal exists" are three
different statements and the verdict text keeps them apart.

The `--threshold` value (default 0.70) is an **exploratory reporting threshold**,
a project convention calibrated against nothing but the authors' judgement. It is
not a validated standard of practical usability.
