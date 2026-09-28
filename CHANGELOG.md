# Changelog

This file records every change to the audit software, including the
corrections that moved published numbers. The corrections are kept in full:
a reader who saw an earlier value needs to be able to find out what happened
to it and why.

The evidence bundle is versioned with the software, because the two must not
drift apart — the anchor a version computes depends on the construction it is
given.

## 0.2.1b — 2026-09-28

### Ties are now decided by a tolerance, not by BLAS reduction order

`specificity_auc` compared the diagonal with each off-diagonal entry using a
strict `>`. When a predictor is *exactly* collapsed — every predicted profile in
a cell line identical — those two quantities are mathematically equal, and the
comparison is then decided by the order in which the products happen to be
summed. A matrix product does not accumulate `C[i, i]` and `C[i, j]` the same
way, so a backend is free to return entries differing in the last bit.

Measured consequence: on Windows the structured 6-row test panel produces
bit-identical entries and the anchor scored **0.0**; the same code on macOS
Accelerate (ARM64, all three CI Python versions) produced last-bit differences
and scored **16 of 30 comparisons as wins, 0.2667**. A tool whose entire purpose
is to recognise an exactly collapsed predictor cannot have its answer depend on
which BLAS is installed.

Two similarities now count as tied when they differ by no more than
`TIE_ATOL = 1e-12` — thousands of times the float64 rounding error at these
magnitudes, and far below any difference a reader would call real (the shipped
panels separate by ~1e-2). Ties remain non-discriminating; only the boundary
moved. The same rule is applied in `per_anchor_scores`.

**This changes no published number.** Every quantity in the README, the
reproduction report and the evidence documentation was recomputed after the
change and is identical to the last digit: pooled 0.5093, per-pair 0.5694,
anchors 0.5000 / 0.9583, oracle 1.0000, permutation p 0.4026, bootstrap interval
[0.3934, 0.6157], synthetic collapsed 0.4491. The shipped panels have margins
around 1e-2, four orders of magnitude above the new tie band.

Two tests cover it: the existing exact-collapse test now documents why a strict
comparison is not portable, and `test_last_bit_noise_is_treated_as_a_tie_not_as_a_win`
perturbs a diagonal entry by one ULP and asserts it is still a tie.

## 0.2.1a — 2026-09-28 (submission cleanup)

Text and packaging only; no claim, formula, figure or result changed.

* The technical abstract's final slide and the report's requirements row still
  said "three minutes from a fresh install". That figure was never measured.
  Both now state the measured quickstart: **8.8 s** for `make-example` plus a
  1000+1000-draw audit, **12.9 s** including the install when the dependencies
  are already present. The phrase is gone from every `.tex`, `.md` and PDF; the
  only remaining occurrence is in this changelog, quoted as history.
* The archive no longer carries build intermediates: `code/ddc_audit/build/`
  (a stray setuptools `build/lib/` tree) and `code/regeneration/audit_check/`
  and `final_review_check/` (review scratch directories) were removed from the
  working tree before packaging, and the packaging step now refuses to include
  them. The ZIP went from 133 entries / 2.13 MB to **121 entries / 2.11 MB**,
  and its inventory was checked entry by entry for `/build/`, `*.egg-info`,
  `__pycache__`, `.pytest_cache`, LaTeX intermediates and temporary audit
  directories.
* Re-verified from a fresh extraction of the cleaned archive: install,
  `--version`, synthetic quickstart, real pooled audit, the regeneration
  pipeline at 1000/1000, and the test suite.

## 0.2.1 — 2026-09-28 (this revision)

A second pass over the 0.2.0 package. Three defects, one of them scientific.

### The anchor ignored the pseudobulks it was given

`calibration_ladder()` required `vehicle_treated` and then never used it. The
drug-blind predictor was assembled from `cell_line_means(targets)` and the
vehicle offsets — an approximation — instead of from the supplied counts-space
pseudobulks. On the shipped bundle the two disagree:

| construction | anchor vs per-pair targets | anchor vs pooled targets |
|---|---:|---:|
| 0.2.0 approximation | 0.064815 | — |
| **direct, from the supplied pseudobulks** | **0.958333** | **0.509259** |
| direct, converted into the pooled response space | — | 0.500000 |

The maximum elementwise difference between the two predictors is 1.02; the
approximation was wrong in sign as well as magnitude. Everything is now built by
one function, `anchors.drug_blind_predictions(vehicle_treated, vehicle_profiles,
cl)`, which both the CLI and the regeneration pipeline call, so the two entry
points cannot drift apart again. New tests fail against the old implementation
(`test_vehicle_treated_actually_changes_the_anchor`,
`test_drug_blind_predictor_matches_the_hand_computed_formula`,
`test_log1p_of_mean_is_not_mean_of_log1p`).

The correction changes the reported per-pair anchor from 0.065 to **0.958** and
the pooled-space anchor from 0.518 to **0.500**. Both are *stronger* statements
of the package's central finding, and both are updated everywhere.

### Calibration was allowed to pass while the anchor was biased

0.2.0 marked the construction `VERIFIED` when the anchor left chance but the
model scored higher. That conflates two questions. Now:

* **construction calibration** asks only whether the response construction
  manufactures discrimination for a drug-blind predictor. It reads the anchors
  and nothing else.
* `model_minus_anchor` is reported **separately**, as a performance statement.
  It never rescues a failed calibration.
* a material departure from 0.500 is `FAILED` regardless of the model;
* the tolerance is configurable (`--anchor-tolerance`, default 0.06) and its
  justification is written down in `anchors.py`, not hidden.

Status vocabulary is now `VERIFIED` / `FAILED` / `NOT_ASSESSED`, with a separate
`calibration_scope` recording which response spaces were considered
(`per_pair`, `pooled`, or `both_space_worst_case` when no manifest declares one).

**Schema version 2 → 3.** The flat `anchors` block is replaced by
`calibration` (status, scope, both anchors, tolerance, offsets, the
model-minus-anchor comparison) plus `metric_checks` (chance, random, cell-line
mean, oracle — the properties of the *metric*). Migration: any reader of
`anchors.vehicle_calibration` should read `calibration.calibration_status`;
`anchors.no_drug_info_anchor_auc` splits into
`calibration.anchor_pooled_response_auc` and
`calibration.anchor_perpair_response_auc`, and neither is a substitute for the
other.

### The bundle needed a pooled vehicle file

A pooled run handed the per-pair controls now contradicts its own manifest and is
rejected (the audit says so rather than guessing). The bundle therefore also
ships `vehicle_profiles_pooled_{control,treated}.npy` — the same counts, one row
per cell line — and the documented pooled command uses them.

### Smaller corrections

* The test suite no longer needs the research repository for its equivalence
  check: `tests/data/reference_metric_outputs.json` freezes the author's
  `eval/metrics.py` output for a documented input, and
  `test_matches_the_frozen_reference_outputs` runs everywhere. The live
  comparison remains and skips where the repository is absent, so the suite is
  **71 passed + 1 skipped** on the author's machine and **70 passed + 2 skipped**
  elsewhere. Exact commands and counts: `docs/validation.md` §2.
* Claims about installation were wrong and are corrected throughout: the
  `--no-build-isolation` flag avoids *pip's* build-environment download, it does
  not make dependency installation offline. Measured behaviour is in
  `docs/validation.md` §5, including the wheelhouse command for a genuinely offline
  install.
* The "three-minute quickstart" was never measured. It is now timed: **8.8 s**
  for `make-example` plus a full audit at 1000+1000 draws, 12.9 s including the
  install (details and machine in `docs/validation.md` §5).
* "the same prediction matrix" was imprecise — the two shipped `.npy` files are
  the same underlying predictions re-expressed in two response spaces and are
  not byte-identical. Corrected everywhere.
* The regeneration pipeline now computes uncertainty in **both** response
  spaces, so its per-pair numbers can be compared with the CLI's. They agree to
  the last digit (`docs/validation.md` §4). The superseded row-resample estimator is
  numerically unstable by construction — a last-bit difference in its input
  moves its quantiles by ~1e-3 — which is documented at the estimator and
  reported with a stated tolerance rather than hidden.
* PDFs: report density reduced, abstract slides 5/6/8 figures enlarged, a
  malformed glyph-spacing artifact fixed.

## 0.2.0 — 2026-09-28

The previous package (0.1.0, 2026-08-14) claimed that all numbers could be
recomputed by the shipped commands. Several of those commands could not run, the
statistic behind two headline uncertainties was measured wrongly, and one
"anchor" was a sanity check wearing the wrong label. This revision repairs each
of those and records the numbers that consequently changed.

### Reproduction chain

* **`code/regeneration/regenerate_paper_numbers.py` imported `scripts.regenerate`,
  which the package shipped as `scripts_regenerate`, and defaulted its expected
  values to a path in the author's own repository.** It now resolves both, via a
  shim that installs the packaged `ddc_audit.metrics` as `eval.metrics` and
  aliases `scripts.regenerate` onto the shipped package, and it defaults to the
  bundle and expected-values file that ship beside it. It runs from a fresh
  extraction with no environment variables and no absolute paths.
* **The bundled evidence now actually exists.** 0.1.0 shipped a README note that
  the "pooled regeneration bundle" was not included, while the supplementary
  materials claimed every number was recomputable. This package ships
  `code/regeneration/bundle/` — per-pair control and treated pseudobulks, both
  target constructions, the stored predictions in both spaces, the per-pair
  vehicle files, and a manifest with the SHA-256 of every file. The pooled
  construction is now derived by a fixed formula; it is **not** selected by
  proximity to any expected value.
* **The pipeline reports what it cannot reproduce.** The previous export tool
  chose its pooled-truth candidate by picking whichever candidate landed closest
  to the paper's printed AUC. That is reverse-engineering a number from its
  expected value and it has been removed. The new script classifies every cited
  quantity as *recomputed*, *differs*, or *needs an additional input*.
* **Missing inputs are named.** `DATA_AND_SCOPE.md` lists every quantity that
  needs data the release does not contain, with the file that would be required.

### Statistics

* **The bootstrap resampled the wrong thing and was measurably biased.**
  Resampling rows with replacement puts a duplicated drug's copies into each
  other's off-diagonal sets, where the strict comparison scores them as losses.
  On the shipped matrices the observed 0.5694 sat at the 97th percentile of its
  own bootstrap distribution (mean 0.5049, bias −0.065). The default is now a
  drug-clustered **per-anchor** bootstrap — resample the nine drugs, average the
  drawn drugs' per-anchor scores — which is the estimator the paper's Section IV
  describes and the one the author's own `compute_valid_bootstrap.py` adopts.
  The superseded estimator is still computed and reported (`bootstrap_legacy`)
  so earlier published intervals stay auditable, and it is never used for a
  verdict. Both are plotted side by side.
* **The estimand is now stated.** `results.json` carries `estimand` and `unit`
  for each bootstrap, so a reader can tell what was resampled without reading
  the source.
* **Ties are documented rather than asserted.** The old docstring claimed the
  paper "reports that half-counting ties leaves its values unchanged". That is
  true on this artifact (increments of 1/8, zero exact ties in 216 comparisons)
  but it is not a general property, and the tool no longer presents it as one.
  A test pins the strict rule where ties do occur.

### Calibration

* **The "no-drug-information anchor" was a cell-line-mean sanity check.**
  `anchors.py` built every drug's prediction from the same per-cell-line mean,
  which is constant within a cell line and therefore scores exactly 0.500 — a
  property of the metric, not evidence about the data. The tool now reports two
  constructs separately: the collapsed-prediction check (always computable, says
  nothing about the data) and the vehicle-construction control (computable only
  when the per-pair vehicle pseudobulks are supplied).
* **A three-state calibration verdict replaces the boolean warning.**
  `VERIFIED` / `FAILED` / `NOT_ASSESSED`. `NOT_ASSESSED` is not a pass, and the
  terminal report says in the same breath that a 0.500 cell-line-mean result is
  not evidence that the vehicle construction is clean.
* **The verdict no longer collapses four questions into one word.** It is one of
  `ABOVE THRESHOLD`, `BELOW THRESHOLD`, `NO DETECTABLE DISCRIMINATION`, or
  `INDETERMINATE`, and the report prints the permutation p, the directional
  Wilcoxon p, the interval and the threshold as separate rows. The 0.70 value is
  labelled an exploratory reporting threshold, not a usability standard.
* **"Below the threshold", "no signal detected" and "no signal exists" are
  distinguished** in the verdict text and in the HTML report.

### Engineering

* **`--vehicle-profiles`, `--vehicle-treated`, `--manifest`, `--threshold`** were
  added; `check-input` now reports the column names it used, the pairs per cell
  line, whether vehicle pseudobulks were supplied, and warnings for constant
  rows and duplicate `(drug, cell_line)` keys.
* **A self-contained HTML audit report** is written on every run.
* **The figures are data, not screenshots.** 0.1.0's report pasted whole pages of
  the paper as its Figures 1–3, which did not match their captions. Every figure
  now ships as a PDF drawn from the verified bundle, with the underlying table
  beside it (`figures/fig1_data.csv`).
* **A third example predictor** (`predictions_oracle`) and a `make-example`
  banner that says the data is synthetic in the console, the filenames, the
  captions and every report.
* **The cross-correlation is vectorised**, which took the test suite from 390 s
  to about 10 s. That is what makes the suite usable in a judging environment.

### Testing

* **The reference-equivalence test no longer hard-codes the author's absolute
  path.** It resolves the research repository from `DDC_AUDIT_REFERENCE_REPO`,
  then `~/cytobridge-benchmark`, then a sibling checkout, and **skips** when
  none is present instead of silently passing.
* **The suite grew from 22 to 56 tests.** New coverage: the bootstrap bias is
  pinned by a test that fails if the two estimators ever coincide; the three
  calibration states; the tie rule; input aliases; duplicate condition keys;
  manifest validation; the HTML report's self-containment; and the threshold
  flag.
* **No test was deleted, weakened, or re-seeded to protect a conclusion.** The
  one test that changed behaviour is the reference-equivalence test, which
  changed from "runs only on the author's machine" to "runs where the reference
  exists, skips elsewhere, and says so".

### Numbers that changed

| quantity | 0.1.0 reported | this revision | why |
|---|---|---|---|
| per-pair loss-only DDC AUC | 0.5694 | 0.5694 | unchanged; the endpoint itself was always right |
| per-pair 95% bootstrap CI | [0.41, 0.57] | [0.472, 0.658] | the old interval came from the biased row-resampling estimator |
| per-pair permutation p | 0.001 | 0.001 | unchanged |
| pooled loss-only AUC | not reproducible | 0.5093 | the pooled construction is now shipped and derived by formula |
| pooled 95% bootstrap CI | — | [0.393, 0.616] | printed in the paper as [0.37, 0.51]; the difference is reported, not hidden |
| "no-drug-info anchor" | 0.500 (mislabelled) | 0.500 pooled / 0.958 per-pair (0.963 in 0.2.0, corrected in 0.2.1) | two constructs, reported separately; the 0.2.0 value came from the approximation described above |
| test count | "22 tests passing" | 56 passed, measured | the old claim was not measured anywhere |

## 0.1.0 — 2026-08-14

First packaged release of the audit tool, produced for a competition
submission. Superseded. It is kept in this changelog only as the baseline the
corrections above are measured against: it shipped an anchor built from the
targets rather than from the supplied pseudobulks, a row-resampling bootstrap,
and a test suite whose one equivalence test hard-coded an absolute path.

Nothing in this repository still uses any of it. The version numbers above are
this repository's own; the research artifact that 0.1.0 was prepared for is
described in `evidence/README.md`.
