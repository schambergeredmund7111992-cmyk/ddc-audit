# Evidence and limitations

## What this repository recomputes

From the shipped `evidence/` alone, with `scripts/reproduce_evidence.py`:

| quantity | value | status |
|---|---:|---|
| pooled loss-only AUC (the paper's headline space) | 0.5093 | recomputed; printed 0.509 |
| pooled on-off gap | 0.0135 | recomputed; printed 0.014 |
| pooled-space drug-blind anchor | 0.5000 | recomputed (must be 0.500) |
| pooled oracle | 1.0000 | recomputed (must be 1.000) |
| pooled permutation p | 0.4026 | recomputed; printed 0.39 |
| per-pair loss-only AUC | 0.5694 | recomputed exactly (the release's own grid value) |
| per-pair on-off gap | 0.0358 | recomputed; printed 0.036 |
| per-pair-space drug-blind anchor | 0.9583 | **differs** from the printed 0.588 |
| pooled bootstrap CI, drug-clustered | [0.3934, 0.6157] | **differs** from the printed [0.37, 0.51] |
| model − anchor, per-pair | −0.3889 | recomputed |

Stochastic rows move in their third decimal with the number of draws: at 1000
draws the pooled CI and p are as above; at 300 draws the same command gives
[0.3840, 0.6157] and 0.4120. No conclusion depends on that decimal — the interval
includes 0.5 and the p-value is nowhere near 0.05 either way.

## The two quantities that do not reconcile

**The printed pooled bootstrap interval [0.37, 0.51].** This repository's
per-anchor drug-clustered estimator gives [0.3934, 0.6157] for the same point
estimate; the superseded row-resampling estimator gives [0.3611, 0.5185] and is
biased low by 0.060. Neither reproduces the printed interval, which came from an
earlier analysis pipeline the release does not contain. The point estimate and
the permutation p *do* reproduce.

**The printed per-pair drug-blind anchor 0.588 (on-off gap 0.092).** Measured on
the released pseudobulks through this tool's canonical constructor, the same
quantity reaches **0.9583** against the per-pair targets and **0.5000** against
the pooled targets. Neither is 0.588, so the released pseudobulk matrices and the
paper's per-pair analysis do not share a construction.

The **qualitative** claim survives and is stronger on this data: under a per-pair
vehicle the drug-blind predictor outscores every audited model (0.9583 against
0.5694, `model − anchor = −0.3889`), and the tool marks the construction
`FAILED`. The **number** 0.588 is treated as printed-and-unverified; this
repository does not lean on it and does not substitute its own value for it.

## What needs an input this repository does not ship

These belong to earlier stages of the research pipeline. Every one is labelled
*needs an additional input* in the reproduction report rather than approximated.

| quantity | printed | file that would be needed |
|---|---:|---|
| cross-plate biological replicate ceiling | 0.810 | `replicates/crossplate_rep{1,2}.npy` |
| hindsight-retrieval oracle | 0.926 | the 160-compound training-response library |
| target-matched oracle (15 of 27 pairs) | 0.717 | the same library plus the vendor target field |
| Tanimoto 1-NN oracle | 0.509 | Morgan fingerprints of the 172-compound library |
| Morgan-ridge oracle | 0.495 | as above, plus `rdkit` and `scikit-learn` |
| technical split-half ceiling | 0.885 ± 0.022 | the cell-level sci-Plex h5ad (~2.4 GB) |
| pathway-level on-off gap | 0.00006 | the pathway-level prediction and target matrices |
| pooled Mean baseline Spearman@50 | 0.491 | the author's pooled Mean predictor |

When reading the ladder, three caveats stay attached to the numbers:

- the **0.717** target-matched oracle covers **15 of 27** anchors, not all 27. It
  is not normalised against the 27-anchor ceiling: target availability is not
  random and dividing one population by the other would mix two different things;
- the **0.926** hindsight oracle chooses a training compound *after seeing the
  held-out response*. It is a retrospective upper bound on any method whose output
  is a training compound's measured response — **not a deployable predictor**;
- 0.810 is the level a cross-plate replicate of the same experiment reaches under
  this design. It is a measured reference level, **not a mathematical bound** that
  no algorithm may exceed.

## Limitations

- **One dataset.** sci-Plex, one frozen 65/8/9 drug split, 3000 training-only
  HVGs, one dose and time point (10 µM, 24 h), nine held-out compounds, three
  cell lines. Every conclusion is a statement about that cohort.
- **Independent reconstructions.** The audited model families were scored in
  independent reconstructions. The shared failure is evidence against a
  single-architecture artifact; it is **not a ranking** and supports no leaderboard.
- **A negative result is a failure to detect.** The permutation test does not
  reject the null on this cohort. That is not a proof that no signal exists — only
  that none clears this control at this sample size. The oracle rungs show the
  information is present in the data; the audited models do not extract it.
- **Scope of the claim.** "Molecular structure carries no drug-identity
  information here" is a statement about two fingerprint encodings, nine
  compounds, and one benchmark. It is not a general claim about structure-based
  modelling.
- **The threshold is a convention.** 0.70 is an exploratory reporting threshold,
  not a validated usability standard.
- **No model is retrained.** The audit reads stored predictions. That is the point
  of the tool and also the limit of what it can check about the models themselves.

## Provenance

Every file in `evidence/` is listed with its SHA-256 in
`evidence/bundle_manifest.json` and described in `evidence/README.md`. The bundle
derives from the research release at commit
`87189db29653d0ee760714c84f012e494e5e761a`. The correction history — what
changed, why, and which numbers moved — is in `CHANGELOG.md`.

The accompanying paper's current supported status is **preprint submitted to
Elsevier, 2026**. It is not claimed to be accepted, published, or publicly posted.
