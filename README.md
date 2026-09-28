# DDC-Audit

[![tests](https://github.com/schambergeredmund7111992-cmyk/ddc-audit/actions/workflows/tests.yml/badge.svg)](https://github.com/schambergeredmund7111992-cmyk/ddc-audit/actions/workflows/tests.yml)
[![Python 3.10–3.12](https://img.shields.io/badge/python-3.10–3.12-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

**High correlation does not mean a model learned the drug. DDC-Audit finds out
whether it did.**

Single-cell perturbation models can look accurate under standard correlation
metrics while producing nearly the same response for different compounds.
DDC-Audit turns stored predictions into an evidence-backed drug-discrimination
audit: it exposes prediction collapse, detects artificial signal introduced by
response construction, and produces a calibrated report with uncertainty and
statistical controls.

Run it before submission, publication, or deployment. It needs only prediction
files and a CPU — no retraining, model access, or GPU.

---

## Why per-pair correlation is not enough

Perturbation models are usually scored by correlating each predicted response
with its own measured response. That number never asks whether the model
distinguishes drugs: a model that emits one drug-invariant response template per
cell line still scores well, because shared cell-line structure dominates the
correlation. On the sci-Plex drug-holdout benchmark, the **drug-blind cell-line
mean baseline has the best mean per-pair Spearman@50 of any row we audited
(0.484)** — better than every configuration of the model under study — while the
same predictions score at chance when asked to rank their own drug above the
others. Per-pair correlation ranked predictors backwards.

DDC-Audit measures the missing property, and — this is the part that took a
correction to get right — it checks whether the *response construction itself*
is manufacturing that property before it blames or credits any model.

## Install

```bash
git clone https://github.com/schambergeredmund7111992-cmyk/ddc-audit
cd ddc-audit
pip install . --no-build-isolation
```

`--no-build-isolation` avoids pip's build-environment download; it does **not**
make dependency installation offline. With numpy/scipy/pandas/matplotlib already
present the install needs no network at all; with one missing, pip will fetch it.
For a genuinely offline install, build a wheelhouse first:

```bash
pip download -d wheelhouse -r requirements.txt   # from a machine with network
pip install --no-index --find-links wheelhouse . --no-build-isolation
```

Requires Python ≥ 3.10. Dependencies are pinned in both `pyproject.toml` and
`requirements.txt`
(numpy 2.2.5, scipy 1.15.3, pandas 2.3.3, matplotlib 3.10.9 — matplotlib only for
the figures). The tool is **not** published on PyPI, so `pip install ddc-audit`
will not work. A `Dockerfile` is included as a container **build configuration**;
no image is published.

## Quickstart on synthetic data (about 10 s)

```bash
ddc-audit run \
  --predictions examples/synthetic/predictions_collapsed.npy \
  --targets     examples/synthetic/targets.npy \
  --meta        examples/synthetic/meta.csv \
  --out         /tmp/audit_collapsed
```

The synthetic files are seeded random numbers. They are labelled `synthetic`
in their filenames, in the console, in the JSON and in the HTML report, and are
never presented as experimental results. Three predictors ship:
`predictions_collapsed` (near-identical response per drug — the illusion this
tool exists to expose), `predictions_drug_aware`, and `predictions_oracle`.

## The real case

```bash
# pooled construction — one shared vehicle per cell line
ddc-audit run \
  --predictions evidence/prediction_loss_only_pooled.npy \
  --targets     evidence/target_pooled.npy \
  --meta        evidence/pair_order.csv \
  --vehicle-profiles evidence/vehicle_profiles_pooled_control.npy \
  --vehicle-treated  evidence/vehicle_profiles_pooled_treated.npy \
  --manifest    evidence/manifests/input_manifest.json \
  --out         /tmp/audit_pooled
```

```bash
# per-pair construction — each pair's own vehicle (the release's stored space)
ddc-audit run \
  --predictions evidence/prediction_loss_only_perpair.npy \
  --targets     evidence/target_perpair.npy \
  --meta        evidence/pair_order.csv \
  --vehicle-profiles evidence/vehicle_profiles_perpair_control.npy \
  --vehicle-treated  evidence/vehicle_profiles_perpair_treated.npy \
  --manifest    evidence/manifests/input_manifest_perpair.json \
  --out         /tmp/audit_perpair
```

Measured on the shipped evidence (27 pairs, 9 drugs, 3 cell lines, 3000 genes):

| run | DDC AUC | drug-blind anchor | calibration | model − anchor |
|---|---:|---:|---|---:|
| pooled | **0.5093** | 0.5000 | `VERIFIED` | +0.009 |
| per-pair | **0.5694** | **0.9583** | `FAILED` | −0.3889 |

Both runs are the *same underlying predictions re-expressed in two response
spaces*. The two shipped `.npy` files are not byte-identical — each is the other
converted through the vehicle term — and the two AUCs must never appear in one
table without their construction.

In the per-pair space a predictor that receives **no drug information at all**
scores 0.958, above every audited model. The tool reports that as a failed
calibration and refuses to issue a pass or a fail.

## What you get

Every run writes into `--out`:

- `results.json` — machine-readable (schema v3): `ddc`, `bootstrap`,
  `bootstrap_legacy`, `permutation`, `calibration`, `metric_checks`, `per_pair`
- `terminal_summary.txt` — the human-readable summary printed to the console
- `audit_report.html` — a self-contained page, no network, no server
- `onoff_distribution.pdf`, `calibration_ladder.pdf`,
  `bootstrap_distribution.pdf`, `bootstrap_comparison.pdf`

## The three calibration states

The report answers four questions separately. `calibration_status` is the answer
to question 1b, and it depends on the drug-blind anchor and nothing else:

| state | meaning |
|---|---|
| `VERIFIED` | the drug-blind predictor sits at chance (within the tolerance, 0.06 by default, `--anchor-tolerance`): the construction does not manufacture discrimination |
| `FAILED` | the drug-blind predictor departs from chance: the construction is biased, **whatever the model scores**. `model − anchor` is still reported, as a performance statement, and never rescues the calibration |
| `NOT_ASSESSED` | the per-pair control and treated pseudobulks were not supplied, so the question cannot be answered. **This is not a pass.** |

A cell-line-mean prediction landing on exactly 0.500 is *not* evidence that the
construction is clean — it is a property of the metric. The tool says so in every
report where the vehicle pseudobulks were absent.

## ⚠ The 0.70 threshold is exploratory

The `--threshold` value (default **0.70**) is a **project convention for
reporting, not a validated standard of practical usability.** It is calibrated
against nothing but the authors' judgement. The tool prints it in every report so
that a reader can disagree with it explicitly. "Below threshold", "no signal
detected" and "no signal exists" are three different statements and the verdict
text keeps them apart.

## Tests and measured timing

```bash
python -m pytest -q          # 73 passed when both optional evidence sources are present (~10 s)
```

Two tests are optional and are labelled rather than silently skipped:
`test_numerically_identical_to_reference_implementation` needs the research
repository's `eval/metrics.py` importable (set `DDC_AUDIT_REFERENCE_REPO`), and
`test_the_shipped_bundle_reproduces_the_documented_anchor_values` needs
`evidence/` beside the package — so it is skipped when the tool is installed on
its own. **Equivalence does not depend on either**: the packaged metric is
compared on every machine against a frozen record of the reference
implementation's output, `tests/data/reference_metric_outputs.json`.

Measured on the authors' machine (Python 3.12.9, the pinned dependency versions,
warm filesystem):

| step | time |
|---|---:|
| `pip install . --no-build-isolation` (dependencies present) | 3.4 s |
| `ddc-audit make-example` | 1.7 s |
| `ddc-audit run` on the synthetic quickstart, 1000 bootstrap + 1000 permutation draws | 6.5 s |
| `make-example` + that run | **8.2 s** |
| the same including the install | 12.9 s |

## Reproducing the evidence

```bash
python scripts/reproduce_evidence.py --out /tmp/repro --n-boot 1000 --n-perm 1000
```

Rebuilds both vehicle constructions from the shipped pseudobulks, aborts if the
rebuild does not reproduce the shipped targets (it matches to `2.2e-07`), and
classifies every cited quantity as **recomputed**, **differs**, or **needs an
additional input**. See `evidence/README.md`,
`docs/evidence_and_limitations.md`, and `docs/validation.md`.

### Two printed values this repository does not reproduce

The accompanying paper prints a per-pair drug-blind anchor of **0.588** and a
pooled bootstrap interval of **[0.37, 0.51]**. Measured on the released
artifacts, the same constructs give **0.9583** and **[0.3934, 0.6157]**. Neither
printed value reproduces, and this repository says so rather than silently
substituting or claiming reproduction:

- the printed interval comes from a different estimator in an earlier analysis
  pipeline that the release does not contain. The point estimate (0.5093 vs
  printed 0.509) and the permutation p (0.4026 vs printed 0.39) *do* reproduce;
- the printed anchor does not reconcile in either response space, which means the
  released pseudobulk matrices and the paper's per-pair analysis do not share a
  construction. The *qualitative* claim survives and is stronger on this data.

## Relationship to CytoBridge

Two repositories, two jobs:

- **[cytobridge-benchmark](https://github.com/schambergeredmund7111992-cmyk/cytobridge-benchmark)**
  is the **research and evidence repository**: the full model experiments, the
  historical paper pipeline, the frozen research evidence, and the archived
  release (Zenodo DOI [10.5281/zenodo.21912287](https://doi.org/10.5281/zenodo.21912287)).
- **ddc-audit** (this repository) is the **standalone audit tool**: a small,
  installable, CPU-only package that runs on any model's stored predictions. It
  carries only the minimum evidence needed to reproduce the two real examples,
  not the research pipeline.

The audit software here is the corrected, competition-facing implementation. The
evidence bundle under `evidence/` derives from the frozen split published in the
research repository; the derived values and the correction history are recorded
in `CHANGELOG.md`.

If you use this work, please cite the paper (see `CITATION.cff`). The current
supported status of that paper is **preprint submitted to Elsevier, 2026** — it
is not claimed to be accepted, published, or publicly posted.

## Scope

One dataset (sci-Plex), one frozen 65/8/9 drug split, 3000 training-only HVGs,
one dose and time point (10 µM, 24 h), nine held-out compounds, three cell lines.
Every conclusion is a statement about that cohort. The audited model families
were scored in **independent reconstructions**, so the shared failure is evidence
against a single-architecture artifact — **not a ranking**, and not a leaderboard.
A non-significant AUC is a failure to detect, not a proof that no signal exists.

## Licence

MIT for this repository's code and for the derived matrices it computes — see
`LICENSE`. Upstream inputs keep their own terms (sci-Plex, GEO GSE139944; MSigDB
Hallmark; the author's release supplement). Provenance per file is recorded in
`THIRD_PARTY_NOTICES.md`; no blanket licence is applied to upstream data.
