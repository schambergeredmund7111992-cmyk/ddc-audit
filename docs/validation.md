# Validation record

This record describes checks performed on the standalone DDC-Audit repository.
Values that do not reproduce the accompanying paper are reported as differences,
not silently substituted.

## 1. Supported environments

GitHub Actions runs the package on Ubuntu, Windows, and macOS with Python 3.10,
3.11, and 3.12. The matrix contains nine jobs and is required to run the test
suite, the synthetic quickstart, and a real per-pair audit. The run for commit
`a2d7428` passed all nine jobs.

The audit is CPU-only. Runtime dependencies are pinned in `pyproject.toml` and
`requirements.txt`: numpy 2.2.5, scipy 1.15.3, pandas 2.3.3, and matplotlib
3.10.9.

## 2. Test suite

```bash
python -m pytest -q -rs
```

The full local checkout reports `73 passed` in about 10 seconds when both the
shipped `evidence/` bundle and the research repository are available. Two tests
are optional and state why they skip:

1. the live comparison with the research implementation needs
   `eval/metrics.py`; set `DDC_AUDIT_REFERENCE_REPO` to its checkout;
2. the shipped-bundle anchor test needs `evidence/`, which is intentionally not
   included in the Python wheel.

The portable equivalence check does not depend on either optional input. It
compares the package on every machine with frozen outputs produced by the
research implementation in `tests/data/reference_metric_outputs.json`.

## 3. Evidence reproduction

```bash
python scripts/reproduce_evidence.py --out repro --n-boot 1000 --n-perm 1000
```

The script rebuilds the response constructions and stops if the reconstructed
per-pair target differs from the shipped target by more than `1e-5`. The measured
maximum difference is `2.20e-07`.

| quantity | paper | shipped evidence | status |
|---|---:|---:|---|
| pooled loss-only DDC AUC | 0.509 | 0.5093 | recomputed |
| pooled permutation p | 0.39 | 0.4026 | recomputed |
| per-pair loss-only DDC AUC | 0.5694 | 0.5694 | recomputed |
| pooled-space drug-blind anchor | 0.500 | 0.5000 | recomputed |
| per-pair-space drug-blind anchor | 0.588 | 0.9583 | differs |
| pooled drug-clustered 95% CI | [0.37, 0.51] | [0.3934, 0.6157] | differs |

The missing-input rows and the interpretation of these differences are listed
in `docs/evidence_and_limitations.md`.

## 4. CLI and reproduction pipeline

On the same per-pair inputs and random seeds, the CLI and reproduction pipeline
agree exactly on the model AUC (`0.569444`), per-pair anchor (`0.958333`), pooled
anchor (`0.500000`), model-minus-anchor (`-0.388889`), permutation p
(`0.000999`), and drug-clustered interval (`[0.472222, 0.657523]`). The
superseded row-resampling interval is retained for auditability and is not used
for a verdict.

Similarity differences within `1e-12` are treated as ties. This removes
last-bit variation between BLAS implementations; the published evidence values
are separated by about `1e-2` and are unchanged by the tolerance.

## 5. Build and installation

```bash
python -m build
pip install . --no-build-isolation
```

The source distribution and universal wheel build successfully. The wheel
contains only the installable package and licence; the repository carries the
larger examples, evidence, tests, and reproduction scripts.

`--no-build-isolation` prevents a separate build-environment download but does
not make missing runtime dependencies available offline. Prepare a wheelhouse
on a connected machine for an offline installation:

```bash
pip download -d wheelhouse -r requirements.txt
pip install --no-index --find-links wheelhouse . --no-build-isolation
```

The package is not published on PyPI and no container image is published. The
repository contains a Docker build configuration only.
