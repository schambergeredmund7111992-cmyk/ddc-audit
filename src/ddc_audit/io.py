"""Input loading and validation.

Input contract
--------------
Three files are required:

* ``predictions`` -- [n_pairs, n_genes] matrix of predicted responses,
  ``.npy`` or ``.csv``. ``.csv`` is read headerless.
* ``targets``     -- [n_pairs, n_genes] matrix of measured responses, same
  shape, same response space as the predictions.
* ``meta``        -- ``.csv``, one row per pair, with ``drug_id`` and
  ``cell_line`` columns (``drug``/``cell`` and ``cell_line``/``cellline`` are
  accepted aliases; extra columns are ignored). Rows align *positionally* with
  the matrix rows; the file is not joined, so a reordered meta silently
  mislabels every pair. :func:`check` reports the column names it used.

Two optional files complete the calibration:

* ``vehicle_profiles`` -- [n_pairs, n_genes] per-pair *control pseudobulk* in
  counts space, i.e. the mean of the vehicle cells matched to that pair.
* ``vehicle_treated``  -- [n_pairs, n_genes] per-pair *treated pseudobulk* in
  counts space. The no-drug-information predictor needs both: it emits the
  cell-line mean treated profile measured against each pair's own vehicle.

Without them the audit still runs and reports the AUC, but the vehicle
calibration is reported as NOT_ASSESSED -- the response space cannot be
checked and no clean PASS can be issued.

A ``manifest`` (JSON) is optional and purely declarative: it records the
response space, the vehicle definition, and the row-alignment convention so a
third party can tell what was audited. Nothing in it changes the arithmetic.

Each drug must appear in at least two cell lines and every cell line must hold
at least two held-out drugs; otherwise the off-diagonal control is undefined.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

DRUG_ALIASES = ("drug_id", "drug")
CELL_ALIASES = ("cell_line", "cellline", "cell")

MANIFEST_KEYS = (
    "response_space",
    "vehicle_construction",
    "vehicle_construction_detail",
    "row_alignment",
    "feature_names",
    "notes",
)


class InputError(ValueError):
    """Raised when the input files do not satisfy the contract above."""


def load_matrix(path: Path, *, what: str) -> np.ndarray:
    path = Path(path)
    if not path.is_file():
        raise InputError(f"{what}: file not found: {path}")
    if path.suffix.lower() == ".npy":
        try:
            array = np.load(path)
        except Exception as error:  # noqa: BLE001
            raise InputError(f"{what}: cannot read .npy file {path}: {error}") from error
    elif path.suffix.lower() == ".csv":
        try:
            array = pd.read_csv(path, header=None).to_numpy(dtype=float)
        except Exception as error:  # noqa: BLE001
            raise InputError(f"{what}: cannot read .csv file {path}: {error}") from error
    else:
        raise InputError(
            f"{what}: unsupported suffix {path.suffix!r}; use .npy or .csv"
        )
    if array.ndim != 2:
        raise InputError(f"{what}: expected a 2-D matrix, got shape {array.shape}")
    if not np.issubdtype(array.dtype, np.number):
        raise InputError(f"{what}: matrix must be numeric, got dtype {array.dtype}")
    return array.astype(np.float64)


def _pick_column(meta: pd.DataFrame, aliases, what: str) -> str:
    for name in aliases:
        if name in meta.columns:
            return name
    raise InputError(
        f"meta: no {what} column; looked for {list(aliases)}, "
        f"found {list(meta.columns)}"
    )


def load_meta(path: Path) -> pd.DataFrame:
    path = Path(path)
    if not path.is_file():
        raise InputError(f"meta: file not found: {path}")
    if path.suffix.lower() != ".csv":
        raise InputError(f"meta: expected a .csv file, got {path.suffix!r}")
    try:
        meta = pd.read_csv(path)
    except Exception as error:  # noqa: BLE001
        raise InputError(f"meta: cannot read {path}: {error}") from error
    _pick_column(meta, DRUG_ALIASES, "drug")
    _pick_column(meta, CELL_ALIASES, "cell-line")
    return meta


def load_manifest(path: Path | None) -> dict | None:
    """Read the optional declarative manifest. It records; it never computes."""
    if path is None:
        return None
    path = Path(path)
    if not path.is_file():
        raise InputError(f"manifest: file not found: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as error:  # noqa: BLE001
        raise InputError(f"manifest: cannot parse {path}: {error}") from error
    if not isinstance(payload, dict):
        raise InputError("manifest: expected a JSON object")
    unknown = sorted(set(payload) - set(MANIFEST_KEYS))
    if unknown:
        raise InputError(
            f"manifest: unknown key(s) {unknown}; allowed keys are "
            f"{list(MANIFEST_KEYS)}"
        )
    return payload


def validate(
    predictions: np.ndarray,
    targets: np.ndarray,
    meta: pd.DataFrame,
    vehicle_profiles: np.ndarray | None = None,
    vehicle_treated: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Validate alignment and return (pred, true, drugs, cell_lines)."""
    if predictions.shape != targets.shape:
        raise InputError(
            f"shape mismatch: predictions {predictions.shape} != targets {targets.shape}"
        )
    n_pairs = predictions.shape[0]
    if len(meta) != n_pairs:
        raise InputError(
            f"meta has {len(meta)} rows but matrices have {n_pairs} pairs"
        )
    if n_pairs < 4:
        raise InputError(
            f"need at least 4 (drug, cell line) pairs, got {n_pairs}"
        )
    if predictions.shape[1] < 2:
        raise InputError("matrices must have at least 2 genes/features")

    if not np.isfinite(predictions).all():
        raise InputError(
            "predictions contain NaN or infinite values; the audit has no "
            "imputation rule and will not guess one"
        )
    if not np.isfinite(targets).all():
        raise InputError("targets contain NaN or infinite values")

    for name, matrix in (
        ("vehicle_profiles", vehicle_profiles),
        ("vehicle_treated", vehicle_treated),
    ):
        if matrix is None:
            continue
        if matrix.shape != targets.shape:
            raise InputError(
                f"{name} shape {matrix.shape} != targets shape {targets.shape}; "
                "the pseudobulk matrices must be per pair, like the targets"
            )
        if not np.isfinite(matrix).all():
            raise InputError(f"{name} contains NaN or infinite values")
        if (matrix < 0).any():
            raise InputError(f"{name} contains negative counts")

    drugs = meta[_pick_column(meta, DRUG_ALIASES, "drug")].astype(str).to_numpy()
    cells = meta[_pick_column(meta, CELL_ALIASES, "cell-line")].astype(str).to_numpy()
    unique_cells = np.unique(cells)
    if unique_cells.size < 2:
        raise InputError(
            f"the panel has only one cell line ({unique_cells[0]!r}); the "
            "off-diagonal control is a within-cell-line comparison and needs "
            "at least two contexts"
        )
    for cell in unique_cells:
        rows = np.flatnonzero(cells == cell)
        if rows.size < 2:
            raise InputError(
                f"cell line {cell!r} has only {rows.size} pair(s); the "
                "off-diagonal control needs at least 2 held-out drugs per line"
            )
    for drug in np.unique(drugs):
        rows = np.flatnonzero(drugs == drug)
        if rows.size < 2:
            raise InputError(
                f"drug {drug!r} appears in only {rows.size} cell line(s); "
                "each drug should be scored in at least 2 contexts"
            )
    return predictions, targets, drugs, cells


def check(
    predictions: np.ndarray,
    targets: np.ndarray,
    meta: pd.DataFrame,
    vehicle_profiles: np.ndarray | None = None,
    vehicle_treated: np.ndarray | None = None,
) -> dict:
    """The report :func:`load_all` hands back: what was read, and what to warn about."""
    warnings: list[str] = []
    drug_col = _pick_column(meta, DRUG_ALIASES, "drug")
    cell_col = _pick_column(meta, CELL_ALIASES, "cell-line")
    drugs = meta[drug_col].astype(str).to_numpy()
    cells = meta[cell_col].astype(str).to_numpy()

    constant_rows = int(np.sum(targets.var(axis=1) < 1e-12))
    if constant_rows:
        warnings.append(
            f"{constant_rows} target row(s) are constant across genes; their "
            "anchors carry no gene ranking and contribute ties"
        )
    constant_pred = int(np.sum(predictions.var(axis=1) < 1e-12))
    if constant_pred:
        warnings.append(
            f"{constant_pred} prediction row(s) are constant across genes"
        )
    keys = list(zip(drugs.tolist(), cells.tolist()))
    repeated_conditions = len(keys) - len(set(keys))
    if repeated_conditions:
        warnings.append(
            f"{repeated_conditions} duplicate (drug, cell_line) key(s): the panel "
            "repeats a condition, and the drug-clustered bootstrap treats each "
            "copy as its own anchor"
        )
    multi_context = sum(1 for d in set(drugs.tolist())
                        if sum(1 for x in drugs.tolist() if x == d) > 1)
    if multi_context and not repeated_conditions:
        warnings.append(
            f"{multi_context} drug(s) are scored in more than one cell line. That "
            "is the intended design of a within-cell-line control, not a defect; "
            "it is reported so a reader can confirm the panel is the expected one."
        )

    # NOTE: no heuristic is applied to guess the response space.
    #
    # Two attempts were made and both were removed. "Predictions and targets
    # differ in row mean" is not diagnostic: a collapsed model's row means are
    # near zero, so any difference is large *relative* to them, and the check
    # fired at 119% on the shipped pooled run, whose matrices are demonstrably
    # in one space (the oracle and the cell-line mean both land exactly where
    # they must). "Prediction row means are flat" is equally weak: a model that
    # agrees with the pooled target and a model that is constant within a cell
    # line differ by 0.049 and 0.015, which does not separate the two examples.
    #
    # The response space is therefore declared, not guessed: the caller supplies
    # a manifest, and the calibration then checks the declaration against the
    # vehicle pseudobulks and rejects a contradiction rather than resolving it.

    grid = {c: int(np.sum(cells == c)) for c in sorted(set(cells.tolist()))}
    return {
        "drug_column": drug_col,
        "cell_line_column": cell_col,
        "n_pairs": int(predictions.shape[0]),
        "n_genes": int(predictions.shape[1]),
        "n_drugs": int(len(set(drugs.tolist()))),
        "n_cell_lines": int(len(set(cells.tolist()))),
        "pairs_per_cell_line": grid,
        "vehicle_supplied": vehicle_profiles is not None,
        "vehicle_treated_supplied": vehicle_treated is not None,
        "warnings": warnings,
    }


def load_all(
    predictions_path: Path,
    targets_path: Path,
    meta_path: Path,
    vehicle_profiles_path: Path | None = None,
    vehicle_treated_path: Path | None = None,
):
    predictions = load_matrix(predictions_path, what="predictions")
    targets = load_matrix(targets_path, what="targets")
    meta = load_meta(meta_path)
    profiles = (
        load_matrix(vehicle_profiles_path, what="vehicle_profiles")
        if vehicle_profiles_path is not None
        else None
    )
    treated = (
        load_matrix(vehicle_treated_path, what="vehicle_treated")
        if vehicle_treated_path is not None
        else None
    )
    pred, true, drugs, cells = validate(predictions, targets, meta, profiles, treated)
    return pred, true, drugs, cells, profiles, treated
