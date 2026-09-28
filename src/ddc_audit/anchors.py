"""Calibration: is the response construction manufacturing discrimination?

Three different checks live here and they answer three different questions.
Conflating them is the defect this module exists to avoid.

*collapsed-prediction check* (always computable)
    Replace the prediction with the cell line's mean measured profile. The
    endpoint must be exactly 0.500, because a predictor that is constant within
    a cell line cannot rank one drug above another. This checks the *metric*.
    It says nothing whatsoever about the data.

*construction calibration* (needs the per-pair vehicle counts)
    A predictor that receives no drug information at all --- it knows only the
    cell line --- must be pushed through the same response construction as the
    model. Its response is, per pair,

        pred[pair] = log1p(cell-line mean treated pseudobulk)
                     - log1p(this pair's own vehicle pseudobulk)

    The first term is constant within a cell line; the second is not, when the
    vehicle is estimated per pair. So under a per-pair vehicle the drug-blind
    predictor is *not* constant, and its estimation noise enters both arguments
    of the on-diagonal similarity and neither off-diagonal one. Whether the
    anchor then leaves chance is a property of the *data*, and it is the one
    thing a model's AUC must be read against.

    Note the mean: ``log1p(mean(counts))``, not ``mean(log1p(counts))``. The
    two differ, they are not interchangeable, and the pipeline that produced
    the shipped matrices uses the former.

*model-versus-anchor comparison* (a performance statement, not a calibration one)
    ``model_minus_anchor`` is reported separately. A model that outscores a
    biased anchor is still being scored in a biased response space, so this
    comparison never rescues a failed calibration.

Status vocabulary
-----------------
``calibration_status``
    ``VERIFIED``      the anchors sit at chance in every response space
                      considered; the construction does not manufacture
                      discrimination.
    ``FAILED``        the anchor departs from chance by more than the
                      tolerance. The construction is biased, whatever the
                      model scores.
    ``NOT_ASSESSED``  the vehicle/treated pseudobulks were not supplied, so the
                      question cannot be answered. This is not a pass.

``calibration_scope`` records which response spaces were considered:
``both_space_worst_case`` (the default; conservative), ``per_pair`` or
``pooled`` when a manifest declares the construction, or ``none`` when nothing
could be assessed.
"""
from __future__ import annotations

import re

import numpy as np

from ddc_audit.metrics import drug_discrimination_score

TOP_K = 50
METRIC = "pearson"

VEHICLE_VERIFIED = "VERIFIED"
VEHICLE_FAILED = "FAILED"
VEHICLE_NOT_ASSESSED = "NOT_ASSESSED"

SCOPE_WORST_CASE = "both_space_worst_case"
SCOPE_PER_PAIR = "per_pair"
SCOPE_POOLED = "pooled"
SCOPE_NONE = "none"

#: How far from 0.500 a drug-blind anchor may sit before the construction is
#: called biased. Configurable through ``--anchor-tolerance``.
#:
#: Justification for the default 0.06: the Monte Carlo standard error of the
#: endpoint at 1000 resamples is about 0.015 on this panel, so 0.06 is four
#: standard errors -- a departure that large is not sampling noise. The
#: manuscript's own per-pair anchor effect is +0.088 (printed 0.588), and the
#: value measured here is +0.458 (0.958), so a boundary anywhere in
#: [0.02, 0.10] separates the two regimes on every dataset in this package.
#: It is a documented convention, not a derived constant, and the tool prints
#: it in every report so a reader can disagree with it explicitly.
ANCHOR_TOLERANCE = 0.06

#: Numerical slack for deciding whether the supplied vehicles are pooled
#: (identical within a cell line) or per pair.
OFFSET_EPSILON = 1e-12


def cell_line_means(matrix: np.ndarray, cell_lines) -> np.ndarray:
    """Per-cell-line mean profile broadcast to every row of that cell line.

    Applied to a counts-space matrix this computes ``mean(counts)`` per cell
    line; the ``log1p`` is applied by the caller, never folded in here.
    """
    matrix = np.asarray(matrix, dtype=float)
    cl = np.asarray(cell_lines)
    out = np.zeros_like(matrix)
    for cell in np.unique(cl):
        mask = cl == cell
        out[mask] = matrix[mask].mean(axis=0)
    return out


def _ddc(pred, true, cl, top_k=TOP_K, metric=METRIC) -> dict:
    return drug_discrimination_score(
        np.asarray(pred, dtype=float),
        np.asarray(true, dtype=float),
        np.asarray(cl),
        top_k=top_k,
        metric=metric,
    )


def _auc(pred, true, cl, top_k=TOP_K, metric=METRIC) -> float:
    return float(_ddc(pred, true, cl, top_k, metric)["specificity_auc"])


def pool_vehicle(vehicle_profiles, cl) -> np.ndarray:
    """One vehicle per cell line: the mean of that line's per-pair vehicles."""
    return cell_line_means(vehicle_profiles, cl)


def vehicle_offsets(vehicle_profiles, cl) -> np.ndarray:
    """``log1p(ctrl_pair) - log1p(ctrl_cell_line)``, the per-pair vehicle term.

    Zero under a pooled construction, non-zero under a per-pair one. This is
    the term that enters both arguments of the on-diagonal similarity and
    neither off-diagonal one.
    """
    profiles = np.asarray(vehicle_profiles, dtype=float)
    return np.log1p(profiles) - np.log1p(cell_line_means(profiles, cl))


def _validate_counts(name: str, matrix, shape) -> np.ndarray:
    array = np.asarray(matrix, dtype=float)
    if array.shape != shape:
        raise ValueError(
            f"{name} shape {array.shape} != targets shape {shape}; pseudobulk "
            "matrices are indexed per pair, exactly like the targets"
        )
    if not np.isfinite(array).all():
        raise ValueError(f"{name} contains NaN or infinite values")
    if (array < 0).any():
        raise ValueError(f"{name} contains negative counts")
    return array


class AnchorPredictions:
    """The drug-blind predictor expressed in both response spaces.

    Attributes
    ----------
    per_pair
        ``log1p(line-mean treated) - log1p(this pair's own vehicle)`` --- the
        response as a pipeline that estimates a vehicle per pair would emit it.
    pooled
        ``log1p(line-mean treated) - log1p(shared cell-line vehicle)`` --- the
        same predictor when one vehicle is shared by every pair of a line.
    treated_line_mean
        The cell-line mean treated pseudobulk in counts space (the ``mu_line``
        used to build both).
    """

    __slots__ = ("per_pair", "pooled", "treated_line_mean", "vehicle_line_mean")

    def __init__(self, per_pair, pooled, treated_line_mean, vehicle_line_mean):
        self.per_pair = per_pair
        self.pooled = pooled
        self.treated_line_mean = treated_line_mean
        self.vehicle_line_mean = vehicle_line_mean


def drug_blind_predictions(
    vehicle_treated: np.ndarray,
    vehicle_profiles: np.ndarray,
    cl,
) -> AnchorPredictions:
    """The one canonical construction of the drug-blind predictor.

    Every anchor reported anywhere in this package comes from this function.

    Parameters
    ----------
    vehicle_treated
        ``[n_pairs, n_genes]`` per-pair **treated** pseudobulk, counts space.
    vehicle_profiles
        ``[n_pairs, n_genes]`` per-pair **control** pseudobulk, counts space.
    cl
        ``[n_pairs]`` cell-line label, positionally aligned with both.

    The cell-line treated profile is the mean of the *counts* of that line's
    pairs, and ``log1p`` is applied to that mean. ``mean(log1p(counts))`` is a
    different quantity and is not used.
    """
    treated = np.asarray(vehicle_treated, dtype=float)
    profiles = np.asarray(vehicle_profiles, dtype=float)
    if treated.shape != profiles.shape:
        raise ValueError(
            f"treated pseudobulks {treated.shape} != control pseudobulks "
            f"{profiles.shape}"
        )
    treated = _validate_counts("vehicle_treated", treated, treated.shape)
    profiles = _validate_counts("vehicle_profiles", profiles, profiles.shape)
    cl = np.asarray(cl)
    if cl.shape[0] != treated.shape[0]:
        raise ValueError(
            f"cell-line labels have {cl.shape[0]} entries but the pseudobulks "
            f"have {treated.shape[0]} rows; alignment is positional"
        )

    treated_line = cell_line_means(treated, cl)          # counts-space mean
    vehicle_line = cell_line_means(profiles, cl)
    log_treated_line = np.log1p(treated_line)
    return AnchorPredictions(
        per_pair=log_treated_line - np.log1p(profiles),
        pooled=log_treated_line - np.log1p(vehicle_line),
        treated_line_mean=treated_line,
        vehicle_line_mean=vehicle_line,
    )


def no_drug_information_predictor(
    vehicle_treated: np.ndarray, vehicle_profiles: np.ndarray, cl
) -> np.ndarray:
    """Deprecated alias for ``drug_blind_predictions(...).per_pair``.

    Kept because earlier revisions of this package and its documentation refer
    to it by name. New code should call :func:`drug_blind_predictions` so the
    response space of the returned array is explicit.
    """
    return drug_blind_predictions(vehicle_treated, vehicle_profiles, cl).per_pair


def pooled_truth_from_perpair(
    true: np.ndarray, cl, vehicle_profiles: np.ndarray
) -> np.ndarray:
    """Re-express per-pair targets against one vehicle per cell line.

    ``logfc_pp = log1p(treated) - log1p(ctrl_pair)``; the pooled version
    replaces ``ctrl_pair`` with the cell line's shared vehicle, so
    ``logfc_po = logfc_pp + log1p(ctrl_pair) - log1p(ctrl_line)``.
    """
    true = np.asarray(true, dtype=float)
    profiles = np.asarray(vehicle_profiles, dtype=float)
    return true + vehicle_offsets(profiles, cl)


def _declared_space(manifest: dict | None) -> str | None:
    """Read the response space a manifest declares, if any.

    The manifest is the only source of a declaration: the file name is never
    consulted. Unrecognised values raise rather than being guessed at.
    """
    if not manifest:
        return None
    value = str(manifest.get("vehicle_construction", "")).strip().lower()
    if not value:
        return None
    # Normalise punctuation so "per-(drug, cell line) vehicle" and
    # "per_pair" both reduce to plain words.
    text = re.sub(r"[^a-z0-9]+", " ", value)
    # "per (drug, cell line) vehicle" normalises to "per drug cell line ...",
    # which is the per-pair case; check it before the pooled patterns.
    if "per pair" in text or "pairwise" in text or "per drug cell line" in text:
        return SCOPE_PER_PAIR
    if "per cell line" in text or "pooled" in text or "shared" in text:
        return SCOPE_POOLED
    raise ValueError(
        f"manifest vehicle_construction {value!r} not understood; write either "
        "'per pair' or 'per cell line' so the audit knows which response space "
        "the targets are in"
    )


def construction_calibration(
    pred,
    true,
    cl,
    *,
    vehicle_profiles: np.ndarray | None = None,
    vehicle_treated: np.ndarray | None = None,
    manifest: dict | None = None,
    tolerance: float = ANCHOR_TOLERANCE,
    top_k=TOP_K,
    metric=METRIC,
    vehicle_source: str = "not supplied",
) -> dict:
    """Does the response construction manufacture drug discrimination?

    Returns the drug-blind anchor in both response spaces, the model's AUC in
    the same spaces, the calibration status, and the model-minus-anchor
    comparison as a separate field. The status depends only on the anchors.
    """
    result = {
        "tolerance": float(tolerance),
        "vehicle_source": vehicle_source,
        "declared_space": _declared_space(manifest),
        "calibration_status": VEHICLE_NOT_ASSESSED,
        "calibration_scope": SCOPE_NONE,
        "anchor_perpair_response_auc": None,
        "anchor_pooled_response_auc": None,
        "anchor_target_response_auc": None,
        "anchor_worst_abs_deviation": None,
        "model_perpair_response_auc": None,
        "model_pooled_response_auc": None,
        "model_target_response_auc": None,
        "model_minus_anchor": None,
        "vehicle_offset_abs_max": None,
        "vehicle_offset_abs_mean": None,
        "note": "",
    }

    true = np.asarray(true, dtype=float)
    cl = np.asarray(cl)
    pred = np.asarray(pred, dtype=float)

    if vehicle_profiles is None or vehicle_treated is None:
        result["note"] = (
            "no vehicle information supplied: the audit scores the profiles in "
            "whatever response space they were built in and cannot tell whether "
            "per-pair vehicle noise is inflating the AUC. Supply the per-pair "
            "control and treated pseudobulks (--vehicle-profiles and "
            "--vehicle-treated) to have this assessed. A cell-line-mean "
            "prediction of exactly 0.500 is NOT evidence that the construction "
            "is clean."
        )
        return result

    blind = drug_blind_predictions(vehicle_treated, vehicle_profiles, cl)
    offsets = vehicle_offsets(vehicle_profiles, cl)
    offset_abs_max = float(np.abs(offsets).max())
    pooled_vehicle = bool(offset_abs_max <= OFFSET_EPSILON)

    anchor_pp = _auc(blind.per_pair, true, cl, top_k, metric)
    anchor_po = _auc(blind.pooled, true, cl, top_k, metric)
    model_pp = _auc(pred, true, cl, top_k, metric)
    model_po = _auc(
        pred + offsets, pooled_truth_from_perpair(true, cl, vehicle_profiles),
        cl, top_k, metric,
    )

    # Which spaces count towards the calibration verdict.
    scope = result["declared_space"]
    if scope is None:
        # No declaration: be conservative and fail if *either* space is biased.
        # A drug-blind predictor that picks up discrimination in either one is
        # a reason not to read the model's AUC against 0.5.
        scope = SCOPE_POOLED if pooled_vehicle else SCOPE_WORST_CASE
    elif scope == SCOPE_POOLED and not pooled_vehicle:
        raise ValueError(
            "the manifest declares a pooled (per-cell-line) vehicle but the "
            f"supplied control pseudobulks differ within a cell line "
            f"(max |offset| {offset_abs_max:.6f}); the declaration and the "
            "inputs disagree"
        )

    considered = {
        SCOPE_PER_PAIR: [anchor_pp],
        SCOPE_POOLED: [anchor_po],
        SCOPE_WORST_CASE: [anchor_pp, anchor_po],
    }[scope]
    worst = float(max(considered))
    deviation = abs(worst - 0.5)

    # The anchors are what the targets are scored in; take the one matching the
    # declared space, or the more extreme one when nothing is declared.
    target_anchor = {
        SCOPE_PER_PAIR: anchor_pp,
        SCOPE_POOLED: anchor_po,
        SCOPE_WORST_CASE: worst,
    }[scope]
    target_model = {
        SCOPE_PER_PAIR: model_pp,
        SCOPE_POOLED: model_po,
        SCOPE_WORST_CASE: model_pp if worst == anchor_pp else model_po,
    }[scope]

    if deviation > tolerance:
        status = VEHICLE_FAILED
        note = (
            f"the drug-blind predictor departs from chance by {deviation:.3f} "
            f"(anchors {anchor_pp:.3f} per-pair space, {anchor_po:.3f} pooled "
            f"space; tolerance {tolerance:.3f}). This construction manufactures "
            "drug discrimination for a predictor that never sees the drug, so "
            "the model's AUC cannot be read against 0.5. The construction is "
            "biased whatever the model scores."
        )
    else:
        status = VEHICLE_VERIFIED
        note = (
            f"the drug-blind predictor sits at chance in the response space(s) "
            f"considered (anchors {anchor_pp:.3f} per-pair, {anchor_po:.3f} "
            f"pooled; tolerance {tolerance:.3f}); the construction does not "
            "manufacture discrimination."
        )

    result.update({
        "calibration_status": status,
        "calibration_scope": scope,
        "anchor_perpair_response_auc": anchor_pp,
        "anchor_pooled_response_auc": anchor_po,
        "anchor_target_response_auc": float(target_anchor),
        "anchor_worst_abs_deviation": float(deviation),
        "model_perpair_response_auc": model_pp,
        "model_pooled_response_auc": model_po,
        "model_target_response_auc": float(target_model),
        "model_minus_anchor": float(target_model - target_anchor),
        "vehicle_offset_abs_max": offset_abs_max,
        "vehicle_offset_abs_mean": float(np.abs(offsets).mean()),
        "note": note,
    })
    if pooled_vehicle and result["declared_space"] is None:
        result["note"] += (
            " The supplied vehicles are identical within each cell line "
            "(pooled construction), so the two response spaces coincide."
        )
    return result


def calibration_ladder(
    pred, true, cl, *, n_random: int = 50, seed: int = 7301,
    top_k=TOP_K, metric=METRIC,
    vehicle_profiles: np.ndarray | None = None,
    vehicle_treated: np.ndarray | None = None,
    manifest: dict | None = None,
    tolerance: float = ANCHOR_TOLERANCE,
    vehicle_source: str = "not supplied",
) -> dict:
    """The metric checks plus the construction calibration, in one record.

    ``oracle_auc`` and ``cell_line_mean_auc`` are metric checks: they must be
    1.000 and 0.500 and say nothing about the data. Everything under
    ``calibration`` answers the data question and is what the verdict reads.
    """
    rng = np.random.default_rng(seed)
    random_aucs = []
    shuffled = np.asarray(true, dtype=float).copy()
    for _ in range(n_random):
        for cell in np.unique(cl):
            mask = np.asarray(cl) == cell
            shuffled[mask] = rng.permutation(shuffled[mask])
        random_aucs.append(_auc(shuffled, true, cl, top_k, metric))

    mean_score = _ddc(cell_line_means(true, cl), true, cl, top_k, metric)
    oracle = _ddc(true, true, cl, top_k, metric)
    calibration = construction_calibration(
        pred, true, cl,
        vehicle_profiles=vehicle_profiles, vehicle_treated=vehicle_treated,
        manifest=manifest, tolerance=tolerance, top_k=top_k, metric=metric,
        vehicle_source=vehicle_source,
    )

    return {
        "chance": 0.5,
        "random_auc": float(np.mean(random_aucs)),
        "cell_line_mean_auc": float(mean_score["specificity_auc"]),
        "cell_line_mean_gap": float(mean_score["gap"]),
        "oracle_auc": float(oracle["specificity_auc"]),
        "calibration": calibration,
        "n_random": n_random,
        "seed": seed,
    }
