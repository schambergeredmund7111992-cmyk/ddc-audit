"""The construction calibration: three checks that must not be conflated.

The distinction the tests protect:

* a cell-line-mean prediction landing on 0.500 is a property of the *metric*;
* the drug-blind anchor is a property of the *data*, built from the supplied
  counts-space treated and control pseudobulks;
* a model that outscores a biased anchor is a performance statement, not
  evidence that the construction is sound.
"""
from __future__ import annotations

import numpy as np
import pytest

from ddc_audit.anchors import (
    SCOPE_PER_PAIR,
    SCOPE_POOLED,
    VEHICLE_FAILED,
    VEHICLE_NOT_ASSESSED,
    VEHICLE_VERIFIED,
    calibration_ladder,
    cell_line_means,
    construction_calibration,
    drug_blind_predictions,
    pool_vehicle,
    vehicle_offsets,
)

CELL_LINES = ("A549", "K562", "MCF7")
CL = np.array([c for c in CELL_LINES for _ in range(9)])


def _toy(seed=0, n_genes=200, per_pair_vehicle=True):
    """Counts-space pseudobulks whose targets follow the shipped formulas."""
    rng = np.random.default_rng(seed)
    gene = rng.uniform(0.01, 1.5, size=n_genes)          # shared per cell line
    control = np.stack([gene + rng.uniform(0.001, 0.2, n_genes) for _ in range(27)])
    treated = np.stack([gene + rng.uniform(0.001, 0.8, n_genes) for _ in range(27)])
    if not per_pair_vehicle:
        control = cell_line_means(control, CL)
    true_perpair = np.log1p(treated) - np.log1p(control)
    true_pooled = np.log1p(treated) - np.log1p(cell_line_means(control, CL))
    return dict(control=control, treated=treated, true_perpair=true_perpair,
                true_pooled=true_pooled)


# --------------------------------------------------------------------------
# the predictor itself
# --------------------------------------------------------------------------

def test_drug_blind_predictor_matches_the_hand_computed_formula():
    toy = _toy()
    blind = drug_blind_predictions(toy["treated"], toy["control"], CL)
    expected_perpair = (np.log1p(cell_line_means(toy["treated"], CL))
                        - np.log1p(toy["control"]))
    expected_pooled = (np.log1p(cell_line_means(toy["treated"], CL))
                       - np.log1p(cell_line_means(toy["control"], CL)))
    assert np.allclose(blind.per_pair, expected_perpair, atol=1e-12)
    assert np.allclose(blind.pooled, expected_pooled, atol=1e-12)


def test_pooled_predictor_is_constant_within_a_cell_line():
    """Constant to within float rounding: the mean of a repeated row is that
    row, so the difference is an artifact of summing, not of the formula."""
    toy = _toy()
    blind = drug_blind_predictions(toy["treated"], toy["control"], CL)
    assert np.abs(blind.pooled - cell_line_means(blind.pooled, CL)).max() < 1e-15


def test_the_two_spaces_differ_when_the_vehicle_is_per_pair():
    toy = _toy(per_pair_vehicle=True)
    blind = drug_blind_predictions(toy["treated"], toy["control"], CL)
    assert np.abs(blind.per_pair - blind.pooled).max() > 1e-3


def test_the_two_spaces_coincide_when_the_vehicle_is_pooled():
    toy = _toy(per_pair_vehicle=False)
    blind = drug_blind_predictions(toy["treated"], toy["control"], CL)
    assert np.abs(blind.per_pair - blind.pooled).max() < 1e-12


def test_vehicle_treated_actually_changes_the_anchor():
    """The regression this revision fixes: ``vehicle_treated`` was accepted and
    then ignored, so the anchor was derived from the targets instead of the
    supplied counts."""
    toy = _toy()
    base = drug_blind_predictions(toy["treated"], toy["control"], CL).per_pair
    scaled = drug_blind_predictions(2.0 * toy["treated"], toy["control"], CL).per_pair
    assert np.abs(scaled - base).max() > 1e-3


def test_log1p_of_mean_is_not_mean_of_log1p():
    """The two differ; the implementation must use the former."""
    toy = _toy()
    blind = drug_blind_predictions(toy["treated"], toy["control"], CL)
    log_of_mean = np.log1p(cell_line_means(toy["treated"], CL))
    mean_of_log = cell_line_means(np.log1p(toy["treated"]), CL)
    assert np.abs(log_of_mean - mean_of_log).max() > 1e-6
    assert np.allclose(blind.treated_line_mean,
                       cell_line_means(toy["treated"], CL), atol=0)


# --------------------------------------------------------------------------
# alignment
# --------------------------------------------------------------------------

def test_misaligned_treated_control_rejected():
    toy = _toy()
    with pytest.raises(ValueError, match="treated pseudobulks"):
        drug_blind_predictions(toy["treated"][:26], toy["control"], CL)


def test_misaligned_labels_rejected():
    toy = _toy()
    with pytest.raises(ValueError, match="cell-line labels"):
        drug_blind_predictions(toy["treated"], toy["control"], CL[:26])


def test_negative_counts_rejected():
    toy = _toy()
    bad = toy["control"].copy()
    bad[0, 0] = -1.0
    with pytest.raises(ValueError, match="negative counts"):
        drug_blind_predictions(toy["treated"], bad, CL)


# --------------------------------------------------------------------------
# calibration status
# --------------------------------------------------------------------------

def test_calibration_not_assessed_without_vehicles():
    toy = _toy()
    cal = construction_calibration(np.zeros_like(toy["true_perpair"]),
                                   toy["true_perpair"], CL)
    assert cal["calibration_status"] == VEHICLE_NOT_ASSESSED
    assert cal["anchor_perpair_response_auc"] is None
    assert cal["anchor_pooled_response_auc"] is None
    assert "NOT evidence" in cal["note"]


def test_calibration_verified_under_a_pooled_vehicle():
    toy = _toy(per_pair_vehicle=False)
    cal = construction_calibration(
        cell_line_means(toy["true_pooled"], CL), toy["true_pooled"], CL,
        vehicle_profiles=toy["control"], vehicle_treated=toy["treated"],
        manifest={"vehicle_construction": "per cell line"},
    )
    assert cal["calibration_status"] == VEHICLE_VERIFIED
    assert cal["calibration_scope"] == SCOPE_POOLED
    assert cal["anchor_pooled_response_auc"] == pytest.approx(0.5, abs=1e-9)


def test_calibration_fails_under_a_per_pair_vehicle():
    toy = _toy(seed=3)
    blind = drug_blind_predictions(toy["treated"], toy["control"], CL)
    cal = construction_calibration(
        blind.per_pair, toy["true_perpair"], CL,
        vehicle_profiles=toy["control"], vehicle_treated=toy["treated"],
        manifest={"vehicle_construction": "per pair"},
    )
    assert cal["calibration_status"] == VEHICLE_FAILED
    assert cal["anchor_perpair_response_auc"] > 0.5 + cal["tolerance"]
    assert cal["model_minus_anchor"] is not None


def test_model_outsourcing_a_biased_anchor_does_not_rescue_the_calibration():
    """Three required behaviours in one test: the construction stays FAILED,
    the anchor is unchanged by the model, and the comparison is still reported."""
    toy = _toy(seed=3)
    blind = drug_blind_predictions(toy["treated"], toy["control"], CL)
    strong = blind.per_pair * 0.2 + toy["true_perpair"] * 0.8   # clearly better
    weak = construction_calibration(
        blind.per_pair, toy["true_perpair"], CL,
        vehicle_profiles=toy["control"], vehicle_treated=toy["treated"],
        manifest={"vehicle_construction": "per pair"},
    )
    better = construction_calibration(
        strong, toy["true_perpair"], CL,
        vehicle_profiles=toy["control"], vehicle_treated=toy["treated"],
        manifest={"vehicle_construction": "per pair"},
    )
    assert better["calibration_status"] == VEHICLE_FAILED
    assert better["anchor_perpair_response_auc"] == pytest.approx(
        weak["anchor_perpair_response_auc"], abs=1e-12
    )
    assert better["model_minus_anchor"] > weak["model_minus_anchor"]


def test_constant_prediction_scores_exactly_half_in_the_pooled_space():
    toy = _toy(per_pair_vehicle=False)
    cal = construction_calibration(
        cell_line_means(toy["true_pooled"], CL), toy["true_pooled"], CL,
        vehicle_profiles=toy["control"], vehicle_treated=toy["treated"],
        manifest={"vehicle_construction": "per cell line"},
    )
    assert cal["anchor_pooled_response_auc"] == pytest.approx(0.5, abs=1e-12)


def test_tolerance_is_honoured_and_reported():
    toy = _toy(seed=3)
    blind = drug_blind_predictions(toy["treated"], toy["control"], CL)
    strict = construction_calibration(
        blind.per_pair, toy["true_perpair"], CL,
        vehicle_profiles=toy["control"], vehicle_treated=toy["treated"],
        manifest={"vehicle_construction": "per pair"}, tolerance=0.01,
    )
    loose = construction_calibration(
        blind.per_pair, toy["true_perpair"], CL,
        vehicle_profiles=toy["control"], vehicle_treated=toy["treated"],
        manifest={"vehicle_construction": "per pair"}, tolerance=0.9,
    )
    assert strict["calibration_status"] == VEHICLE_FAILED
    assert loose["calibration_status"] == VEHICLE_VERIFIED
    assert strict["tolerance"] == 0.01 and loose["tolerance"] == 0.9


def test_declared_pooled_vehicle_with_per_pair_controls_is_rejected():
    toy = _toy(per_pair_vehicle=True)
    with pytest.raises(ValueError, match="declaration and the inputs disagree"):
        construction_calibration(
            np.zeros_like(toy["true_perpair"]), toy["true_perpair"], CL,
            vehicle_profiles=toy["control"], vehicle_treated=toy["treated"],
            manifest={"vehicle_construction": "per cell line"},
        )


def test_manifest_per_pair_declaration_selects_the_per_pair_space():
    toy = _toy(seed=3)
    cal = construction_calibration(
        np.zeros_like(toy["true_perpair"]), toy["true_perpair"], CL,
        vehicle_profiles=toy["control"], vehicle_treated=toy["treated"],
        manifest={"vehicle_construction": "per pair"},
    )
    assert cal["calibration_scope"] == SCOPE_PER_PAIR


def test_unknown_manifest_declaration_raises():
    toy = _toy()
    with pytest.raises(ValueError, match="not understood"):
        construction_calibration(
            np.zeros_like(toy["true_perpair"]), toy["true_perpair"], CL,
            vehicle_profiles=toy["control"], vehicle_treated=toy["treated"],
            manifest={"vehicle_construction": "something else entirely"},
        )


def test_the_shipped_bundle_reproduces_the_documented_anchor_values():
    """End-to-end pin on the real artifact: these are the values reported in
    VALIDATION.md and in the report."""
    from pathlib import Path

    import pandas as pd

    # Locate the evidence by content, not by a fixed depth: the tool ships both
    # as the `ddc-audit` repository (evidence/ beside tests/) and inside the
    # competition package (code/regeneration/bundle/), and it must skip cleanly
    # when installed from a wheel or an sdist that carries no evidence at all.
    here = Path(__file__).resolve()
    bundle = next(
        (q / rel
         for q in here.parents
         for rel in ("evidence", "regeneration/bundle", "code/regeneration/bundle")
         if (q / rel / "pair_order.csv").is_file()),
        None,
    )
    if bundle is None:
        pytest.skip("evidence bundle not present in this installation")
    meta = pd.read_csv(bundle / "pair_order.csv")
    cl = meta["cell_line"].to_numpy()
    control = np.load(bundle / "vehicle_profiles_perpair_control.npy")
    treated = np.load(bundle / "vehicle_profiles_perpair_treated.npy")
    perpair = np.load(bundle / "target_perpair.npy")
    pooled = np.load(bundle / "target_pooled.npy")
    model = np.load(bundle / "prediction_loss_only_perpair.npy")

    cal = construction_calibration(
        model, perpair, cl,
        vehicle_profiles=control, vehicle_treated=treated,
        manifest={"vehicle_construction": "per pair"},
    )
    assert cal["anchor_perpair_response_auc"] == pytest.approx(0.958333, abs=1e-5)
    assert cal["anchor_pooled_response_auc"] == pytest.approx(0.5, abs=1e-9)
    assert cal["calibration_status"] == VEHICLE_FAILED
    assert cal["model_minus_anchor"] == pytest.approx(0.569444 - 0.958333, abs=1e-5)

    # the pooled section must be handed a *pooled* vehicle: one row per cell
    # line. Handing it the per-pair controls is the contradiction the manifest
    # check now rejects, so the test loads the pooled files the bundle ships.
    def _load(name):
        path = bundle / name
        return np.load(path) if path.is_file() else None

    pooled_control = _load("vehicle_profiles_pooled_control.npy")
    pooled_treated = _load("vehicle_profiles_pooled_treated.npy")
    pooled_model = _load("prediction_loss_only_pooled.npy")
    if pooled_control is None or pooled_treated is None or pooled_model is None:
        pytest.skip("pooled vehicle files not present in this bundle")
    pooled_cal = construction_calibration(
        pooled_model, pooled, cl,
        vehicle_profiles=pooled_control, vehicle_treated=pooled_treated,
        manifest={"vehicle_construction": "per cell line"},
    )
    assert pooled_cal["calibration_status"] == VEHICLE_VERIFIED
    assert pooled_cal["anchor_pooled_response_auc"] == pytest.approx(0.5, abs=1e-9)


# --------------------------------------------------------------------------
# the ladder
# --------------------------------------------------------------------------

def test_ladder_metric_checks_are_data_independent():
    toy = _toy()
    ladder = calibration_ladder(np.zeros_like(toy["true_perpair"]),
                                toy["true_perpair"], CL)
    assert ladder["chance"] == 0.5
    assert ladder["cell_line_mean_auc"] == pytest.approx(0.5, abs=1e-9)
    assert ladder["oracle_auc"] == 1.0
    assert ladder["calibration"]["calibration_status"] == VEHICLE_NOT_ASSESSED


def test_ladder_carries_the_calibration_record():
    toy = _toy(per_pair_vehicle=False)
    ladder = calibration_ladder(
        cell_line_means(toy["true_pooled"], CL), toy["true_pooled"], CL,
        vehicle_profiles=toy["control"], vehicle_treated=toy["treated"],
        manifest={"vehicle_construction": "per cell line"},
    )
    cal = ladder["calibration"]
    assert cal["calibration_status"] == VEHICLE_VERIFIED
    assert cal["anchor_pooled_response_auc"] == pytest.approx(0.5, abs=1e-9)


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------

def test_vehicle_offsets_zero_under_pooled_and_nonzero_under_perpair():
    pooled = _toy(per_pair_vehicle=False)
    perpair = _toy(per_pair_vehicle=True)
    assert np.abs(vehicle_offsets(pooled["control"], CL)).max() < 1e-12
    assert np.abs(vehicle_offsets(perpair["control"], CL)).max() > 1e-3


def test_pool_vehicle_matches_a_hand_computed_mean():
    control = np.arange(27 * 4, dtype=float).reshape(27, 4)
    pooled = pool_vehicle(control, CL)
    for cell in np.unique(CL):
        assert np.allclose(pooled[CL == cell], control[CL == cell].mean(0))


def test_zero_genes_in_the_pseudobulk_are_tolerated():
    toy = _toy()
    control = toy["control"].copy()
    control[:, :5] = 0.0
    blind = drug_blind_predictions(toy["treated"], control, CL)
    assert np.isfinite(blind.per_pair).all()
