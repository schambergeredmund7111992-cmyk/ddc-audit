"""End-to-end CLI behaviour on the example datasets."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from ddc_audit.cli import main


def _run(example_dir, which, out_dir, **extra):
    argv = [
        "run",
        "--predictions", str(example_dir / f"predictions_{which}.npy"),
        "--targets", str(example_dir / "targets.npy"),
        "--meta", str(example_dir / "meta.csv"),
        "--out", str(out_dir),
        "--n-boot", "40", "--n-perm", "40",
    ]
    for key, value in extra.items():
        argv += [f"--{key.replace('_', '-')}", str(value)]
    return main(argv)


def test_make_example_and_run_end_to_end(tmp_path, capsys):
    example_dir = tmp_path / "example"
    assert main(["make-example", "--out", str(example_dir)]) == 0

    out_collapsed = tmp_path / "audit_collapsed"
    assert _run(example_dir, "collapsed", out_collapsed) == 0
    result = json.loads((out_collapsed / "results.json").read_text(encoding="utf-8"))
    assert abs(result["ddc"]["auc"] - 0.5) < 0.12
    for name in ("onoff_distribution.pdf", "calibration_ladder.pdf",
                 "bootstrap_distribution.pdf", "bootstrap_comparison.pdf",
                 "terminal_summary.txt", "audit_report.html", "results.json"):
        assert (out_collapsed / name).is_file(), name
    assert "DDC AUC" in capsys.readouterr().out

    out_aware = tmp_path / "audit_aware"
    assert _run(example_dir, "drug_aware", out_aware) == 0
    aware = json.loads((out_aware / "results.json").read_text(encoding="utf-8"))
    assert aware["ddc"]["auc"] > 0.8


def test_verdict_is_indeterminate_without_vehicle_information(tmp_path):
    example_dir = tmp_path / "example"
    main(["make-example", "--out", str(example_dir)])
    out = tmp_path / "audit"
    assert _run(example_dir, "collapsed", out) == 0
    summary = (out / "terminal_summary.txt").read_text(encoding="utf-8")
    assert "NOT_ASSESSED" in summary
    assert "INDETERMINATE" in summary
    assert "does NOT show" in summary  # the Mean = 0.5 confusion is called out
    result = json.loads((out / "results.json").read_text(encoding="utf-8"))
    assert result["schema_version"] == 3
    assert result["calibration"]["calibration_status"] == "NOT_ASSESSED"


def test_anchor_tolerance_flag_is_honoured(tmp_path):
    """A tolerance wide enough to swallow the anchor flips the status, and the
    report says which tolerance was used."""
    example_dir = tmp_path / "example"
    main(["make-example", "--out", str(example_dir)])
    out = tmp_path / "audit"
    _run(example_dir, "collapsed", out, anchor_tolerance=0.9)
    result = json.loads((out / "results.json").read_text(encoding="utf-8"))
    assert result["calibration"]["tolerance"] == 0.9
    assert result["calibration"]["calibration_status"] == "NOT_ASSESSED"


def test_vehicle_profiles_enable_the_calibration(tmp_path):
    """With per-pair vehicle pseudobulks the ladder assesses the construction."""
    example_dir = tmp_path / "example"
    main(["make-example", "--out", str(example_dir)])

    # the example's targets are built as log1p(treated) - log1p(control), so a
    # positive counts matrix reproduces a valid per-pair construction
    rng = np.random.default_rng(0)
    control = rng.uniform(0.0, 1.0, size=(27, 300))
    treated = control + rng.uniform(0.0, 0.5, size=(27, 300))
    np.save(tmp_path / "control.npy", control)
    np.save(tmp_path / "treated.npy", treated)

    out = tmp_path / "audit"
    assert _run(example_dir, "drug_aware", out,
                vehicle_profiles=tmp_path / "control.npy",
                vehicle_treated=tmp_path / "treated.npy") == 0
    result = json.loads((out / "results.json").read_text(encoding="utf-8"))
    cal = result["calibration"]
    assert cal["calibration_status"] in {"VERIFIED", "FAILED"}
    assert cal["anchor_perpair_response_auc"] is not None
    assert cal["anchor_pooled_response_auc"] is not None
    assert cal["model_minus_anchor"] is not None
    assert cal["calibration_scope"] in {"both_space_worst_case", "per_pair", "pooled"}


def test_vehicle_profiles_without_treated_is_an_input_error(tmp_path, capsys):
    example_dir = tmp_path / "example"
    main(["make-example", "--out", str(example_dir)])
    rng = np.random.default_rng(0)
    np.save(tmp_path / "control.npy", rng.uniform(size=(27, 300)))
    code = _run(example_dir, "collapsed", tmp_path / "out",
                vehicle_profiles=tmp_path / "control.npy")
    assert code == 2
    assert "vehicle" in capsys.readouterr().err


def test_html_report_is_self_contained(tmp_path):
    example_dir = tmp_path / "example"
    main(["make-example", "--out", str(example_dir)])
    out = tmp_path / "audit"
    _run(example_dir, "collapsed", out)
    html = (out / "audit_report.html").read_text(encoding="utf-8")
    assert html.startswith("<!doctype html>")
    assert "http://" not in html and "https://" not in html  # no remote assets
    assert "Statistical evidence" in html
    assert "Metric checks" in html
    assert "Construction calibration" in html


def test_threshold_flag_changes_only_the_threshold_row(tmp_path):
    example_dir = tmp_path / "example"
    main(["make-example", "--out", str(example_dir)])
    out = tmp_path / "audit"
    _run(example_dir, "drug_aware", out, threshold=0.99)
    result = json.loads((out / "results.json").read_text(encoding="utf-8"))
    assert result["inputs"]["threshold"] == 0.99
    summary = (out / "terminal_summary.txt").read_text(encoding="utf-8")
    assert "0.99" in summary


def test_check_input_rejects_bad_files(tmp_path, capsys):
    bad = tmp_path / "bad.npy"
    bad.write_bytes(b"not a npy")
    meta = tmp_path / "meta.csv"
    meta.write_text("drug_id,cell_line\nd,A549\n", encoding="utf-8")
    code = main([
        "check-input",
        "--predictions", str(bad),
        "--targets", str(bad),
        "--meta", str(meta),
    ])
    assert code == 2
    assert "input error" in capsys.readouterr().err


def test_check_input_reports_columns_and_warnings(tmp_path, capsys):
    pred = np.random.default_rng(0).normal(size=(27, 300))
    true = np.random.default_rng(1).normal(size=(27, 300))
    meta = pd.DataFrame({
        "drug": [f"d{i // 3}" for i in range(27)],
        "cell_line": ["A549", "K562", "MCF7"] * 9,
    })
    np.save(tmp_path / "p.npy", pred)
    np.save(tmp_path / "t.npy", true)
    meta.to_csv(tmp_path / "m.csv", index=False)
    code = main(["check-input", "--predictions", str(tmp_path / "p.npy"),
                 "--targets", str(tmp_path / "t.npy"),
                 "--meta", str(tmp_path / "m.csv")])
    assert code == 0
    out = capsys.readouterr().out
    assert "drug='drug'" in out
    assert "NOT supplied" in out


def test_manifest_is_recorded_in_the_result(tmp_path):
    example_dir = tmp_path / "example"
    main(["make-example", "--out", str(example_dir)])
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({
        "response_space": "logFC vs pooled vehicle",
        "vehicle_construction": "one vehicle per cell line",
        "row_alignment": "positional",
    }), encoding="utf-8")
    out = tmp_path / "audit"
    _run(example_dir, "collapsed", out, manifest=manifest)
    result = json.loads((out / "results.json").read_text(encoding="utf-8"))
    assert result["inputs"]["manifest"]["vehicle_construction"].startswith("one vehicle")
