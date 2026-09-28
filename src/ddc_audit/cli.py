"""Command-line interface.

Subcommands:
    run           -- full audit: DDC AUC, bootstrap CI, permutation p,
                     calibration ladder, figures, JSON, terminal summary and a
                     self-contained HTML report
    check-input   -- validate the input files against the contract
    make-example  -- write the example datasets used by the quickstart

Exit codes: 0 = audit completed, 2 = input contract violated. The verdict is in
the report, not in the exit code -- an "INDETERMINATE" audit still exits 0.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from ddc_audit import __version__
from ddc_audit.io import InputError, check, load_all, load_manifest
from ddc_audit.metrics import (
    drug_discrimination_score,
    inter_drug_pearson,
    per_pair_spearman,
)
from ddc_audit.anchors import ANCHOR_TOLERANCE, calibration_ladder
from ddc_audit.statistics import (
    between_drug_sd,
    delete_one_drug_sd,
    drug_clustered_anchor_bootstrap,
    permutation_null,
    row_resample_bootstrap,
)
from ddc_audit.report import (
    DEFAULT_THRESHOLD,
    build_result,
    html_report,
    terminal_summary,
    write_json,
)
from ddc_audit.plotting import write_figures

DEFAULT_TOP_K = 50
DEFAULT_METRIC = "pearson"
DEFAULT_SEED = 7301


def _add_common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K,
                        help="DEG threshold (default 50; use 0 for all genes)")
    parser.add_argument("--metric", choices=("pearson", "spearman"),
                        default=DEFAULT_METRIC)


def cmd_run(args: argparse.Namespace) -> int:
    top_k = None if args.top_k <= 0 else args.top_k
    pred, true, drugs, cells, profiles, treated = load_all(
        args.predictions, args.targets, args.meta,
        args.vehicle_profiles, args.vehicle_treated,
    )
    manifest = load_manifest(args.manifest)
    if (profiles is None) != (treated is None):
        raise InputError(
            "--vehicle-profiles and --vehicle-treated must be given together: "
            "the drug-blind predictor is built from both pseudobulk matrices"
        )

    ddc = drug_discrimination_score(pred, true, cells, top_k=top_k, metric=args.metric)
    boot = drug_clustered_anchor_bootstrap(
        pred, true, cells, drugs, n_boot=args.n_boot, seed=args.seed,
        top_k=top_k, metric=args.metric,
    )
    legacy = row_resample_bootstrap(
        pred, true, cells, drugs, n_boot=args.n_boot, seed=args.seed,
        top_k=top_k, metric=args.metric,
    )
    perm = permutation_null(
        pred, true, cells, n_perm=args.n_perm, seed=args.seed,
        top_k=top_k, metric=args.metric,
    )
    ladder = calibration_ladder(
        pred, true, cells, n_random=50, seed=args.seed,
        top_k=top_k, metric=args.metric,
        vehicle_profiles=profiles, vehicle_treated=treated,
        manifest=manifest,
        tolerance=args.anchor_tolerance,
        vehicle_source=(
            "control and treated pseudobulks supplied on the command line"
            if profiles is not None else "not supplied"
        ),
    )
    try:
        spearman50 = float(
            per_pair_spearman(true, pred, top_k=top_k or DEFAULT_TOP_K).mean()
        )
    except ValueError as error:
        print(f"note: per-pair Spearman@50 not reported ({error})")
        spearman50 = float("nan")
    result = build_result(
        ddc=ddc,
        bootstrap=boot,
        legacy_bootstrap=legacy,
        permutation=perm,
        ladder=ladder,
        spearman50_mean=spearman50,
        inter_drug_r=float(inter_drug_pearson(pred, cells)),
        loo_sd=float(delete_one_drug_sd(pred, true, cells, drugs, top_k=top_k, metric=args.metric)),
        between_drug_sd=float(between_drug_sd(pred, true, cells, drugs, top_k=top_k, metric=args.metric)),
        n_pairs=pred.shape[0],
        n_genes=pred.shape[1],
        n_drugs=int(len(set(drugs.tolist()))),
        n_cell_lines=int(len(set(cells.tolist()))),
        top_k=top_k or 0,
        metric=args.metric,
        seed=args.seed,
        threshold=args.threshold,
        input_manifest=manifest,
    )
    out_dir = Path(args.out)
    write_json(result, out_dir / "results.json")
    figures = write_figures(
        out_dir, pred=pred, true=true, cl=cells, bootstrap=boot, legacy=legacy,
        ladder=ladder, top_k=top_k, metric=args.metric,
    )
    summary = terminal_summary(result)
    (out_dir / "terminal_summary.txt").write_text(summary + "\n", encoding="utf-8")
    names = [figure.name for figure in figures]
    (out_dir / "audit_report.html").write_text(
        html_report(result, names), encoding="utf-8"
    )
    print(summary)
    print(f"\noutputs written to {out_dir}:")
    print("  results.json")
    print("  audit_report.html")
    print("  terminal_summary.txt")
    for name in names:
        print(f"  {name}")
    return 0


def cmd_check_input(args: argparse.Namespace) -> int:
    pred, true, drugs, cells, profiles, treated = load_all(
        args.predictions, args.targets, args.meta,
        args.vehicle_profiles, args.vehicle_treated,
    )
    report = check(pred, true, load_meta_for(args), profiles, treated)
    print(f"input OK: {report['n_pairs']} pairs x {report['n_genes']} genes, "
          f"{report['n_drugs']} drugs, {report['n_cell_lines']} cell lines")
    print(f"columns used: drug='{report['drug_column']}', "
          f"cell line='{report['cell_line_column']}'")
    print(f"pairs per cell line: {report['pairs_per_cell_line']}")
    print(f"vehicle pseudobulks: {'supplied' if report['vehicle_supplied'] else 'NOT supplied'}")
    if report["warnings"]:
        print("warnings:")
        for warning in report["warnings"]:
            print(f"  - {warning}")
    else:
        print("warnings: none")
    return 0


def load_meta_for(args: argparse.Namespace):
    from ddc_audit.io import load_meta

    return load_meta(args.meta)


def cmd_make_example(args: argparse.Namespace) -> int:
    from ddc_audit.example import build

    build(args.out, args.seed)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ddc-audit",
        description="Pre-submission drug-discrimination audit for single-cell "
                    "perturbation prediction models (Zhao & Chen, preprint).",
    )
    parser.add_argument("--version", action="version", version=__version__)
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="run the full audit")
    run.add_argument("--predictions", type=Path, required=True)
    run.add_argument("--targets", type=Path, required=True)
    run.add_argument("--meta", type=Path, required=True)
    run.add_argument("--vehicle-profiles", type=Path, default=None,
                     help="[n_pairs, n_genes] per-pair control pseudobulk in "
                          "counts space; enables the vehicle calibration")
    run.add_argument("--vehicle-treated", type=Path, default=None,
                     help="[n_pairs, n_genes] per-pair treated pseudobulk in "
                          "counts space; required with --vehicle-profiles")
    run.add_argument("--manifest", type=Path, default=None,
                     help="optional JSON recording the response space, vehicle "
                          "definition and row-alignment convention")
    run.add_argument("--out", type=Path, default=Path("audit_out"))
    run.add_argument("--n-boot", type=int, default=1000)
    run.add_argument("--n-perm", type=int, default=1000)
    run.add_argument("--seed", type=int, default=DEFAULT_SEED)
    run.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD,
                     help="exploratory reporting threshold (default 0.70); a "
                          "project convention, not a validated standard")
    run.add_argument("--anchor-tolerance", type=float, default=ANCHOR_TOLERANCE,
                     help="how far the drug-blind anchor may sit from 0.500 "
                          "before the construction is called biased (default "
                          "0.06; see ddc_audit.anchors for the justification)")
    _add_common(run)
    run.set_defaults(func=cmd_run)

    check = sub.add_parser("check-input", help="validate the input files only")
    check.add_argument("--predictions", type=Path, required=True)
    check.add_argument("--targets", type=Path, required=True)
    check.add_argument("--meta", type=Path, required=True)
    check.add_argument("--vehicle-profiles", type=Path, default=None)
    check.add_argument("--vehicle-treated", type=Path, default=None)
    check.set_defaults(func=cmd_check_input)

    example = sub.add_parser("make-example", help="write the example dataset")
    example.add_argument("--out", type=Path, default=Path("example"))
    example.add_argument("--seed", type=int, default=0)
    example.set_defaults(func=cmd_make_example)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except InputError as error:
        print(f"input error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
