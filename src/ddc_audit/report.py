"""Machine-readable JSON result, the terminal summary, and the HTML audit report.

The report keeps four questions apart, because a single PASS/FAIL hides them:

1. **Data and calibration** -- are the inputs well formed, and does the response
   construction itself manufacture drug discrimination? The second question is
   answered by the drug-blind anchor, and by nothing else.
2. **Statistical evidence** -- does the model beat the permutation null?
3. **Effect size and uncertainty** -- how large is the AUC, how wide the
   interval, and how does the model compare with the anchor as a performance
   statement (a separate question from whether the construction is sound).
4. **Threshold** -- does the AUC clear the project's reporting threshold?

A failed or unassessed calibration can never yield a clean PASS. Nor can a model
that outscores a biased anchor: that comparison is reported, but it is not
evidence that the response space is trustworthy.
"""
from __future__ import annotations

import html
import json
from pathlib import Path

from ddc_audit.anchors import (
    VEHICLE_FAILED,
    VEHICLE_NOT_ASSESSED,
    VEHICLE_VERIFIED,
)

THRESHOLD_LABEL = "exploratory reporting threshold"
DEFAULT_THRESHOLD = 0.70
SCHEMA_VERSION = 3


def build_result(
    *,
    ddc: dict,
    bootstrap: dict,
    legacy_bootstrap: dict | None,
    permutation: dict,
    ladder: dict,
    spearman50_mean: float,
    inter_drug_r: float,
    loo_sd: float | None,
    between_drug_sd: float | None,
    n_pairs: int,
    n_genes: int,
    n_drugs: int,
    n_cell_lines: int,
    top_k: int,
    metric: str,
    seed: int,
    threshold: float = DEFAULT_THRESHOLD,
    input_manifest: dict | None = None,
) -> dict:
    calibration = dict(ladder["calibration"])
    return {
        "schema_version": SCHEMA_VERSION,
        "inputs": {
            "n_pairs": n_pairs,
            "n_genes": n_genes,
            "n_drugs": n_drugs,
            "n_cell_lines": n_cell_lines,
            "top_k": top_k,
            "metric": metric,
            "seed": seed,
            "threshold": threshold,
            "manifest": input_manifest,
        },
        "ddc": {
            "auc": ddc["specificity_auc"],
            "gap": ddc["gap"],
            "on_diag_mean": ddc["on_diag_mean"],
            "off_diag_mean": ddc["off_diag_mean"],
            "n_pairs_scored": ddc["n_pairs_scored"],
            "wilcoxon_p_on_gt_off": ddc["wilcoxon_p_on_gt_off"],
            "estimand": "mean over anchors of the within-cell-line win rate",
            "response_space": calibration.get("declared_space"),
        },
        "bootstrap": {
            "estimand": bootstrap["estimand"],
            "unit": bootstrap["unit"],
            "ci_lo": bootstrap["ci_lo"],
            "ci_hi": bootstrap["ci_hi"],
            "mean": bootstrap["bootstrap_mean"],
            "bias": bootstrap["bias"],
            "n_boot": bootstrap["n_boot"],
            "seed": bootstrap["seed"],
        },
        "bootstrap_legacy": None if legacy_bootstrap is None else {
            "estimand": legacy_bootstrap["estimand"],
            "unit": legacy_bootstrap["unit"],
            "ci_lo": legacy_bootstrap["ci_lo"],
            "ci_hi": legacy_bootstrap["ci_hi"],
            "mean": legacy_bootstrap["bootstrap_mean"],
            "bias": legacy_bootstrap["bias"],
            "n_boot": legacy_bootstrap["n_boot"],
            "seed": legacy_bootstrap["seed"],
        },
        "permutation": {
            "test": permutation["test"],
            "p_value": permutation["p_value"],
            "null_mean": permutation["null_mean"],
            "null_sd": permutation["null_sd"],
            "n_perm": permutation["n_perm"],
            "seed": permutation["seed"],
        },
        "calibration": calibration,
        "metric_checks": {
            "chance": ladder["chance"],
            "random_auc": ladder["random_auc"],
            "cell_line_mean_auc": ladder["cell_line_mean_auc"],
            "cell_line_mean_gap": ladder["cell_line_mean_gap"],
            "oracle_auc": ladder["oracle_auc"],
            "n_random": ladder["n_random"],
            "seed": ladder["seed"],
        },
        "per_pair": {
            "spearman50_mean": spearman50_mean,
            "inter_drug_pearson": inter_drug_r,
            "delete_one_drug_sd": loo_sd,
            "between_drug_sd": between_drug_sd,
        },
    }


def write_json(result: dict, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _verdict(result: dict) -> tuple[str, str]:
    """Return (verdict, one-line reason). Four questions, kept apart."""
    d = result["ddc"]
    p = result["permutation"]
    b = result["bootstrap"]
    cal = result["calibration"]
    threshold = result["inputs"]["threshold"]
    status = cal["calibration_status"]

    if status == VEHICLE_NOT_ASSESSED:
        return "INDETERMINATE (construction calibration not assessed)", (
            "the AUC is reported, but without the per-pair control and treated "
            "pseudobulks the audit cannot tell whether the response construction "
            "is manufacturing discrimination. Supply --vehicle-profiles and "
            "--vehicle-treated to complete the calibration."
        )
    if status == VEHICLE_FAILED:
        return "INDETERMINATE (construction calibration failed)", (
            f"a predictor that receives no drug information reaches "
            f"{cal['anchor_target_response_auc']:.3f} against these targets "
            f"(tolerance {cal['tolerance']:.2f} around 0.500), so the "
            "construction is biased. The model's AUC cannot be read against 0.5 "
            "and no clean verdict is issued; the model-minus-anchor comparison "
            f"({cal['model_minus_anchor']:+.3f}) is an effect statement, not a "
            "calibration one."
        )
    if p["p_value"] > 0.05:
        return "NO DETECTABLE DISCRIMINATION", (
            f"the observed AUC {d['auc']:.3f} sits inside the permutation null "
            f"(p = {p['p_value']:.3f}); this is a failure to detect, not a proof "
            "that no signal exists."
        )
    if d["auc"] >= threshold:
        return "ABOVE THRESHOLD", (
            f"AUC {d['auc']:.3f} >= {threshold:.2f}, interval "
            f"[{b['ci_lo']:.2f}, {b['ci_hi']:.2f}], permutation p = "
            f"{p['p_value']:.3f}. The threshold is a project convention, not a "
            "validated standard of usability."
        )
    tail = (
        ""
        if b["ci_lo"] > 0.5
        else f"; the 95% interval [{b['ci_lo']:.2f}, {b['ci_hi']:.2f}] includes 0.5"
    )
    return "BELOW THRESHOLD", (
        f"discrimination is detectable (p = {p['p_value']:.3f}) but the AUC "
        f"{d['auc']:.3f} stays below the {threshold:.2f} {THRESHOLD_LABEL}{tail}"
    )


def _fmt(value, spec="{:.3f}"):
    if value is None:
        return "n/a"
    try:
        if value != value:  # NaN
            return "n/a"
    except TypeError:
        return "n/a"
    return spec.format(value)


_SCOPE_LABEL = {
    "both_space_worst_case": "both response spaces, worst case",
    "per_pair": "per-pair response space (declared)",
    "pooled": "pooled response space (declared)",
    "none": "not assessed",
}


def terminal_summary(result: dict) -> str:
    d = result["ddc"]; b = result["bootstrap"]; bl = result["bootstrap_legacy"]
    p = result["permutation"]; cal = result["calibration"]; m = result["metric_checks"]
    pp = result["per_pair"]; inp = result["inputs"]
    head, why = _verdict(result)

    lines = [
        "=" * 74,
        " ddc-audit  --  drug-discrimination control",
        "=" * 74,
        f" inputs   : {inp['n_pairs']} pairs, {inp['n_drugs']} drugs, "
        f"{inp['n_cell_lines']} cell lines, {inp['n_genes']} genes",
        f" metric   : {inp['metric']} on the per-cell-line union top-{inp['top_k']} genes",
        "",
        " 1a. metric checks (properties of the control, not of the data)",
        f"   chance (definition)          : {m['chance']:.3f}",
        f"   random predictions           : {_fmt(m['random_auc'])}",
        f"   cell-line-mean prediction    : {_fmt(m['cell_line_mean_auc'])}"
        "   must be 0.500",
        f"   oracle (truth as prediction) : {_fmt(m['oracle_auc'])}   must be 1.000",
        "",
        " 1b. construction calibration (a property of the data)",
        f"   status                       : {cal['calibration_status']}",
        "   scope                        : "
        f"{_SCOPE_LABEL.get(cal['calibration_scope'], cal['calibration_scope'])}",
        f"   drug-blind anchor, per-pair  : {_fmt(cal['anchor_perpair_response_auc'])}",
        f"   drug-blind anchor, pooled    : {_fmt(cal['anchor_pooled_response_auc'])}",
        f"   anchor used for the verdict  : {_fmt(cal['anchor_target_response_auc'])}",
        f"   tolerance around 0.500       : {cal['tolerance']:.3f}",
        f"   vehicle source               : {cal['vehicle_source']}",
    ]
    if cal["vehicle_offset_abs_max"] is not None:
        lines.append(
            f"   per-pair vehicle |offset|    : max "
            f"{cal['vehicle_offset_abs_max']:.4f}, mean "
            f"{cal['vehicle_offset_abs_mean']:.4f}"
        )
    lines += [
        "",
        " 2. statistical evidence",
        f"   permutation p (plus-one)        : {_fmt(p['p_value'])}"
        "   [within-cell-line permutation]",
        f"   one-sided Wilcoxon p (on > off) : {_fmt(d['wilcoxon_p_on_gt_off'])}"
        "   [paired, directional only]",
        f"   permutation null                : mean {_fmt(p['null_mean'])}, "
        f"sd {_fmt(p['null_sd'])}  ({p['n_perm']} draws, seed {p['seed']})",
        "",
        " 3. effect size and uncertainty",
        f"   DDC AUC                    : {_fmt(d['auc'])}",
        f"   on vs off diagonal gap     : {_fmt(d['gap'])}",
        f"   per-pair Spearman@50 (mean): {_fmt(pp['spearman50_mean'])}",
        f"   inter-drug prediction r    : {_fmt(pp['inter_drug_pearson'])}",
        f"   95% CI, drug-clustered     : [{_fmt(b['ci_lo'])}, {_fmt(b['ci_hi'])}]"
        f"   (bootstrap bias {_fmt(b['bias'])})",
    ]
    if bl is not None:
        lines.append(
            f"   95% CI, row-resample legacy: [{_fmt(bl['ci_lo'])}, {_fmt(bl['ci_hi'])}]"
            f"   (superseded; bias {_fmt(bl['bias'])})"
        )
    lines += [
        f"   delete-one-drug SD         : {_fmt(pp['delete_one_drug_sd'])}",
        f"   between-drug SD            : {_fmt(pp['between_drug_sd'])}",
        f"   model - anchor             : {_fmt(cal['model_minus_anchor'])}"
        "   [performance, not calibration]",
        "",
        " 4. threshold",
        f"   exploratory threshold      : {inp['threshold']:.2f}"
        "   (project convention, not a validated standard)",
        "",
        f" verdict: {head}",
        f"   {why}",
    ]
    if cal["calibration_status"] == VEHICLE_NOT_ASSESSED:
        lines += [
            "",
            " NOTE: the cell-line-mean prediction sitting at 0.500 does NOT show "
            "that the construction is clean. It is a property of the metric.",
        ]
    lines.append("=" * 74)
    return "\n".join(lines)


def html_report(result: dict, figure_names: list[str]) -> str:
    """A single self-contained HTML page; no server, no network."""
    d = result["ddc"]; b = result["bootstrap"]; p = result["permutation"]
    cal = result["calibration"]; m = result["metric_checks"]
    pp = result["per_pair"]; inp = result["inputs"]
    head, why = _verdict(result)

    def row(label, value, note=""):
        return (
            f"<tr><th>{html.escape(label)}</th><td>{html.escape(str(value))}</td>"
            f"<td class='note'>{html.escape(note)}</td></tr>"
        )

    status_class = {VEHICLE_VERIFIED: "ok", VEHICLE_FAILED: "bad"}.get(
        cal["calibration_status"], "warn"
    )
    figures = "".join(
        f"<li><a href='{html.escape(name)}'>{html.escape(name)}</a></li>"
        for name in figure_names
    )
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<title>ddc-audit report</title>
<style>
 body{{font-family:system-ui,-apple-system,Segoe UI,sans-serif;max-width:54rem;
      margin:2rem auto;padding:0 1rem;color:#1f2937;line-height:1.5}}
 h1{{font-size:1.4rem;margin-bottom:.2rem}} h2{{font-size:1.05rem;margin-top:2rem}}
 .verdict{{border-left:4px solid #b45309;background:#fffbeb;padding:.75rem 1rem;margin:1rem 0}}
 table{{border-collapse:collapse;width:100%;margin:.5rem 0}}
 th,td{{text-align:left;padding:.35rem .5rem;border-bottom:1px solid #e5e7eb;
        vertical-align:top;font-size:.92rem}}
 th{{width:15rem;font-weight:600}} td.note{{color:#6b7280;width:22rem;font-size:.85rem}}
 code{{background:#f3f4f6;padding:.1rem .3rem;border-radius:.2rem}}
 .pill{{display:inline-block;padding:.1rem .5rem;border-radius:.6rem;font-size:.8rem;
        font-weight:600;color:#fff}}
 .ok{{background:#2A9D8F}} .bad{{background:#E76F51}} .warn{{background:#B45309}}
</style></head><body>
<h1>ddc-audit &mdash; drug-discrimination control</h1>
<p>Does the audited model rank a held-out drug above the other held-out drugs in
the same cell line? A model that emits a drug-invariant response template cannot,
however well its per-pair correlations read.</p>
<div class="verdict"><strong>{html.escape(head)}</strong><br>{html.escape(why)}</div>

<h2>1a. Metric checks <span style="font-weight:400">(properties of the control)</span></h2>
<table>
{row("cell-line-mean prediction", _fmt(m["cell_line_mean_auc"]), "must be 0.500; a property of the metric, NOT a construction check")}
{row("oracle (truth as prediction)", _fmt(m["oracle_auc"]), "must be 1.000")}
{row("random predictions", _fmt(m["random_auc"]), f"{m['n_random']} draws, seed {m['seed']}")}
</table>

<h2>1b. Construction calibration <span style="font-weight:400">(a property of the data)</span></h2>
<p><span class="pill {status_class}">{html.escape(cal["calibration_status"])}</span>
scope: {html.escape(_SCOPE_LABEL.get(cal["calibration_scope"], str(cal["calibration_scope"])))}</p>
<table>
{row("drug-blind anchor, per-pair space", _fmt(cal["anchor_perpair_response_auc"]), "log1p(line-mean treated) - log1p(this pair's own vehicle)")}
{row("drug-blind anchor, pooled space", _fmt(cal["anchor_pooled_response_auc"]), "the same predictor against one shared vehicle per cell line")}
{row("anchor used for the verdict", _fmt(cal["anchor_target_response_auc"]), f"tolerance {cal['tolerance']:.3f} around 0.500")}
{row("per-pair vehicle |offset|", "n/a" if cal["vehicle_offset_abs_max"] is None else f"max {cal['vehicle_offset_abs_max']:.4f}, mean {cal['vehicle_offset_abs_mean']:.4f}", "how far each pair's vehicle sits from its cell line's")}
{row("vehicle source", cal["vehicle_source"], "")}
</table>
<p class="note">{html.escape(cal["note"])}</p>

<h2>2. Statistical evidence</h2>
<table>
{row("permutation p (plus-one)", _fmt(p["p_value"]), "within-cell-line permutation of the predicted rows destroys the drug pairing")}
{row("permutation null", f"mean {_fmt(p['null_mean'])}, sd {_fmt(p['null_sd'])}", f"{p['n_perm']} draws, seed {p['seed']}")}
{row("one-sided Wilcoxon p (on &gt; off)", _fmt(d["wilcoxon_p_on_gt_off"]), "paired and directional only; a different question from the permutation p")}
</table>

<h2>3. Effect size and uncertainty</h2>
<table>
{row("DDC AUC", _fmt(d["auc"]), "mean over anchors of the within-cell-line win rate")}
{row("on vs off diagonal gap", _fmt(d["gap"]), "")}
{row("95% CI, drug-clustered", f"[{_fmt(b['ci_lo'])}, {_fmt(b['ci_hi'])}]", b["unit"] + f"; bootstrap bias {_fmt(b['bias'])}")}
{row("model &minus; anchor", _fmt(cal["model_minus_anchor"]), "a performance statement; it never rescues a failed calibration")}
{row("per-pair Spearman@50", _fmt(pp["spearman50_mean"]), "the metric that is blind to drug identity")}
{row("inter-drug prediction r", _fmt(pp["inter_drug_pearson"]), "collapse meter")}
{row("delete-one-drug SD", _fmt(pp["delete_one_drug_sd"]), "")}
{row("between-drug SD", _fmt(pp["between_drug_sd"]), "")}
</table>

<h2>4. Threshold</h2>
<p>AUC compared against the project's exploratory reporting threshold
<code>{inp['threshold']:.2f}</code>. This number is a convention of the project,
not a validated standard of practical usability; it is calibrated against
nothing but the authors' judgement.</p>

<h2>Figures</h2>
<ul>{figures}</ul>

<h2>Limits</h2>
<ul>
<li>One dataset, one split, one gene panel: the verdict is about this cohort.</li>
<li>A non-significant AUC is a failure to detect, not proof that no signal exists.</li>
<li>Models scored in independent reconstructions are not a ranking.</li>
<li>A failed or unassessed construction calibration means no clean pass or fail
    can be issued, however the model compares with the anchor.</li>
</ul>
</body></html>
"""
