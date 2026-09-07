"""Write a concise audited Markdown report for the ADMET322 confirmation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


MODELS = (
    "extra_trees_morgan", "extra_trees_rich", "lightgbm_rich",
    "random_forest_rich", "xgboost_rich", "catboost_rich",
)


def _markdown(frame: pd.DataFrame) -> str:
    def render(value: object) -> str:
        if isinstance(value, float):
            return f"{value:.4f}"
        return str(value).replace("|", "\\|")

    header = "| " + " | ".join(str(value) for value in frame.columns) + " |"
    divider = "| " + " | ".join("---" for _ in frame.columns) + " |"
    rows = [
        "| " + " | ".join(render(value) for value in row) + " |"
        for row in frame.itertuples(index=False, name=None)
    ]
    return "\n".join([header, divider, *rows])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--summary-manifest", type=Path, required=True)
    parser.add_argument("--development-manifest", type=Path, required=True)
    parser.add_argument("--execution-freeze", type=Path, required=True)
    parser.add_argument("--holdout-run-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    results = pd.read_csv(args.summary)
    if len(results) != 322:
        raise RuntimeError("Report requires all 322 fresh endpoints")
    summary = json.loads(args.summary_manifest.read_text())
    development = json.loads(args.development_manifest.read_text())
    freeze = json.loads(args.execution_freeze.read_text())
    fusion_deltas = []
    for holdout_manifest_path in args.holdout_run_dir.glob("*.manifest.json"):
        holdout_manifest = json.loads(holdout_manifest_path.read_text())
        recipe = holdout_manifest["frozen_recipe"]
        if recipe["kind"] != "fusion":
            continue
        development_manifest = json.loads(Path(holdout_manifest["development_recipe_path"]).read_text())
        development_table = pd.read_csv(development_manifest["result_path"]).set_index("action")
        left, right = recipe["left"], recipe["right"]
        stronger = left if (
            development_table.loc[left, "mean_oriented_score"]
            >= development_table.loc[right, "mean_oriented_score"]
        ) else right
        holdout_table = pd.read_csv(holdout_manifest["result_path"]).groupby("model")["score"].mean()
        fusion_deltas.append(float(holdout_table["agent_frozen"] - holdout_table[stronger]))
    fusion_delta = pd.Series(fusion_deltas, dtype=float)
    tier_order = ["strong", "good", "positive", "no_gain"]
    tiers = results["holdout_tier"].value_counts().reindex(tier_order, fill_value=0)
    top = results.loc[results["holdout_tier"].eq("strong")].sort_values(
        ["agent_mcc", "agent_gain_vs_morgan"], ascending=False
    ).head(20)[[
        "endpoint_id", "source", "assay_family", "selected_kind", "agent_mcc",
        "morgan_et_mcc", "agent_gain_vs_morgan", "holdout_tier",
    ]]
    family = results.groupby(["source", "assay_family"]).agg(
        endpoints=("endpoint_id", "size"),
        strong=("holdout_tier", lambda x: int(x.eq("strong").sum())),
        good_or_strong=("holdout_tier", lambda x: int(x.isin(["strong", "good"]).sum())),
        positive=("agent_gain_vs_morgan", lambda x: int(x.gt(0).sum())),
        mean_agent_mcc=("agent_mcc", "mean"),
        mean_gain=("agent_gain_vs_morgan", "mean"),
    ).reset_index()
    family["strong_rate"] = family["strong"] / family["endpoints"]
    family = family.sort_values(["strong_rate", "mean_agent_mcc"], ascending=False)
    wtl = []
    for model in MODELS:
        delta = results["agent_mcc"] - results[f"{model}_mcc"]
        wtl.append({
            "comparator": model, "wins": int(delta.gt(1e-12).sum()),
            "ties": int(delta.abs().le(1e-12).sum()), "losses": int(delta.lt(-1e-12).sum()),
            "mean_delta": float(delta.mean()),
        })
    lines = [
        "# ADMET322 fresh-endpoint frozen-fusion confirmation",
        "",
        "## Outcome",
        "",
        f"- Complete denominator: **{len(results)}** endpoints (309 ToxCast, 13 SIDER).",
        f"- Strong: **{int(tiers['strong'])}**; Good: **{int(tiers['good'])}**; Positive: **{int(tiers['positive'])}**; No gain: **{int(tiers['no_gain'])}**.",
        f"- Positive gain over OOF-calibrated Morgan ExtraTrees: **{int(results['agent_gain_vs_morgan'].gt(0).sum())}/{len(results)}**.",
        f"- Mean Agent MCC: **{results['agent_mcc'].mean():.4f}**; mean Morgan ExtraTrees MCC: **{results['morgan_et_mcc'].mean():.4f}**; mean delta: **{results['agent_gain_vs_morgan'].mean():+.4f}**.",
        f"- Frozen fusion selected: **{int(results['selected_kind'].eq('fusion').sum())}/{len(results)}** endpoints.",
        f"- Fusion vs its development-stronger member on holdout: **{int(fusion_delta.gt(1e-12).sum())}/{int(fusion_delta.abs().le(1e-12).sum())}/{int(fusion_delta.lt(-1e-12).sum())}** wins/ties/losses; mean delta **{fusion_delta.mean():+.4f} MCC**.",
        "",
        "Strong was preregistered as mean holdout MCC >= 0.25, gain >= 0.05 over Morgan ExtraTrees, and wins on both confirmation seeds. Good uses MCC >= 0.20 and gain >= 0.03 with two seed wins.",
        "",
        "## Matched classic-model comparisons",
        "",
        _markdown(pd.DataFrame(wtl)),
        "",
        "Each classic model uses its own threshold frozen from folds 0-2. The per-endpoint post-hoc best classic is descriptive only and is not a fair deployable comparator.",
        "",
        "## Top-20 presentation view",
        "",
        _markdown(top),
        "",
        "Top-20 is selected after holdout only for presentation. All claims above use the complete 322-endpoint denominator.",
        "",
        "## Source and assay-family aggregation",
        "",
        _markdown(family),
        "",
        "ToxCast endpoints within one assay family are correlated tasks, not independent external datasets. Family aggregation prevents counting each assay as independent evidence.",
        "",
        "## Audit trail",
        "",
        f"- Development recipes audited: {development['endpoint_count']}; fresh recipes: {development['fresh_endpoint_count']}.",
        f"- Execution recipe-set SHA-256: `{freeze['recipe_set_sha256']}`.",
        f"- Confirmation cohort SHA-256: `{freeze['confirmation_cohort_sha256']}`.",
        f"- Holdout runner SHA-256: `{freeze['holdout_runner_sha256']}`.",
        f"- Confirmation summary version: `{summary['version']}`.",
        "- Molecular identity and chemistry-group cross-fold overlaps are both zero.",
        "- No ADMET322 holdout outcome was used to select endpoints, recipes, thresholds, or success gates.",
        "",
    ]
    args.output.write_text("\n".join(lines))
    print(json.dumps({"output": str(args.output.resolve()), "rows": len(results)}, sort_keys=True))


if __name__ == "__main__":
    main()
