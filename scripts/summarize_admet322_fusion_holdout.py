"""Build the complete fresh-endpoint holdout denominator and success tiers."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


CLASSIC = (
    "extra_trees_morgan", "extra_trees_rich", "lightgbm_rich",
    "random_forest_rich", "xgboost_rich", "catboost_rich",
)


def _family(endpoint_id: str) -> str:
    source, name = endpoint_id.split("::", 1)
    return name.split("_", 1)[0] if source == "toxcast" else source


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cohort", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()

    cohort = pd.read_csv(args.cohort)
    endpoints = sorted(cohort["endpoint_id"].astype(str).unique())
    metadata = cohort.groupby("endpoint_id").agg(
        rows=("target", "size"),
        positives=("target", "sum"),
        structures=("inchikey", "nunique"),
    )
    metadata["positive_rate"] = metadata["positives"] / metadata["rows"]
    by_endpoint: dict[str, Path] = {}
    for path in args.run_dir.glob("*.manifest.json"):
        payload = json.loads(path.read_text())
        endpoint_id = str(payload["endpoint_id"])
        if endpoint_id in by_endpoint:
            raise RuntimeError(f"Duplicate manifest for {endpoint_id}")
        by_endpoint[endpoint_id] = path
    missing = sorted(set(endpoints) - set(by_endpoint))
    extra = sorted(set(by_endpoint) - set(endpoints))
    if missing or extra:
        raise RuntimeError(f"Artifact mismatch: missing={len(missing)}, extra={len(extra)}")

    rows: list[dict] = []
    for endpoint_id in endpoints:
        manifest = json.loads(by_endpoint[endpoint_id].read_text())
        if manifest.get("integrity_passed") is not True:
            raise RuntimeError(f"Integrity failure: {endpoint_id}")
        if manifest.get("confirmation_labels_modified_recipe") is not False:
            raise RuntimeError(f"Recipe firewall failure: {endpoint_id}")
        table = pd.read_csv(manifest["result_path"])
        mean_scores = table.groupby("model")["score"].mean()
        if set(CLASSIC) | {"agent_frozen"} != set(mean_scores.index):
            raise RuntimeError(f"Model table mismatch: {endpoint_id}")
        agent = float(mean_scores["agent_frozen"])
        baseline = float(mean_scores["extra_trees_morgan"])
        gain = agent - baseline
        wins = int(manifest["summary"]["agent_wins_by_seed"])
        best_classic = mean_scores[list(CLASSIC)].idxmax()
        best_classic_score = float(mean_scores[best_classic])
        if agent >= 0.25 and gain >= 0.05 and wins == 2:
            tier = "strong"
        elif agent >= 0.20 and gain >= 0.03 and wins == 2:
            tier = "good"
        elif gain > 0:
            tier = "positive"
        else:
            tier = "no_gain"
        source = endpoint_id.split("::", 1)[0]
        row = {
            "endpoint_id": endpoint_id,
            "source": source,
            "assay_family": _family(endpoint_id),
            "rows": int(metadata.loc[endpoint_id, "rows"]),
            "structures": int(metadata.loc[endpoint_id, "structures"]),
            "positive_rate": float(metadata.loc[endpoint_id, "positive_rate"]),
            "selected_kind": manifest["frozen_recipe"]["kind"],
            "selected_action": manifest["frozen_recipe"]["action"],
            "agent_mcc": agent,
            "morgan_et_mcc": baseline,
            "agent_gain_vs_morgan": gain,
            "agent_wins_by_seed": wins,
            "best_classic_model_posthoc": best_classic,
            "best_classic_mcc_posthoc": best_classic_score,
            "agent_minus_best_classic_posthoc": agent - best_classic_score,
            "holdout_tier": tier,
        }
        for model in CLASSIC:
            row[f"{model}_mcc"] = float(mean_scores[model])
        rows.append(row)

    result = pd.DataFrame(rows).sort_values(
        ["holdout_tier", "agent_mcc", "agent_gain_vs_morgan"],
        ascending=[True, False, False], kind="stable",
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output, index=False)
    counts = result["holdout_tier"].value_counts().to_dict()
    summary = {
        "version": "admet322-frozen-fusion-holdout-summary-v1",
        "endpoint_count": len(result),
        "tier_definition": {
            "strong": "mean holdout MCC >= 0.25, gain vs Morgan ExtraTrees >= 0.05, wins on both seeds",
            "good": "mean holdout MCC >= 0.20, gain vs Morgan ExtraTrees >= 0.03, wins on both seeds",
            "positive": "positive mean gain vs Morgan ExtraTrees",
        },
        "tier_counts": counts,
        "positive_gain_count": int(result["agent_gain_vs_morgan"].gt(0).sum()),
        "fusion_selected_count": int(result["selected_kind"].eq("fusion").sum()),
        "mean_agent_mcc": float(result["agent_mcc"].mean()),
        "mean_morgan_et_mcc": float(result["morgan_et_mcc"].mean()),
        "mean_agent_gain_vs_morgan": float(result["agent_gain_vs_morgan"].mean()),
        "output": str(args.output.resolve()),
    }
    args.manifest.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
