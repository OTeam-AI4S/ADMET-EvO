"""Audit and summarize all endpoint-level OOF development artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def _source(endpoint_id: str) -> str:
    return endpoint_id.split("::", 1)[0]


def _family(endpoint_id: str) -> str:
    source, name = endpoint_id.split("::", 1)
    return name.split("_", 1)[0] if source == "toxcast" else source


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cohort", type=Path, required=True)
    parser.add_argument("--previous-cohort", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()

    endpoints = sorted(
        pd.read_csv(args.cohort, usecols=["endpoint_id"])["endpoint_id"].astype(str).unique()
    )
    previous = set(
        pd.read_csv(args.previous_cohort, usecols=["endpoint_id"])["endpoint_id"].astype(str)
    )
    manifests = list(args.run_dir.glob("*.manifest.json"))
    by_endpoint: dict[str, Path] = {}
    for path in manifests:
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
    fusion_count = 0
    for endpoint_id in endpoints:
        manifest = json.loads(by_endpoint[endpoint_id].read_text())
        if manifest.get("integrity_passed") is not True:
            raise RuntimeError(f"Integrity failure: {endpoint_id}")
        if manifest.get("holdout_labels_loaded") is not False:
            raise RuntimeError(f"Holdout firewall failure: {endpoint_id}")
        table = pd.read_csv(manifest["result_path"])
        if len(table) != 51 or int(table["selected"].sum()) != 1:
            raise RuntimeError(f"Unexpected candidate table: {endpoint_id}")
        selected = table.loc[table["selected"]].iloc[0]
        baseline = table.loc[table["action"].eq("extra_trees_morgan")].iloc[0]
        wins = sum(float(selected[f"fold{fold}"]) > float(baseline[f"fold{fold}"]) for fold in range(3))
        if str(selected["kind"]) == "fusion":
            fusion_count += 1
        score = float(selected["mean_oriented_score"])
        gain = score - float(baseline["mean_oriented_score"])
        if score >= 0.25 and gain >= 0.05 and wins >= 2:
            tier = "development_strong"
        elif score >= 0.20 and gain >= 0.03 and wins >= 2:
            tier = "development_good"
        elif gain > 0:
            tier = "development_positive"
        else:
            tier = "development_no_gain"
        rows.append({
            "endpoint_id": endpoint_id,
            "source": _source(endpoint_id),
            "assay_family": _family(endpoint_id),
            "previously_exposed_in_admet50": endpoint_id in previous,
            "task_type": manifest.get("task_type", selected["task_type"]),
            "selected_kind": str(selected["kind"]),
            "selected_action": str(selected["action"]),
            "development_score": score,
            "development_baseline": float(baseline["mean_oriented_score"]),
            "development_gain": gain,
            "fold_wins_vs_baseline": wins,
            "development_tier": tier,
        })
    result = pd.DataFrame(rows).sort_values(
        ["previously_exposed_in_admet50", "development_score"], ascending=[True, False]
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output, index=False)
    fresh = result.loc[~result["previously_exposed_in_admet50"]]
    summary = {
        "version": "admet372-fusion-development-summary-v1",
        "endpoint_count": len(result),
        "fresh_endpoint_count": len(fresh),
        "previously_exposed_endpoint_count": int(result["previously_exposed_in_admet50"].sum()),
        "fusion_selected_count": fusion_count,
        "fresh_fusion_selected_count": int(fresh["selected_kind"].eq("fusion").sum()),
        "fresh_tier_counts": fresh["development_tier"].value_counts().to_dict(),
        "fresh_positive_gain_count": int(fresh["development_gain"].gt(0).sum()),
        "holdout_outcomes_used": False,
        "output": str(args.output.resolve()),
    }
    args.manifest.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
