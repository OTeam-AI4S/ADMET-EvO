"""Summarize multi-seed fixed-holdout shards without endpoint selection."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run(args: argparse.Namespace) -> dict:
    manifests = []
    for path in sorted(args.result_dir.glob("*.manifest.json")):
        record = json.loads(path.read_text())
        record["manifest_path"] = str(path.resolve())
        manifests.append(record)
    if not manifests:
        raise RuntimeError(f"No manifests found in {args.result_dir}")
    failed = [record for record in manifests if not record.get("integrity_passed", False)]
    if failed:
        raise RuntimeError(f"Found {len(failed)} failed manifests")

    expected = {
        Path(record["output_path"]).resolve()
        for record in manifests
    }
    frames = [pd.read_csv(path) for path in sorted(expected)]
    results = pd.concat(frames, ignore_index=True)
    if results.duplicated(["endpoint_id", "seed", "action"]).any():
        raise RuntimeError("Duplicate endpoint/seed/action result rows")
    endpoint_seeds = results.groupby("endpoint_id")["seed"].nunique()
    if endpoint_seeds.nunique() != 1:
        raise RuntimeError(f"Inconsistent seed coverage: {endpoint_seeds.to_dict()}")

    action_summary = (
        results.groupby(["endpoint_id", "source_id", "action", "metric"], as_index=False)
        .agg(
            mean_score=("raw_score", "mean"),
            seed_sd=("raw_score", "std"),
            seeds=("seed", "nunique"),
            train_rows=("train_rows", "first"),
            test_rows=("test_rows", "first"),
        )
    )
    baseline = action_summary.loc[action_summary["action"].eq("baseline"), [
        "endpoint_id", "mean_score"
    ]].rename(columns={"mean_score": "baseline_score"})
    action_summary = action_summary.merge(baseline, on="endpoint_id", how="left")
    action_summary["mae_gain_vs_baseline"] = (
        action_summary["baseline_score"] - action_summary["mean_score"]
    )

    primary = action_summary.loc[action_summary["action"].eq(args.primary_action)].copy()
    if len(primary) != results["endpoint_id"].nunique():
        raise RuntimeError("Primary action is missing for at least one endpoint")
    primary = primary.sort_values("mae_gain_vs_baseline", ascending=False, kind="stable")
    posthoc = (
        action_summary.sort_values(
            ["endpoint_id", "mean_score", "action"], kind="stable"
        )
        .groupby("endpoint_id", as_index=False)
        .first()
        .rename(columns={"action": "posthoc_best_action", "mean_score": "posthoc_best_score"})
    )
    primary = primary.merge(
        posthoc[["endpoint_id", "posthoc_best_action", "posthoc_best_score"]],
        on="endpoint_id",
        how="left",
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    action_path = args.output_dir / "action_summary.csv"
    primary_path = args.output_dir / "primary_endpoint_summary.csv"
    results_path = args.output_dir / "shard_results.csv"
    action_summary.to_csv(action_path, index=False)
    primary.to_csv(primary_path, index=False)
    results.to_csv(results_path, index=False)
    manifest_path = args.output_dir / "summary_manifest.json"
    manifest = {
        "version": args.version,
        "result_dir": str(args.result_dir.resolve()),
        "complete_shard_manifests": len(manifests),
        "failed_shard_manifests": 0,
        "endpoints": int(results["endpoint_id"].nunique()),
        "seeds": sorted(int(value) for value in results["seed"].unique()),
        "actions": sorted(results["action"].astype(str).unique()),
        "primary_action": args.primary_action,
        "primary_macro_mae": float(primary["mean_score"].mean()),
        "baseline_macro_mae": float(primary["baseline_score"].mean()),
        "primary_endpoints_improved_vs_baseline": int(
            primary["mae_gain_vs_baseline"].gt(0).sum()
        ),
        "endpoint_denominator": int(len(primary)),
        "endpoint_selection_for_primary_claim": "none; full denominator retained",
        "posthoc_best_role": "descriptive only; not used for primary claim",
        "test_labels_used_for_training_or_model_selection": False,
        "leaderboard_feedback_used": False,
        "files": {
            "action_summary": {"path": str(action_path.resolve()), "sha256": _sha256(action_path)},
            "primary_endpoint_summary": {"path": str(primary_path.resolve()), "sha256": _sha256(primary_path)},
            "shard_results": {"path": str(results_path.resolve()), "sha256": _sha256(results_path)},
        },
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps(manifest, sort_keys=True))
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--primary-action", default="boosting_anchor")
    parser.add_argument("--version", default="fixed-holdout-summary-v1")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
