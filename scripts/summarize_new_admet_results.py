"""Summarize development-selected, multi-seed-confirmed new ADMET experiments."""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_shards(result_dirs: list[Path]) -> tuple[pd.DataFrame, list[dict]]:
    frames = []
    manifests = []
    for directory in result_dirs:
        for raw in sorted(glob.glob(str(directory / "*.manifest.json"))):
            path = Path(raw)
            manifest = json.loads(path.read_text())
            manifests.append(manifest)
            if manifest.get("integrity_passed") is not True:
                continue
            fallback = path.with_name(path.name.removesuffix(".manifest.json") + ".csv")
            output = Path(manifest.get("output_path", fallback))
            if manifest.get("output_sha256") and _sha256(output) != manifest["output_sha256"]:
                raise RuntimeError(f"Result hash mismatch: {output}")
            frames.append(pd.read_csv(output))
    if not frames:
        raise RuntimeError("No integrity-passing result shards found")
    return pd.concat(frames, ignore_index=True), manifests


def run(args: argparse.Namespace) -> dict:
    cohort_path = args.cohort.resolve()
    cohort = pd.read_csv(cohort_path)
    results, manifests = _load_shards(args.result_dir)
    if args.legacy_seed17 is not None:
        legacy = pd.read_csv(args.legacy_seed17)
        legacy["confirmation_seed"] = 17
        results = pd.concat([results, legacy], ignore_index=True)
    if "fits" not in results:
        results["fits"] = 1
    else:
        results["fits"] = results["fits"].fillna(1)
    key = ["endpoint_id", "confirmation_seed", "outer_fold", "action"]
    results = (
        results.sort_values(key + ["wall_seconds"], kind="stable")
        .drop_duplicates(key, keep="last")
        .reset_index(drop=True)
    )
    expected_endpoints = set(cohort["endpoint_id"].astype(str).unique())
    results = results.loc[results["endpoint_id"].astype(str).isin(expected_endpoints)].copy()
    endpoint_seed_action = (
        results.groupby(
            ["endpoint_id", "source_id", "task_type", "metric", "confirmation_seed", "action"],
            as_index=False,
        )
        .agg(
            oriented_score=("oriented_score", "mean"),
            fold_sd=("oriented_score", "std"),
            folds=("outer_fold", "nunique"),
            fits=("fits", "sum"),
            wall_seconds=("wall_seconds", "sum"),
        )
    )
    development = endpoint_seed_action.loc[
        endpoint_seed_action["confirmation_seed"].eq(args.development_seed)
    ].copy()
    if set(development["endpoint_id"]) != expected_endpoints:
        missing = sorted(expected_endpoints - set(development["endpoint_id"]))
        raise RuntimeError(f"Development seed is incomplete; missing={missing}")
    choices = (
        development.sort_values(
            ["endpoint_id", "oriented_score", "action"],
            ascending=[True, False, True],
            kind="stable",
        )
        .groupby("endpoint_id", sort=True)
        .head(1)[["endpoint_id", "action", "oriented_score"]]
        .rename(columns={
            "action": "selected_action",
            "oriented_score": "development_score",
        })
    )
    confirmation_seeds = tuple(
        int(value) for value in args.confirmation_seeds.split(",") if value
    )
    confirmation = endpoint_seed_action.loc[
        endpoint_seed_action["confirmation_seed"].isin(confirmation_seeds)
    ].copy()
    selected = confirmation.merge(
        choices,
        left_on=["endpoint_id", "action"],
        right_on=["endpoint_id", "selected_action"],
        how="inner",
    )
    selected_summary = (
        selected.groupby(
            ["endpoint_id", "source_id", "task_type", "metric", "selected_action", "development_score"],
            as_index=False,
        )
        .agg(
            confirmation_score=("oriented_score", "mean"),
            confirmation_seed_sd=("oriented_score", "std"),
            confirmation_seeds=("confirmation_seed", "nunique"),
        )
    )
    baseline = (
        confirmation.loc[confirmation["action"].eq("baseline")]
        .groupby("endpoint_id", as_index=False)
        .agg(
            confirmation_baseline=("oriented_score", "mean"),
            baseline_seed_sd=("oriented_score", "std"),
        )
    )
    endpoint_summary = selected_summary.merge(baseline, on="endpoint_id", how="left")
    endpoint_summary["oriented_gain"] = (
        endpoint_summary["confirmation_score"]
        - endpoint_summary["confirmation_baseline"]
    )
    scale = np.where(
        endpoint_summary["task_type"].eq("regression"),
        np.maximum(np.abs(endpoint_summary["confirmation_baseline"]), 1e-12),
        np.maximum(1.0 - endpoint_summary["confirmation_baseline"], 1e-12),
    )
    endpoint_summary["relative_gain"] = endpoint_summary["oriented_gain"] / scale
    endpoint_summary["confirmed_improvement"] = endpoint_summary["oriented_gain"] > 0
    endpoint_summary = endpoint_summary.sort_values(
        ["relative_gain", "endpoint_id"], ascending=[False, True], kind="stable"
    ).reset_index(drop=True)
    endpoint_summary["descriptive_rank"] = np.arange(1, len(endpoint_summary) + 1)

    action_summary = (
        endpoint_seed_action.groupby(["endpoint_id", "action"], as_index=False)
        .agg(
            mean_oriented_score=("oriented_score", "mean"),
            seed_sd=("oriented_score", "std"),
            seeds=("confirmation_seed", "nunique"),
            total_fits=("fits", "sum"),
            total_wall_seconds=("wall_seconds", "sum"),
        )
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    fold_path = args.output_dir / "fold_results.csv"
    action_path = args.output_dir / "action_summary.csv"
    endpoint_path = args.output_dir / "endpoint_confirmation.csv"
    results.to_csv(fold_path, index=False)
    action_summary.to_csv(action_path, index=False)
    endpoint_summary.to_csv(endpoint_path, index=False)
    payload = {
        "version": "new-admet-multiseed-summary-v1",
        "cohort_path": str(cohort_path),
        "cohort_sha256": _sha256(cohort_path),
        "development_seed": args.development_seed,
        "confirmation_seeds": list(confirmation_seeds),
        "selection_rule": "maximize mean oriented outer-fold score on development seed only",
        "endpoint_selection_for_primary_claim": args.endpoint_selection_role,
        "top_k_role": (
            "development-selected before untouched holdout evaluation"
            if args.endpoint_selection_role != "none; full denominator retained"
            else "descriptive display only; ranked after confirmation and not inferential"
        ),
        "endpoints": len(expected_endpoints),
        "confirmed_improvements": int(endpoint_summary["confirmed_improvement"].sum()),
        "complete_shard_manifests": int(sum(item.get("integrity_passed") is True for item in manifests)),
        "failed_shard_manifests": int(sum(item.get("integrity_passed") is not True for item in manifests)),
        "fold_results": {"path": str(fold_path.resolve()), "sha256": _sha256(fold_path)},
        "action_summary": {"path": str(action_path.resolve()), "sha256": _sha256(action_path)},
        "endpoint_confirmation": {"path": str(endpoint_path.resolve()), "sha256": _sha256(endpoint_path)},
        "official_test_labels_loaded": False,
        "leaderboard_feedback_used": False,
        "confirmation_evaluation_protocols": sorted({
            item.get("evaluation_protocol", "unspecified")
            for item in manifests
            if item.get("confirmation_seed") in confirmation_seeds
        }),
        "confirmation_labels_used_for_selection": False,
    }
    manifest_path = args.output_dir / "summary_manifest.json"
    manifest_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, sort_keys=True))
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cohort", type=Path, required=True)
    parser.add_argument("--result-dir", type=Path, action="append", required=True)
    parser.add_argument("--legacy-seed17", type=Path)
    parser.add_argument("--development-seed", type=int, default=17)
    parser.add_argument("--confirmation-seeds", default="29,43")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--endpoint-selection-role",
        default="none; full denominator retained",
    )
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
