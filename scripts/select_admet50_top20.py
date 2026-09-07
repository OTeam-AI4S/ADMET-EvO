"""Select Top-20 endpoints using seed-17 development results only.

This script is intentionally written before the 33 new endpoint runs finish.
Seeds 29 and 43 are rejected so that they remain untouched confirmation data.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd


AGENT_ACTIONS = {
    "baseline", "feature_rich", "boosting_anchor", "automl_expanded",
    "nested_imbalance", "gpu_catboost", "random_forest", "linear_svm", "xgboost",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_result_files(directories: list[Path], *, seed: int) -> pd.DataFrame:
    tables = []
    for directory in directories:
        for path in sorted(directory.glob("*.csv")):
            if not path.stem.endswith(f"__seed{seed}"):
                continue
            table = pd.read_csv(path)
            if len(table):
                tables.append(table)
    return pd.concat(tables, ignore_index=True) if tables else pd.DataFrame()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cohort", type=Path, required=True)
    parser.add_argument("--confirmation-cohort", type=Path)
    parser.add_argument("--v5-development", type=Path)
    parser.add_argument("--development-dir", type=Path, action="append", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    cohort = pd.read_csv(args.cohort)
    if cohort["endpoint_id"].nunique() != 50:
        raise RuntimeError("Selection denominator must contain exactly 50 endpoints")
    v5 = pd.DataFrame()
    if args.v5_development is not None:
        v5 = pd.read_csv(args.v5_development)
        v5["confirmation_seed"] = 17
    additional = _read_result_files(args.development_dir, seed=17)
    results = pd.concat([v5, additional], ignore_index=True)
    results = results.loc[results["action"].isin(AGENT_ACTIONS)].copy()
    results["confirmation_seed"] = pd.to_numeric(results["confirmation_seed"], errors="raise").astype(int)
    if set(results["confirmation_seed"]) != {17}:
        raise RuntimeError("Top20 selection may use seed 17 only")
    results = results.drop_duplicates(
        ["endpoint_id", "confirmation_seed", "outer_fold", "action"], keep="last"
    )
    expected = set(cohort["endpoint_id"].astype(str))
    observed = set(results["endpoint_id"].astype(str))
    if expected != observed:
        raise RuntimeError(
            f"Incomplete 50-endpoint development denominator; missing={sorted(expected-observed)} "
            f"unexpected={sorted(observed-expected)}"
        )
    fold_sets = results.groupby(["endpoint_id", "action"])["outer_fold"].agg(
        lambda values: tuple(sorted(set(int(value) for value in values)))
    )
    if int(fold_sets.map(lambda value: value != (0, 1, 2)).sum()):
        raise RuntimeError("Every development action must contain selection folds 0,1,2 only")

    action = (
        results.groupby(["endpoint_id", "action", "task_type", "metric"], as_index=False)
        ["oriented_score"].mean()
        .rename(columns={"oriented_score": "development_oriented_score"})
    )
    baseline = (
        action.loc[action["action"].eq("baseline"), ["endpoint_id", "development_oriented_score"]]
        .rename(columns={"development_oriented_score": "development_baseline"})
    )
    ordered = action.sort_values(
        ["endpoint_id", "development_oriented_score", "action"],
        ascending=[True, False, True], kind="stable",
    )
    best = ordered.groupby("endpoint_id", as_index=False).first().rename(columns={
        "action": "selected_action", "development_oriented_score": "development_agent_score"
    })
    screen = best.merge(baseline, on="endpoint_id", validate="one_to_one")
    screen["development_oriented_gain"] = (
        screen["development_agent_score"] - screen["development_baseline"]
    )
    # Rank within task type so MCC and negative MAE remain comparable. The score
    # gives equal weight to absolute development quality and baseline uplift.
    screen["quality_percentile"] = screen.groupby("task_type")["development_agent_score"].rank(
        method="average", pct=True
    )
    screen["gain_percentile"] = screen.groupby("task_type")["development_oriented_gain"].rank(
        method="average", pct=True
    )
    screen["selection_index"] = 0.5 * (
        screen["quality_percentile"] + screen["gain_percentile"]
    )
    screen = screen.sort_values(
        ["selection_index", "development_oriented_gain", "development_agent_score", "endpoint_id"],
        ascending=[False, False, False, True], kind="stable",
    ).reset_index(drop=True)
    screen["development_rank"] = range(1, len(screen) + 1)
    screen["selected_top20"] = screen["development_rank"].le(20)
    sources = cohort[["endpoint_id", "source_id"]].drop_duplicates("endpoint_id")
    screen = screen.merge(sources, on="endpoint_id", how="left", validate="one_to_one")
    if len(screen) != 50 or int(screen["selected_top20"].sum()) != 20:
        raise RuntimeError("Selection must preserve 50 rows and select exactly 20")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    screen_path = args.output_dir / "admet50_seed17_development_screen.csv"
    screen.to_csv(screen_path, index=False)
    top_ids = set(screen.loc[screen["selected_top20"], "endpoint_id"])
    confirmation_cohort_path = args.confirmation_cohort or args.cohort
    confirmation_cohort = pd.read_csv(confirmation_cohort_path)
    if set(confirmation_cohort["endpoint_id"].astype(str)) != expected:
        raise RuntimeError("Confirmation cohort must contain the same 50-endpoint denominator")
    top_cohort = confirmation_cohort.loc[
        confirmation_cohort["endpoint_id"].isin(top_ids)
    ].copy()
    top_path = args.output_dir / "admet50_top20_confirmation_cohort.csv"
    top_cohort.to_csv(top_path, index=False)
    manifest = {
        "version": "admet50-top20-selection-v1",
        "cohort_path": str(args.cohort.resolve()),
        "cohort_sha256": _sha256(args.cohort),
        "development_seed": 17,
        "prohibited_confirmation_seeds": [29, 43],
        "denominator_endpoints": 50,
        "selected_endpoints": 20,
        "selection_rule": (
            "Using three-fold CV wholly inside frozen folds 0,1,2, select the highest "
            "mean oriented seed-17 action for each endpoint. "
            "Within task type, percentile-rank its absolute oriented score and its "
            "oriented gain over baseline; selection_index is their unweighted mean. "
            "Select the 20 highest indices with deterministic gain, score, endpoint_id ties."
        ),
        "screen_path": str(screen_path.resolve()),
        "screen_sha256": _sha256(screen_path),
        "top20_cohort_path": str(top_path.resolve()),
        "top20_cohort_sha256": _sha256(top_path),
        "confirmation_cohort_path": str(confirmation_cohort_path.resolve()),
        "confirmation_cohort_sha256": _sha256(confirmation_cohort_path),
        "top20_endpoint_ids": screen.loc[screen["selected_top20"], "endpoint_id"].tolist(),
        "model_selection_folds": [0, 1, 2],
        "untouched_confirmation_folds": [3, 4],
        "confirmation_results_loaded": False,
        "leaderboard_feedback_used": False,
    }
    manifest_path = args.output_dir / "admet50_top20_selection_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "denominator": 50, "selected": 20,
        "screen_sha256": manifest["screen_sha256"],
        "top20_cohort_sha256": manifest["top20_cohort_sha256"],
    }, sort_keys=True))


if __name__ == "__main__":
    main()
