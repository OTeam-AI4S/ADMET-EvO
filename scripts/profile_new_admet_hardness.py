"""Apply the canonical six-axis Hardness profiler to a frozen new-task cohort."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from admet_agent.hardness import profile_hardness
from admet_agent.tasks import TaskSpec


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run(args: argparse.Namespace) -> dict:
    cohort_path = args.cohort.resolve()
    cohort = pd.read_csv(cohort_path)
    required = {
        "endpoint_id", "task_type", "metric", "standard_smiles", "target",
        "outer_fold",
    }
    if missing := required - set(cohort):
        raise ValueError(f"Missing cohort columns: {sorted(missing)}")
    endpoint_cards = {}
    summary_rows = []
    for endpoint_id, endpoint in cohort.groupby("endpoint_id", sort=True):
        task_type = str(endpoint["task_type"].iloc[0])
        metric = str(endpoint["metric"].iloc[0])
        spec = TaskSpec(str(endpoint_id), task_type, metric, "new_admet")
        cards = []
        for outer_fold in sorted(endpoint["outer_fold"].astype(int).unique()):
            train = (
                endpoint.loc[endpoint["outer_fold"].astype(int).ne(outer_fold)]
                .rename(columns={"standard_smiles": "Drug", "target": "Y"})
            )
            valid = (
                endpoint.loc[endpoint["outer_fold"].astype(int).eq(outer_fold)]
                .rename(columns={"standard_smiles": "Drug", "target": "Y"})
            )
            card = profile_hardness(
                train,
                valid,
                spec,
                sample_limit=args.sample_limit,
                seed=args.seed + int(outer_fold),
            )
            cards.append({"outer_fold": int(outer_fold), **card.to_dict()})
        dimensions = {
            name: float(np.mean([
                card["dimensions"][name]["severity"] for card in cards
            ]))
            for name in cards[0]["dimensions"]
        }
        overall = float(np.mean([card["overall"] for card in cards]))
        dominant = [
            name for name, _ in sorted(
                dimensions.items(), key=lambda item: (-item[1], item[0])
            )[:3]
        ]
        endpoint_cards[str(endpoint_id)] = {
            "task_type": task_type,
            "metric": metric,
            "rows": len(endpoint),
            "overall_mean": overall,
            "dominant_factors": dominant,
            "dimension_mean_severity": dimensions,
            "fold_cards": cards,
        }
        summary_rows.append({
            "endpoint_id": endpoint_id,
            "task_type": task_type,
            "metric": metric,
            "rows": len(endpoint),
            "hardness": overall,
            "dominant_factors": ";".join(dominant),
            **dimensions,
        })
    summary = pd.DataFrame(summary_rows).sort_values(
        ["hardness", "endpoint_id"], ascending=[False, True], kind="stable"
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(args.output, index=False)
    payload = {
        "version": "new-admet-hardness-v1",
        "profiler_version": "hardness-v1",
        "cohort_path": str(cohort_path),
        "cohort_sha256": _sha256(cohort_path),
        "sample_limit": args.sample_limit,
        "seed": args.seed,
        "endpoints": len(endpoint_cards),
        "endpoint_cards": endpoint_cards,
        "summary_path": str(args.output.resolve()),
        "summary_sha256": _sha256(args.output),
        "discovery_only": True,
        "official_test_labels_loaded": False,
        "leaderboard_feedback_used": False,
    }
    args.manifest.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "endpoints": len(endpoint_cards),
        "mean_hardness": float(summary["hardness"].mean()),
        "summary_sha256": payload["summary_sha256"],
    }, sort_keys=True))
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cohort", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--sample-limit", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20260827)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
