"""Evaluate the single preregistered MiniMol/LightGBM equal-rank fusion."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

import numpy as np
from scipy.stats import rankdata
from sklearn.metrics import average_precision_score


def percentile_ranks(values: np.ndarray) -> np.ndarray:
    return rankdata(values, method="average") / len(values)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root", type=Path, default=Path("runs/v24-minimol-validation")
    )
    parser.add_argument("--task", default="CYP2C9_Substrate_CarbonMangels")
    args = parser.parse_args()
    task_root = args.root / args.task
    split_payload = json.loads((task_root / "splits.json").read_text())
    minimol = np.load(task_root / "results.predictions.npz")
    scores = {"2d_lightgbm": [], "minimol": [], "equal_rank_fusion": []}
    for seed in split_payload["seeds"]:
        labels = np.asarray(split_payload["splits"][str(seed)]["valid_y"])
        baseline = np.load(
            Path("runs/unimol2-pilot-v22")
            / args.task
            / f"seed_{seed}"
            / "validation_predictions.npz"
        )["2d_lightgbm"]
        mini = minimol[f"seed_{seed}"]
        fused = (percentile_ranks(baseline) + percentile_ranks(mini)) / 2
        scores["2d_lightgbm"].append(average_precision_score(labels, baseline))
        scores["minimol"].append(average_precision_score(labels, mini))
        scores["equal_rank_fusion"].append(average_precision_score(labels, fused))
    summary = {
        name: {
            "scores": values,
            "mean": statistics.mean(values),
            "std": statistics.pstdev(values),
        }
        for name, values in scores.items()
    }
    result = {
        "task": args.task,
        "metric": "auprc",
        "candidate": "rank(2d_lightgbm@0.50+minimol@0.50)",
        "summary": summary,
        "gain_over_baseline": (
            summary["equal_rank_fusion"]["mean"] - summary["2d_lightgbm"]["mean"]
        ),
        "selection_scope": "one preregistered equal-rank fusion; no weight sweep",
        "official_test_loaded": False,
    }
    (task_root / "rank_fusion.json").write_text(
        json.dumps(result, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
