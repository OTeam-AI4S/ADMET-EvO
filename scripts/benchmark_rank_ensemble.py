"""Confirm a fixed heterogeneous rank ensemble on fresh validation seeds."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
from scipy.stats import rankdata
from tdc.benchmark_group import admet_group

from admet_agent.features import FeatureCache
from admet_agent.runner import _columns
from admet_agent.schemas import ExperimentSpec
from admet_agent.selection import build_model, predict, score
from admet_agent.tasks import resolve_task

TASKS = ("CYP2C9_Substrate_CarbonMangels", "Bioavailability_Ma")
REPRESENTATIONS = ("morgan", "avalon", "erg", "rdkit2d")
# Seeds 1-5 were used in discovery. These are untouched confirmation splits.
SEEDS = (6, 7, 8, 9, 10)
MODELS = {
    "extra_trees": ("extra_trees", {}),
    "lightgbm": ("lightgbm", {}),
    "svm": ("svm_rbf", {"C": 3.0, "gamma": "scale"}),
    "xgboost": ("xgboost", {}),
    "random_forest": ("random_forest", {}),
}
ENSEMBLES = {
    "rank_lgbm_svm": ("lightgbm", "svm"),
    "rank_et_lgbm_svm": ("extra_trees", "lightgbm", "svm"),
    "rank_all5": tuple(MODELS),
}


def _rank_average(predictions: dict[str, np.ndarray], members: tuple[str, ...]) -> np.ndarray:
    normalized = [
        rankdata(predictions[name], method="average") / len(predictions[name])
        for name in members
    ]
    return np.mean(normalized, axis=0)


def _summarize(values: dict[str, list[float]], elapsed: dict[str, list[float]]) -> dict:
    return {
        name: {
            "scores": scores,
            "mean": float(np.mean(scores)),
            "std": float(np.std(scores)),
            "elapsed_seconds": float(np.sum(elapsed[name])),
        }
        for name, scores in values.items()
    }


def run(
    data_dir: Path,
    output_dir: Path,
    workers: int,
    task_names: tuple[str, ...] = TASKS,
    seeds: tuple[int, ...] = SEEDS,
) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    cache = FeatureCache(output_dir / "feature_cache")
    group = admet_group(path=str(data_dir))
    output = {
        "protocol": {
            "split": "TDC default scaffold validation",
            "phase": "fresh-seed confirmation after seeds 1-5 discovery",
            "seeds": list(seeds),
            "representations": list(REPRESENTATIONS),
            "test_labels_used": False,
            "ensemble_weighting": "fixed equal-weight percentile-rank average",
        },
        "tasks": {},
    }
    for task_name in task_names:
        task = resolve_task(task_name)
        values = {name: [] for name in (*MODELS, *ENSEMBLES)}
        elapsed = {name: [] for name in values}
        for seed in seeds:
            train, valid = group.get_train_valid_split(
                benchmark=task_name,
                split_type="default",
                seed=seed,
            )
            smiles_col, y_col = _columns(train)
            combined = train[smiles_col].tolist() + valid[smiles_col].tolist()
            x = cache.get_or_compute(combined, REPRESENTATIONS)
            x_train, x_valid = x[: len(train)], x[len(train) :]
            y_train, y_valid = train[y_col].to_numpy(), valid[y_col].to_numpy()
            predictions = {}
            for name, (model_name, hyperparameters) in MODELS.items():
                spec = ExperimentSpec(
                    task=task_name,
                    seed=seed,
                    representations=REPRESENTATIONS,
                    model=model_name,
                    imbalance="class_weight",
                    n_jobs=workers,
                    hyperparameters=hyperparameters,
                    hypothesis="fixed heterogeneous confirmation benchmark",
                )
                started = time.monotonic()
                model = build_model(spec, task, y_train)
                model.fit(x_train, y_train)
                predictions[name] = predict(model, x_valid, task)
                elapsed[name].append(time.monotonic() - started)
                values[name].append(score(y_valid, predictions[name], task.metric))
            for name, members in ENSEMBLES.items():
                started = time.monotonic()
                ensemble_prediction = _rank_average(predictions, members)
                elapsed[name].append(time.monotonic() - started)
                values[name].append(score(y_valid, ensemble_prediction, task.metric))
            print(
                task_name,
                f"seed={seed}",
                " ".join(f"{name}={values[name][-1]:.4f}" for name in values),
                flush=True,
            )
        summaries = _summarize(values, elapsed)
        best = max(summaries, key=lambda name: summaries[name]["mean"])
        best_single = max(MODELS, key=lambda name: summaries[name]["mean"])
        output["tasks"][task_name] = {
            "metric": task.metric,
            "results": summaries,
            "best": best,
            "best_single_model": best_single,
            "best_delta_vs_single": summaries[best]["mean"] - summaries[best_single]["mean"],
        }
        (output_dir / "results.json").write_text(
            json.dumps(output, indent=2), encoding="utf-8"
        )
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--output-dir", type=Path, default=Path("runs/validation-rank-v22"))
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--task", action="append", choices=TASKS)
    parser.add_argument("--seeds", default=",".join(str(seed) for seed in SEEDS))
    args = parser.parse_args()
    task_names = tuple(args.task) if args.task else TASKS
    seeds = tuple(int(value) for value in args.seeds.split(","))
    print(
        json.dumps(
            run(args.data_dir, args.output_dir, args.workers, task_names, seeds), indent=2
        )
    )


if __name__ == "__main__":
    main()
