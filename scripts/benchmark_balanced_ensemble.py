"""Validation-only benchmark for imbalance-aware tree ensembles.

This script intentionally calls only ``get_train_valid_split``. It never loads
the official TDC test split or uses test labels for model selection.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
from tdc.benchmark_group import admet_group

from admet_agent.features import FeatureCache
from admet_agent.runner import _columns
from admet_agent.schemas import ExperimentSpec
from admet_agent.selection import build_model, predict, score
from admet_agent.tasks import resolve_task

TASKS = (
    "CYP2C9_Substrate_CarbonMangels",
    "Bioavailability_Ma",
)
REPRESENTATIONS = ("morgan", "avalon", "erg", "rdkit2d")
SEEDS = (1, 2, 3, 4, 5)

# These configurations are declared before looking at this benchmark's results.
CANDIDATES = {
    "extra_trees_current": {
        "model": "extra_trees",
        "hyperparameters": {},
    },
    "lightgbm_current": {
        "model": "lightgbm",
        "hyperparameters": {},
    },
    "svm_tuned_c3": {
        "model": "svm_rbf",
        "hyperparameters": {"C": 3.0, "gamma": "scale"},
    },
    "balanced_et_r1": {
        "model": "balanced_extra_trees",
        "hyperparameters": {
            "n_estimators": 300,
            "n_members": 9,
            "majority_ratio": 1.0,
            "min_samples_leaf": 1,
            "max_features": 0.5,
            "criterion": "gini",
        },
    },
    "balanced_et_r2": {
        "model": "balanced_extra_trees",
        "hyperparameters": {
            "n_estimators": 300,
            "n_members": 7,
            "majority_ratio": 2.0,
            "min_samples_leaf": 2,
            "max_features": "sqrt",
            "criterion": "entropy",
        },
    },
    "balanced_et_r3": {
        "model": "balanced_extra_trees",
        "hyperparameters": {
            "n_estimators": 400,
            "n_members": 5,
            "majority_ratio": 3.0,
            "min_samples_leaf": 3,
            "max_features": 0.8,
            "criterion": "log_loss",
        },
    },
}


def _summary(values: list[float], elapsed: list[float]) -> dict:
    return {
        "scores": values,
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
        "elapsed_seconds": float(np.sum(elapsed)),
    }


def run(
    data_dir: Path, output_dir: Path, workers: int, task_names: tuple[str, ...] = TASKS
) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    cache = FeatureCache(output_dir / "feature_cache", device="cpu")
    group = admet_group(path=str(data_dir))
    results = {
        "protocol": {
            "split": "TDC default scaffold validation",
            "seeds": list(SEEDS),
            "representations": list(REPRESENTATIONS),
            "test_labels_used": False,
            "selection_note": "Candidate recipes and blend weight were fixed before this run.",
        },
        "tasks": {},
    }

    for task_name in task_names:
        task = resolve_task(task_name)
        raw_scores = {name: [] for name in CANDIDATES}
        raw_elapsed = {name: [] for name in CANDIDATES}
        blend_scores = {f"lightgbm_50_{name}": [] for name in CANDIDATES if name.startswith("balanced_")}
        blend_elapsed = {name: [] for name in blend_scores}

        for seed in SEEDS:
            train, valid = group.get_train_valid_split(
                benchmark=task_name,
                split_type="default",
                seed=seed,
            )
            smiles_col, y_col = _columns(train)
            combined_smiles = train[smiles_col].tolist() + valid[smiles_col].tolist()
            x_values = cache.get_or_compute(combined_smiles, REPRESENTATIONS)
            x_train, x_valid = x_values[: len(train)], x_values[len(train) :]
            y_train = train[y_col].to_numpy()
            y_valid = valid[y_col].to_numpy()
            predictions = {}

            for candidate_name, recipe in CANDIDATES.items():
                experiment = ExperimentSpec(
                    task=task_name,
                    seed=seed,
                    representations=REPRESENTATIONS,
                    model=recipe["model"],
                    imbalance="class_weight",
                    device="cpu",
                    n_jobs=workers,
                    hyperparameters=recipe["hyperparameters"],
                    hypothesis="fixed validation-only imbalance benchmark",
                )
                started = time.monotonic()
                model = build_model(experiment, task, y_train)
                model.fit(x_train, y_train)
                candidate_predictions = predict(model, x_valid, task)
                elapsed = time.monotonic() - started
                candidate_score = score(y_valid, candidate_predictions, task.metric)
                predictions[candidate_name] = candidate_predictions
                raw_scores[candidate_name].append(candidate_score)
                raw_elapsed[candidate_name].append(elapsed)
                print(
                    f"{task_name} seed={seed} {candidate_name}: "
                    f"{candidate_score:.6f} ({elapsed:.1f}s)",
                    flush=True,
                )

            for balanced_name in (
                name for name in CANDIDATES if name.startswith("balanced_")
            ):
                blend_name = f"lightgbm_50_{balanced_name}"
                started = time.monotonic()
                blended = 0.5 * (
                    predictions["lightgbm_current"] + predictions[balanced_name]
                )
                blend_scores[blend_name].append(score(y_valid, blended, task.metric))
                blend_elapsed[blend_name].append(time.monotonic() - started)

        candidate_summary = {
            name: _summary(raw_scores[name], raw_elapsed[name])
            for name in CANDIDATES
        }
        blend_summary = {
            name: _summary(blend_scores[name], blend_elapsed[name])
            for name in blend_scores
        }
        all_summaries = {**candidate_summary, **blend_summary}
        best_name = max(all_summaries, key=lambda name: all_summaries[name]["mean"])
        conventional_names = (
            "extra_trees_current",
            "lightgbm_current",
            "svm_tuned_c3",
        )
        best_conventional = max(
            conventional_names,
            key=lambda name: candidate_summary[name]["mean"],
        )
        results["tasks"][task_name] = {
            "metric": task.metric,
            "candidates": candidate_summary,
            "fixed_blends": blend_summary,
            "best": best_name,
            "best_conventional": best_conventional,
            "best_delta_vs_conventional": (
                all_summaries[best_name]["mean"]
                - candidate_summary[best_conventional]["mean"]
            ),
        }
        output_path = output_dir / "results.json"
        output_path.write_text(json.dumps(results, indent=2), encoding="utf-8")

    return results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--output-dir", type=Path, default=Path("runs/validation-balanced-v22"))
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--task", action="append", choices=TASKS)
    args = parser.parse_args()
    task_names = tuple(args.task) if args.task else TASKS
    results = run(args.data_dir, args.output_dir, args.workers, task_names)
    print(json.dumps(results, indent=2), flush=True)


if __name__ == "__main__":
    main()
