"""Targeted metric-aligned ablations for CYP2C9 substrate and half-life."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.stats import rankdata
from tdc.benchmark_group import admet_group

from admet_agent.features import FeatureCache
from admet_agent.runner import _columns
from admet_agent.schemas import ExperimentSpec
from admet_agent.selection import build_model, predict, score
from admet_agent.tasks import resolve_task

RICH = ("morgan", "avalon", "erg", "rdkit2d")
EXTENDED = (
    "morgan",
    "morgan_r3",
    "morgan_count",
    "avalon",
    "erg",
    "maccs",
    "rdkit2d",
)


def _recipes(task_name: str) -> list[dict]:
    if task_name == "CYP2C9_Substrate_CarbonMangels":
        return [
            {"name": "lgb_default", "model": "lightgbm", "repr": RICH},
            {
                "name": "lgb_shallow",
                "model": "lightgbm",
                "repr": RICH,
                "params": {"n_estimators": 1000, "num_leaves": 7, "max_depth": 3,
                           "learning_rate": 0.02, "min_child_samples": 10,
                           "reg_lambda": 5.0},
            },
            {
                "name": "lgb_regularized",
                "model": "lightgbm",
                "repr": RICH,
                "params": {"n_estimators": 1000, "num_leaves": 15, "max_depth": 5,
                           "learning_rate": 0.025, "min_child_samples": 20,
                           "reg_alpha": 0.1, "reg_lambda": 5.0},
            },
            {
                "name": "lgb_pr_weight_05",
                "model": "lightgbm",
                "repr": RICH,
                "weight_multiplier": 0.5,
            },
            {
                "name": "lgb_pr_weight_20",
                "model": "lightgbm",
                "repr": RICH,
                "weight_multiplier": 2.0,
            },
            {"name": "lgb_extended", "model": "lightgbm", "repr": EXTENDED},
            {
                "name": "xgb_depth3",
                "model": "xgboost",
                "repr": RICH,
                "params": {"n_estimators": 1000, "max_depth": 3,
                           "learning_rate": 0.025, "min_child_weight": 4.0,
                           "reg_lambda": 5.0},
            },
            {
                "name": "xgb_depth4_extended",
                "model": "xgboost",
                "repr": EXTENDED,
                "params": {"n_estimators": 800, "max_depth": 4,
                           "learning_rate": 0.035, "min_child_weight": 2.0,
                           "reg_lambda": 3.0},
            },
            {
                "name": "cat_depth5",
                "model": "catboost",
                "repr": RICH,
                "params": {"iterations": 1200, "depth": 5, "learning_rate": 0.025,
                           "l2_leaf_reg": 8.0},
            },
            {
                "name": "balanced_et",
                "model": "balanced_extra_trees",
                "repr": EXTENDED,
                "params": {"n_estimators": 400, "n_members": 9,
                           "majority_ratio": 2.0, "min_samples_leaf": 2},
            },
        ]
    base = [
        {"name": "svr_c3", "model": "svm_rbf", "repr": RICH,
         "params": {"C": 3.0, "gamma": "scale", "epsilon": 0.1}},
        {"name": "svr_c10", "model": "svm_rbf", "repr": RICH,
         "params": {"C": 10.0, "gamma": 0.001, "epsilon": 0.05}},
        {"name": "svr_c30", "model": "svm_rbf", "repr": RICH,
         "params": {"C": 30.0, "gamma": 0.0001, "epsilon": 0.2}},
        {"name": "svr_extended", "model": "svm_rbf", "repr": EXTENDED,
         "params": {"C": 3.0, "gamma": "scale", "epsilon": 0.1}},
        {"name": "lgb_extended", "model": "lightgbm", "repr": EXTENDED},
        {"name": "xgb_depth3", "model": "xgboost", "repr": RICH,
         "params": {"n_estimators": 1000, "max_depth": 3,
                    "learning_rate": 0.025, "min_child_weight": 4.0}},
        {"name": "cat_depth5", "model": "catboost", "repr": RICH,
         "params": {"iterations": 1200, "depth": 5, "learning_rate": 0.025,
                    "l2_leaf_reg": 8.0}},
        {"name": "extra_trees_extended", "model": "extra_trees", "repr": EXTENDED,
         "params": {"n_estimators": 800, "min_samples_leaf": 2,
                    "max_features": 0.5}},
    ]
    return base + [{**item, "name": f"rank_target_{item['name']}", "rank_target": True}
                   for item in base]


def main() -> None:
    root = Path("runs/hard-task-ablation")
    root.mkdir(parents=True, exist_ok=True)
    cache = FeatureCache(root / "feature_cache")
    group = admet_group(path="data")
    report = {"test_labels_used": False, "tasks": {}}
    for task_name in ("CYP2C9_Substrate_CarbonMangels", "Half_Life_Obach"):
        task = resolve_task(task_name)
        recipes = _recipes(task.name)
        scores = {recipe["name"]: [] for recipe in recipes}
        for seed in range(1, 6):
            train, valid = group.get_train_valid_split(
                benchmark=task.name, split_type="default", seed=seed
            )
            smiles_col, y_col = _columns(train)
            all_smiles = train[smiles_col].tolist() + valid[smiles_col].tolist()
            features = {}
            y_train = train[y_col].to_numpy()
            for recipe in recipes:
                representations = tuple(recipe["repr"])
                if representations not in features:
                    features[representations] = cache.get_or_compute(
                        all_smiles, representations
                    )
                x_all = features[representations]
                params = dict(recipe.get("params", {}))
                imbalance = "class_weight" if task.task_type == "classification" else "none"
                if "weight_multiplier" in recipe:
                    positives = max(float(np.sum(y_train == 1)), 1.0)
                    negatives = float(np.sum(y_train == 0))
                    params["scale_pos_weight"] = (
                        negatives / positives * recipe["weight_multiplier"]
                    )
                    imbalance = "none"
                spec = ExperimentSpec(
                    task=task.name,
                    seed=seed,
                    representations=representations,
                    model=recipe["model"],
                    imbalance=imbalance,
                    device="cpu",
                    n_jobs=8,
                    hyperparameters=params,
                    hypothesis="metric-aligned hard-task ablation",
                )
                target = (
                    rankdata(y_train, method="average") / len(y_train)
                    if recipe.get("rank_target")
                    else y_train
                )
                model = build_model(spec, task, target)
                model.fit(x_all[: len(train)], target)
                predictions = predict(model, x_all[len(train) :], task)
                value = score(valid[y_col].to_numpy(), predictions, task.metric)
                scores[recipe["name"]].append(value)
                print(task.name, seed, recipe["name"], value, flush=True)
        summary = {
            name: {"mean": float(np.mean(values)), "std": float(np.std(values)),
                   "scores": values}
            for name, values in scores.items()
        }
        winner = max(summary, key=lambda name: summary[name]["mean"])
        report["tasks"][task.name] = {"metric": task.metric, "winner": winner,
                                          "summary": summary}
        (root / "results.partial.json").write_text(
            json.dumps(report, indent=2, sort_keys=True), encoding="utf-8"
        )
    (root / "results.json").write_text(
        json.dumps(report, indent=2, sort_keys=True), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
