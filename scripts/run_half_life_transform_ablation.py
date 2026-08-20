"""Ablate leakage-safe target transforms for the skewed Half-Life endpoint.

Every transform is fitted on the outer training split only.  The official TDC
test set is never loaded by this script.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.stats import rankdata
from sklearn.preprocessing import PowerTransformer, QuantileTransformer, StandardScaler
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

MODELS = (
    ("svr_c3", "svm_rbf", RICH, {"C": 3.0, "gamma": "scale", "epsilon": 0.1}),
    (
        "svr_extended",
        "svm_rbf",
        EXTENDED,
        {"C": 3.0, "gamma": "scale", "epsilon": 0.1},
    ),
    ("lightgbm", "lightgbm", EXTENDED, {}),
    (
        "catboost",
        "catboost",
        RICH,
        {"iterations": 1200, "depth": 5, "learning_rate": 0.025, "l2_leaf_reg": 8.0},
    ),
)


def transform_target(y: np.ndarray, name: str, seed: int) -> np.ndarray:
    """Fit a transform using only y and return a one-dimensional target."""
    values = np.asarray(y, dtype=float).reshape(-1, 1)
    if name == "raw":
        return values.ravel()
    if name == "standard":
        return StandardScaler().fit_transform(values).ravel()
    if name == "log1p":
        offset = max(0.0, -float(values.min()) + 1e-8)
        return np.log1p(values.ravel() + offset)
    if name == "box_cox":
        offset = max(0.0, -float(values.min()) + 1e-8)
        positive = values + offset + 1e-8
        return (
            PowerTransformer(method="box-cox", standardize=True)
            .fit_transform(positive)
            .ravel()
        )
    if name == "yeo_johnson":
        return (
            PowerTransformer(method="yeo-johnson", standardize=True)
            .fit_transform(values)
            .ravel()
        )
    if name == "rank":
        return rankdata(values.ravel(), method="average") / len(values)
    if name == "quantile_normal":
        n_quantiles = min(100, len(values))
        return (
            QuantileTransformer(
                n_quantiles=n_quantiles,
                output_distribution="normal",
                random_state=seed,
            )
            .fit_transform(values)
            .ravel()
        )
    raise ValueError(f"Unknown transform: {name}")


def main() -> None:
    root = Path("runs/half-life-transform-ablation")
    root.mkdir(parents=True, exist_ok=True)
    cache = FeatureCache(root / "feature_cache")
    group = admet_group(path="data")
    task = resolve_task("Half_Life_Obach")
    transforms = (
        "raw",
        "standard",
        "log1p",
        "box_cox",
        "yeo_johnson",
        "rank",
        "quantile_normal",
    )
    scores = {
        f"{model}_{transform}": []
        for model, _, _, _ in MODELS
        for transform in transforms
    }
    report = {
        "test_labels_used": False,
        "split": "TDC default scaffold train/validation, seeds 1-5",
        "transform_fit_scope": "outer training split only",
        "metric": task.metric,
    }
    for seed in range(1, 6):
        train, valid = group.get_train_valid_split(
            benchmark=task.name, split_type="default", seed=seed
        )
        smiles_col, y_col = _columns(train)
        y_train = train[y_col].to_numpy(dtype=float)
        y_valid = valid[y_col].to_numpy(dtype=float)
        all_smiles = train[smiles_col].tolist() + valid[smiles_col].tolist()
        feature_sets = {
            representations: cache.get_or_compute(all_smiles, representations)
            for representations in {recipe[2] for recipe in MODELS}
        }
        for transform in transforms:
            target = transform_target(y_train, transform, seed)
            for model_label, model_name, representations, params in MODELS:
                name = f"{model_label}_{transform}"
                spec = ExperimentSpec(
                    task=task.name,
                    seed=seed,
                    representations=representations,
                    model=model_name,
                    imbalance="none",
                    device="cpu",
                    n_jobs=8,
                    hyperparameters=params,
                    hypothesis="leakage-safe target-transform ablation",
                )
                x_all = feature_sets[representations]
                model = build_model(spec, task, target)
                model.fit(x_all[: len(train)], target)
                predictions = predict(model, x_all[len(train) :], task)
                value = score(y_valid, predictions, task.metric)
                scores[name].append(value)
                print(seed, name, value, flush=True)
        report["partial_scores"] = scores
        (root / "results.partial.json").write_text(
            json.dumps(report, indent=2, sort_keys=True), encoding="utf-8"
        )
    summary = {
        name: {
            "mean": float(np.mean(values)),
            "std": float(np.std(values)),
            "scores": values,
        }
        for name, values in scores.items()
    }
    report.pop("partial_scores", None)
    report["winner"] = max(summary, key=lambda name: summary[name]["mean"])
    report["summary"] = summary
    (root / "results.json").write_text(
        json.dumps(report, indent=2, sort_keys=True), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
