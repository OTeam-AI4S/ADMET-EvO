"""Matched predictive actions for the EC-VOI v5 public development cohort."""

from __future__ import annotations

import time
import hashlib
from typing import Any

import numpy as np
from lightgbm import LGBMClassifier, LGBMRegressor
from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor
from sklearn.metrics import matthews_corrcoef, mean_absolute_error
from sklearn.model_selection import GroupKFold, StratifiedGroupKFold

from .imbalance_action import fit_nested_threshold_action


ACTIONS = (
    "baseline",
    "feature_rich",
    "nested_imbalance",
    "boosting_anchor",
    "automl_expanded",
)

AUTOML_CONFIGS = (
    {"num_leaves": 15, "min_child_samples": 10, "learning_rate": 0.03},
    {"num_leaves": 15, "min_child_samples": 30, "learning_rate": 0.05},
    {"num_leaves": 31, "min_child_samples": 10, "learning_rate": 0.05},
    {"num_leaves": 31, "min_child_samples": 30, "learning_rate": 0.03},
)


def sampled_structural_support(
    train_bits: np.ndarray,
    train_ids: np.ndarray,
    valid_bits: np.ndarray,
    valid_ids: np.ndarray,
    *,
    sample_size: int = 256,
    seed: int = 17,
) -> float:
    """Median nearest-train Tanimoto on deterministic label-blind samples."""
    def selected(values: np.ndarray) -> np.ndarray:
        order = sorted(
            range(len(values)),
            key=lambda index: hashlib.sha256(f"{seed}:{values[index]}".encode()).hexdigest(),
        )
        return np.asarray(order[:sample_size], dtype=int)

    train = np.asarray(train_bits[selected(np.asarray(train_ids))] > 0, dtype=np.float32)
    valid = np.asarray(valid_bits[selected(np.asarray(valid_ids))] > 0, dtype=np.float32)
    intersection = valid @ train.T
    union = valid.sum(axis=1)[:, None] + train.sum(axis=1)[None, :] - intersection
    similarity = np.divide(intersection, union, out=np.zeros_like(intersection), where=union > 0)
    return float(np.median(similarity.max(axis=1)))


def _classifier(seed: int, *, balanced: bool, trees: int, workers: int):
    return ExtraTreesClassifier(
        n_estimators=trees,
        min_samples_leaf=2,
        max_features="sqrt",
        class_weight="balanced" if balanced else None,
        random_state=seed,
        n_jobs=workers,
    )


def _regressor(seed: int, *, trees: int, workers: int):
    return ExtraTreesRegressor(
        n_estimators=trees,
        min_samples_leaf=2,
        max_features="sqrt",
        random_state=seed,
        n_jobs=workers,
    )


def _boosting(task_type: str, seed: int, workers: int, config: dict[str, float | int]):
    common = {
        "n_estimators": 250,
        "random_state": seed,
        "n_jobs": workers,
        "verbosity": -1,
        "deterministic": True,
        "force_col_wise": True,
        **config,
    }
    if task_type == "classification":
        return LGBMClassifier(**common)
    return LGBMRegressor(**common)


def _automl_fit_predict(
    x_train: np.ndarray,
    target_train: np.ndarray,
    groups_train: np.ndarray,
    x_valid: np.ndarray,
    *,
    task_type: str,
    seed: int,
    workers: int,
    inner_splits: int,
) -> tuple[np.ndarray, int, int]:
    target = np.asarray(target_train)
    if task_type == "classification":
        splitter = StratifiedGroupKFold(n_splits=inner_splits, shuffle=True, random_state=seed)
        splits = list(splitter.split(x_train, target.astype(int), groups_train))
    else:
        splitter = GroupKFold(n_splits=inner_splits)
        splits = list(splitter.split(x_train, target, groups_train))
    candidate_scores = []
    for candidate_index, config in enumerate(AUTOML_CONFIGS):
        scores = []
        for fold, (fit_index, score_index) in enumerate(splits):
            model = _boosting(task_type, seed + candidate_index * inner_splits + fold, workers, config)
            model.fit(x_train[fit_index], target[fit_index])
            if task_type == "classification":
                prediction = (model.predict_proba(x_train[score_index])[:, 1] >= 0.5).astype(int)
                scores.append(float(matthews_corrcoef(target[score_index].astype(int), prediction)))
            else:
                scores.append(-float(mean_absolute_error(target[score_index], model.predict(x_train[score_index]))))
        candidate_scores.append(float(np.mean(scores)))
    best = max(range(len(AUTOML_CONFIGS)), key=lambda index: (candidate_scores[index], -index))
    final_model = _boosting(task_type, seed + 10_000 + best, workers, AUTOML_CONFIGS[best])
    final_model.fit(x_train, target)
    if task_type == "classification":
        prediction = (final_model.predict_proba(x_valid)[:, 1] >= 0.5).astype(int)
    else:
        prediction = final_model.predict(x_valid)
    return np.asarray(prediction), len(AUTOML_CONFIGS) * inner_splits + 1, best


def evaluate_action(
    x_train: np.ndarray,
    target_train: np.ndarray,
    groups_train: np.ndarray,
    x_valid: np.ndarray,
    target_valid: np.ndarray,
    *,
    action: str,
    task_type: str,
    seed: int,
    trees: int = 200,
    workers: int = 4,
    inner_splits: int = 5,
) -> dict[str, Any]:
    """Fit one outer-fold action and return an oriented, cost-audited score."""
    if action not in ACTIONS:
        raise ValueError(f"Unknown action: {action}")
    if action == "nested_imbalance" and task_type != "classification":
        raise ValueError("nested_imbalance is classification-only")
    started = time.monotonic()
    threshold = None
    selected_candidate = None
    if action == "automl_expanded":
        prediction, fits, selected_candidate = _automl_fit_predict(
            x_train,
            target_train,
            groups_train,
            x_valid,
            task_type=task_type,
            seed=seed,
            workers=workers,
            inner_splits=inner_splits,
        )
        if task_type == "classification":
            raw_score = float(matthews_corrcoef(np.asarray(target_valid, dtype=int), prediction.astype(int)))
            oriented_score, metric = raw_score, "mcc"
        elif task_type == "regression":
            raw_score = float(mean_absolute_error(target_valid, prediction))
            oriented_score, metric = -raw_score, "mae"
        else:
            raise ValueError(f"Unsupported task type: {task_type}")
    elif action == "boosting_anchor":
        model = _boosting(task_type, seed, workers, AUTOML_CONFIGS[2])
        model.fit(x_train, target_train)
        fits = 1
        if task_type == "classification":
            prediction = (model.predict_proba(x_valid)[:, 1] >= 0.5).astype(int)
            raw_score = float(matthews_corrcoef(np.asarray(target_valid, dtype=int), prediction))
            oriented_score, metric = raw_score, "mcc"
        elif task_type == "regression":
            raw_score = float(mean_absolute_error(target_valid, model.predict(x_valid)))
            oriented_score, metric = -raw_score, "mae"
        else:
            raise ValueError(f"Unsupported task type: {task_type}")
    elif task_type == "classification":
        train_y = np.asarray(target_train, dtype=int)
        valid_y = np.asarray(target_valid, dtype=int)
        if action == "nested_imbalance":
            nested = fit_nested_threshold_action(
                x_train,
                train_y,
                groups_train,
                estimator_factory=lambda local_seed: _classifier(
                    local_seed, balanced=True, trees=trees, workers=workers
                ),
                splits=inner_splits,
                seed=seed,
            )
            prediction = nested.predict(x_valid)
            threshold = float(nested.threshold)
            fits = int(nested.inner_fits + 1)
        else:
            model = _classifier(seed, balanced=False, trees=trees, workers=workers)
            model.fit(x_train, train_y)
            prediction = (model.predict_proba(x_valid)[:, 1] >= 0.5).astype(int)
            fits = 1
        raw_score = float(matthews_corrcoef(valid_y, prediction))
        oriented_score = raw_score
        metric = "mcc"
    elif task_type == "regression":
        model = _regressor(seed, trees=trees, workers=workers)
        model.fit(x_train, np.asarray(target_train, dtype=float))
        prediction = model.predict(x_valid)
        raw_score = float(mean_absolute_error(target_valid, prediction))
        oriented_score = -raw_score
        metric = "mae"
        fits = 1
    else:
        raise ValueError(f"Unsupported task type: {task_type}")
    return {
        "action": action,
        "task_type": task_type,
        "metric": metric,
        "raw_score": raw_score,
        "oriented_score": oriented_score,
        "threshold": threshold,
        "selected_automl_candidate": selected_candidate,
        "fits": fits,
        "wall_seconds": float(time.monotonic() - started),
        "outer_validation_labels_used_for_training_or_threshold": False,
    }


__all__ = ["ACTIONS", "AUTOML_CONFIGS", "evaluate_action", "sampled_structural_support"]
