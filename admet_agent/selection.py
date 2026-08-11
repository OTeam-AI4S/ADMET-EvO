"""Auditable model construction, prediction and metric functions."""

from __future__ import annotations

import numpy as np
from scipy.stats import spearmanr
from sklearn.ensemble import (
    ExtraTreesClassifier,
    ExtraTreesRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import average_precision_score, mean_absolute_error, roc_auc_score

from .schemas import ExperimentSpec
from .tasks import TaskSpec


def build_model(experiment: ExperimentSpec, task: TaskSpec, y_train: np.ndarray):
    weighted = experiment.imbalance == "class_weight"
    common = dict(n_estimators=400, n_jobs=experiment.n_jobs, random_state=experiment.seed, min_samples_leaf=2)
    overrides = dict(experiment.hyperparameters)
    if experiment.model == "extra_trees":
        params = {**common, **overrides}
        if task.task_type == "classification":
            params.setdefault("class_weight", "balanced" if weighted else None)
            return ExtraTreesClassifier(**params)
        return ExtraTreesRegressor(**params)
    if experiment.model == "random_forest":
        params = {**common, "max_features": "sqrt" if task.task_type == "classification" else 0.7}
        params.update(overrides)
        if task.task_type == "classification":
            weights = "balanced_subsample" if weighted else None
            params.setdefault("class_weight", weights)
            return RandomForestClassifier(**params)
        return RandomForestRegressor(**params)
    if experiment.model == "xgboost":
        try:
            from xgboost import XGBClassifier, XGBRegressor
        except ImportError as exc:
            raise RuntimeError("Install the `boosters` project extra to use XGBoost") from exc
        if experiment.device == "cuda":
            from importlib.metadata import PackageNotFoundError, version
            try:
                version("xgboost-cpu")
            except PackageNotFoundError:
                pass
            else:
                raise RuntimeError(
                    "CUDA XGBoost requires the `xgboost` wheel in a separate GPU "
                    "environment; this environment contains `xgboost-cpu`"
                )
        params = dict(
            n_estimators=600, max_depth=6, learning_rate=0.05,
            subsample=0.85, colsample_bytree=0.7, min_child_weight=2.0,
            reg_lambda=2.0, n_jobs=experiment.n_jobs, random_state=experiment.seed,
            tree_method="hist", device=experiment.device,
        )
        params.update(overrides)
        if task.task_type == "classification":
            params.setdefault("eval_metric", "aucpr" if task.metric == "auprc" else "auc")
            if weighted:
                positives = float(np.sum(y_train == 1))
                negatives = float(np.sum(y_train == 0))
                params.setdefault("scale_pos_weight", negatives / max(positives, 1.0))
            return XGBClassifier(**params)
        params.setdefault("objective", "reg:absoluteerror" if task.metric == "mae" else "reg:squarederror")
        return XGBRegressor(**params)
    if experiment.model == "lightgbm":
        try:
            from lightgbm import LGBMClassifier, LGBMRegressor
        except ImportError as exc:
            raise RuntimeError("Install the `boosters` project extra to use LightGBM") from exc
        params = dict(
            n_estimators=600, num_leaves=31, learning_rate=0.04,
            subsample=0.85, subsample_freq=1, colsample_bytree=0.7,
            min_child_samples=15, reg_lambda=2.0, n_jobs=experiment.n_jobs,
            random_state=experiment.seed, verbosity=-1,
        )
        params.update(overrides)
        if task.task_type == "classification":
            params.setdefault("objective", "binary")
            params.setdefault("class_weight", "balanced" if weighted else None)
            return LGBMClassifier(**params)
        params.setdefault("objective", "regression_l1" if task.metric == "mae" else "regression")
        return LGBMRegressor(**params)
    if experiment.model == "svm_rbf":
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
        from sklearn.svm import SVC, SVR

        params = dict(C=10.0, gamma="scale")
        params.update(overrides)
        if task.task_type == "classification":
            params.pop("epsilon", None)
            params.setdefault("class_weight", "balanced" if weighted else None)
            calibrated = CalibratedClassifierCV(
                SVC(**params), cv=3, ensemble=False, n_jobs=experiment.n_jobs
            )
            return make_pipeline(StandardScaler(), calibrated)
        return make_pipeline(StandardScaler(), SVR(**params))
    if experiment.model == "catboost":
        try:
            from catboost import CatBoostClassifier, CatBoostRegressor
        except ImportError as exc:
            raise RuntimeError("Install the `catboost` project extra to use this model") from exc
        params = dict(iterations=600, depth=8, learning_rate=0.05,
                      random_seed=experiment.seed, verbose=False,
                      thread_count=experiment.n_jobs)
        if experiment.device == "cuda":
            params.update(task_type="GPU", devices="0")
        params.update(overrides)
        if task.task_type == "classification":
            params["eval_metric"] = "PRAUC" if task.metric == "auprc" else "AUC"
            if weighted:
                params["auto_class_weights"] = "Balanced"
            return CatBoostClassifier(**params)
        params["loss_function"] = "MAE" if task.metric == "mae" else "RMSE"
        return CatBoostRegressor(**params)
    raise ValueError(f"Unknown model {experiment.model!r}")


def predict(model: object, x: np.ndarray, task: TaskSpec) -> np.ndarray:
    if task.task_type == "classification":
        return np.asarray(model.predict_proba(x))[:, 1]
    return np.asarray(model.predict(x))


def score(y_true: np.ndarray, y_pred: np.ndarray, metric: str) -> float:
    if metric == "mae":
        return float(mean_absolute_error(y_true, y_pred))
    if metric == "auroc":
        return float(roc_auc_score(y_true, y_pred))
    if metric == "auprc":
        return float(average_precision_score(y_true, y_pred))
    if metric == "spearman":
        value = spearmanr(y_true, y_pred).statistic
        return float(0.0 if np.isnan(value) else value)
    raise ValueError(f"Unsupported metric: {metric}")
