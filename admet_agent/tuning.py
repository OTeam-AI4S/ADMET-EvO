"""Stateful Optuna refiner used inside a validation-only agent run."""

from __future__ import annotations

from collections import defaultdict
from typing import Sequence

from .schemas import RunResult, TaskProfile


class OptunaRefiner:
    def __init__(self, seed: int = 2026):
        self.seed = seed
        self._studies = {}
        self._pending = defaultdict(dict)

    @staticmethod
    def _signature(parameters: dict) -> tuple:
        return tuple(sorted(parameters.items()))

    def _study(self, profile: TaskProfile, model: str, representations: tuple[str, ...]):
        key = (profile.task, model, representations)
        if key not in self._studies:
            import optuna

            sampler = optuna.samplers.TPESampler(seed=self.seed)
            self._studies[key] = optuna.create_study(
                direction="minimize" if profile.metric == "mae" else "maximize",
                sampler=sampler,
            )
        return key, self._studies[key]

    def suggest(
        self,
        profile: TaskProfile,
        history: Sequence[RunResult],
        model: str,
        representations: tuple[str, ...],
        count: int = 3,
    ) -> list[dict]:
        try:
            import optuna
        except ImportError:
            return []
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        key, study = self._study(profile, model, representations)
        by_signature = {
            self._signature(item.hyperparameters): item
            for item in history
            if item.model == model and tuple(item.representations) == representations
        }
        for signature, trial in list(self._pending[key].items()):
            result = by_signature.get(signature)
            if result is None:
                continue
            if result.status == "completed":
                study.tell(trial, result.validation_score)
            else:
                study.tell(trial, state=optuna.trial.TrialState.FAIL)
            del self._pending[key][signature]

        suggestions = []
        attempts = 0
        existing = set(by_signature) | set(self._pending[key])
        while len(suggestions) < count and attempts < count * 5:
            attempts += 1
            trial = study.ask()
            parameters = self._sample(trial, model, profile.task_type)
            signature = self._signature(parameters)
            if signature in existing:
                study.tell(trial, state=optuna.trial.TrialState.PRUNED)
                continue
            existing.add(signature)
            self._pending[key][signature] = trial
            suggestions.append(parameters)
        return suggestions

    @staticmethod
    def _sample(trial, model: str, task_type: str) -> dict:
        if model in {"extra_trees", "random_forest"}:
            criteria = (
                ["gini", "entropy", "log_loss"]
                if task_type == "classification"
                else ["squared_error", "absolute_error"]
            )
            return {
                "n_estimators": trial.suggest_int("n_estimators", 300, 1200, step=100),
                "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
                "max_features": trial.suggest_float("max_features", 0.2, 1.0),
                "criterion": trial.suggest_categorical("criterion", criteria),
            }
        if model == "catboost":
            return {
                "iterations": trial.suggest_int("iterations", 300, 1600, step=100),
                "depth": trial.suggest_int("depth", 4, 10),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.15, log=True),
                "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 0.1, 20.0, log=True),
                "random_strength": trial.suggest_float("random_strength", 0.0, 5.0),
            }
        if model == "xgboost":
            return {
                "n_estimators": trial.suggest_int("n_estimators", 300, 1200, step=100),
                "max_depth": trial.suggest_int("max_depth", 3, 10),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.15, log=True),
                "subsample": trial.suggest_float("subsample", 0.6, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.3, 1.0),
                "min_child_weight": trial.suggest_float("min_child_weight", 0.5, 12.0, log=True),
                "reg_alpha": trial.suggest_float("reg_alpha", 1e-4, 10.0, log=True),
                "reg_lambda": trial.suggest_float("reg_lambda", 0.1, 20.0, log=True),
            }
        if model == "lightgbm":
            return {
                "n_estimators": trial.suggest_int("n_estimators", 300, 1200, step=100),
                "num_leaves": trial.suggest_int("num_leaves", 15, 127),
                "max_depth": trial.suggest_int("max_depth", 4, 12),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.15, log=True),
                "subsample": trial.suggest_float("subsample", 0.6, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.3, 1.0),
                "min_child_samples": trial.suggest_int("min_child_samples", 5, 60),
                "reg_alpha": trial.suggest_float("reg_alpha", 1e-4, 10.0, log=True),
                "reg_lambda": trial.suggest_float("reg_lambda", 0.1, 20.0, log=True),
            }
        if model == "svm_rbf":
            values = {
                "C": trial.suggest_float("C", 0.1, 300.0, log=True),
                "gamma": trial.suggest_float("gamma", 1e-6, 0.1, log=True),
            }
            if task_type == "regression":
                values["epsilon"] = trial.suggest_float("epsilon", 0.01, 0.5, log=True)
            return values
        raise ValueError(f"Unsupported Optuna model: {model}")
