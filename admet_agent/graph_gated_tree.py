"""OOF-safe decision-tree gating across 2D, GNN and MPNN feature views."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Mapping

import numpy as np
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.tree import DecisionTreeClassifier


@dataclass(frozen=True)
class GraphGateAudit:
    view_names: tuple[str, ...]
    rows: int
    folds: int
    gate_depth: int
    minimum_gate_leaf: int
    gate_trained_on_oof_predictions: bool
    base_refit_on_full_training: bool
    test_labels_loaded: bool = False

    def to_dict(self) -> dict:
        return asdict(self)


class OOFGraphGatedTreeClassifier:
    """Select a per-molecule tree expert using only OOF training predictions.

    Expected views are typically ``two_d``, ``gin`` and ``mpnn``.  The class is
    backend-agnostic: graph embeddings must be produced by independently hashed
    feature providers, so a missing MPNN runtime cannot silently masquerade as
    another graph model.
    """

    def __init__(
        self,
        *,
        n_estimators: int = 400,
        folds: int = 5,
        gate_depth: int = 2,
        random_state: int = 1,
        n_jobs: int = 8,
    ):
        self.n_estimators = n_estimators
        self.folds = folds
        self.gate_depth = gate_depth
        self.random_state = random_state
        self.n_jobs = n_jobs

    def _base(self, seed: int) -> ExtraTreesClassifier:
        return ExtraTreesClassifier(
            n_estimators=self.n_estimators,
            min_samples_leaf=2,
            max_features="sqrt",
            class_weight="balanced",
            random_state=seed,
            n_jobs=self.n_jobs,
        )

    @staticmethod
    def _validate_views(views: Mapping[str, np.ndarray], rows: int) -> tuple[str, ...]:
        names = tuple(sorted(views))
        if len(names) < 2:
            raise ValueError("Graph gating requires at least two distinct feature views")
        for name in names:
            values = np.asarray(views[name])
            if values.ndim != 2 or values.shape[0] != rows or values.shape[1] == 0:
                raise ValueError(f"Invalid feature matrix for view {name!r}")
            if not np.isfinite(values).all():
                raise ValueError(f"Non-finite features in view {name!r}")
        return names

    @staticmethod
    def _gate_features(probabilities: np.ndarray) -> np.ndarray:
        return np.hstack([probabilities, np.abs(probabilities - 0.5)])

    def fit(self, views: Mapping[str, np.ndarray], y: np.ndarray):
        labels = np.asarray(y, dtype=int)
        if labels.ndim != 1 or set(np.unique(labels)) != {0, 1}:
            raise ValueError("OOF graph gate currently requires binary labels with both classes")
        self.view_names_ = self._validate_views(views, len(labels))
        minimum_class = int(np.bincount(labels).min())
        folds = min(self.folds, minimum_class)
        if folds < 2:
            raise ValueError("At least two examples per class are required")
        splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=self.random_state)
        oof = np.zeros((len(labels), len(self.view_names_)), dtype=float)
        for view_index, name in enumerate(self.view_names_):
            matrix = np.asarray(views[name], dtype=np.float32)
            for fold_index, (train_rows, valid_rows) in enumerate(splitter.split(matrix, labels)):
                model = self._base(self.random_state * 1000 + view_index * 100 + fold_index)
                model.fit(matrix[train_rows], labels[train_rows])
                oof[valid_rows, view_index] = model.predict_proba(matrix[valid_rows])[:, 1]
        probability_for_true_class = np.where(labels[:, None] == 1, oof, 1.0 - oof)
        gate_target = np.argmax(probability_for_true_class, axis=1)
        minimum_leaf = max(5, len(labels) // 20)
        self.gate_ = DecisionTreeClassifier(
            max_depth=self.gate_depth,
            min_samples_leaf=minimum_leaf,
            random_state=self.random_state,
            class_weight="balanced",
        ).fit(self._gate_features(oof), gate_target)
        self.base_models_ = {}
        for view_index, name in enumerate(self.view_names_):
            model = self._base(self.random_state * 1000 + view_index)
            model.fit(np.asarray(views[name], dtype=np.float32), labels)
            self.base_models_[name] = model
        self.audit_ = GraphGateAudit(
            view_names=self.view_names_,
            rows=len(labels),
            folds=folds,
            gate_depth=self.gate_depth,
            minimum_gate_leaf=minimum_leaf,
            gate_trained_on_oof_predictions=True,
            base_refit_on_full_training=True,
        )
        return self

    def _expert_probabilities(self, views: Mapping[str, np.ndarray]) -> np.ndarray:
        if not hasattr(self, "base_models_"):
            raise RuntimeError("Graph gate is not fitted")
        if tuple(sorted(views)) != self.view_names_:
            raise ValueError("Prediction views differ from fitted views")
        rows = len(np.asarray(views[self.view_names_[0]]))
        self._validate_views(views, rows)
        return np.column_stack(
            [
                self.base_models_[name].predict_proba(np.asarray(views[name], dtype=np.float32))[:, 1]
                for name in self.view_names_
            ]
        )

    def predict_proba(self, views: Mapping[str, np.ndarray]) -> np.ndarray:
        probabilities = self._expert_probabilities(views)
        chosen = self.gate_.predict(self._gate_features(probabilities)).astype(int)
        positive = probabilities[np.arange(len(probabilities)), chosen]
        return np.column_stack([1.0 - positive, positive])

    def gate_decision(self, views: Mapping[str, np.ndarray]) -> tuple[str, ...]:
        probabilities = self._expert_probabilities(views)
        chosen = self.gate_.predict(self._gate_features(probabilities)).astype(int)
        return tuple(self.view_names_[index] for index in chosen)


__all__ = ["GraphGateAudit", "OOFGraphGatedTreeClassifier"]
