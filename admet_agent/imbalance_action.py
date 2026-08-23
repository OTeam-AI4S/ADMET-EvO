"""Group-safe nested threshold action for imbalanced classification endpoints."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Protocol

import numpy as np
from sklearn.metrics import matthews_corrcoef
from sklearn.model_selection import StratifiedGroupKFold


class ProbabilisticClassifier(Protocol):
    def fit(self, x: np.ndarray, y: np.ndarray): ...
    def predict_proba(self, x: np.ndarray) -> np.ndarray: ...


def select_mcc_threshold(
    target: np.ndarray,
    probability: np.ndarray,
    *,
    grid: tuple[float, ...] = tuple(np.linspace(0.05, 0.95, 91)),
) -> float:
    """Select a deterministic MCC threshold from training-only OOF predictions."""
    target = np.asarray(target, dtype=int)
    probability = np.asarray(probability, dtype=float)
    if len(target) != len(probability) or len(target) == 0:
        raise ValueError("Target and probability must have equal non-zero length")
    if set(np.unique(target)) - {0, 1}:
        raise ValueError("MCC threshold selection requires binary targets")
    if not np.isfinite(probability).all() or np.any((probability < 0) | (probability > 1)):
        raise ValueError("Probabilities must be finite and within [0, 1]")
    scored = []
    for threshold in grid:
        prediction = (probability >= threshold).astype(int)
        scored.append((float(matthews_corrcoef(target, prediction)), -abs(threshold - 0.5), -threshold))
    best = max(range(len(grid)), key=lambda index: scored[index])
    return float(grid[best])


@dataclass
class NestedThresholdResult:
    model: ProbabilisticClassifier
    threshold: float
    oof_probability: np.ndarray
    inner_fits: int

    def predict(self, x: np.ndarray) -> np.ndarray:
        probability = self.model.predict_proba(x)[:, 1]
        return (probability >= self.threshold).astype(int)


def fit_nested_threshold_action(
    x: np.ndarray,
    target: np.ndarray,
    groups: np.ndarray,
    *,
    estimator_factory: Callable[[int], ProbabilisticClassifier],
    splits: int = 5,
    seed: int = 20260823,
) -> NestedThresholdResult:
    """Choose threshold on group-disjoint inner OOF data, then refit all outer train."""
    x = np.asarray(x)
    target = np.asarray(target, dtype=int)
    groups = np.asarray(groups)
    if not (len(x) == len(target) == len(groups)) or len(x) == 0:
        raise ValueError("x, target, and groups must have equal non-zero length")
    splitter = StratifiedGroupKFold(n_splits=splits, shuffle=True, random_state=seed)
    oof = np.full(len(target), np.nan, dtype=float)
    inner_fits = 0
    for fold, (train_index, validation_index) in enumerate(
        splitter.split(x, target, groups)
    ):
        if set(groups[train_index]) & set(groups[validation_index]):
            raise RuntimeError("Inner group leakage detected")
        model = estimator_factory(seed + fold)
        model.fit(x[train_index], target[train_index])
        oof[validation_index] = model.predict_proba(x[validation_index])[:, 1]
        inner_fits += 1
    if not np.isfinite(oof).all():
        raise RuntimeError("Nested OOF schedule did not predict every outer-train row")
    threshold = select_mcc_threshold(target, oof)
    final_model = estimator_factory(seed + splits)
    final_model.fit(x, target)
    return NestedThresholdResult(
        model=final_model,
        threshold=threshold,
        oof_probability=oof,
        inner_fits=inner_fits,
    )


__all__ = [
    "NestedThresholdResult",
    "fit_nested_threshold_action",
    "select_mcc_threshold",
]
