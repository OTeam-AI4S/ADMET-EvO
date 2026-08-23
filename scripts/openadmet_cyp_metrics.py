"""Training-side replicas of frozen OpenADMET CYP primary metrics."""

from __future__ import annotations

import numpy as np


def soft_threshold_distance(
    prediction: np.ndarray,
    target: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
) -> np.ndarray:
    has_interval = np.isfinite(lower) & np.isfinite(upper)
    resolved_lower = np.where(has_interval, lower, target)
    resolved_upper = np.where(has_interval, upper, target)
    return np.maximum(resolved_lower - prediction, 0) + np.maximum(
        prediction - resolved_upper, 0
    )


def official_st_rae(
    *,
    target: np.ndarray,
    prediction: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
) -> float | None:
    numerator = soft_threshold_distance(prediction, target, lower, upper).sum()
    baseline = np.full(len(target), float(np.mean(target)))
    denominator = soft_threshold_distance(baseline, target, lower, upper).sum()
    return float(numerator / denominator) if denominator > 0 else None


__all__ = ["official_st_rae", "soft_threshold_distance"]
