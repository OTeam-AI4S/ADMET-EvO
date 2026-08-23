"""Blind-safe applicability and uncertainty diagnostics for molecular OOF data."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr
from sklearn.metrics import roc_auc_score


def percentile_rank(values: np.ndarray) -> np.ndarray:
    """Return deterministic ranks in [0, 1], with ties receiving mean rank."""
    values = np.asarray(values, dtype=float)
    if values.ndim != 1 or len(values) == 0 or not np.isfinite(values).all():
        raise ValueError("percentile_rank requires a finite non-empty vector")
    if len(values) == 1:
        return np.asarray([0.5])
    return (rankdata(values, method="average") - 1.0) / (len(values) - 1.0)


def combined_uncertainty(
    *,
    distance: np.ndarray,
    disagreement: np.ndarray,
    confidence_uncertainty: np.ndarray | None = None,
) -> np.ndarray:
    """Equal-weight, scale-free combination of label-free uncertainty signals."""
    components = [percentile_rank(distance), percentile_rank(disagreement)]
    if confidence_uncertainty is not None:
        components.append(percentile_rank(confidence_uncertainty))
    return np.mean(np.vstack(components), axis=0)


def interval_aware_loss(
    target: np.ndarray,
    prediction: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
) -> np.ndarray:
    """Official ST-RAE numerator contribution for each observation."""
    target = np.asarray(target, dtype=float)
    prediction = np.asarray(prediction, dtype=float)
    lower = np.asarray(lower, dtype=float)
    upper = np.asarray(upper, dtype=float)
    fallback = np.abs(target - prediction)
    valid = np.isfinite(lower) & np.isfinite(upper) & (lower <= upper)
    loss = fallback.copy()
    loss[valid] = np.where(
        prediction[valid] < lower[valid],
        lower[valid] - prediction[valid],
        np.where(
            prediction[valid] > upper[valid],
            prediction[valid] - upper[valid],
            0.0,
        ),
    )
    return loss


def risk_coverage(
    loss: np.ndarray,
    uncertainty: np.ndarray,
    coverages: tuple[float, ...] = tuple(np.arange(0.1, 1.01, 0.1)),
) -> dict:
    """Compute a lower-is-better area under the selective risk curve."""
    loss = np.asarray(loss, dtype=float)
    uncertainty = np.asarray(uncertainty, dtype=float)
    if len(loss) != len(uncertainty) or len(loss) == 0:
        raise ValueError("loss and uncertainty must have equal non-zero length")
    order = np.argsort(uncertainty, kind="stable")
    curve = []
    for coverage in coverages:
        count = max(1, int(math.ceil(len(loss) * coverage)))
        curve.append(
            {
                "coverage": float(coverage),
                "rows": count,
                "risk": float(np.mean(loss[order[:count]])),
            }
        )
    aurc = float(np.mean([item["risk"] for item in curve]))
    full_risk = float(np.mean(loss))
    return {
        "aurc": aurc,
        "full_risk": full_risk,
        "relative_aurc_improvement_vs_random": (
            float(1.0 - aurc / full_risk) if full_risk > 0 else 0.0
        ),
        "curve": curve,
    }


def error_detection(loss: np.ndarray, uncertainty: np.ndarray) -> dict:
    """Measure whether uncertainty detects the endpoint-local worst-loss quartile."""
    loss = np.asarray(loss, dtype=float)
    uncertainty = np.asarray(uncertainty, dtype=float)
    threshold = float(np.quantile(loss, 0.75))
    high_error = loss > threshold
    if not high_error.any() or high_error.all():
        auroc = None
    else:
        auroc = float(roc_auc_score(high_error.astype(int), uncertainty))
    correlation = spearmanr(loss, uncertainty).statistic
    return {
        "high_error_definition": "strictly_above_endpoint_loss_q75",
        "high_error_rows": int(high_error.sum()),
        "auroc": auroc,
        "spearman_loss_uncertainty": float(
            0.0 if np.isnan(correlation) else correlation
        ),
    }


def similarity_quartiles(similarity: np.ndarray) -> np.ndarray:
    """Assign label-free endpoint-local quartiles (0 is lowest similarity)."""
    rank = percentile_rank(np.asarray(similarity, dtype=float))
    return np.minimum((rank * 4).astype(int), 3)


def cross_conformal_by_similarity(
    frame: pd.DataFrame,
    *,
    scale_column: str,
    alpha: float = 0.1,
) -> dict:
    """Cross-conformal regression coverage, including low-similarity coverage."""
    required = {
        "target",
        "prediction",
        "outer_fold_evaluated",
        "similarity_quartile",
        scale_column,
    }
    if missing := required - set(frame.columns):
        raise ValueError(f"Missing conformal columns: {sorted(missing)}")
    fold_rows = []
    covered_all = []
    widths_all = []
    quartiles_all = []
    for fold in sorted(frame["outer_fold_evaluated"].unique()):
        calibration = frame.loc[frame["outer_fold_evaluated"] != fold]
        evaluation = frame.loc[frame["outer_fold_evaluated"] == fold]
        cal_scale = calibration[scale_column].to_numpy(dtype=float).clip(min=1e-6)
        scores = np.abs(
            calibration["target"].to_numpy(dtype=float)
            - calibration["prediction"].to_numpy(dtype=float)
        ) / cal_scale
        level = min(1.0, math.ceil((len(scores) + 1) * (1 - alpha)) / len(scores))
        quantile = float(np.quantile(scores, level, method="higher"))
        eval_scale = evaluation[scale_column].to_numpy(dtype=float).clip(min=1e-6)
        half_width = quantile * eval_scale
        target = evaluation["target"].to_numpy(dtype=float)
        prediction = evaluation["prediction"].to_numpy(dtype=float)
        covered = (target >= prediction - half_width) & (
            target <= prediction + half_width
        )
        fold_rows.append(
            {
                "outer_fold": int(fold),
                "coverage": float(np.mean(covered)),
                "mean_width": float(np.mean(2 * half_width)),
                "rows": len(evaluation),
            }
        )
        covered_all.append(covered)
        widths_all.append(2 * half_width)
        quartiles_all.append(evaluation["similarity_quartile"].to_numpy(dtype=int))
    covered = np.concatenate(covered_all)
    widths = np.concatenate(widths_all)
    quartiles = np.concatenate(quartiles_all)
    conditional = {}
    for quartile in range(4):
        selected = quartiles == quartile
        conditional[str(quartile)] = {
            "coverage": float(np.mean(covered[selected])),
            "mean_width": float(np.mean(widths[selected])),
            "rows": int(selected.sum()),
        }
    return {
        "target_coverage": float(1 - alpha),
        "pooled_coverage": float(np.mean(covered)),
        "pooled_mean_width": float(np.mean(widths)),
        "similarity_quartiles": conditional,
        "folds": fold_rows,
    }
