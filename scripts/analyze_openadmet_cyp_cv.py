"""Aggregate OpenADMET CYP OOF metrics, calibration and selective risk."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import brier_score_loss, matthews_corrcoef

from scripts.openadmet_cyp_metrics import official_st_rae

PRIMARY_FIDELITIES = {"primary_direct", "primary_classification"}
COVERAGES = (0.5, 0.6, 0.7, 0.8, 0.9, 1.0)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _collapse_seeds(frame: pd.DataFrame) -> pd.DataFrame:
    keys = [
        "inchikey",
        "endpoint_key",
        "source_table",
        "source_row",
        "outer_fold_evaluated",
        "chemistry_group",
        "fidelity",
        "task_type",
        "target",
        "credible_low",
        "credible_high",
    ]
    rows = []
    for values, group in frame.groupby(keys, dropna=False, sort=True):
        row = dict(zip(keys, values, strict=True))
        prediction = group["prediction"].to_numpy(dtype=float)
        reported_variance = np.square(
            group["predictive_std"].to_numpy(dtype=float)
        )
        finite_reported = reported_variance[np.isfinite(reported_variance)]
        seed_variance = float(np.var(prediction, ddof=1)) if len(prediction) > 1 else 0.0
        if len(finite_reported) or len(prediction) > 1:
            total_variance = (
                float(np.mean(finite_reported)) if len(finite_reported) else 0.0
            ) + seed_variance
            predictive_std = math.sqrt(max(total_variance, 0.0))
        else:
            predictive_std = float("nan")
        rows.append(
            {
                **row,
                "prediction": float(np.mean(prediction)),
                "predictive_std": predictive_std,
                "seeds": int(group["training_seed"].nunique()),
            }
        )
    return pd.DataFrame(rows)


def _ece(target: np.ndarray, probability: np.ndarray, bins: int = 10) -> float:
    boundaries = np.linspace(0.0, 1.0, bins + 1)
    total = len(target)
    value = 0.0
    for index in range(bins):
        left, right = boundaries[index], boundaries[index + 1]
        selected = (probability >= left) & (
            probability <= right if index == bins - 1 else probability < right
        )
        if selected.any():
            value += selected.mean() * abs(
                float(target[selected].mean()) - float(probability[selected].mean())
            )
    return float(value if total else np.nan)


def _endpoint_metrics(frame: pd.DataFrame) -> dict:
    if frame["task_type"].iloc[0] == "classification":
        target = frame["target"].to_numpy(dtype=int)
        probability = frame["prediction"].to_numpy(dtype=float)
        binary = (probability >= 0.5).astype(int)
        return {
            "mcc": float(matthews_corrcoef(target, binary)),
            "brier": float(brier_score_loss(target, probability)),
            "ece_10": _ece(target, probability),
            "rows": len(frame),
        }
    target = frame["target"].to_numpy(dtype=float)
    prediction = frame["prediction"].to_numpy(dtype=float)
    correlation = spearmanr(target, prediction).statistic
    return {
        "st_rae": official_st_rae(
            target=target,
            prediction=prediction,
            lower=frame["credible_low"].to_numpy(dtype=float),
            upper=frame["credible_high"].to_numpy(dtype=float),
        ),
        "mae": float(np.mean(np.abs(target - prediction))),
        "spearman": float(0.0 if np.isnan(correlation) else correlation),
        "rows": len(frame),
    }


def _cluster_bootstrap(
    frame: pd.DataFrame, *, samples: int, seed: int
) -> dict[str, dict]:
    if samples <= 0:
        raise ValueError("Cluster bootstrap samples must be positive")
    generator = np.random.default_rng(seed)
    output = {}
    for endpoint, endpoint_frame in frame.groupby("endpoint_key"):
        endpoint_frame = endpoint_frame.reset_index(drop=True)
        groups = sorted(endpoint_frame["chemistry_group"].unique())
        group_index = {group: index for index, group in enumerate(groups)}
        row_group_codes = endpoint_frame["chemistry_group"].map(group_index).to_numpy()
        row_positions = np.arange(len(endpoint_frame))
        values: dict[str, list[float]] = {}
        for _ in range(samples):
            # Sampling whole groups with replacement is exactly represented by
            # repeating each row by the draw count of its group. This avoids
            # concatenating O(number_of_groups) DataFrames per replicate.
            selected = generator.integers(0, len(groups), size=len(groups))
            group_counts = np.bincount(selected, minlength=len(groups))
            repeated_positions = np.repeat(
                row_positions, group_counts[row_group_codes]
            )
            resample = endpoint_frame.iloc[repeated_positions]
            for name, value in _endpoint_metrics(resample).items():
                if name == "rows" or value is None:
                    continue
                values.setdefault(name, []).append(float(value))
        output[endpoint] = {
            name: {
                "mean": float(np.mean(metric_values)),
                "ci95": [
                    float(np.quantile(metric_values, 0.025)),
                    float(np.quantile(metric_values, 0.975)),
                ],
            }
            for name, metric_values in values.items()
        }
    return output


def _selective_risk(frame: pd.DataFrame) -> list[dict]:
    classification = frame["task_type"].iloc[0] == "classification"
    if classification:
        confidence = np.abs(frame["prediction"].to_numpy(dtype=float) - 0.5)
        order = np.argsort(-confidence)
    else:
        uncertainty = frame["predictive_std"].to_numpy(dtype=float)
        if not np.isfinite(uncertainty).all():
            return []
        order = np.argsort(uncertainty)
    output = []
    for coverage in COVERAGES:
        count = max(1, math.ceil(len(frame) * coverage))
        selected = frame.iloc[order[:count]]
        metrics = _endpoint_metrics(selected)
        primary_name = "mcc" if classification else "st_rae"
        output.append(
            {
                "coverage": coverage,
                "rows": count,
                "primary_metric": primary_name,
                "primary_value": metrics[primary_name],
            }
        )
    return output


def _cross_conformal(frame: pd.DataFrame, alpha: float = 0.1) -> dict | None:
    if frame["task_type"].iloc[0] == "classification":
        return None
    if not np.isfinite(frame["predictive_std"]).all():
        return None
    fold_results = []
    for fold in sorted(frame["outer_fold_evaluated"].unique()):
        calibration = frame.loc[frame["outer_fold_evaluated"] != fold]
        evaluation = frame.loc[frame["outer_fold_evaluated"] == fold]
        if calibration.empty or evaluation.empty:
            continue
        scale = calibration["predictive_std"].to_numpy(dtype=float).clip(min=1e-6)
        scores = np.abs(calibration["target"] - calibration["prediction"]) / scale
        quantile_level = min(1.0, math.ceil((len(scores) + 1) * (1 - alpha)) / len(scores))
        quantile = float(np.quantile(scores, quantile_level, method="higher"))
        half_width = quantile * evaluation["predictive_std"].to_numpy(dtype=float)
        target = evaluation["target"].to_numpy(dtype=float)
        prediction = evaluation["prediction"].to_numpy(dtype=float)
        covered = (target >= prediction - half_width) & (target <= prediction + half_width)
        fold_results.append(
            {
                "outer_fold": int(fold),
                "coverage": float(covered.mean()),
                "mean_width": float((2 * half_width).mean()),
                "calibration_rows": len(calibration),
                "evaluation_rows": len(evaluation),
            }
        )
    if not fold_results:
        return None
    total = sum(item["evaluation_rows"] for item in fold_results)
    return {
        "target_coverage": 1 - alpha,
        "pooled_coverage": float(
            sum(item["coverage"] * item["evaluation_rows"] for item in fold_results)
            / total
        ),
        "pooled_mean_width": float(
            sum(item["mean_width"] * item["evaluation_rows"] for item in fold_results)
            / total
        ),
        "folds": fold_results,
    }


def analyze(label: str, result_path: Path, *, bootstrap_samples: int, seed: int) -> dict:
    result_path = result_path.resolve()
    result = json.loads(result_path.read_text())
    if result.get("training_only") is not True or result.get("blind_test_opened") is not False:
        raise RuntimeError("Result is not a training-only blind-safe artifact")
    prediction_path = Path(result["prediction_path"]).resolve()
    if result["prediction_sha256"] != _sha256(prediction_path):
        raise RuntimeError("Prediction hash mismatch")
    predictions = pd.read_csv(prediction_path)
    predictions = predictions.loc[
        predictions["fidelity"].isin(PRIMARY_FIDELITIES)
    ].copy()
    collapsed = _collapse_seeds(predictions)
    endpoints = {
        endpoint: _endpoint_metrics(frame)
        for endpoint, frame in collapsed.groupby("endpoint_key")
    }
    return {
        "label": label,
        "result_path": str(result_path),
        "result_sha256": _sha256(result_path),
        "prediction_path": str(prediction_path),
        "prediction_sha256": _sha256(prediction_path),
        "collapsed_rows": len(collapsed),
        "endpoints": endpoints,
        "cluster_bootstrap": _cluster_bootstrap(
            collapsed, samples=bootstrap_samples, seed=seed
        ),
        "selective_risk": {
            endpoint: _selective_risk(frame)
            for endpoint, frame in collapsed.groupby("endpoint_key")
        },
        "cross_conformal_90": {
            endpoint: value
            for endpoint, frame in collapsed.groupby("endpoint_key")
            if (value := _cross_conformal(frame)) is not None
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--result",
        action="append",
        required=True,
        help="label=path/to/cv_result.json; repeat for multiple frozen models",
    )
    parser.add_argument("--bootstrap-samples", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20260821)
    parser.add_argument(
        "--output", type=Path, default=Path("reports/openadmet_cyp_cv_analysis_v1.json")
    )
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    models = []
    for item in arguments.result:
        label, raw_path = item.split("=", 1)
        models.append(
            analyze(
                label,
                Path(raw_path),
                bootstrap_samples=arguments.bootstrap_samples,
                seed=arguments.seed,
            )
        )
    payload = {
        "version": "openadmet-cyp-cv-analysis-v1",
        "training_only": True,
        "blind_test_opened": False,
        "bootstrap_unit": "chemistry_group",
        "bootstrap_samples": arguments.bootstrap_samples,
        "seed": arguments.seed,
        "models": models,
        "test_labels_loaded": False,
        "leaderboard_feedback_used": False,
    }
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True))
