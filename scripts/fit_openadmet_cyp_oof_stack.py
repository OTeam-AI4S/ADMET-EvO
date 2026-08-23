"""Fit a nested, non-negative OOF stack for frozen OpenADMET CYP components."""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import matthews_corrcoef

from scripts.analyze_openadmet_cyp_cv import _collapse_seeds
from scripts.openadmet_cyp_metrics import official_st_rae

REQUIRED_FOLDS = (0, 1, 2, 3, 4)
REQUIRED_SEEDS = (1, 2, 3)
GRID_DENOMINATOR = 20
KEYS = [
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


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_component(label: str, result_path: Path) -> tuple[pd.DataFrame, dict]:
    result_path = result_path.resolve()
    result = json.loads(result_path.read_text(encoding="utf-8"))
    if result.get("training_only") is not True or result.get("blind_test_opened") is not False:
        raise RuntimeError(f"Component {label} is not training-only")
    if result.get("sealed_certification_opened") is not False:
        raise RuntimeError(f"Component {label} opened sealed certification")
    if tuple(result.get("outer_folds", [])) != REQUIRED_FOLDS:
        raise RuntimeError(f"Component {label} must cover outer folds {REQUIRED_FOLDS}")
    if tuple(result.get("seeds", [])) != REQUIRED_SEEDS:
        raise RuntimeError(f"Component {label} must cover seeds {REQUIRED_SEEDS}")
    prediction_path = Path(result["prediction_path"]).resolve()
    if result["prediction_sha256"] != _sha256(prediction_path):
        raise RuntimeError(f"Component {label} prediction hash mismatch")
    frame = pd.read_csv(prediction_path)
    missing = set(KEYS + ["training_seed", "prediction", "predictive_std"]) - set(frame)
    if missing:
        raise RuntimeError(f"Component {label} predictions missing {sorted(missing)}")
    primary = frame.loc[
        frame["fidelity"].isin({"primary_direct", "primary_classification"})
    ].copy()
    collapsed = _collapse_seeds(primary)
    if set(collapsed["outer_fold_evaluated"].astype(int)) != set(REQUIRED_FOLDS):
        raise RuntimeError(f"Component {label} collapsed OOF coverage is incomplete")
    return collapsed, {
        "result_path": str(result_path),
        "result_sha256": _sha256(result_path),
        "prediction_path": str(prediction_path),
        "prediction_sha256": _sha256(prediction_path),
    }


def _weight_grid(components: int) -> list[np.ndarray]:
    if not 2 <= components <= 4:
        raise ValueError("OOF stack requires two to four frozen components")
    output = []
    for values in itertools.product(range(GRID_DENOMINATOR + 1), repeat=components - 1):
        used = sum(values)
        if used > GRID_DENOMINATOR:
            continue
        weights = (*values, GRID_DENOMINATOR - used)
        output.append(np.asarray(weights, dtype=float) / GRID_DENOMINATOR)
    return output


def _score(frame: pd.DataFrame, prediction: np.ndarray, task_type: str) -> float:
    work = frame.copy()
    work["stack_prediction"] = prediction
    values = []
    for _, group in work.groupby("endpoint_key"):
        if task_type == "classification":
            binary = (group["stack_prediction"].to_numpy() >= 0.5).astype(int)
            values.append(float(matthews_corrcoef(group["target"].astype(int), binary)))
        else:
            value = official_st_rae(
                target=group["target"].to_numpy(),
                prediction=group["stack_prediction"].to_numpy(),
                lower=group["credible_low"].to_numpy(),
                upper=group["credible_high"].to_numpy(),
            )
            if value is not None:
                values.append(float(value))
    if not values:
        raise RuntimeError(f"No scorable {task_type} endpoints in OOF stack")
    return float(np.mean(values))


def _fit_weights(frame: pd.DataFrame, labels: list[str], task_type: str) -> tuple[np.ndarray, float]:
    predictions = frame[[f"prediction__{label}" for label in labels]].to_numpy(dtype=float)
    best_weights = None
    best_score = None
    for weights in _weight_grid(len(labels)):
        score = _score(frame, predictions @ weights, task_type)
        objective = -score if task_type == "classification" else score
        # Threshold metrics can leave a broad plateau. Prefer the least
        # concentrated convex combination on exact ties, then lexical order.
        concentration = float(np.square(weights - 1.0 / len(labels)).sum())
        candidate = (objective, concentration, tuple(-value for value in weights))
        if best_score is None or candidate < best_score:
            best_score = candidate
            best_weights = weights
    assert best_weights is not None
    return best_weights, _score(frame, predictions @ best_weights, task_type)


def _stack_prediction(frame: pd.DataFrame, labels: list[str], weights: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    means = frame[[f"prediction__{label}" for label in labels]].to_numpy(dtype=float)
    variances = np.column_stack(
        [
            np.square(frame[f"predictive_std__{label}"].to_numpy(dtype=float))
            for label in labels
        ]
    )
    variances = np.nan_to_num(variances, nan=0.0, posinf=0.0, neginf=0.0)
    prediction = means @ weights
    second_moment = (variances + np.square(means)) @ weights
    total_variance = np.maximum(0.0, second_moment - np.square(prediction))
    return prediction, np.sqrt(total_variance)


def fit_stack(component_paths: dict[str, Path], output_dir: Path) -> dict[str, Any]:
    labels = sorted(component_paths)
    if not 2 <= len(labels) <= 4:
        raise ValueError("E0 requires two to four frozen component result files")
    merged = None
    manifests = {}
    for label in labels:
        frame, manifest = _load_component(label, component_paths[label])
        selected = frame[KEYS + ["prediction", "predictive_std"]].rename(
            columns={
                "prediction": f"prediction__{label}",
                "predictive_std": f"predictive_std__{label}",
            }
        )
        if merged is None:
            merged = selected
        else:
            before = len(merged)
            merged = merged.merge(selected, on=KEYS, how="inner", validate="one_to_one")
            if len(merged) != before or len(merged) != len(selected):
                raise RuntimeError("Frozen component OOF rows do not align exactly")
        manifests[label] = manifest
    assert merged is not None
    nested_frames = []
    nested_records = []
    final_weights = {}
    for task_type in ("regression", "classification"):
        type_frame = merged.loc[merged["task_type"] == task_type].copy()
        if type_frame.empty:
            raise RuntimeError(f"Missing primary {task_type} observations")
        weights, all_oof_score = _fit_weights(type_frame, labels, task_type)
        final_weights[task_type] = {
            label: float(weight) for label, weight in zip(labels, weights, strict=True)
        }
        final_weights[task_type]["all_oof_optimization_score"] = all_oof_score
        for fold in REQUIRED_FOLDS:
            training = type_frame.loc[type_frame["outer_fold_evaluated"] != fold]
            evaluation = type_frame.loc[type_frame["outer_fold_evaluated"] == fold].copy()
            fold_weights, training_score = _fit_weights(training, labels, task_type)
            prediction, uncertainty = _stack_prediction(evaluation, labels, fold_weights)
            evaluation["prediction"] = prediction
            evaluation["predictive_std"] = uncertainty
            evaluation["stack_task_type"] = task_type
            nested_frames.append(evaluation)
            nested_records.append(
                {
                    "task_type": task_type,
                    "outer_fold": fold,
                    "training_folds": [value for value in REQUIRED_FOLDS if value != fold],
                    "weights": {
                        label: float(weight)
                        for label, weight in zip(labels, fold_weights, strict=True)
                    },
                    "training_score": training_score,
                    "evaluation_score": _score(evaluation, prediction, task_type),
                    "evaluation_rows": len(evaluation),
                }
            )
    nested = pd.concat(nested_frames, ignore_index=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    prediction_path = output_dir / "e0_nested_oof_predictions.csv"
    nested.to_csv(prediction_path, index=False)
    result = {
        "version": "openadmet-cyp-e0-nested-oof-stack-v1",
        "training_only": True,
        "blind_test_opened": False,
        "sealed_certification_opened": False,
        "components": manifests,
        "component_order": labels,
        "required_outer_folds": list(REQUIRED_FOLDS),
        "required_seeds": list(REQUIRED_SEEDS),
        "nonnegative_grid_step": 1 / GRID_DENOMINATOR,
        "separate_weights_by_task_type": True,
        "nested_fold_results": nested_records,
        "final_all_oof_weights_for_future_frozen_refit": final_weights,
        "prediction_path": str(prediction_path.resolve()),
        "prediction_sha256": _sha256(prediction_path),
        "test_labels_loaded": False,
        "leaderboard_feedback_used": False,
    }
    result_path = output_dir / "e0_stack_result.json"
    result_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return {**result, "result_path": str(result_path.resolve())}


def _assign(values: list[str]) -> dict[str, Path]:
    output = {}
    for value in values:
        label, separator, path = value.partition("=")
        if not separator or not label or not path:
            raise ValueError("Components must use LABEL=RESULT_PATH")
        if label in output:
            raise ValueError(f"Duplicate component label {label}")
        output[label] = Path(path)
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--component", action="append", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(fit_stack(_assign(args.component), args.output_dir), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
