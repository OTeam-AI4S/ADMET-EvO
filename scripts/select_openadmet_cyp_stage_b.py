"""Preregistered Stage-B component gate for the OpenADMET CYP recipe."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

REQUIRED_FOLDS = (0, 1, 2, 3, 4)
REQUIRED_SEEDS = (1, 2, 3)
ST_RAE_NONINFERIORITY_MARGIN = 0.05
MCC_NONINFERIORITY_MARGIN = 0.03
WORST_FOLD_ST_RAE_MARGIN = 0.10
WORST_FOLD_MCC_MARGIN = 0.10


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("training_only") is not True:
        raise RuntimeError(f"Stage-B artifact is not training-only: {path}")
    if payload.get("blind_test_opened") is not False:
        raise RuntimeError(f"Stage-B artifact indicates blind-test access: {path}")
    if payload.get("sealed_certification_opened") is not False:
        raise RuntimeError(f"Stage-B artifact indicates certification access: {path}")
    if tuple(payload.get("outer_folds", ())) != REQUIRED_FOLDS:
        raise RuntimeError(f"Stage-B artifact lacks five-fold coverage: {path}")
    if tuple(payload.get("seeds", ())) != REQUIRED_SEEDS:
        raise RuntimeError(f"Stage-B artifact lacks three-seed coverage: {path}")
    return payload


def _metrics(row: dict[str, Any]) -> tuple[float, float]:
    values = row.get("metrics", row)
    return float(values["macro_st_rae"]), float(values["macro_mcc"])


def _fold_means(payload: dict[str, Any]) -> dict[int, tuple[float, float]]:
    records: dict[tuple[int, int], tuple[float, float]] = {}
    for row in payload["fold_seed_results"]:
        key = (int(row["outer_fold"]), int(row["seed"]))
        if key in records:
            raise RuntimeError(f"Duplicate Stage-B fold/seed record: {key}")
        records[key] = _metrics(row)
    expected = {(fold, seed) for fold in REQUIRED_FOLDS for seed in REQUIRED_SEEDS}
    if set(records) != expected:
        raise RuntimeError("Stage-B fold/seed result grid is incomplete")
    return {
        fold: (
            float(np.mean([records[(fold, seed)][0] for seed in REQUIRED_SEEDS])),
            float(np.mean([records[(fold, seed)][1] for seed in REQUIRED_SEEDS])),
        )
        for fold in REQUIRED_FOLDS
    }


def select(baseline_path: Path, candidate_paths: dict[str, Path]) -> dict[str, Any]:
    if not candidate_paths:
        raise ValueError("At least one Stage-B candidate is required")
    baseline_path = baseline_path.resolve()
    baseline = _load(baseline_path)
    baseline_folds = _fold_means(baseline)
    decisions = {}
    selected = ["B1"]
    for label, path in sorted(candidate_paths.items()):
        path = path.resolve()
        candidate = _load(path)
        candidate_folds = _fold_means(candidate)
        st_gain = np.asarray(
            [baseline_folds[fold][0] - candidate_folds[fold][0] for fold in REQUIRED_FOLDS]
        )
        mcc_gain = np.asarray(
            [candidate_folds[fold][1] - baseline_folds[fold][1] for fold in REQUIRED_FOLDS]
        )
        regression_noninferior = float(st_gain.mean()) >= -ST_RAE_NONINFERIORITY_MARGIN
        classification_noninferior = float(mcc_gain.mean()) >= -MCC_NONINFERIORITY_MARGIN
        worst_fold_safe = bool(
            float(st_gain.min()) >= -WORST_FOLD_ST_RAE_MARGIN
            and float(mcc_gain.min()) >= -WORST_FOLD_MCC_MARGIN
        )
        primary_improved = bool(float(st_gain.mean()) > 0 or float(mcc_gain.mean()) > 0)
        admitted = bool(
            regression_noninferior
            and classification_noninferior
            and worst_fold_safe
            and primary_improved
        )
        if admitted:
            selected.append(label)
        decisions[label] = {
            "result_path": str(path),
            "result_sha256": _sha256(path),
            "fold_mean_st_rae_gain_vs_b1": st_gain.tolist(),
            "fold_mean_mcc_gain_vs_b1": mcc_gain.tolist(),
            "mean_st_rae_gain_vs_b1": float(st_gain.mean()),
            "mean_mcc_gain_vs_b1": float(mcc_gain.mean()),
            "worst_fold_st_rae_gain_vs_b1": float(st_gain.min()),
            "worst_fold_mcc_gain_vs_b1": float(mcc_gain.min()),
            "regression_noninferior": regression_noninferior,
            "classification_noninferior": classification_noninferior,
            "worst_fold_safe": worst_fold_safe,
            "primary_improved": primary_improved,
            "admit_as_stack_component": admitted,
            "blind_leaderboard_can_override": False,
        }
    return {
        "protocol": "openadmet-cyp-stage-b-component-gate-v1",
        "training_only": True,
        "blind_test_opened": False,
        "sealed_certification_opened": False,
        "baseline": {
            "label": "B1",
            "path": str(baseline_path),
            "sha256": _sha256(baseline_path),
        },
        "required_outer_folds": list(REQUIRED_FOLDS),
        "required_seeds": list(REQUIRED_SEEDS),
        "statistical_unit_for_gate": "outer_fold_after_seed_averaging",
        "thresholds": {
            "mean_st_rae_noninferiority_margin": ST_RAE_NONINFERIORITY_MARGIN,
            "mean_mcc_noninferiority_margin": MCC_NONINFERIORITY_MARGIN,
            "worst_fold_st_rae_margin": WORST_FOLD_ST_RAE_MARGIN,
            "worst_fold_mcc_margin": WORST_FOLD_MCC_MARGIN,
        },
        "decisions": decisions,
        "selected_components": selected,
        "recipe_mode": "nested_oof_stack" if len(selected) >= 2 else "single_component_B1",
        "leaderboard_feedback_used": False,
    }


def _assign(values: list[str]) -> dict[str, Path]:
    output = {}
    for value in values:
        label, separator, path = value.partition("=")
        if not separator or not label or not path:
            raise ValueError("Candidate assignments must use LABEL=PATH")
        if label in output:
            raise ValueError(f"Duplicate candidate label {label}")
        output[label] = Path(path)
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = select(args.baseline, _assign(args.candidate))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
