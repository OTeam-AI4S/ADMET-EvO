"""Apply the frozen OpenADMET CYP Stage-A promotion gate."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

ST_RAE_NONINFERIORITY_MARGIN = 0.05
MCC_NONINFERIORITY_MARGIN = 0.03
BRIER_IMPROVEMENT = 0.01
ECE_IMPROVEMENT = 0.02
REQUIRED_FOLDS = (0, 1, 2)
REQUIRED_SEEDS = (1,)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_result(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("training_only") is not True or payload.get("blind_test_opened") is not False:
        raise RuntimeError(f"Stage-A result is not training-only: {path}")
    if payload.get("sealed_certification_opened") is not False:
        raise RuntimeError(f"Stage-A result opened sealed certification: {path}")
    if tuple(payload.get("outer_folds", [])) != REQUIRED_FOLDS:
        raise RuntimeError(f"Stage-A result must use folds {REQUIRED_FOLDS}: {path}")
    if tuple(payload.get("seeds", [])) != REQUIRED_SEEDS:
        raise RuntimeError(f"Stage-A result must use seeds {REQUIRED_SEEDS}: {path}")
    return payload


def _record_map(payload: dict[str, Any]) -> dict[tuple[int, int], dict]:
    output = {
        (int(row["outer_fold"]), int(row["seed"])): row
        for row in payload["fold_seed_results"]
    }
    expected = {(fold, seed) for fold in REQUIRED_FOLDS for seed in REQUIRED_SEEDS}
    if set(output) != expected:
        raise RuntimeError("Stage-A fold/seed record grid is incomplete")
    return output


def _macro_classification_uq(payload: dict[str, Any]) -> dict[str, float] | None:
    endpoints = payload.get("endpoints", {})
    rows = [
        value
        for value in endpoints.values()
        if isinstance(value, dict) and "brier" in value and "ece_10" in value
    ]
    if not rows:
        return None
    return {
        "brier": float(np.mean([float(row["brier"]) for row in rows])),
        "ece_10": float(np.mean([float(row["ece_10"]) for row in rows])),
    }


def _analysis_for_label(payload: dict[str, Any], label: str) -> dict[str, Any]:
    """Accept either a single-model analysis or the analyzer's multi-model file."""

    if "endpoints" in payload:
        recorded = payload.get("label")
        if recorded not in {None, label}:
            raise RuntimeError(
                f"Stage-A analysis label mismatch: expected {label}, found {recorded}"
            )
        return payload
    models = payload.get("models")
    if not isinstance(models, list):
        raise TypeError("Stage-A analysis contains neither endpoints nor models")
    matches = [row for row in models if row.get("label") == label]
    if len(matches) != 1:
        raise RuntimeError(
            f"Stage-A analysis must contain exactly one model labeled {label}"
        )
    return matches[0]


def _uq_gate(
    baseline: dict[str, Any] | None,
    candidate: dict[str, Any] | None,
) -> dict[str, Any]:
    baseline_uq = _macro_classification_uq(baseline or {})
    candidate_uq = _macro_classification_uq(candidate or {})
    if baseline_uq is None or candidate_uq is None:
        return {"available": False, "passed": False, "reason": "paired_classification_uq_missing"}
    brier_gain = baseline_uq["brier"] - candidate_uq["brier"]
    ece_gain = baseline_uq["ece_10"] - candidate_uq["ece_10"]
    return {
        "available": True,
        "baseline": baseline_uq,
        "candidate": candidate_uq,
        "brier_gain": brier_gain,
        "ece_10_gain": ece_gain,
        "passed": brier_gain >= BRIER_IMPROVEMENT or ece_gain >= ECE_IMPROVEMENT,
    }


def promote(
    baseline_path: Path,
    candidate_paths: dict[str, Path],
    *,
    analysis_paths: dict[str, Path] | None = None,
) -> dict[str, Any]:
    if not candidate_paths:
        raise ValueError("At least one Stage-A candidate is required")
    baseline_path = baseline_path.resolve()
    baseline = _load_result(baseline_path)
    baseline_records = _record_map(baseline)
    analyses = analysis_paths or {}
    baseline_analysis = None
    if "B1" in analyses:
        baseline_analysis = _analysis_for_label(
            json.loads(analyses["B1"].read_text(encoding="utf-8")), "B1"
        )
    decisions = {}
    for label, source_path in sorted(candidate_paths.items()):
        source_path = source_path.resolve()
        candidate = _load_result(source_path)
        records = _record_map(candidate)
        st_rae_gain = np.asarray(
            [
                float(baseline_records[key]["macro_st_rae"])
                - float(records[key]["metrics"]["macro_st_rae"] if "metrics" in records[key] else records[key]["macro_st_rae"])
                for key in sorted(records)
            ]
        )
        mcc_gain = np.asarray(
            [
                float(records[key]["metrics"]["macro_mcc"] if "metrics" in records[key] else records[key]["macro_mcc"])
                - float(baseline_records[key]["macro_mcc"])
                for key in sorted(records)
            ]
        )
        candidate_analysis = (
            _analysis_for_label(
                json.loads(analyses[label].read_text(encoding="utf-8")), label
            )
            if label in analyses
            else None
        )
        uq = _uq_gate(baseline_analysis, candidate_analysis)
        regression_noninferior = float(st_rae_gain.mean()) >= -ST_RAE_NONINFERIORITY_MARGIN
        classification_noninferior = float(mcc_gain.mean()) >= -MCC_NONINFERIORITY_MARGIN
        primary_improved = float(st_rae_gain.mean()) > 0 or float(mcc_gain.mean()) > 0
        passed = regression_noninferior and classification_noninferior and (
            primary_improved or uq["passed"]
        )
        decisions[label] = {
            "result_path": str(source_path),
            "result_sha256": _sha256(source_path),
            "mean_st_rae_gain_vs_b1": float(st_rae_gain.mean()),
            "mean_mcc_gain_vs_b1": float(mcc_gain.mean()),
            "regression_noninferior": regression_noninferior,
            "classification_noninferior": classification_noninferior,
            "primary_improved": primary_improved,
            "uq_gate": uq,
            "promote_to_stage_b": passed,
            "blind_leaderboard_can_override": False,
        }
    return {
        "protocol": "openadmet_cyp_stage_a_promotion_v1",
        "training_only": True,
        "blind_test_opened": False,
        "sealed_certification_opened": False,
        "baseline": {"label": "B1", "path": str(baseline_path), "sha256": _sha256(baseline_path)},
        "required_outer_folds": list(REQUIRED_FOLDS),
        "required_seeds": list(REQUIRED_SEEDS),
        "thresholds": {
            "st_rae_noninferiority_margin": ST_RAE_NONINFERIORITY_MARGIN,
            "mcc_noninferiority_margin": MCC_NONINFERIORITY_MARGIN,
            "brier_improvement": BRIER_IMPROVEMENT,
            "ece_10_improvement": ECE_IMPROVEMENT,
        },
        "decisions": decisions,
    }


def _assign(values: list[str]) -> dict[str, Path]:
    output = {}
    for value in values:
        label, separator, path = value.partition("=")
        if not separator or not label or not path:
            raise ValueError("Assignments must use LABEL=PATH")
        if label in output:
            raise ValueError(f"Duplicate assignment label {label}")
        output[label] = Path(path).resolve()
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", action="append", required=True)
    parser.add_argument("--analysis", action="append", default=[])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = promote(
        args.baseline,
        _assign(args.candidate),
        analysis_paths=_assign(args.analysis),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
