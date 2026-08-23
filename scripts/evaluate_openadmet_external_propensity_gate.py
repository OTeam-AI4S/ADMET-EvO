"""Evaluate the frozen OpenADMET/Veith auxiliary propensity promotion gate."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _protocol(path: Path) -> dict[str, Any]:
    value = _load(path)
    claimed = value.get("sha256")
    payload = {key: item for key, item in value.items() if key != "sha256"}
    actual = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
    if claimed != actual:
        raise RuntimeError("External propensity protocol SHA-256 is invalid")
    return value


def _check(name: str, observed: Any, requirement: str, passed: bool) -> dict[str, Any]:
    return {
        "name": name,
        "observed": observed,
        "requirement": requirement,
        "passed": bool(passed),
    }


def evaluate(protocol: dict[str, Any], result: dict[str, Any]) -> dict[str, Any]:
    gate = protocol["promotion_gate"]
    control = result["arm_means"]["b1_control"]
    treatment = result["arm_means"]["b1_external_propensity"]
    mean_gains = {
        "st_rae": float(control["macro_st_rae"] - treatment["macro_st_rae"]),
        "mcc": float(treatment["macro_mcc"] - control["macro_mcc"]),
        "brier": float(control["macro_brier"] - treatment["macro_brier"]),
    }
    fold_rows = result["outer_fold_mean_deltas"]
    observed_schedule = sorted(
        {
            (int(item["outer_fold"]), int(item["seed"]))
            for item in result["fold_seed_results"]
        }
    )
    expected_schedule = [
        (fold, seed) for fold in protocol["outer_folds"] for seed in protocol["seeds"]
    ]
    forbidden_accesses = int(result.get("test_labels_loaded") is not False) + int(
        result.get("leaderboard_feedback_used") is not False
    ) + int(result.get("blind_test_opened") is not False) + int(
        result.get("sealed_certification_opened") is not False
    )
    source_rows = [
        metrics
        for item in result["fold_seed_results"]
        for metrics in item["source_heldout_metrics"].values()
    ]
    source_summary = {
        "mean_auroc": float(np.mean([item["auroc"] for item in source_rows])),
        "mean_average_precision": float(
            np.mean([item["average_precision"] for item in source_rows])
        ),
        "mean_brier": float(np.mean([item["brier"] for item in source_rows])),
    }
    checks = [
        _check(
            "protocol_binding",
            result.get("protocol_sha256"),
            protocol["sha256"],
            result.get("protocol_sha256") == protocol["sha256"]
            and result.get("implementation_source_sha256")
            == protocol["implementation_source_sha256"],
        ),
        _check(
            "complete_fold_seed_schedule",
            observed_schedule,
            str(expected_schedule),
            observed_schedule == expected_schedule,
        ),
        _check(
            "baseline_reproduction",
            result["baseline_reference_max_abs_difference"],
            f"<= {protocol['baseline_reproduction_tolerance']}",
            result["baseline_reference_max_abs_difference"]
            <= protocol["baseline_reproduction_tolerance"],
        ),
        _check(
            "mean_st_rae_gain",
            mean_gains["st_rae"],
            f">= {gate['mean_st_rae_gain_min']}",
            mean_gains["st_rae"] >= gate["mean_st_rae_gain_min"],
        ),
        _check(
            "mean_mcc_gain",
            mean_gains["mcc"],
            f">= {gate['mean_mcc_gain_min']}",
            mean_gains["mcc"] >= gate["mean_mcc_gain_min"],
        ),
        _check(
            "one_primary_metric_improves",
            {"st_rae_gain": mean_gains["st_rae"], "mcc_gain": mean_gains["mcc"]},
            "at least one gain > 0",
            mean_gains["st_rae"] > 0 or mean_gains["mcc"] > 0,
        ),
        _check(
            "worst_fold_st_rae_gain",
            min(item["st_rae_gain"] for item in fold_rows),
            f">= {gate['worst_fold_st_rae_gain_min']}",
            min(item["st_rae_gain"] for item in fold_rows)
            >= gate["worst_fold_st_rae_gain_min"],
        ),
        _check(
            "worst_fold_mcc_gain",
            min(item["mcc_gain"] for item in fold_rows),
            f">= {gate['worst_fold_mcc_gain_min']}",
            min(item["mcc_gain"] for item in fold_rows)
            >= gate["worst_fold_mcc_gain_min"],
        ),
        _check(
            "mean_brier_gain",
            mean_gains["brier"],
            f">= {gate['mean_brier_gain_min']}",
            mean_gains["brier"] >= gate["mean_brier_gain_min"],
        ),
        _check(
            "forbidden_blind_access",
            forbidden_accesses,
            f"<= {gate['test_or_leaderboard_access_max']}",
            forbidden_accesses <= gate["test_or_leaderboard_access_max"],
        ),
    ]
    promoted = all(item["passed"] for item in checks)
    return {
        "version": "openadmet-external-propensity-gate-v1",
        "protocol_sha256": protocol["sha256"],
        "scope": protocol["scope"],
        "mean_gains": mean_gains,
        "source_heldout_summary": source_summary,
        "checks": checks,
        "promotion_passed": promoted,
        "external_data_action_admitted": promoted,
        "blind_test_can_override": False,
        "sealed_certification_can_override": False,
        "test_labels_loaded": False,
        "leaderboard_feedback_used": False,
        "interpretation": (
            "The auxiliary Veith propensity bridge passed the frozen training-only gate."
            if promoted
            else "The auxiliary Veith propensity bridge failed the frozen gate and remains excluded from the blind recipe."
        ),
    }


def render_markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# OpenADMET external propensity gate",
        "",
        f"Protocol: `{payload['protocol_sha256']}`  ",
        f"Promotion passed: **{'yes' if payload['promotion_passed'] else 'no'}**.",
        "",
        "| Check | Observed | Requirement | Passed |",
        "|---|---|---|---:|",
    ]
    for item in payload["checks"]:
        lines.append(
            f"| `{item['name']}` | `{json.dumps(item['observed'], sort_keys=True)}` | "
            f"{item['requirement']} | {'yes' if item['passed'] else 'no'} |"
        )
    lines.extend(["", "## Decision", "", payload["interpretation"], ""])
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--markdown", type=Path)
    args = parser.parse_args()
    protocol = _protocol(args.protocol)
    result = _load(args.result)
    payload = evaluate(protocol, result)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    if args.markdown is not None:
        args.markdown.write_text(render_markdown(payload), encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
