"""Prospectively score cost-constrained v4 action allocation on fresh seeds."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from admet_agent.ecvoi_v4_budget_policy import (
    allocate_actions,
    oracle_allocation,
    score_allocation,
)
from admet_agent.ecvoi_v4_policy import evaluate_leave_one_endpoint_out
from scripts.analyze_ecvoi_v4_matched_states import _effect_rows


def _digest(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _load_bound(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    claimed = value.pop("sha256", None)
    if claimed != _digest(value):
        raise ValueError(f"Invalid SHA-256: {path}")
    return {**value, "sha256": claimed}


def analyze(screen_registry: dict, confirmation_registry: dict, matched_protocol: dict, allocation_protocol: dict) -> dict:
    screen_rows = _effect_rows(screen_registry, matched_protocol)
    all_rows = _effect_rows(confirmation_registry, matched_protocol)
    confirmation_seeds = set(matched_protocol["confirmation_seeds"])
    confirmation_rows = [row for row in all_rows if row["seed"] in confirmation_seeds]
    if {row["seed"] for row in confirmation_rows} != confirmation_seeds:
        raise ValueError("Fresh-seed confirmation grid is incomplete")
    options = allocation_protocol["policy_options"]
    predictions = {
        "full_v4": evaluate_leave_one_endpoint_out(screen_rows, diagnosis_enabled=True, **options),
        "no_diagnosis": evaluate_leave_one_endpoint_out(screen_rows, diagnosis_enabled=False, **options),
        "shuffled_diagnosis": evaluate_leave_one_endpoint_out(
            screen_rows,
            diagnosis_enabled=True,
            shuffled_diagnosis=True,
            shuffle_seed=allocation_protocol["shuffle_seed"],
            **options,
        ),
    }
    actual: dict[str, dict[str, dict[str, float]]] = {}
    for task in matched_protocol["tasks"]:
        actual[task] = {}
        for state in matched_protocol["states"]:
            actual[task][state] = {}
            for action in matched_protocol["actions"]:
                values = [
                    row["oriented_effect"]
                    for row in confirmation_rows
                    if row["task"] == task and row["state"] == state and row["action"] == action
                ]
                if len(values) != len(confirmation_seeds):
                    raise ValueError(f"Incomplete confirmation effects: {task}/{state}/{action}")
                actual[task][state][action] = float(np.mean(values))

    results = {name: [] for name in (*predictions, "oracle")}
    for task in matched_protocol["tasks"]:
        policy_maps = {}
        for name, rows in predictions.items():
            task_rows = [row for row in rows if row["task"] == task]
            policy_maps[name] = {row["state"]: row["predictions"] for row in task_rows}
        feature_all_cost = sum(
            policy_maps["full_v4"][state]["feature_rich"]["cost_ratio"]
            for state in matched_protocol["states"]
        )
        for fraction in allocation_protocol["budget_fractions"]:
            budget = fraction * feature_all_cost
            for name, state_predictions in policy_maps.items():
                allocation = allocate_actions(state_predictions, budget=budget)
                scored = score_allocation(
                    allocation,
                    actual[task],
                    state_predictions,
                    harm_margin=options["harm_margin"],
                )
                results[name].append({"task": task, "budget_fraction": fraction, **scored})
            oracle = oracle_allocation(
                policy_maps["full_v4"], actual[task], budget=budget
            )
            results["oracle"].append({
                "task": task,
                "budget_fraction": fraction,
                **score_allocation(
                    oracle,
                    actual[task],
                    policy_maps["full_v4"],
                    harm_margin=options["harm_margin"],
                ),
            })

    primary = allocation_protocol["primary_budget_fraction"]
    summary = {}
    for name, rows in results.items():
        selected = [row for row in rows if row["budget_fraction"] == primary]
        summary[name] = {
            "endpoints": len(selected),
            "mean_total_oriented_effect": float(np.mean([row["total_oriented_effect"] for row in selected])),
            "mean_state_effect": float(np.mean([row["mean_oriented_effect"] for row in selected])),
            "harmful_interventions": int(sum(row["harmful_interventions"] for row in selected)),
            "mean_normalized_cost": float(np.mean([row["realized_normalized_cost"] for row in selected])),
        }
    oracle_gain = summary["oracle"]["mean_total_oriented_effect"]
    for name in predictions:
        summary[name]["mean_oracle_regret"] = oracle_gain - summary[name]["mean_total_oriented_effect"]
    checks = [
        {
            "name": "full_gain_exceeds_shuffled",
            "observed": {"full": summary["full_v4"]["mean_total_oriented_effect"], "shuffled": summary["shuffled_diagnosis"]["mean_total_oriented_effect"]},
            "passed": summary["full_v4"]["mean_total_oriented_effect"] > summary["shuffled_diagnosis"]["mean_total_oriented_effect"],
        },
        {
            "name": "full_regret_below_no_diagnosis",
            "observed": {"full": summary["full_v4"]["mean_oracle_regret"], "no_diagnosis": summary["no_diagnosis"]["mean_oracle_regret"]},
            "passed": summary["full_v4"]["mean_oracle_regret"] < summary["no_diagnosis"]["mean_oracle_regret"],
        },
        {
            "name": "full_nonnegative_fresh_seed_gain",
            "observed": summary["full_v4"]["mean_total_oriented_effect"],
            "passed": summary["full_v4"]["mean_total_oriented_effect"] >= 0,
        },
        {
            "name": "full_zero_harm",
            "observed": summary["full_v4"]["harmful_interventions"],
            "passed": summary["full_v4"]["harmful_interventions"] == 0,
        },
    ]
    body = {
        "version": "ecvoi-v4-prospective-budget-allocation-analysis-v1",
        "matched_protocol_sha256": matched_protocol["sha256"],
        "allocation_protocol_sha256": allocation_protocol["sha256"],
        "training_scope": "screen_seed_17_other_endpoints_only",
        "evaluation_scope": "fresh_confirmation_seeds_29_43",
        "primary_budget_fraction": primary,
        "results": results,
        "primary_summary": summary,
        "checks": checks,
        "budget_gate_passed": all(check["passed"] for check in checks),
        "test_loaded": False,
        "leaderboard_loaded": False,
        "interpretation_guard": "Fresh split seeds test prospective budget allocation on the same three development endpoints; they do not unlock TDC-19 without the separate full mechanism gate.",
    }
    return {**body, "sha256": _digest(body)}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--screen-registry", type=Path, required=True)
    parser.add_argument("--confirmation-registry", type=Path, required=True)
    parser.add_argument("--matched-protocol", type=Path, required=True)
    parser.add_argument("--allocation-protocol", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--markdown", type=Path, required=True)
    args = parser.parse_args()
    result = analyze(
        json.loads(args.screen_registry.read_text()),
        json.loads(args.confirmation_registry.read_text()),
        _load_bound(args.matched_protocol),
        _load_bound(args.allocation_protocol),
    )
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    lines = [
        "# EC-VOI v4 prospective budget allocation",
        "",
        f"Primary budget fraction: **{result['primary_budget_fraction']:.2f}**",
        f"Budget gate: **{'PASS' if result['budget_gate_passed'] else 'FAIL'}**",
        "",
        "| Policy | Total effect | Oracle regret | Harm | Normalized cost |",
        "|---|---:|---:|---:|---:|",
    ]
    for name, row in result["primary_summary"].items():
        lines.append(
            f"| {name} | {row['mean_total_oriented_effect']:.6f} | "
            f"{row.get('mean_oracle_regret', 0.0):.6f} | {row['harmful_interventions']} | {row['mean_normalized_cost']:.3f} |"
        )
    lines.extend(["", "## Checks", ""])
    lines.extend(f"- {'PASS' if row['passed'] else 'FAIL'} — `{row['name']}`: {row['observed']}" for row in result["checks"])
    lines.extend(["", result["interpretation_guard"], ""])
    args.markdown.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"primary_summary": result["primary_summary"], "checks": result["checks"], "budget_gate_passed": result["budget_gate_passed"], "sha256": result["sha256"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
