"""Analyze EC-VOI v4 matched states and apply the preregistered mechanism gate."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from admet_agent.ecvoi_v4_policy import (
    evaluate_leave_one_endpoint_out,
    fixed_rule_results,
    seeded_random_results,
)


def _digest(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _load_bound(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    claimed = value.pop("sha256", None)
    if claimed != _digest(value):
        raise ValueError(f"Invalid SHA-256: {path}")
    return {**value, "sha256": claimed}


def _effect_rows(registry: dict, protocol: dict) -> list[dict[str, Any]]:
    records = registry["records"]
    grouped: dict[tuple[str, int, str], dict[str, dict]] = {}
    for record in records.values():
        if record.get("test_loaded") is not False or record.get("leaderboard_loaded") is not False:
            raise ValueError("Forbidden evaluation access in matched-state registry")
        if record["result"]["status"] != "completed":
            raise ValueError("Matched-state analysis requires completed records")
        key = (record["task"], int(record["seed"]), record["state"])
        grouped.setdefault(key, {})[record["action"]] = record
    rows = []
    for (task, seed, state), action_rows in sorted(grouped.items()):
        if set(action_rows) != set(protocol["actions"]):
            raise ValueError(f"Incomplete action grid: {task}/{seed}/{state}")
        baseline = float(action_rows["baseline"]["result"]["validation_score"])
        for action in protocol["actions"]:
            record = action_rows[action]
            rows.append({
                "task": task,
                "seed": seed,
                "state": state,
                "action": action,
                "validation_score": float(record["result"]["validation_score"]),
                "oriented_effect": float(record["result"]["validation_score"]) - baseline,
                "elapsed_seconds": float(record["result"]["elapsed_seconds"]),
                "state_audit": record["state_audit"],
                "profile": record["profile"],
                "run_id": record["result"]["run_id"],
                "data_policy": record["spec"]["data_policy"],
            })
    return rows


def _summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    effects = np.asarray([row["selected_oriented_effect"] for row in rows], dtype=float)
    regrets = np.asarray([row["selection_regret"] for row in rows], dtype=float)
    return {
        "states": len(rows),
        "mean_selected_oriented_effect": float(np.mean(effects)),
        "median_selected_oriented_effect": float(np.median(effects)),
        "mean_selection_regret": float(np.mean(regrets)),
        "harmful_interventions": int(sum(bool(row["harmful_intervention"]) for row in rows)),
        "harmful_intervention_rate": float(np.mean([bool(row["harmful_intervention"]) for row in rows])),
        "stop_rate": float(np.mean([row["selected_action"] == "baseline" for row in rows])),
        "selected_actions": {
            action: sum(row["selected_action"] == action for row in rows)
            for action in ("baseline", "feature_rich", "imbalance_aware")
        },
    }


def analyze(registry: dict, protocol: dict, analysis_protocol: dict) -> dict:
    if registry.get("protocol_sha256") != protocol["sha256"]:
        raise ValueError("Registry/protocol binding mismatch")
    rows = _effect_rows(registry, protocol)
    options = analysis_protocol["policy_options"]
    full = evaluate_leave_one_endpoint_out(rows, diagnosis_enabled=True, **options)
    no_diagnosis = evaluate_leave_one_endpoint_out(rows, diagnosis_enabled=False, **options)
    shuffled = evaluate_leave_one_endpoint_out(
        rows,
        diagnosis_enabled=True,
        shuffled_diagnosis=True,
        shuffle_seed=analysis_protocol["shuffle_seed"],
        **options,
    )
    rules = fixed_rule_results(rows, harm_margin=options["harm_margin"])
    random = seeded_random_results(
        rows,
        seed=analysis_protocol["random_seed"],
        harm_margin=options["harm_margin"],
    )
    stop = [
        {
            "task": task,
            "state": state,
            "selected_action": "baseline",
            "selected_oriented_effect": 0.0,
            "selection_regret": max(
                row["oriented_effect"]
                for row in rows
                if row["task"] == task and row["state"] == state
            ),
            "harmful_intervention": False,
        }
        for task, state in sorted({(row["task"], row["state"]) for row in rows})
    ]
    policies = {
        "full_v4": full,
        "no_diagnosis": no_diagnosis,
        "shuffled_diagnosis": shuffled,
        "fixed_rules": rules,
        "seeded_random": random,
        "stop_baseline": stop,
    }
    summaries = {name: _summary(value) for name, value in policies.items()}
    checks = [
        {
            "name": "target_endpoint_firewall",
            "observed": max(row["target_endpoint_training_rows"] for row in full),
            "requirement": "exactly 0",
            "passed": all(row["target_endpoint_training_rows"] == 0 for row in full),
        },
        {
            "name": "full_nonnegative_mean_effect",
            "observed": summaries["full_v4"]["mean_selected_oriented_effect"],
            "requirement": ">= 0",
            "passed": summaries["full_v4"]["mean_selected_oriented_effect"] >= 0,
        },
        {
            "name": "full_harm_no_worse_than_no_diagnosis_and_rules",
            "observed": {
                name: summaries[name]["harmful_intervention_rate"]
                for name in ("full_v4", "no_diagnosis", "fixed_rules")
            },
            "requirement": "full <= both controls",
            "passed": summaries["full_v4"]["harmful_intervention_rate"] <= min(
                summaries["no_diagnosis"]["harmful_intervention_rate"],
                summaries["fixed_rules"]["harmful_intervention_rate"],
            ),
        },
        {
            "name": "full_regret_below_random",
            "observed": {
                "full": summaries["full_v4"]["mean_selection_regret"],
                "random": summaries["seeded_random"]["mean_selection_regret"],
            },
            "requirement": "full < random",
            "passed": summaries["full_v4"]["mean_selection_regret"] < summaries["seeded_random"]["mean_selection_regret"],
        },
        {
            "name": "diagnosis_identifiable_against_shuffle",
            "observed": {
                "full_regret": summaries["full_v4"]["mean_selection_regret"],
                "shuffled_regret": summaries["shuffled_diagnosis"]["mean_selection_regret"],
            },
            "requirement": "full regret < shuffled regret",
            "passed": summaries["full_v4"]["mean_selection_regret"] < summaries["shuffled_diagnosis"]["mean_selection_regret"],
        },
    ]
    complete_seeds = sorted({row["seed"] for row in rows})
    required_final_seeds = sorted(protocol["screen_seeds"] + protocol["confirmation_seeds"])
    evidence_stage = "full_development" if complete_seeds == required_final_seeds else "screen_only"
    body = {
        "version": "ecvoi-v4-matched-state-analysis-v1",
        "protocol_sha256": protocol["sha256"],
        "analysis_protocol_sha256": analysis_protocol["sha256"],
        "evidence_stage": evidence_stage,
        "seeds": complete_seeds,
        "tasks": sorted({row["task"] for row in rows}),
        "states": sorted({row["state"] for row in rows}),
        "effect_rows": rows,
        "policy_results": policies,
        "policy_summary": summaries,
        "checks": checks,
        "mechanism_gate_passed": evidence_stage == "full_development" and all(check["passed"] for check in checks),
        "confirmation_19_unlocked": evidence_stage == "full_development" and all(check["passed"] for check in checks),
        "test_loaded": False,
        "leaderboard_loaded": False,
        "interpretation_guard": "Development matched states test mechanism identifiability; they do not establish TDC-19 or blind superiority.",
    }
    return {**body, "sha256": _digest(body)}


def _markdown(result: dict) -> str:
    lines = [
        "# EC-VOI v4 matched-state analysis",
        "",
        f"Evidence stage: **{result['evidence_stage']}**",
        f"Mechanism gate: **{'PASS' if result['mechanism_gate_passed'] else 'FAIL / NOT YET ELIGIBLE'}**",
        f"TDC-19 unlocked: **{result['confirmation_19_unlocked']}**",
        "",
        "| Policy | Mean effect | Mean regret | Harm rate | Stop rate |",
        "|---|---:|---:|---:|---:|",
    ]
    for name, row in result["policy_summary"].items():
        lines.append(
            f"| {name} | {row['mean_selected_oriented_effect']:.6f} | "
            f"{row['mean_selection_regret']:.6f} | {row['harmful_intervention_rate']:.3f} | {row['stop_rate']:.3f} |"
        )
    lines.extend(["", "## Gate checks", ""])
    for check in result["checks"]:
        lines.append(f"- {'PASS' if check['passed'] else 'FAIL'} — `{check['name']}`: {check['observed']}")
    lines.extend(["", result["interpretation_guard"], ""])
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--analysis-protocol", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--markdown", type=Path, required=True)
    args = parser.parse_args()
    registry = json.loads(args.registry.read_text(encoding="utf-8"))
    protocol = _load_bound(args.protocol)
    analysis_protocol = _load_bound(args.analysis_protocol)
    result = analyze(registry, protocol, analysis_protocol)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    args.markdown.write_text(_markdown(result), encoding="utf-8")
    print(json.dumps({
        "evidence_stage": result["evidence_stage"],
        "mechanism_gate_passed": result["mechanism_gate_passed"],
        "confirmation_19_unlocked": result["confirmation_19_unlocked"],
        "policy_summary": result["policy_summary"],
        "checks": result["checks"],
        "sha256": result["sha256"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
