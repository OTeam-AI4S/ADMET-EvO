"""Summarize the completed validation-only corrective smoke episode."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from admet_agent.research_process_metrics import extract_research_process_metrics
from admet_agent.self_correction import ValidationSelfCorrectionPlanner


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build(run_dir: Path, historical_run_dir: Path) -> dict:
    smoke_path = run_dir / "corrective_smoke_report.json"
    smoke = json.loads(smoke_path.read_text(encoding="utf-8"))
    if smoke.get("test_loaded") is not False:
        raise ValueError("Corrective smoke report must be validation-only")
    planner = ValidationSelfCorrectionPlanner()
    parent_outcomes = []
    child_outcomes = []
    final_decisions = []
    task_rows = []
    for task_report in smoke["tasks"]:
        task = task_report["task"]
        historical_path = historical_run_dir / "tasks" / task / "search_seed_1.json"
        historical = json.loads(historical_path.read_text(encoding="utf-8"))
        parent = {"task": task, **historical["action_evidence"]}
        child = task_report["action_evidence"]
        parent_outcomes.append(parent)
        child_outcomes.append(child)
        final = planner.decide(
            task_report["research_plan"],
            child,
            attempted_action_ids=(
                parent["action_id"],
                child["action_id"],
            ),
            remaining_action_budget=1,
            sequence_index=3,
        ).to_dict()
        final_decisions.append(final)
        task_rows.append(
            {
                "task": task,
                "parent_action_id": parent["action_id"],
                "parent_status": parent["status"],
                "correction_kind": task_report["research_plan"]["actions"][0][
                    "correction_kind"
                ],
                "child_action_id": child["action_id"],
                "child_status": child["status"],
                "mean_oriented_delta": child["mean_oriented_delta"],
                "positive_delta_rate": child["positive_delta_rate"],
                "single_axis_verified": child["single_axis_verified"],
                "paired_seeds": len(child["validation_seeds"]),
                "post_followup_kind": final["kind"],
                "post_followup_reasons": final["reason_codes"],
                "fits": len(task_report["results"]),
                "fit_seconds": sum(
                    result["elapsed_seconds"] for result in task_report["results"]
                ),
                "test_loaded": False,
            }
        )
    process = extract_research_process_metrics(
        search_summaries=smoke["tasks"],
        action_outcomes=[*parent_outcomes, *child_outcomes],
    ).to_dict()
    completed = sum(
        row["action_evidence"].get("execution_status") == "completed"
        for row in smoke["tasks"]
    )
    evidence_repairs = sum(
        row["correction_kind"] == "repair_evidence" for row in task_rows
    )
    resolved_children = sum(
        row["child_status"] in {"supported", "refuted"} for row in task_rows
    )
    return {
        "protocol": "validation-only-self-correction-smoke-evidence-v1",
        "run_dir": str(run_dir),
        "historical_run_dir": str(historical_run_dir),
        "scheduled_actions": len(smoke["tasks"]),
        "completed_actions": completed,
        "completion_rate": completed / len(smoke["tasks"]),
        "completed_evidence_repairs": evidence_repairs,
        "resolved_child_hypotheses": resolved_children,
        "supported_child_hypotheses": sum(
            row["child_status"] == "supported" for row in task_rows
        ),
        "all_post_followup_stopped": all(
            row["post_followup_kind"] == "stop" for row in task_rows
        ),
        "task_rows": task_rows,
        "process_metrics": process,
        "total_fits": smoke["total_fits"],
        "total_fit_seconds": smoke["total_fit_seconds"],
        "source_sha256": smoke["source_sha256"],
        "evidence_hashes": {
            "corrective_smoke_report": _sha256(smoke_path),
            "evidence_ledger": _sha256(run_dir / "evidence_ledger.jsonl"),
        },
        "test_loaded": False,
        "selection_feedback_to_discovery": False,
        "claim_boundary": (
            "demonstrates completed linked corrective behavior and bounded stopping; "
            "does not demonstrate a supported new predictive hypothesis or an "
            "Agent-over-rules/AutoML advantage"
        ),
    }


def render(payload: dict) -> str:
    self_correction = payload["process_metrics"]["self_correction_rate"]
    rows = [
        "# Prospective validation-only self-correction smoke",
        "",
        f"- Scheduled/completed actions: {payload['completed_actions']}/{payload['scheduled_actions']}",
        f"- Linked behavioral self-correction: {self_correction['numerator']}/{self_correction['denominator']}",
        f"- Completed evidence repairs: {payload['completed_evidence_repairs']}",
        f"- Supported corrective hypotheses: {payload['supported_child_hypotheses']}/2",
        f"- Resolved corrective hypotheses: {payload['resolved_child_hypotheses']}/2",
        f"- Fits: {payload['total_fits']}; fit-seconds: {payload['total_fit_seconds']:.1f}",
        "- Uni-Mol2: 0; tuning trials: 0; ensembles: 0; test loaded: false",
        "",
        "| Endpoint | Parent outcome | Corrective action | Child evidence | Mean oriented delta | Positive seeds | Next |",
        "|---|---|---|---|---:|---:|---|",
    ]
    for row in payload["task_rows"]:
        rows.append(
            f"| {row['task']} | {row['parent_status']} `{row['parent_action_id']}` | "
            f"{row['correction_kind']} | {row['child_status']} | "
            f"{row['mean_oriented_delta']:.6f} | {row['positive_delta_rate']:.3f} | "
            f"{row['post_followup_kind']} |"
        )
    rows.extend(
        (
            "",
            "Interpretation: the system executed both scheduled matched follow-ups and",
            "linked them to the action that created the opportunity. CYP2C9 repaired a",
            "missing model-pair evidence gap; hERG changed from the refuted data axis to",
            "the model axis. Both child hypotheses remained inconclusive, so the bounded",
            "planner stopped instead of escalating to high-cost feature/Uni-Mol2 search.",
            "",
            "This is positive evidence for *completed corrective behavior and stopping*,",
            "not for discovery of a better predictive recipe. It also lacks a matched rules",
            "or AutoML feedback control, so no comparative autonomy claim is made yet.",
            "",
        )
    )
    return "\n".join(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run-dir", type=Path, default=Path("runs/v5-self-correction-smoke")
    )
    parser.add_argument(
        "--historical-run-dir",
        type=Path,
        default=Path("runs/v3-agent-tdc22-v5"),
    )
    parser.add_argument(
        "--output-prefix",
        type=Path,
        default=Path("reports/self_correction_smoke_v1"),
    )
    args = parser.parse_args()
    payload = build(args.run_dir, args.historical_run_dir)
    args.output_prefix.parent.mkdir(parents=True, exist_ok=True)
    args.output_prefix.with_suffix(".json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    args.output_prefix.with_suffix(".md").write_text(render(payload), encoding="utf-8")


if __name__ == "__main__":
    main()
