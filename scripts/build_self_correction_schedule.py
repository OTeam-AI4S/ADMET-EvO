"""Build a minimal validation-only follow-up queue from historical outcomes."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

from admet_agent.self_correction import ValidationSelfCorrectionPlanner


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build(run_dir: Path) -> dict:
    paths = sorted(run_dir.glob("tasks/*/search_seed_*.json"))
    if not paths:
        raise ValueError(f"No search summaries found under {run_dir}")
    planner = ValidationSelfCorrectionPlanner()
    decisions = []
    for path in paths:
        summary = json.loads(path.read_text(encoding="utf-8"))
        decision = planner.decide(
            summary["research_plan"],
            summary["action_evidence"],
            attempted_action_ids=(summary["action_evidence"]["action_id"],),
            remaining_action_budget=1,
            sequence_index=2,
        ).to_dict()
        decisions.append(
            {
                "task": summary["profile"]["task"],
                "source_summary": str(path),
                "trigger_action_id": decision["trigger_action_id"],
                "trigger_status": decision["trigger_status"],
                "decision_kind": decision["kind"],
                "selected_action_id": decision["selected_action_id"],
                "selected_axis": decision["selected_axis"],
                "estimated_additional_fits": decision["estimated_additional_fits"],
                "execution_status": decision["execution_status"],
                "reason_codes": decision["reason_codes"],
                "decision_sha256": decision["decision_sha256"],
                "next_plan": decision["next_plan"],
                "test_loaded": False,
            }
        )
    counts = Counter(row["decision_kind"] for row in decisions)
    status_counts = Counter(row["trigger_status"] for row in decisions)
    scheduled = [row for row in decisions if row["execution_status"] == "scheduled"]
    return {
        "protocol": "retrospective-validation-self-correction-schedule-v1",
        "run_dir": str(run_dir),
        "task_count": len(decisions),
        "trigger_status_counts": dict(sorted(status_counts.items())),
        "decision_counts": dict(sorted(counts.items())),
        "scheduled_task_count": len(scheduled),
        "estimated_additional_fits": sum(
            row["estimated_additional_fits"] for row in scheduled
        ),
        "scheduled_tasks": [row["task"] for row in scheduled],
        "decisions": decisions,
        "source_hashes": {str(path): _sha256(path) for path in paths},
        "maximum_action_compute_cost": planner.maximum_action_compute_cost,
        "test_loaded": False,
        "training_started": False,
        "counts_as_completed_self_correction": False,
        "evidence_scope": (
            "retrospective scheduling only; follow-up matched experiments are required"
        ),
    }


def render(payload: dict) -> str:
    rows = [
        "# Minimal validation-only self-correction schedule",
        "",
        f"Source: `{payload['run_dir']}`",
        "",
        f"- Tasks audited: {payload['task_count']}/22",
        f"- Scheduled follow-ups: {payload['scheduled_task_count']}",
        f"- Estimated additional fits: {payload['estimated_additional_fits']}",
        (
            "- Maximum normalized action compute cost: "
            f"{payload['maximum_action_compute_cost']:.2f}"
        ),
        "- Test loaded: false; training started: false",
        "",
        "| Endpoint | Trigger | Decision | Next axis | Est. fits | Status |",
        "|---|---|---|---|---:|---|",
    ]
    for row in payload["decisions"]:
        if row["trigger_status"] == "supported" and row["decision_kind"] == "stop":
            continue
        rows.append(
            f"| {row['task']} | {row['trigger_status']} "
            f"`{row['trigger_action_id']}` | {row['decision_kind']} | "
            f"{row['selected_axis'] or '—'} | {row['estimated_additional_fits']} | "
            f"{row['execution_status']} |"
        )
    rows.extend(
        (
            "",
            "Only scheduled follow-ups are candidates for a prospective smoke run. The",
            "18 supported endpoints stop immediately; high-cost follow-ups above the 0.60",
            "normalized compute gate are not launched merely to create more activity.",
            "",
            "This artifact is a queue, not evidence of successful self-correction. A task",
            "counts only after its linked matched experiment is executed and retained.",
            "",
        )
    )
    return "\n".join(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, default=Path("runs/v3-agent-tdc22-v5"))
    parser.add_argument(
        "--output-prefix",
        type=Path,
        default=Path("reports/self_correction_schedule_v1"),
    )
    args = parser.parse_args()
    payload = build(args.run_dir)
    args.output_prefix.parent.mkdir(parents=True, exist_ok=True)
    args.output_prefix.with_suffix(".json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    args.output_prefix.with_suffix(".md").write_text(render(payload), encoding="utf-8")


if __name__ == "__main__":
    main()
