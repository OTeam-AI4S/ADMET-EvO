import json

import pytest

from admet_agent.cross_task_memory_readiness import (
    audit_cross_task_memory_readiness,
)
from admet_agent.tasks import TASKS


def test_historical_selected_actions_remain_audit_only(tmp_path):
    run_dir = tmp_path / "run"
    memory_path = run_dir / "action_outcomes.jsonl"
    memory_rows = []
    for index, task in enumerate(task.name for task in TASKS.values()):
        action = "feature-domain-shift" if index < 18 else "model-imbalance-aware"
        axis = "feature" if index < 18 else "model"
        task_dir = run_dir / "tasks" / task
        task_dir.mkdir(parents=True)
        summary = {
            "profile": {"task": task},
            "research_plan": {"selected_action_id": action},
            "action_evidence": {
                "action_id": action,
                "axis": axis,
                "status": "supported" if index < 20 else "refuted",
                "single_axis_verified": True,
            },
        }
        (task_dir / "search_seed_1.json").write_text(json.dumps(summary))
        memory_rows.append(
            json.dumps(
                {
                    "task": task,
                    "action_id": action,
                    "status": summary["action_evidence"]["status"],
                    "single_axis_verified": True,
                }
            )
        )
    memory_path.write_text("\n".join(memory_rows) + "\n")

    report = audit_cross_task_memory_readiness(run_dir, memory_path)
    assert report["task_count"] == 22
    assert report["single_axis_verified_tasks"] == 22
    assert report["transport_eligible_record_count"] == 0
    assert report["routing_ready"] is False
    assert report["largest_action_share"] == pytest.approx(18 / 22)
    assert report["enforcement"]["observational_memory_can_enable_unimol2"] is False
    assert report["recommended_next_evidence"]["test_feedback"] is False
