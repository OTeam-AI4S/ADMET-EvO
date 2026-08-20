"""Audit whether cross-task action outcomes may causally influence routing."""

from __future__ import annotations

import json
import math
from collections import Counter
from pathlib import Path
from typing import Any

from .tasks import TASKS

READINESS_VERSION = "cross-task-memory-readiness-v1"
MINIMUM_ENDPOINTS_PER_ACTION = 5


def _normalized_entropy(counts: Counter[str]) -> float:
    total = sum(counts.values())
    if total == 0 or len(counts) <= 1:
        return 0.0
    probabilities = [count / total for count in counts.values()]
    entropy = -sum(value * math.log(value) for value in probabilities)
    return entropy / math.log(len(counts))


def audit_cross_task_memory_readiness(
    run_dir: Path, memory_path: Path
) -> dict[str, Any]:
    expected = {task.name for task in TASKS.values()}
    summaries: dict[str, dict[str, Any]] = {}
    for path in sorted(run_dir.glob("tasks/*/search_seed_1.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        task = payload.get("profile", {}).get("task")
        if not task or task in summaries:
            raise ValueError(f"Missing or duplicate task in {path}")
        summaries[task] = payload
    if set(summaries) != expected:
        raise ValueError("Readiness audit requires exactly the canonical TDC 22 tasks")

    action_counts: Counter[str] = Counter()
    axis_counts: Counter[str] = Counter()
    status_counts: Counter[str] = Counter()
    task_rows = []
    for task in sorted(expected):
        summary = summaries[task]
        plan = summary.get("research_plan", {})
        evidence = summary.get("action_evidence") or {}
        action_id = plan.get("selected_action_id")
        if not action_id or evidence.get("action_id") != action_id:
            raise ValueError(f"Selected action and evidence do not match for {task}")
        action_counts[action_id] += 1
        axis_counts[str(evidence.get("axis"))] += 1
        status_counts[str(evidence.get("status"))] += 1
        task_rows.append(
            {
                "task": task,
                "category": TASKS[task.lower()].category,
                "selected_action_id": action_id,
                "axis": evidence.get("axis"),
                "status": evidence.get("status"),
                "single_axis_verified": evidence.get("single_axis_verified") is True,
                "test_loaded": False,
            }
        )

    memory_rows = []
    if memory_path.exists():
        memory_rows = [
            json.loads(line)
            for line in memory_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    eligible = [row for row in memory_rows if row.get("transport_eligible") is True]
    eligible_by_action: Counter[str] = Counter(
        str(row.get("action_id")) for row in eligible
    )
    eligible_status_by_action: dict[str, Counter[str]] = {}
    for row in eligible:
        eligible_status_by_action.setdefault(str(row.get("action_id")), Counter())[
            str(row.get("status"))
        ] += 1

    action_readiness = {}
    for action_id in sorted(action_counts):
        outcomes = eligible_status_by_action.get(action_id, Counter())
        eligible_count = eligible_by_action[action_id]
        has_variation = len([count for count in outcomes.values() if count]) >= 2
        blockers = []
        if eligible_count < MINIMUM_ENDPOINTS_PER_ACTION:
            blockers.append("insufficient_transport_eligible_endpoints")
        if not has_variation:
            blockers.append("transport_outcome_variation_missing")
        action_readiness[action_id] = {
            "historical_selected_endpoints": action_counts[action_id],
            "transport_eligible_endpoints": eligible_count,
            "transport_status_counts": dict(sorted(outcomes.items())),
            "routing_eligible": not blockers,
            "blockers": blockers,
        }

    routing_ready = bool(action_readiness) and all(
        row["routing_eligible"] for row in action_readiness.values()
    )
    global_blockers = []
    if not eligible:
        global_blockers.append("no_prospective_randomized_or_complete_grid_evidence")
    if max(action_counts.values()) / len(task_rows) >= 0.75:
        global_blockers.append("selected_action_positivity_violation")
    global_blockers.append("historical_selected_action_outcomes_are_observational_only")

    return {
        "version": READINESS_VERSION,
        "task_count": len(task_rows),
        "test_loaded": False,
        "historical_design": "policy_selected_actions_only",
        "historical_action_counts": dict(sorted(action_counts.items())),
        "historical_axis_counts": dict(sorted(axis_counts.items())),
        "historical_status_counts": dict(sorted(status_counts.items())),
        "selected_action_normalized_entropy": _normalized_entropy(action_counts),
        "largest_action_share": max(action_counts.values()) / len(task_rows),
        "single_axis_verified_tasks": sum(
            row["single_axis_verified"] for row in task_rows
        ),
        "memory_record_count": len(memory_rows),
        "transport_eligible_record_count": len(eligible),
        "action_readiness": action_readiness,
        "routing_ready": routing_ready,
        "global_blockers": global_blockers,
        "enforcement": {
            "observational_effect_multiplier": 1.0,
            "observational_memory_can_change_ranking": False,
            "observational_memory_can_enable_unimol2": False,
            "minimum_unique_endpoints_per_action": MINIMUM_ENDPOINTS_PER_ACTION,
            "allowed_designs": [
                "complete_action_grid",
                "randomized_action_assignment",
            ],
            "requires_prospective": True,
            "requires_same_budget": True,
            "requires_shared_protocol_sha256": True,
            "requires_outcome_variation": True,
        },
        "recommended_next_evidence": {
            "design": "prospective_complete_action_grid",
            "minimum_sentinel_endpoints": MINIMUM_ENDPOINTS_PER_ACTION,
            "actions": [
                "data-quality-filter",
                "feature-domain-shift",
                "model-estimator-family",
            ],
            "minimum_matched_action_episodes": 15,
            "selection_data": "train_and_validation_only",
            "test_feedback": False,
            "purpose": (
                "Estimate action-by-hardness transport effects; do not optimize the "
                "already-qualified frozen application models."
            ),
        },
        "task_rows": task_rows,
        "interpretation": (
            "The 22 historical outcomes demonstrate matched single-axis execution and "
            "negative-result retention, but not causal cross-task learning. The policy "
            "selected which action was observed, so missing counterfactual actions and "
            "action imbalance prevent routing claims."
        ),
    }
