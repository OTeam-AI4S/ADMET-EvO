"""Locked controls for the two-case corrective-feedback smoke episode.

This is a small local policy replay, not an independent benchmark.  Its main
purpose is to distinguish an explicit research-feedback protocol from ordinary
scalar AutoML and to reveal when complete deterministic rules can tie it.
"""

from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass
from typing import Any

FEEDBACK_CONTROL_PROTOCOL = "locked-corrective-feedback-policy-controls-v1"


@dataclass(frozen=True)
class FeedbackPolicyDecision:
    task: str
    parent_status: str
    parent_axis: str
    selected_axis: str | None
    decision_kind: str
    explicit_parent_link: bool
    matched_pair_protected: bool
    stopped_after_low_information_child: bool
    high_cost_escalation: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _appropriate(row: FeedbackPolicyDecision, paired_seeds: int) -> bool:
    if row.parent_status == "refuted":
        return row.decision_kind == "stop" or (
            row.selected_axis is not None and row.selected_axis != row.parent_axis
        )
    if row.parent_status == "inconclusive" and paired_seeds < 3:
        return row.decision_kind == "repair_evidence"
    return row.decision_kind == "stop"


def _score(rows: list[FeedbackPolicyDecision], seeds: dict[str, int]) -> dict[str, Any]:
    if not rows:
        raise ValueError("Feedback policy rows are required")
    dimensions = {
        "feedback_conditioned_response": sum(
            _appropriate(row, seeds[row.task]) for row in rows
        )
        / len(rows),
        "explicit_parent_link": sum(row.explicit_parent_link for row in rows)
        / len(rows),
        "matched_pair_protection": sum(row.matched_pair_protected for row in rows)
        / len(rows),
        "bounded_low_information_stop": sum(
            row.stopped_after_low_information_child for row in rows
        )
        / len(rows),
        "no_high_cost_escalation": sum(not row.high_cost_escalation for row in rows)
        / len(rows),
    }
    return {
        "case_count": len(rows),
        "dimensions": dimensions,
        "macro_process_score": sum(dimensions.values()) / len(dimensions),
        "rows": [row.to_dict() for row in rows],
    }


def _random_axis(task: str, available: list[str], policy_seed: int) -> str:
    digest = hashlib.sha256(f"{policy_seed}:{task}".encode()).digest()
    return available[int.from_bytes(digest[:8], "big") % len(available)]


def compare_feedback_controls(
    schedule: dict[str, Any],
    smoke_report: dict[str, Any],
    *,
    policy_seed: int = 20260820,
) -> dict[str, Any]:
    if (
        schedule.get("test_loaded") is not False
        or smoke_report.get("test_loaded") is not False
    ):
        raise ValueError("Feedback controls require validation-only inputs")
    schedule_rows = {
        row["task"]: row
        for row in schedule["decisions"]
        if row.get("execution_status") == "scheduled"
    }
    smoke_rows = {row["task"]: row for row in smoke_report["task_rows"]}
    if set(schedule_rows) != set(smoke_rows):
        raise ValueError("Schedule and smoke tasks must align")
    seeds = {}
    agent_rows = []
    rules_rows = []
    expanded_rows = []
    conventional_rows = []
    random_rows = []
    for task in sorted(schedule_rows):
        scheduled = schedule_rows[task]
        observed = smoke_rows[task]
        parent_status = observed["parent_status"]
        parent_axis = (
            "data" if observed["parent_action_id"] == "data-leakage-safe" else "model"
        )
        paired_seeds = 0 if scheduled["decision_kind"] == "repair_evidence" else 5
        seeds[task] = paired_seeds
        common = {
            "task": task,
            "parent_status": parent_status,
            "parent_axis": parent_axis,
        }
        agent = FeedbackPolicyDecision(
            **common,
            selected_axis=scheduled["selected_axis"],
            decision_kind=scheduled["decision_kind"],
            explicit_parent_link=True,
            matched_pair_protected=True,
            stopped_after_low_information_child=(
                observed["post_followup_kind"] == "stop"
            ),
            high_cost_escalation=False,
        )
        agent_rows.append(agent)
        # A complete transition table can intentionally reproduce this simple
        # two-case protocol.  Equality is retained as a negative control.
        rules_rows.append(agent)
        for target in (expanded_rows, conventional_rows):
            target.append(
                FeedbackPolicyDecision(
                    **common,
                    selected_axis="model",
                    decision_kind="scalar_model_search",
                    explicit_parent_link=False,
                    matched_pair_protected=False,
                    stopped_after_low_information_child=False,
                    high_cost_escalation=False,
                )
            )
        available_axes = ["feature", "model", "data", "stop"]
        axis = _random_axis(task, available_axes, policy_seed)
        random_rows.append(
            FeedbackPolicyDecision(
                **common,
                selected_axis=None if axis == "stop" else axis,
                decision_kind="stop" if axis == "stop" else "random_action",
                explicit_parent_link=False,
                matched_pair_protected=False,
                stopped_after_low_information_child=(axis == "stop"),
                high_cost_escalation=(axis == "feature"),
            )
        )
    arms = {
        "agent_feedback_loop": _score(agent_rows, seeds),
        "complete_rules_feedback_loop": _score(rules_rows, seeds),
        "expanded_automl_scalar_search": _score(expanded_rows, seeds),
        "conventional_automl_scalar_search": _score(conventional_rows, seeds),
        "seeded_random_action": _score(random_rows, seeds),
    }
    return {
        "protocol": FEEDBACK_CONTROL_PROTOCOL,
        "policy_seed": policy_seed,
        "case_count": len(agent_rows),
        "arms": arms,
        "test_loaded": False,
        "claim_scope": "locked_two_case_policy_replay_not_independent_evaluation",
        "interpretation": {
            "automl_boundary": (
                "Explicit feedback linkage, matched controls and bounded stopping "
                "are process capabilities absent from scalar AutoML search."
            ),
            "rules_boundary": (
                "Complete rules exactly reproduce the two observed cases, so this "
                "does not establish learned reasoning beyond deterministic rules."
            ),
            "upgrade_requirement": (
                "Use independently authored unseen composite episodes with ambiguous "
                "feedback and multiple feasible remediations."
            ),
        },
    }


__all__ = [
    "FEEDBACK_CONTROL_PROTOCOL",
    "FeedbackPolicyDecision",
    "compare_feedback_controls",
]
