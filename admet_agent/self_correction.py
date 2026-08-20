"""Validation-only planning for auditable research self-correction.

The planner turns matched-action evidence into a *scheduled* next action.  A
scheduled action is deliberately not counted as a successful correction: the
follow-up must later be executed and retained as a linked action outcome.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from typing import Any, Literal

from .action_evidence import MatchedActionEvidence

CorrectionKind = Literal["stop", "repair_evidence", "switch_axis"]
_INTERVENTION_AXES = frozenset({"feature", "model", "data"})
_SAFE_TEST_KEYS = frozenset({"falsification_test", "falsification_tests"})
SELF_CORRECTION_POLICY_VERSION = "validation-self-correction-v1"


def _canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)


def _sha256(value: Any) -> str:
    return hashlib.sha256(_canonical(value).encode("utf-8")).hexdigest()


def _forbidden_test_paths(value: Any, path: str = "input") -> tuple[str, ...]:
    forbidden = []
    if isinstance(value, Mapping):
        for raw_key, child in value.items():
            key = str(raw_key)
            normalized = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", key).lower()
            child_path = f"{path}.{key}"
            tokens = tuple(
                token for token in re.split(r"[^a-z0-9]+", normalized) if token
            )
            safe_sentinel = normalized == "test_loaded" and child is False
            if (
                normalized not in _SAFE_TEST_KEYS
                and not safe_sentinel
                and any(
                    token in {"test", "testing", "heldout", "holdout"}
                    for token in tokens
                )
            ):
                forbidden.append(child_path)
            forbidden.extend(_forbidden_test_paths(child, child_path))
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for index, child in enumerate(value):
            forbidden.extend(_forbidden_test_paths(child, f"{path}[{index}]"))
    return tuple(sorted(set(forbidden)))


def _as_evidence(value: MatchedActionEvidence | Mapping[str, Any]) -> dict[str, Any]:
    if isinstance(value, MatchedActionEvidence):
        return value.to_dict()
    if not isinstance(value, Mapping):
        raise TypeError("evidence must be MatchedActionEvidence or a mapping")
    return dict(value)


def _is_executable(action: Mapping[str, Any], *, maximum_compute_cost: float) -> bool:
    if action.get("axis") not in _INTERVENTION_AXES:
        return False
    try:
        compute_cost = float(action.get("compute_cost", 0.0))
    except (TypeError, ValueError):
        return False
    if not 0.0 <= compute_cost <= maximum_compute_cost:
        return False
    payload = action.get("payload", {})
    if not isinstance(payload, Mapping):
        return False
    # External data is a hypothesis until the admission gate has passed.  The
    # correction planner must never turn quarantine into an automatic merge.
    if action.get("action_id") == "data-external-quarantine":
        return payload.get("merge_status") == "admitted"
    return True


@dataclass(frozen=True)
class CorrectiveDecision:
    task: str
    trigger_action_id: str
    trigger_status: str
    kind: CorrectionKind
    selected_action_id: str | None
    selected_axis: str | None
    estimated_additional_fits: int
    requires_execution: bool
    execution_status: str
    reason_codes: tuple[str, ...]
    next_plan: dict[str, Any] | None
    minimum_paired_seeds: int
    maximum_action_compute_cost: float
    policy_version: str
    source_plan_sha256: str
    trigger_evidence_sha256: str
    decision_sha256: str
    test_loaded: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class ValidationSelfCorrectionPlanner:
    """Choose a bounded next experiment from matched validation evidence."""

    def __init__(
        self,
        minimum_paired_seeds: int = 3,
        maximum_action_compute_cost: float = 0.60,
    ):
        if (
            isinstance(minimum_paired_seeds, bool)
            or not isinstance(minimum_paired_seeds, int)
            or minimum_paired_seeds < 3
        ):
            raise ValueError("minimum_paired_seeds must be an integer >= 3")
        if (
            isinstance(maximum_action_compute_cost, bool)
            or not 0.0 <= maximum_action_compute_cost <= 1.0
        ):
            raise ValueError("maximum_action_compute_cost must be within [0, 1]")
        self.minimum_paired_seeds = int(minimum_paired_seeds)
        self.maximum_action_compute_cost = float(maximum_action_compute_cost)

    @staticmethod
    def _selected_action(plan: Mapping[str, Any]) -> dict[str, Any] | None:
        selected = plan.get("selected_action_id")
        return next(
            (
                dict(action)
                for action in plan.get("actions", [])
                if isinstance(action, Mapping) and action.get("action_id") == selected
            ),
            None,
        )

    def _switch_candidate(
        self,
        plan: Mapping[str, Any],
        *,
        trigger_axis: str,
        attempted_action_ids: frozenset[str],
    ) -> dict[str, Any] | None:
        for action in plan.get("actions", []):
            if not isinstance(action, Mapping):
                continue
            action_id = str(action.get("action_id", ""))
            if (
                action_id
                and action_id not in attempted_action_ids
                and action.get("axis") != trigger_axis
                and _is_executable(
                    action,
                    maximum_compute_cost=self.maximum_action_compute_cost,
                )
            ):
                return dict(action)
        return None

    def _scheduled_plan(
        self,
        plan: Mapping[str, Any],
        action: Mapping[str, Any],
        *,
        trigger_action_id: str,
        kind: CorrectionKind,
        sequence_index: int,
    ) -> dict[str, Any]:
        source_action_id = str(action["action_id"])
        suffix = _sha256(
            {
                "task": plan.get("task"),
                "trigger": trigger_action_id,
                "source_action": source_action_id,
                "kind": kind,
                "sequence_index": sequence_index,
            }
        )[:10]
        corrective_id = f"{source_action_id}--correct-{suffix}"
        corrective_action = {
            **dict(action),
            "action_id": corrective_id,
            "source_action_id": source_action_id,
            "action_family": source_action_id,
            "corrects_action_id": trigger_action_id,
            "parent_action_id": trigger_action_id,
            "attempt_id": corrective_id,
            "sequence_index": sequence_index,
            "correction_kind": kind,
        }
        return {
            "task": str(plan.get("task", "unknown")),
            "hardness_evidence_sha256": plan.get("hardness_evidence_sha256"),
            "selected_action_id": corrective_id,
            "effect_priority": plan.get("effect_priority"),
            "actions": [corrective_action],
            "parent_plan_sha256": _sha256(plan),
            "execution_status": "scheduled",
            "test_loaded": False,
        }

    def decide(
        self,
        plan: Mapping[str, Any],
        evidence: MatchedActionEvidence | Mapping[str, Any],
        *,
        attempted_action_ids: Sequence[str] = (),
        remaining_action_budget: int = 1,
        sequence_index: int = 2,
    ) -> CorrectiveDecision:
        if not isinstance(plan, Mapping):
            raise TypeError("plan must be a mapping")
        evidence_dict = _as_evidence(evidence)
        forbidden = _forbidden_test_paths([plan, evidence_dict])
        if forbidden:
            raise ValueError(f"Test-like fields are forbidden: {list(forbidden)}")
        if (
            isinstance(remaining_action_budget, bool)
            or not isinstance(remaining_action_budget, int)
            or remaining_action_budget < 0
        ):
            raise ValueError("remaining_action_budget must be a non-negative integer")
        if (
            isinstance(sequence_index, bool)
            or not isinstance(sequence_index, int)
            or sequence_index < 2
        ):
            raise ValueError("sequence_index must be an integer >= 2")

        selected = self._selected_action(plan)
        trigger_action_id = str(
            evidence_dict.get("action_id")
            or plan.get("selected_action_id")
            or "unknown"
        )
        status = str(evidence_dict.get("status", "unknown"))
        trigger_axis = str(
            evidence_dict.get("axis") or (selected or {}).get("axis") or "unknown"
        )
        action: dict[str, Any] | None = None
        kind: CorrectionKind = "stop"
        reasons: tuple[str, ...]

        if remaining_action_budget == 0:
            reasons = ("action_budget_exhausted", "stop_without_test_feedback")
        elif status == "supported":
            reasons = (
                "preregistered_effect_supported",
                "prediction_performance_already_qualified",
                "avoid_redundant_search",
            )
        elif status == "refuted":
            attempted = frozenset({trigger_action_id, *map(str, attempted_action_ids)})
            action = self._switch_candidate(
                plan,
                trigger_axis=trigger_axis,
                attempted_action_ids=attempted,
            )
            if action is None:
                reasons = (
                    "hypothesis_refuted",
                    "no_unattempted_executable_cross_axis_action",
                    "stop_without_repeating_refuted_family",
                )
            else:
                kind = "switch_axis"
                reasons = (
                    "hypothesis_refuted",
                    "do_not_repeat_refuted_action",
                    "switch_to_highest_ranked_executable_cross_axis_action",
                )
        elif (
            status == "inconclusive"
            and selected
            and _is_executable(
                selected,
                maximum_compute_cost=self.maximum_action_compute_cost,
            )
        ):
            paired_seeds = len(evidence_dict.get("validation_seeds", ()))
            structurally_valid = evidence_dict.get("single_axis_verified") is True
            if paired_seeds < self.minimum_paired_seeds or not structurally_valid:
                action = selected
                kind = "repair_evidence"
                reasons = (
                    "evidence_incomplete",
                    "repair_matched_pair_or_paired_seed_coverage",
                    "do_not_claim_support_from_inconclusive_evidence",
                )
            else:
                attempted = frozenset(
                    {trigger_action_id, *map(str, attempted_action_ids)}
                )
                action = self._switch_candidate(
                    plan,
                    trigger_axis=trigger_axis,
                    attempted_action_ids=attempted,
                )
                if action is None:
                    reasons = (
                        "effect_below_resolution_gate",
                        "no_unattempted_executable_cross_axis_action",
                        "stop_to_avoid_validation_overfitting",
                    )
                else:
                    kind = "switch_axis"
                    reasons = (
                        "effect_below_resolution_gate",
                        "switch_axis_instead_of_repeating_low_information_search",
                    )
        else:
            reasons = (
                "no_correctable_intervention_evidence",
                "stop_without_test_feedback",
            )

        next_plan = (
            self._scheduled_plan(
                plan,
                action,
                trigger_action_id=trigger_action_id,
                kind=kind,
                sequence_index=sequence_index,
            )
            if action is not None
            else None
        )
        if next_plan is None:
            estimated_fits = 0
        elif kind == "repair_evidence":
            observed = len(evidence_dict.get("validation_seeds", ()))
            estimated_fits = 2 * max(1, self.minimum_paired_seeds - observed)
        else:
            estimated_fits = 2 * self.minimum_paired_seeds
        base = {
            "task": str(plan.get("task", "unknown")),
            "trigger_action_id": trigger_action_id,
            "trigger_status": status,
            "kind": kind,
            "selected_action_id": (
                str(next_plan["selected_action_id"]) if next_plan else None
            ),
            "selected_axis": str(action["axis"]) if action else None,
            "estimated_additional_fits": estimated_fits,
            "requires_execution": next_plan is not None,
            "execution_status": "scheduled" if next_plan else "stopped",
            "reason_codes": reasons,
            "next_plan": next_plan,
            "minimum_paired_seeds": self.minimum_paired_seeds,
            "maximum_action_compute_cost": self.maximum_action_compute_cost,
            "policy_version": SELF_CORRECTION_POLICY_VERSION,
            "source_plan_sha256": _sha256(plan),
            "trigger_evidence_sha256": _sha256(evidence_dict),
            "test_loaded": False,
        }
        return CorrectiveDecision(
            **base,
            decision_sha256=_sha256(base),
        )


__all__ = [
    "SELF_CORRECTION_POLICY_VERSION",
    "CorrectiveDecision",
    "ValidationSelfCorrectionPlanner",
]
