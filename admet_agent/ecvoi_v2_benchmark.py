"""Preregistered development and confirmatory protocols for EC-VOI v2."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Literal

from .source_snapshot import production_source_manifest
from .tasks import TASKS

PROTOCOL_VERSION = "metric-valued-ecvoi-v2-matched-benchmark-v1"
DEVELOPMENT_TASKS = ("BBB_Martins", "CYP2C9_Veith", "CYP2D6_Veith")
DISCOVERY_SEEDS = (1, 2, 3)
SHARED_BUDGET = {
    "max_actions_per_task": 8,
    "max_candidates_per_task": 24,
    "max_compute_minutes_per_task": 90,
}
ACTION_MEMORY_PATH = Path("runs/v5-transport-grid/transport_action_outcomes.jsonl")
ACTION_CONTEXT_PATH = Path("reports/admet22_action_context_registry_v1.json")
V2_MEMORY_PATH = Path("reports/ecvoi_v2_matched_effect_memory_v1.jsonl")
FROZEN_MAPPING_PATH = Path("runs/v27-maplight-gnn-freeze/full-22-task.mapping.json")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _arm(
    name: str,
    brain_cli: str,
    router: str,
    comparison_class: str,
) -> dict[str, Any]:
    return {
        "name": name,
        "brain_cli": brain_cli,
        "research_routing_policy": router,
        "comparison_class": comparison_class,
        **SHARED_BUDGET,
        "allowed_axes": ["feature", "model", "data", "replicate", "stop"],
    }


def _arms() -> list[dict[str, Any]]:
    return [
        _arm("ecvoi_v2", "ecvoi-v2-agent", "ecvoi-v2", "primary"),
        _arm("agent", "agent", "legacy", "same_budget_control"),
        _arm("rules", "fixed-rules-control", "legacy", "same_budget_control"),
        _arm("random", "random-control", "legacy", "same_budget_control"),
        _arm("automl", "automl-control", "legacy", "same_budget_control"),
        _arm(
            "expanded_automl",
            "expanded-automl-control",
            "legacy",
            "same_budget_control",
        ),
        _arm(
            "ecvoi_v2_no_context",
            "ecvoi-v2-no-context",
            "ecvoi-v2-no-context",
            "matched_ablation",
        ),
        _arm(
            "ecvoi_v2_no_memory",
            "ecvoi-v2-no-memory",
            "ecvoi-v2-no-memory",
            "matched_ablation",
        ),
        _arm(
            "ecvoi_v2_no_voi",
            "ecvoi-v2-no-voi",
            "ecvoi-v2-no-voi",
            "matched_ablation",
        ),
        _arm(
            "ecvoi_v2_no_cost",
            "ecvoi-v2-no-cost",
            "ecvoi-v2-no-cost",
            "matched_ablation",
        ),
        _arm(
            "ecvoi_v2_no_risk",
            "ecvoi-v2-no-risk",
            "ecvoi-v2-no-risk",
            "matched_ablation",
        ),
        _arm(
            "ecvoi_v2_shuffled_context",
            "ecvoi-v2-shuffled-context",
            "ecvoi-v2-shuffled-context",
            "placebo_control",
        ),
        _arm(
            "ecvoi_v2_shuffled_memory",
            "ecvoi-v2-shuffled-memory",
            "ecvoi-v2-shuffled-memory",
            "placebo_control",
        ),
    ]


def protocol_payload(
    cohort: Literal["development", "confirmation"] = "development",
) -> dict[str, Any]:
    """Build a source-bound protocol with a strict 3/19 task boundary."""

    canonical = tuple(item.name for item in TASKS.values())
    if cohort == "development":
        tasks = DEVELOPMENT_TASKS
    elif cohort == "confirmation":
        tasks = tuple(name for name in canonical if name not in DEVELOPMENT_TASKS)
    else:
        raise ValueError("cohort must be development or confirmation")
    contexts = json.loads(ACTION_CONTEXT_PATH.read_text(encoding="utf-8"))
    payload: dict[str, Any] = {
        "protocol_version": PROTOCOL_VERSION,
        "cohort": cohort,
        "implementation_source_sha256": production_source_manifest()[
            "source_sha256"
        ],
        "tasks": list(tasks),
        "excluded_tasks": [name for name in canonical if name not in tasks],
        "development_tasks": list(DEVELOPMENT_TASKS),
        "discovery_seeds": list(DISCOVERY_SEEDS),
        "arms": _arms(),
        "primary_arm": "ecvoi_v2",
        "statistical_unit": "endpoint_task",
        "primary_estimand": "paired_oriented_validation_gain_per_compute",
        "primary_selection": "effect_first_score_after_multiseed",
        "minimum_action_effect": 0.005,
        "test_policy": "not_loaded_for_routing_benchmark",
        "test_feedback_to_discovery": False,
        "leaderboard_feedback_to_discovery": False,
        "confirmation_can_modify_policy": False,
        "frozen_invariants": {
            "tdc22_mapping_path": str(FROZEN_MAPPING_PATH),
            "tdc22_mapping_sha256": _sha256(FROZEN_MAPPING_PATH),
            "finalize_research_routing_policy": "legacy",
            "action_memory_snapshot_path": str(ACTION_MEMORY_PATH),
            "action_memory_snapshot_sha256": _sha256(ACTION_MEMORY_PATH),
            "action_context_registry_path": str(ACTION_CONTEXT_PATH),
            "action_context_registry_sha256": contexts["sha256"],
            "ecvoi_v2_memory_snapshot_path": str(V2_MEMORY_PATH),
            "ecvoi_v2_memory_snapshot_sha256": _sha256(V2_MEMORY_PATH),
        },
        "pair_contract": {
            "ordered_seeds": list(DISCOVERY_SEEDS),
            "minimum_distinct_seeds": 3,
            "same_seed_treatment_control": True,
            "single_axis_required": True,
            "whole_pair_budget_reservation": True,
            "incomplete_pair_updates_posterior": False,
            "decision_logged_before_execution": True,
            "mandatory_identity_safety_floor": True,
        },
        "claim_gates": {
            "unsafe_actions_max": 0,
            "test_or_leaderboard_access_max": 0,
            "development_results_cannot_support_inference": True,
            "confirmation_task_coverage": 19,
            "paired_endpoint_bootstrap_required": True,
            "frozen_tdc22_mapping_must_remain_unchanged": True,
        },
    }
    validate_protocol(payload)
    digest = hashlib.sha256(
        json.dumps(payload, sort_keys=True).encode("utf-8")
    ).hexdigest()
    return {**payload, "sha256": digest}


def validate_protocol(payload: dict[str, Any]) -> None:
    canonical = {item.name for item in TASKS.values()}
    tasks = set(payload.get("tasks", []))
    development = set(payload.get("development_tasks", []))
    if development != set(DEVELOPMENT_TASKS):
        raise ValueError("The frozen three-task development cohort changed")
    cohort = payload.get("cohort")
    expected = development if cohort == "development" else canonical - development
    if cohort not in {"development", "confirmation"} or tasks != expected:
        raise ValueError("Protocol violates the frozen 3/19 cohort boundary")
    if tasks & set(payload.get("excluded_tasks", [])):
        raise ValueError("Protocol tasks overlap the excluded cohort")
    if payload.get("test_feedback_to_discovery") is not False:
        raise ValueError("Test feedback to discovery must be forbidden")
    if payload.get("leaderboard_feedback_to_discovery") is not False:
        raise ValueError("Leaderboard feedback to discovery must be forbidden")
    if payload.get("statistical_unit") != "endpoint_task":
        raise ValueError("Endpoint, not seed, must be the statistical unit")
    if tuple(payload.get("discovery_seeds", ())) != DISCOVERY_SEEDS:
        raise ValueError("The v2 matched-pair seeds must be exactly 1,2,3")
    names = [arm.get("name") for arm in payload.get("arms", [])]
    required = {
        "ecvoi_v2",
        "agent",
        "rules",
        "random",
        "automl",
        "expanded_automl",
        "ecvoi_v2_no_context",
        "ecvoi_v2_no_memory",
        "ecvoi_v2_no_voi",
        "ecvoi_v2_no_cost",
        "ecvoi_v2_no_risk",
        "ecvoi_v2_shuffled_context",
        "ecvoi_v2_shuffled_memory",
    }
    if not required.issubset(names) or len(names) != len(set(names)):
        raise ValueError("Missing or duplicate EC-VOI v2 benchmark arms")
    budgets = {
        tuple(arm[key] for key in SHARED_BUDGET) for arm in payload["arms"]
    }
    if budgets != {(8, 24, 90)}:
        raise ValueError("Every v2 arm must have the same frozen budget")


def write_protocol(
    path: Path,
    cohort: Literal["development", "confirmation"] = "development",
) -> dict[str, Any]:
    payload = protocol_payload(cohort)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if existing.get("sha256") != payload["sha256"]:
            raise RuntimeError("Refusing to overwrite a different v2 preregistration")
    else:
        path.write_text(
            json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
        )
    return payload


__all__ = [
    "DEVELOPMENT_TASKS",
    "DISCOVERY_SEEDS",
    "protocol_payload",
    "validate_protocol",
    "write_protocol",
]
