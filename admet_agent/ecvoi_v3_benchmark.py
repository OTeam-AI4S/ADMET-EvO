"""Source-bound development/confirmation protocols for EC-VOI v3."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Literal

from .source_snapshot import production_source_manifest
from .tasks import TASKS

PROTOCOL_VERSION = "shared-strong-anchor-ecvoi-v3-matched-benchmark-v1"
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


def _arm(name: str, brain_cli: str, router: str, kind: str) -> dict[str, Any]:
    return {
        "name": name,
        "brain_cli": brain_cli,
        "research_routing_policy": router,
        "comparison_class": kind,
        **SHARED_BUDGET,
        "allowed_axes": ["feature", "model", "replicate", "stop"],
    }


def _arms() -> list[dict[str, Any]]:
    return [
        _arm("ecvoi_v3", "ecvoi-v3-agent", "ecvoi-v3", "primary"),
        _arm(
            "ecvoi_v3_no_initial_exploration",
            "ecvoi-v3-no-initial-exploration",
            "ecvoi-v3-no-initial-exploration",
            "matched_mechanism_ablation",
        ),
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
    ]


def protocol_payload(
    cohort: Literal["development", "confirmation"] = "development",
) -> dict[str, Any]:
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
        "primary_arm": "ecvoi_v3",
        "statistical_unit": "endpoint_task",
        "primary_estimand": "paired_oriented_gain_over_shared_strong_anchor",
        "primary_selection": "best_legal_candidate_with_shared_anchor_fallback",
        "minimum_action_effect": 0.005,
        "test_policy": "not_loaded_for_routing_benchmark",
        "test_feedback_to_discovery": False,
        "leaderboard_feedback_to_discovery": False,
        "confirmation_can_modify_policy": False,
        "shared_anchor_excluded_from_adaptive_action_budget": True,
        "frozen_invariants": {
            "tdc22_mapping_path": str(FROZEN_MAPPING_PATH),
            "tdc22_mapping_sha256": _sha256(FROZEN_MAPPING_PATH),
            "shared_anchor_mapping_path": str(FROZEN_MAPPING_PATH),
            "shared_anchor_mapping_sha256": _sha256(FROZEN_MAPPING_PATH),
            "shared_anchor_data_policy": "identity_safe",
            "mandatory_discovery_data_policy": "identity_safe",
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
            "one_safe_local_exploration_max": 1,
            "maximum_initial_exploration_risk": 0.10,
            "robust_gate_after_first_local_observation": True,
            "mandatory_identity_safety_floor": True,
        },
        "development_promotion_gate": {
            "scope": "development_only_no_inferential_claim",
            "vs_automl_min_nonloss_tasks": 2,
            "vs_automl_mean_oriented_delta_min": -0.002,
            "vs_shared_anchor_min_gain_tasks": 1,
            "vs_shared_anchor_max_harm_tasks": 0,
            "minimum_complete_local_pairs_across_tasks": 2,
            "test_or_leaderboard_access_max": 0,
            "all_frozen_hashes_must_match": True,
        },
        "claim_gates": {
            "unsafe_actions_max": 0,
            "test_or_leaderboard_access_max": 0,
            "development_results_cannot_support_inference": True,
            "confirmation_task_coverage": 19,
            "paired_endpoint_bootstrap_required": True,
            "frozen_tdc22_mapping_must_remain_unchanged": True,
            "shared_anchor_must_be_finalist_eligible_for_every_arm": True,
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
    if tuple(payload.get("discovery_seeds", ())) != DISCOVERY_SEEDS:
        raise ValueError("Matched-pair seeds must remain exactly 1,2,3")
    names = [arm.get("name") for arm in payload.get("arms", [])]
    required = {
        "ecvoi_v3",
        "ecvoi_v3_no_initial_exploration",
        "agent",
        "rules",
        "random",
        "automl",
        "expanded_automl",
    }
    if set(names) != required or len(names) != len(set(names)):
        raise ValueError("Missing or duplicate EC-VOI v3 benchmark arms")
    budgets = {tuple(arm[key] for key in SHARED_BUDGET) for arm in payload["arms"]}
    if budgets != {(8, 24, 90)}:
        raise ValueError("Every v3 arm must have the same frozen adaptive budget")
    invariants = payload.get("frozen_invariants", {})
    if invariants.get("mandatory_discovery_data_policy") != "identity_safe":
        raise ValueError("Every v3 arm requires the identity-safe discovery floor")
    if invariants.get("shared_anchor_mapping_sha256") != invariants.get(
        "tdc22_mapping_sha256"
    ):
        raise ValueError("Shared and frozen TDC-22 anchor mappings must match")


def write_protocol(
    path: Path,
    cohort: Literal["development", "confirmation"] = "development",
) -> dict[str, Any]:
    payload = protocol_payload(cohort)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if existing.get("sha256") != payload["sha256"]:
            raise RuntimeError("Refusing to overwrite a different v3 preregistration")
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
