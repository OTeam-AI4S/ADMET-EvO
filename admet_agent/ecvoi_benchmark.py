"""Preregistered matched benchmark for the EC-VOI research router.

This protocol is intentionally separate from the frozen TDC-22 v5 policy
benchmark.  It compares research-routing policies under the same discovery
tasks, seeds, primitive action space, action count, compute ceiling, promotion
rule and stopping rule.  Test and leaderboard feedback are forbidden.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from .source_snapshot import production_source_manifest
from .tasks import TASKS

PROTOCOL_VERSION = "ecvoi-matched-routing-benchmark-v1"
DISCOVERY_SEEDS = (1, 2, 3, 4, 5)
SHARED_BUDGET = {
    "max_actions_per_task": 8,
    "max_candidates_per_task": 24,
    "max_compute_minutes_per_task": 90,
}
FROZEN_TDC22_MAPPING_SHA256 = (
    "17523aa1d5583437833f7f22aecceff5d0932f466104936b0be4ce5ee363e98d"
)
ACTION_MEMORY_SNAPSHOT_SHA256 = (
    "5257d01873121971fd7cb18a5986dc27f505fa597e70ebac4be0ed08a31989e4"
)
ACTION_CONTEXT_REGISTRY_SHA256 = (
    "ce0628dbb1cb6c93780b8953dcf80ed8e95b8ca4792d25138195ac9e57ae0713"
)


def _arm(
    name: str,
    *,
    brain_cli: str,
    router: str,
    policy: str,
    comparison_class: str,
    uses_hardness: bool,
    uses_memory: bool,
    uses_voi: bool,
    uses_cost: bool,
    uses_risk: bool,
) -> dict[str, Any]:
    return {
        "name": name,
        "brain_cli": brain_cli,
        "research_routing_policy": router,
        "policy": policy,
        "comparison_class": comparison_class,
        **SHARED_BUDGET,
        "uses_hardness": uses_hardness,
        "uses_cross_task_memory": uses_memory,
        "uses_value_of_information": uses_voi,
        "uses_cost_penalty": uses_cost,
        "uses_risk_penalty": uses_risk,
        "allowed_axes": ["feature", "model", "data", "replicate", "stop"],
    }


def protocol_payload() -> dict[str, Any]:
    """Return the immutable, label-blind matched routing contract."""

    arms = [
        _arm(
            "ecvoi",
            brain_cli="ecvoi-agent",
            router="ecvoi",
            policy="evidence_conditioned_value_of_information",
            comparison_class="primary",
            uses_hardness=True,
            uses_memory=True,
            uses_voi=True,
            uses_cost=True,
            uses_risk=True,
        ),
        _arm(
            "agent",
            brain_cli="agent",
            router="legacy",
            policy="legacy_hardness_effect_first",
            comparison_class="same_budget_control",
            uses_hardness=True,
            uses_memory=True,
            uses_voi=False,
            uses_cost=True,
            uses_risk=True,
        ),
        _arm(
            "rules",
            brain_cli="fixed-rules-control",
            router="legacy",
            policy="fixed_action_precedence",
            comparison_class="same_budget_control",
            uses_hardness=True,
            uses_memory=False,
            uses_voi=False,
            uses_cost=False,
            uses_risk=False,
        ),
        _arm(
            "random",
            brain_cli="random-control",
            router="legacy",
            policy="seeded_random_action",
            comparison_class="same_budget_control",
            uses_hardness=False,
            uses_memory=False,
            uses_voi=False,
            uses_cost=False,
            uses_risk=False,
        ),
        _arm(
            "automl",
            brain_cli="automl-control",
            router="legacy",
            policy="model_and_hyperparameter_search",
            comparison_class="same_budget_control",
            uses_hardness=False,
            uses_memory=False,
            uses_voi=False,
            uses_cost=False,
            uses_risk=False,
        ),
        _arm(
            "expanded_automl",
            brain_cli="expanded-automl-control",
            router="legacy",
            policy="expanded_space_scalar_validation_acquisition",
            comparison_class="same_budget_control",
            uses_hardness=False,
            uses_memory=False,
            uses_voi=False,
            uses_cost=True,
            uses_risk=False,
        ),
        _arm(
            "ecvoi_no_context",
            brain_cli="ecvoi-no-context",
            router="ecvoi-no-context",
            policy="ecvoi_without_context_matched_transport",
            comparison_class="matched_ablation",
            uses_hardness=True,
            uses_memory=True,
            uses_voi=True,
            uses_cost=True,
            uses_risk=True,
        ),
        _arm(
            "ecvoi_no_memory",
            brain_cli="ecvoi-no-memory",
            router="ecvoi-no-memory",
            policy="ecvoi_without_cross_task_memory",
            comparison_class="matched_ablation",
            uses_hardness=True,
            uses_memory=False,
            uses_voi=True,
            uses_cost=True,
            uses_risk=True,
        ),
        _arm(
            "ecvoi_no_voi",
            brain_cli="ecvoi-no-voi",
            router="ecvoi-no-voi",
            policy="ecvoi_without_information_bonus",
            comparison_class="matched_ablation",
            uses_hardness=True,
            uses_memory=True,
            uses_voi=False,
            uses_cost=True,
            uses_risk=True,
        ),
        _arm(
            "ecvoi_no_cost",
            brain_cli="ecvoi-no-cost",
            router="ecvoi-no-cost",
            policy="ecvoi_without_compute_cost_penalty",
            comparison_class="matched_ablation",
            uses_hardness=True,
            uses_memory=True,
            uses_voi=True,
            uses_cost=False,
            uses_risk=True,
        ),
        _arm(
            "ecvoi_no_risk",
            brain_cli="ecvoi-no-risk",
            router="ecvoi-no-risk",
            policy="ecvoi_without_scientific_risk_penalty",
            comparison_class="matched_ablation",
            uses_hardness=True,
            uses_memory=True,
            uses_voi=True,
            uses_cost=True,
            uses_risk=False,
        ),
    ]
    payload: dict[str, Any] = {
        "protocol_version": PROTOCOL_VERSION,
        "implementation_source_sha256": production_source_manifest()[
            "source_sha256"
        ],
        "tasks": [item.name for item in TASKS.values()],
        "discovery_seeds": list(DISCOVERY_SEEDS),
        "arms": arms,
        "primary_arm": "ecvoi",
        "statistical_unit": "endpoint_task",
        "primary_estimand": "paired_oriented_validation_gain_per_compute",
        "primary_selection": "effect_first_score_after_multiseed",
        "required_validation_seeds": 5,
        "minimum_action_effect": 0.005,
        "robust_promotion_policy": (
            "matched_pair_plus_pareto_score_plus_model_diversity"
        ),
        "candidate_timeout_policy": "hard_wall_time_per_candidate",
        "robust_futility_policy": (
            "seed2_gap0.005_time3x_worse_both_protect_action_pair"
        ),
        "robust_futility_applies_to_all_arms": True,
        "shared_feature_cache_is_label_free": True,
        "test_policy": "not_loaded_for_routing_benchmark",
        "test_feedback_to_discovery": False,
        "leaderboard_feedback_to_discovery": False,
        "frozen_invariants": {
            "tdc22_mapping_path": (
                "runs/v27-maplight-gnn-freeze/full-22-task.mapping.json"
            ),
            "tdc22_mapping_sha256": FROZEN_TDC22_MAPPING_SHA256,
            "finalize_research_routing_policy": "legacy",
            "action_memory_snapshot_path": (
                "runs/v5-transport-grid/transport_action_outcomes.jsonl"
            ),
            "action_memory_snapshot_sha256": ACTION_MEMORY_SNAPSHOT_SHA256,
            "action_memory_snapshot_scope": "global_admet",
            "action_memory_snapshot_read_only": True,
            "action_context_registry_path": (
                "reports/admet22_action_context_registry_v1.json"
            ),
            "action_context_registry_sha256": ACTION_CONTEXT_REGISTRY_SHA256,
        },
        "contextual_transport": {
            "context_version": "admet-task-context-v1",
            "distance_weights": {
                "hardness": 0.55,
                "task_type": 0.20,
                "metric": 0.10,
                "category": 0.15,
            },
            "similarity_kernel": "exp(-3 * weighted_context_distance)",
            "minimum_raw_endpoints": 5,
            "minimum_effective_endpoints": 3.0,
            "minimum_max_similarity": 0.35,
        },
        "claim_gates": {
            "complete_task_coverage": 22,
            "action_budget_parity_required": True,
            "primary_bootstrap_ci_lower_bound_must_exceed_zero": True,
            "paired_one_sided_sign_p_max": 0.05,
            "unsafe_actions_max": 0,
            "frozen_tdc22_mapping_must_remain_unchanged": True,
        },
    }
    validate_protocol(payload)
    digest = hashlib.sha256(
        json.dumps(payload, sort_keys=True).encode("utf-8")
    ).hexdigest()
    return {**payload, "sha256": digest}


def validate_protocol(payload: dict[str, Any]) -> None:
    expected_tasks = {item.name for item in TASKS.values()}
    if set(payload.get("tasks", [])) != expected_tasks:
        raise ValueError("EC-VOI protocol must contain the canonical 22 tasks")
    if payload.get("test_feedback_to_discovery") is not False:
        raise ValueError("Test feedback to discovery must be forbidden")
    if payload.get("leaderboard_feedback_to_discovery") is not False:
        raise ValueError("Leaderboard feedback to discovery must be forbidden")
    if payload.get("statistical_unit") != "endpoint_task":
        raise ValueError("The statistical unit must be the endpoint task, not seed")
    arms = payload.get("arms", [])
    names = [arm.get("name") for arm in arms]
    if len(names) != len(set(names)):
        raise ValueError("EC-VOI benchmark arm names must be unique")
    required = {
        "ecvoi",
        "agent",
        "rules",
        "random",
        "automl",
        "expanded_automl",
        "ecvoi_no_context",
        "ecvoi_no_memory",
        "ecvoi_no_voi",
        "ecvoi_no_cost",
        "ecvoi_no_risk",
    }
    if not required.issubset(names):
        raise ValueError(f"Missing EC-VOI controls: {sorted(required - set(names))}")
    budgets = {
        tuple(arm[key] for key in SHARED_BUDGET)
        for arm in arms
    }
    if len(budgets) != 1:
        raise ValueError("Every EC-VOI benchmark arm must use the same budget")
    if len(set(payload.get("discovery_seeds", []))) < 5:
        raise ValueError("At least five distinct discovery seeds are required")


def write_protocol(path: Path) -> dict[str, Any]:
    payload = protocol_payload()
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if existing.get("sha256") != payload["sha256"]:
            raise RuntimeError(
                "Refusing to overwrite a different EC-VOI preregistration"
            )
    else:
        path.write_text(
            json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
        )
    return payload


__all__ = ["protocol_payload", "validate_protocol", "write_protocol"]
