"""Preregistered complete-grid protocol for causal cross-task action memory."""

from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from pathlib import Path
from typing import Any

from .tasks import resolve_task

TRANSPORT_PROTOCOL_VERSION = "tdc22-cross-task-transport-grid-v1"
TRANSPORT_ACTIONS = (
    {
        "action_id": "feature-domain-shift",
        "axis": "feature",
        "treatment": {
            "model": "extra_trees",
            "representations": ["morgan", "avalon", "erg", "rdkit2d"],
            "data_policy": "official",
        },
        "control": {
            "model": "extra_trees",
            "representations": ["morgan", "rdkit2d"],
            "data_policy": "official",
        },
    },
    {
        "action_id": "model-estimator-family",
        "axis": "model",
        "treatment": {
            "model": "random_forest",
            "representations": ["morgan", "rdkit2d"],
            "data_policy": "official",
        },
        "control": {
            "model": "extra_trees",
            "representations": ["morgan", "rdkit2d"],
            "data_policy": "official",
        },
    },
    {
        "action_id": "data-quality-filter",
        "axis": "data",
        "treatment": {
            "model": "extra_trees",
            "representations": ["morgan", "rdkit2d"],
            "data_policy": "quality_filtered",
        },
        "control": {
            "model": "extra_trees",
            "representations": ["morgan", "rdkit2d"],
            "data_policy": "official",
        },
    },
)


def _digest(payload: dict[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def build_transport_protocol(
    source_sha256: str, runner_sha256: str = "0" * 64
) -> dict[str, Any]:
    """Return a complete 5-endpoint x 3-action, validation-only design."""

    body = {
        "protocol_version": TRANSPORT_PROTOCOL_VERSION,
        "assignment_design": "complete_action_grid",
        "prospective": True,
        "same_budget": True,
        "source_sha256": source_sha256,
        "runner_sha256": runner_sha256,
        "sentinel_endpoints": [
            "Caco2_Wang",
            "BBB_Martins",
            "CYP2C9_Veith",
            "Clearance_Hepatocyte_AZ",
            "LD50_Zhu",
        ],
        "sentinel_selection_rule": (
            "Frozen before action outcomes: one endpoint from each ADMET category, "
            "all with nonzero train-only data-quality diagnostics and high domain "
            "shift, spanning classification/regression and varied local roughness."
        ),
        "actions": deepcopy(TRANSPORT_ACTIONS),
        "validation_seeds": [1, 2, 3, 4, 5],
        "minimum_effect": 0.005,
        "minimum_direction_agreement": 0.8,
        "fits_per_matched_episode": 10,
        "matched_episodes": 15,
        "shared_control_deduplication": True,
        "unique_configurations_per_endpoint_seed": 4,
        "maximum_total_fits": 100,
        "max_compute_minutes_per_endpoint": 30,
        "model_hyperparameters": {
            "extra_trees": {
                "n_estimators": 300,
                "min_samples_leaf": 2,
                "max_features": "sqrt",
            },
            "random_forest": {
                "n_estimators": 300,
                "min_samples_leaf": 2,
                "max_features": "sqrt",
            },
        },
        "shared_conditions": {
            "selection_policy": "none_complete_grid",
            "n_jobs": 8,
            "device": "cpu",
            "feature_cache_allowed": True,
            "discovery_model_persistence": False,
            "failed_fits_retained": True,
            "negative_outcomes_retained": True,
        },
        "transport_admission": {
            "minimum_unique_endpoints_per_action": 5,
            "requires_outcome_variation": True,
            "requires_protocol_sha256_match": True,
            "requires_source_sha256_match": True,
            "routing_updates_only_after_grid_completion": True,
        },
        "test_loaded": False,
        "test_feedback_to_discovery": False,
        "purpose": (
            "Estimate action-by-hardness transport effects and memory calibration; "
            "do not tune or reopen frozen application recipes."
        ),
    }
    validate_transport_protocol(body)
    return {**body, "sha256": _digest(body)}


def validate_transport_protocol(payload: dict[str, Any]) -> None:
    if payload.get("assignment_design") != "complete_action_grid":
        raise ValueError("Transport evidence requires a complete action grid")
    if payload.get("prospective") is not True or payload.get("same_budget") is not True:
        raise ValueError("Transport grid must be prospective and same-budget")
    endpoints = payload.get("sentinel_endpoints", [])
    if len(endpoints) != 5 or len(set(endpoints)) != 5:
        raise ValueError("Exactly five unique sentinel endpoints are required")
    for endpoint in endpoints:
        resolve_task(endpoint)
    actions = payload.get("actions", [])
    if len(actions) != 3 or {row.get("axis") for row in actions} != {
        "feature",
        "model",
        "data",
    }:
        raise ValueError("Feature, model and data matched actions are required")
    condition_key = {
        "feature": "representations",
        "model": "model",
        "data": "data_policy",
    }
    for action in actions:
        treatment = action.get("treatment", {})
        control = action.get("control", {})
        changed = {
            key
            for key in ("model", "representations", "data_policy")
            if treatment.get(key) != control.get(key)
        }
        if changed != {condition_key[action["axis"]]}:
            raise ValueError(
                f"{action.get('action_id')} is not a single-axis matched contrast"
            )
    if len(payload.get("validation_seeds", [])) != 5:
        raise ValueError("Exactly five validation seeds are required")
    if payload.get("shared_control_deduplication") is not True:
        raise ValueError("Identical matched controls must be deduplicated")
    if payload.get("maximum_total_fits") != 100:
        raise ValueError(
            "The deduplicated complete grid must cap execution at 100 fits"
        )
    if payload.get("test_loaded") is not False:
        raise ValueError("Transport protocol must attest test_loaded=false")
    if payload.get("test_feedback_to_discovery") is not False:
        raise ValueError("Test feedback is forbidden")
    source_sha256 = payload.get("source_sha256")
    if not isinstance(source_sha256, str) or len(source_sha256) != 64:
        raise ValueError("A production source SHA-256 is required")
    runner_sha256 = payload.get("runner_sha256")
    if not isinstance(runner_sha256, str) or len(runner_sha256) != 64:
        raise ValueError("A runner SHA-256 is required")


def load_transport_protocol(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    claimed = payload.pop("sha256", None)
    validate_transport_protocol(payload)
    if claimed != _digest(payload):
        raise ValueError("Transport protocol SHA-256 is invalid")
    return {**payload, "sha256": claimed}
