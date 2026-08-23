"""Minimal category × data-size policy for the fixed TDC ADMET-22 suite.

The default path is deliberately small: validate the endpoint contract, run
one single-axis matched pair plus at most one strong anchor, confirm the pair,
then stop or make one bounded correction.  Optional subsystems are activated
only by explicit failures, not on every task.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from typing import Any, Literal

from .schemas import Budget, ExperimentSpec, TaskProfile
from .tasks import resolve_task

SizeBucket = Literal["small", "medium", "large"]

_CATEGORY_CHECKS = {
    "absorption": (
        "endpoint_and_assay_match",
        "unit_and_transform_contract",
        "scaffold_split_identity_audit",
    ),
    "distribution": (
        "endpoint_and_assay_match",
        "species_and_matrix_contract",
        "protein_binding_or_distribution_unit_contract",
    ),
    "metabolism": (
        "cyp_isoform_contract",
        "substrate_vs_inhibitor_endpoint_contract",
        "class_balance_and_scaffold_audit",
    ),
    "excretion": (
        "clearance_or_half_life_endpoint_contract",
        "species_unit_and_transform_contract",
        "rank_metric_contract",
    ),
    "toxicity": (
        "toxicity_endpoint_and_assay_contract",
        "species_and_censoring_contract",
        "class_balance_or_target_scale_audit",
    ),
}

# Validation-only, hash-bound outcomes from the preregistered exact-pair pilot.
# These are task-local memories: they prevent duplicate work but are never
# transported as proof that the same action works on another endpoint.
_FROZEN_EXACT_PAIR_EVIDENCE = {
    "CYP2C9_Veith": {
        "task": "CYP2C9_Veith",
        "action_id": "model-imbalance-aware",
        "status": "inconclusive",
        "single_axis_verified": True,
        "validation_seeds": [1, 2, 3, 4, 5],
        "mean_oriented_delta": 0.002640980238839452,
        "positive_delta_rate": 0.6,
        "evidence_sha256": "8aa3e23ca06e479e30314e012ac14586b00c41bc281266516cf148ca1af6b55e",
        "test_loaded": False,
    },
    "LD50_Zhu": {
        "task": "LD50_Zhu",
        "action_id": "model-lightgbm-vs-extra-trees",
        "status": "refuted",
        "single_axis_verified": True,
        "validation_seeds": [1, 2],
        "mean_oriented_delta": -0.03801765025294371,
        "positive_delta_rate": 0.0,
        "evidence_sha256": "8aa3e23ca06e479e30314e012ac14586b00c41bc281266516cf148ca1af6b55e",
        "test_loaded": False,
    },
}


def _forbidden_test_paths(value: Any, path: str = "evidence") -> tuple[str, ...]:
    forbidden = []
    if isinstance(value, Mapping):
        for raw_key, child in value.items():
            text = str(raw_key)
            normalized = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", text).lower()
            tokens = tuple(token for token in re.split(r"[^a-z0-9]+", normalized) if token)
            safe = normalized == "test_loaded" and child is False
            if normalized not in {"falsification_test", "falsification_tests"} and not safe and any(
                token in {"test", "testing", "heldout", "holdout"} for token in tokens
            ):
                forbidden.append(f"{path}.{text}")
            forbidden.extend(_forbidden_test_paths(child, f"{path}.{text}"))
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for index, child in enumerate(value):
            forbidden.extend(_forbidden_test_paths(child, f"{path}[{index}]"))
    return tuple(sorted(set(forbidden)))


def size_bucket(n_train: int) -> SizeBucket:
    if isinstance(n_train, bool) or not isinstance(n_train, int) or n_train <= 0:
        raise ValueError("n_train must be a positive integer")
    if n_train < 1000:
        return "small"
    if n_train < 5000:
        return "medium"
    return "large"


@dataclass(frozen=True)
class MinimalTaskPolicy:
    task: str
    category: str
    task_type: str
    metric: str
    n_train: int
    size_bucket: SizeBucket
    endpoint_contract_checks: tuple[str, ...]
    scientific_axis: str
    matched_control: str
    matched_treatment: str
    optional_strong_anchor: str | None
    screen_seeds: tuple[int, ...]
    confirmation_seeds: tuple[int, ...]
    maximum_screen_configs: int
    maximum_hypotheses: int
    maximum_corrections: int
    maximum_hpo_slots: int
    maximum_pretrained_anchor_slots: int
    typical_fit_ceiling: int
    corrected_fit_ceiling: int
    activation_status: str = "shadow_only_pending_validation_noninferiority"
    default_committee: bool = False
    default_ensemble: bool = False
    default_unimol2: bool = False
    default_external_data_search: bool = False
    test_loaded: bool = False

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["policy_sha256"] = hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        return payload


@dataclass(frozen=True)
class MinimalExperimentBundle:
    task: str
    pair_admission: Mapping[str, Any]
    matched_specs: tuple[ExperimentSpec, ...]
    frozen_anchor_recipe: str | None
    maximum_total_configurations: int
    test_loaded: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "task": self.task,
            "pair_admission": dict(self.pair_admission),
            "matched_specs": [spec.to_dict() for spec in self.matched_specs],
            "frozen_anchor_recipe": self.frozen_anchor_recipe,
            "maximum_total_configurations": self.maximum_total_configurations,
            "test_loaded": self.test_loaded,
        }


def _matched_pair(category: str, task_type: str, metric: str, bucket: SizeBucket) -> tuple[str, str, str]:
    if category == "metabolism" and task_type == "classification":
        return (
            "model",
            "rich2d_extra_trees_unweighted",
            "rich2d_extra_trees_class_weighted",
        )
    if category == "excretion" or bucket in {"small", "medium"}:
        return (
            "feature",
            "basic2d_extra_trees",
            "rich2d_extra_trees",
        )
    return (
        "model",
        "rich2d_lightgbm",
        "rich2d_extra_trees",
    )


def build_minimal_task_policy(task_name: str, n_train: int) -> MinimalTaskPolicy:
    task = resolve_task(task_name)
    bucket = size_bucket(n_train)
    axis, control, treatment = _matched_pair(
        task.category, task.task_type, task.metric, bucket
    )
    if task.task_type == "classification" and bucket != "large":
        anchor = "maplight_gnn_frozen"
        pretrained_slots = 1
    elif bucket == "large":
        anchor = None
        pretrained_slots = 0
    else:
        anchor = "rich2d_lightgbm"
        pretrained_slots = 0
    screen_seeds = (1,) if bucket == "large" else (1, 2)
    confirmation_seeds = tuple(seed for seed in (1, 2, 3, 4, 5) if seed not in screen_seeds)
    screen_configs = 3 if anchor is not None else 2
    screen_fits = screen_configs * len(screen_seeds)
    confirmation_fits = 2 * len(confirmation_seeds)
    typical = screen_fits + confirmation_fits
    return MinimalTaskPolicy(
        task=task.name,
        category=task.category,
        task_type=task.task_type,
        metric=task.metric,
        n_train=n_train,
        size_bucket=bucket,
        endpoint_contract_checks=_CATEGORY_CHECKS[task.category],
        scientific_axis=axis,
        matched_control=control,
        matched_treatment=treatment,
        optional_strong_anchor=anchor,
        screen_seeds=screen_seeds,
        confirmation_seeds=confirmation_seeds,
        maximum_screen_configs=screen_configs,
        maximum_hypotheses=1,
        maximum_corrections=1,
        maximum_hpo_slots=0 if bucket == "small" else 1,
        maximum_pretrained_anchor_slots=pretrained_slots,
        typical_fit_ceiling=typical,
        corrected_fit_ceiling=typical + 6,
    )


def build_minimal_pair_specs(profile: TaskProfile) -> tuple[ExperimentSpec, ExperimentSpec]:
    """Materialize the policy's exact matched pair for the shared executor."""

    policy = build_minimal_task_policy(profile.task, profile.n_train)
    rich = ("morgan", "avalon", "erg", "rdkit2d")
    basic = ("morgan", "rdkit2d")
    common = {
        "task": profile.task,
        "seed": 1,
        "data_policy": "official",
        "device": "cpu",
        "n_jobs": 8,
        "budget": Budget(max_candidates=policy.maximum_screen_configs),
    }
    if policy.matched_control == "basic2d_extra_trees":
        control = ExperimentSpec(
            **common,
            representations=basic,
            model="extra_trees",
            imbalance="none",
            hypothesis="matched feature control: basic 2D representation",
        )
        treatment = ExperimentSpec(
            **common,
            representations=rich,
            model="extra_trees",
            imbalance="none",
            hypothesis="feature treatment: rich 2D representation",
        )
    elif policy.matched_control == "rich2d_extra_trees_unweighted":
        control = ExperimentSpec(
            **common,
            representations=rich,
            model="extra_trees",
            imbalance="none",
            hypothesis="matched imbalance control: unweighted ExtraTrees",
        )
        treatment = ExperimentSpec(
            **common,
            representations=rich,
            model="extra_trees",
            imbalance="class_weight",
            hypothesis="imbalance treatment: class-weighted ExtraTrees",
        )
    else:
        control = ExperimentSpec(
            **common,
            representations=rich,
            model="lightgbm",
            imbalance="none",
            hypothesis="matched model control: rich-2D LightGBM",
        )
        treatment = ExperimentSpec(
            **common,
            representations=rich,
            model="extra_trees",
            imbalance="none",
            hypothesis="model treatment: rich-2D ExtraTrees",
        )
    return control, treatment


def minimal_frozen_anchor_recipe(profile: TaskProfile) -> str | None:
    """Return the optional separately executed anchor; never a search loop."""

    policy = build_minimal_task_policy(profile.task, profile.n_train)
    mapping = {
        "maplight_gnn_frozen": "maplight_gnn",
        "rich2d_lightgbm": "2d_lightgbm",
    }
    return mapping.get(policy.optional_strong_anchor)


def admit_minimal_pair(
    policy: MinimalTaskPolicy,
    existing_evidence: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Reuse or block exact task-local evidence before scheduling new fits."""

    evidence = dict(existing_evidence or {})
    forbidden = _forbidden_test_paths(evidence)
    if forbidden:
        raise ValueError(f"Test-like fields are forbidden: {list(forbidden)}")
    expected_action = (
        "feature-domain-shift"
        if policy.matched_control == "basic2d_extra_trees"
        else "model-imbalance-aware"
        if policy.matched_control == "rich2d_extra_trees_unweighted"
        else "model-lightgbm-vs-extra-trees"
    )
    exact = (
        evidence.get("task") == policy.task
        and evidence.get("action_id") == expected_action
        and evidence.get("single_axis_verified") is True
    )
    status = str(evidence.get("status", "missing"))
    seed_count = len(evidence.get("validation_seeds", ()))
    if not exact:
        decision = "screen_new_pair"
        reason = "no_exact_task_local_pair_evidence"
    elif status == "supported":
        decision = "reuse_supported_pair_and_stop"
        reason = "exact_supported_pair_already_resolved"
    elif status == "refuted":
        decision = "block_refuted_pair_use_anchor_or_stop"
        reason = "do_not_repeat_exact_refuted_pair"
    elif status == "inconclusive" and seed_count < 3:
        decision = "repair_incomplete_pair_evidence"
        reason = "exact_pair_has_insufficient_paired_seeds"
    else:
        decision = "block_resolved_inconclusive_pair_use_anchor_or_stop"
        reason = "exact_pair_below_resolution_after_sufficient_seeds"
    core = {
        "task": policy.task,
        "expected_action_id": expected_action,
        "exact_task_local_evidence": exact,
        "evidence_status": status,
        "evidence_seed_count": seed_count,
        "decision": decision,
        "reason": reason,
        "new_training_required": decision in {
            "screen_new_pair",
            "repair_incomplete_pair_evidence",
        },
        "test_loaded": False,
    }
    return {
        **core,
        "decision_sha256": hashlib.sha256(
            json.dumps(core, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest(),
    }


def build_minimal_experiment_bundle(
    profile: TaskProfile,
    existing_evidence: Mapping[str, Any] | None = None,
) -> MinimalExperimentBundle:
    """Build the executable pair plus optional frozen anchor instruction."""

    policy = build_minimal_task_policy(profile.task, profile.n_train)
    effective_evidence = (
        _FROZEN_EXACT_PAIR_EVIDENCE.get(profile.task)
        if existing_evidence is None
        else existing_evidence
    )
    admission = admit_minimal_pair(policy, effective_evidence)
    specs = (
        build_minimal_pair_specs(profile)
        if admission["decision"]
        in {"screen_new_pair", "repair_incomplete_pair_evidence"}
        else ()
    )
    anchor = (
        None
        if admission["decision"] == "reuse_supported_pair_and_stop"
        else minimal_frozen_anchor_recipe(profile)
    )
    total = len(specs) + int(anchor is not None)
    if total > policy.maximum_screen_configs:
        raise ValueError("Minimal bundle exceeds the preregistered config ceiling")
    return MinimalExperimentBundle(
        task=profile.task,
        pair_admission=admission,
        matched_specs=tuple(specs),
        frozen_anchor_recipe=anchor,
        maximum_total_configurations=policy.maximum_screen_configs,
    )


__all__ = [
    "MinimalTaskPolicy",
    "admit_minimal_pair",
    "MinimalExperimentBundle",
    "build_minimal_experiment_bundle",
    "build_minimal_pair_specs",
    "build_minimal_task_policy",
    "minimal_frozen_anchor_recipe",
    "size_bucket",
]
