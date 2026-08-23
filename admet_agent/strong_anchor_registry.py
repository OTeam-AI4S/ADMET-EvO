"""Test-safe registry and routing for reproducible strong model anchors.

Leaderboard observations may motivate which *families* are audited before a
protocol is frozen, but endpoint leaderboard rank and test score are forbidden
routing inputs.  Runtime eligibility depends only on train/validation-visible
metadata and an executable, content-addressed backend.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from typing import Any, Literal

from .tasks import resolve_task

AnchorCost = Literal["low", "moderate", "high"]

_FORBIDDEN_INPUT_TOKENS = frozenset(
    {"test", "testing", "heldout", "holdout", "leaderboard", "rank"}
)
_ALLOWED_SIGNALS = frozenset(
    {
        "domain_shift_severity",
        "imbalance_severity",
        "maplight_runtime_available",
        "minimol_runtime_available",
    }
)


def _sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode()
    ).hexdigest()


def _reject_forbidden_inputs(value: Any, path: str = "signals") -> None:
    if isinstance(value, Mapping):
        for raw_key, child in value.items():
            key = str(raw_key).strip().lower().replace("-", "_")
            tokens = tuple(token for token in key.split("_") if token)
            if any(token in _FORBIDDEN_INPUT_TOKENS for token in tokens):
                raise ValueError(f"Test/leaderboard-derived anchor input forbidden: {path}.{raw_key}")
            _reject_forbidden_inputs(child, f"{path}.{raw_key}")
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            _reject_forbidden_inputs(child, f"{path}[{index}]")


@dataclass(frozen=True)
class StrongAnchor:
    anchor_id: str
    recipe: str
    supported_task_types: tuple[str, ...]
    supported_metrics: tuple[str, ...]
    cost_class: AnchorCost
    backend: str
    source_identity: str
    license: str
    validation_status: str
    universal_default: bool = False
    maximum_train_rows: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


ANCHORS: tuple[StrongAnchor, ...] = (
    StrongAnchor(
        anchor_id="rich2d_lightgbm",
        recipe="2d_lightgbm",
        supported_task_types=("classification", "regression"),
        supported_metrics=("auroc", "auprc", "mae", "spearman"),
        cost_class="low",
        backend="tabular",
        source_identity="internal-rich2d-schema-v1",
        license="project-internal",
        validation_status="existing_22_task_anchor",
        universal_default=True,
    ),
    StrongAnchor(
        anchor_id="rich2d_extra_trees",
        recipe="2d_extra_trees",
        supported_task_types=("classification", "regression"),
        supported_metrics=("auroc", "auprc", "mae", "spearman"),
        cost_class="low",
        backend="tabular",
        source_identity="internal-rich2d-schema-v1",
        license="project-internal",
        validation_status="existing_22_task_anchor",
        universal_default=True,
    ),
    StrongAnchor(
        anchor_id="maplight_gnn_classification",
        recipe="maplight_gnn",
        supported_task_types=("classification",),
        supported_metrics=("auroc", "auprc"),
        cost_class="moderate",
        backend="isolated_maplight_gnn",
        source_identity=(
            "commit:c249378c63232354d17083c83fe94fe728960a27;"
            "source-sha256:74fbd1c98d9afa7fa4bda1add21efd429e20dee0a4b0fb8fa7e9b3825c21fe13;"
            "checkpoint-sha256:6d0f8febad73e437772ebffc2ac32253d79f86ee138cfc233590ae50fb1cfeb9"
        ),
        license="MIT",
        validation_status="fresh_seed_6_10_confirmation_on_two_small_classification_tasks",
    ),
    StrongAnchor(
        anchor_id="minimol_morgan_kernel_small_auroc",
        recipe="minimol_morgan_kernel",
        supported_task_types=("classification",),
        supported_metrics=("auroc",),
        cost_class="moderate",
        backend="precomputed_kernel",
        source_identity="minimol:1.3.5;embedding-width:512;kernel-spec:v1",
        license="MIT",
        validation_status="fresh_scaffold_confirmation_bioavailability_only",
        maximum_train_rows=1000,
    ),
    StrongAnchor(
        anchor_id="focal_minimol_small_auprc",
        recipe="focal_minimol_16",
        supported_task_types=("classification",),
        supported_metrics=("auprc",),
        cost_class="moderate",
        backend="isolated_torch_head",
        source_identity="minimol:1.3.5;focal-head:16x100-v1",
        license="MIT",
        validation_status="metric_aligned_validation_candidate_not_universal",
        maximum_train_rows=2000,
    ),
)


def registry_manifest() -> dict[str, Any]:
    core = {
        "protocol": "test-safe-strong-anchor-registry-v1",
        "anchors": [anchor.to_dict() for anchor in ANCHORS],
        "forbidden_routing_inputs": sorted(_FORBIDDEN_INPUT_TOKENS),
        "unimol2_default_admitted": False,
        "test_loaded": False,
    }
    return {**core, "registry_sha256": _sha256(core)}


def route_strong_anchors(
    task_name: str,
    n_train: int,
    *,
    signals: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Return a bounded anchor screen without consulting leaderboard/test data."""

    if isinstance(n_train, bool) or not isinstance(n_train, int) or n_train <= 0:
        raise ValueError("n_train must be a positive integer")
    values = dict(signals or {})
    _reject_forbidden_inputs(values)
    unknown = set(values) - _ALLOWED_SIGNALS
    if unknown:
        raise ValueError(f"Unknown strong-anchor signals: {sorted(unknown)}")
    for name in ("domain_shift_severity", "imbalance_severity"):
        if name in values:
            value = float(values[name])
            if not 0 <= value <= 1:
                raise ValueError(f"{name} must be within [0, 1]")
    task = resolve_task(task_name)
    maplight_available = values.get("maplight_runtime_available", True) is True
    minimol_available = values.get("minimol_runtime_available", True) is True
    selected = []
    blocked = []
    for anchor in ANCHORS:
        reason = None
        if task.task_type not in anchor.supported_task_types:
            reason = "task_type_unsupported"
        elif task.metric not in anchor.supported_metrics:
            reason = "metric_unsupported"
        elif anchor.maximum_train_rows is not None and n_train > anchor.maximum_train_rows:
            reason = "train_rows_above_preregistered_scope"
        elif anchor.backend == "isolated_maplight_gnn" and not maplight_available:
            reason = "runtime_unavailable"
        elif "minimol" in anchor.anchor_id and not minimol_available:
            reason = "runtime_unavailable"
        if reason is None:
            selected.append(anchor)
        else:
            blocked.append({"anchor_id": anchor.anchor_id, "reason": reason})

    # The first screen remains intentionally bounded.  Universal low-cost
    # anchors are controls; at most one pretrained family is admitted as the
    # treatment, selected by protocol priority rather than leaderboard rank.
    controls = [anchor for anchor in selected if anchor.universal_default]
    pretrained = [anchor for anchor in selected if not anchor.universal_default]
    priority = {
        "maplight_gnn_classification": 0,
        "minimol_morgan_kernel_small_auroc": 1,
        "focal_minimol_small_auprc": 1,
    }
    treatment = min(pretrained, key=lambda item: priority[item.anchor_id], default=None)
    admitted = [*controls, *([treatment] if treatment else [])]
    core = {
        "protocol": "test-safe-strong-anchor-routing-v1",
        "task": task.name,
        "task_type": task.task_type,
        "metric": task.metric,
        "n_train": n_train,
        "admitted_anchor_ids": [item.anchor_id for item in admitted],
        "control_anchor_ids": [item.anchor_id for item in controls],
        "treatment_anchor_id": treatment.anchor_id if treatment else None,
        "blocked": blocked,
        "maximum_pretrained_treatments": 1,
        "leaderboard_or_test_used_for_routing": False,
        "unimol2_admitted": False,
        "required_validation_design": {
            "single_axis": "replace_only_model_or_representation_family",
            "paired_seeds": [1, 2, 3, 4, 5],
            "minimum_oriented_gain": 0.005,
            "minimum_positive_seed_fraction": 0.8,
            "promotion_uses_test": False,
        },
        "test_loaded": False,
    }
    return {**core, "decision_sha256": _sha256(core)}


__all__ = [
    "ANCHORS",
    "StrongAnchor",
    "registry_manifest",
    "route_strong_anchors",
]
