"""Practical noninferiority plus compute-saving certification for a locked router."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class PracticalEfficiencyPolicy:
    noninferiority_margin: float = 0.005
    minimum_nonmaterial_task_rate: float = 0.90
    maximum_aggregate_cost_ratio: float = 0.35
    version: str = "practical-efficiency-gate-v1"

    def __post_init__(self) -> None:
        if (
            not math.isfinite(self.noninferiority_margin)
            or self.noninferiority_margin <= 0
        ):
            raise ValueError("noninferiority_margin must be finite and positive")
        for name in (
            "minimum_nonmaterial_task_rate",
            "maximum_aggregate_cost_ratio",
        ):
            value = getattr(self, name)
            if not math.isfinite(value) or not 0 < value <= 1:
                raise ValueError(f"{name} must be within (0, 1]")


@dataclass(frozen=True)
class PracticalEfficiencyCertification:
    passed: bool
    policy: PracticalEfficiencyPolicy
    task_count: int
    mean_oriented_gain: float
    bootstrap_95_ci: tuple[float, float]
    nonmaterial_task_rate: float
    material_regressions: int
    worst_task: str
    worst_oriented_gain: float
    aggregate_cost_ratio: float
    estimated_cost_reduction: float
    blockers: tuple[str, ...]
    evidence_scope: str
    test_loaded: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def certify_router_efficiency(
    report: dict[str, Any],
    policy: PracticalEfficiencyPolicy | None = None,
) -> PracticalEfficiencyCertification:
    policy = policy or PracticalEfficiencyPolicy()
    if report.get("official_test_loaded") is not False:
        raise ValueError("Efficiency certification requires validation-only evidence")
    if report.get("selection_or_official_freeze_changed") is not False:
        raise ValueError("Locked efficiency evidence cannot change the official freeze")
    tasks = report.get("tasks")
    if not isinstance(tasks, list) or not tasks:
        raise ValueError("Task-level router evidence is required")
    gains = [float(row["guard_minus_score_oriented_gain"]) for row in tasks]
    if any(not math.isfinite(value) for value in gains):
        raise ValueError("Task gains must be finite")
    summary = report["guard_minus_score"]
    interval = tuple(float(value) for value in summary["bootstrap_mean_95_ci"])
    if len(interval) != 2 or interval[0] > interval[1]:
        raise ValueError("Bootstrap interval is invalid")
    cost_ratio = float(report["claim_gate"]["aggregate_guard_cost_ratio_vs_score"])
    if not math.isfinite(cost_ratio) or cost_ratio < 0:
        raise ValueError("Aggregate cost ratio must be finite and non-negative")
    threshold = policy.noninferiority_margin
    material = [value < -threshold for value in gains]
    nonmaterial_rate = 1.0 - sum(material) / len(gains)
    worst_index = min(range(len(gains)), key=gains.__getitem__)
    blockers = []
    if interval[0] < -threshold:
        blockers.append("bootstrap_lower_bound_below_noninferiority_margin")
    if nonmaterial_rate < policy.minimum_nonmaterial_task_rate:
        blockers.append("nonmaterial_task_rate_below_threshold")
    if cost_ratio > policy.maximum_aggregate_cost_ratio:
        blockers.append("aggregate_cost_ratio_above_threshold")
    return PracticalEfficiencyCertification(
        passed=not blockers,
        policy=policy,
        task_count=len(tasks),
        mean_oriented_gain=float(summary["mean"]),
        bootstrap_95_ci=(interval[0], interval[1]),
        nonmaterial_task_rate=nonmaterial_rate,
        material_regressions=sum(material),
        worst_task=str(tasks[worst_index]["task"]),
        worst_oriented_gain=gains[worst_index],
        aggregate_cost_ratio=cost_ratio,
        estimated_cost_reduction=1.0 - cost_ratio,
        blockers=tuple(blockers),
        evidence_scope=(
            "fresh_validation_practical_noninferiority_vs_score_selection; "
            "not Agent-vs-AutoML predictive superiority"
        ),
        test_loaded=False,
    )


__all__ = [
    "PracticalEfficiencyCertification",
    "PracticalEfficiencyPolicy",
    "certify_router_efficiency",
]
