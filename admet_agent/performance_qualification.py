"""Seal existing official TDC-22 outcomes as a non-optimizing qualification gate.

This module deliberately consumes only the aggregate, already-frozen scorecard.
It does not load test labels, rank candidate models, or return information that a
discovery policy may use to modify a finalist.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

_SHA256 = re.compile(r"^[0-9a-f]{64}$")


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


@dataclass(frozen=True)
class PerformanceQualificationPolicy:
    """User-approved application floor, not a leaderboard optimization target."""

    required_tasks: int = 22
    minimum_top5_tasks: int = 15
    minimum_top10_tasks: int = 21
    official_results_must_be_complete: bool = True
    selection_feedback_to_discovery: bool = False
    version: str = "tdc22-performance-qualification-v1"

    def __post_init__(self) -> None:
        if self.required_tasks != 22:
            raise ValueError(
                "The qualification policy is fixed to the canonical TDC-22 suite"
            )
        if not 0 <= self.minimum_top5_tasks <= self.minimum_top10_tasks <= 22:
            raise ValueError("Top-5/Top-10 thresholds are inconsistent")
        if self.selection_feedback_to_discovery:
            raise ValueError("Frozen official results cannot feed back into discovery")


@dataclass(frozen=True)
class PerformanceQualification:
    passed: bool
    policy: PerformanceQualificationPolicy
    tasks_observed: int
    top5_tasks: int
    top10_tasks: int
    exception_tasks: tuple[str, ...]
    full_freeze_sha256: str
    aggregate_scorecard_sha256: str
    official_results_complete: bool
    provenance_coverage: float
    test_labels_loaded: bool
    selection_feedback_to_discovery: bool
    evidence_role: str
    claim_boundary: str
    blockers: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["seal_sha256"] = _canonical_sha256(payload)
        return payload


def qualify_frozen_scorecard(
    scorecard: dict[str, Any],
    policy: PerformanceQualificationPolicy | None = None,
) -> PerformanceQualification:
    """Validate and seal an aggregate scorecard without exposing model feedback."""

    policy = policy or PerformanceQualificationPolicy()
    tasks = scorecard.get("tasks")
    if not isinstance(tasks, list):
        raise TypeError("Scorecard tasks must be a list")
    task_names = [row.get("task") for row in tasks if isinstance(row, dict)]
    if len(task_names) != len(tasks) or any(not name for name in task_names):
        raise ValueError("Every scorecard row must have a task name")
    if len(task_names) != len(set(task_names)):
        raise ValueError("Scorecard contains duplicate tasks")

    freeze_sha256 = scorecard.get("freeze_sha256")
    if not isinstance(freeze_sha256, str) or not _SHA256.fullmatch(freeze_sha256):
        raise ValueError("Scorecard must reference a valid full-freeze SHA-256")

    ranks = [row.get("estimated_rank") for row in tasks]
    if any(not isinstance(rank, int) or rank < 1 for rank in ranks):
        raise ValueError(
            "Every scorecard row must have a positive integer estimated_rank"
        )
    top5 = sum(rank <= 5 for rank in ranks)
    top10 = sum(rank <= 10 for rank in ranks)
    status = scorecard.get("status", {})
    if status.get("top5") != top5 or status.get("top10") != top10:
        raise ValueError("Persisted rank counts do not match scorecard rows")

    provenance = scorecard.get("provenance", {})
    provenance_names = set(provenance) if isinstance(provenance, dict) else set()
    observed_names = set(task_names)
    provenance_coverage = len(observed_names & provenance_names) / policy.required_tasks
    official_complete = scorecard.get("official_results_complete") is True
    exceptions = tuple(
        sorted(row["task"] for row in tasks if row["estimated_rank"] > 10)
    )

    blockers: list[str] = []
    if len(tasks) != policy.required_tasks:
        blockers.append(f"task coverage is {len(tasks)}/{policy.required_tasks}")
    if top5 < policy.minimum_top5_tasks:
        blockers.append(f"Top-5 coverage is {top5}, below {policy.minimum_top5_tasks}")
    if top10 < policy.minimum_top10_tasks:
        blockers.append(
            f"Top-10 coverage is {top10}, below {policy.minimum_top10_tasks}"
        )
    if policy.official_results_must_be_complete and not official_complete:
        blockers.append("official results are incomplete")
    if provenance_coverage != 1.0:
        blockers.append("not every endpoint has frozen provenance")

    # This object is intentionally a one-way certification record.  It cannot
    # authorize candidate selection, new training or post-test reweighting.
    return PerformanceQualification(
        passed=not blockers,
        policy=policy,
        tasks_observed=len(tasks),
        top5_tasks=top5,
        top10_tasks=top10,
        exception_tasks=exceptions,
        full_freeze_sha256=freeze_sha256,
        aggregate_scorecard_sha256=_canonical_sha256(scorecard),
        official_results_complete=official_complete,
        provenance_coverage=provenance_coverage,
        test_labels_loaded=False,
        selection_feedback_to_discovery=False,
        evidence_role="application_performance_qualification_only",
        claim_boundary=(
            "Does not establish predictive superiority, research-process advantage, "
            "or general autonomous-science capability."
        ),
        blockers=tuple(blockers),
    )


def qualify_frozen_scorecard_file(
    path: Path,
    policy: PerformanceQualificationPolicy | None = None,
) -> PerformanceQualification:
    return qualify_frozen_scorecard(
        json.loads(path.read_text(encoding="utf-8")), policy=policy
    )
