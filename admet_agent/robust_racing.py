"""Pure, preregistrable seed-barrier futility decisions.

The runner is deliberately not imported here.  Callers provide the completed
validation fits and any structurally verified protected configuration IDs;
the functions perform no I/O and do not inspect test data.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from typing import Any

from .schemas import RunResult


@dataclass(frozen=True)
class RobustFutilityPolicy:
    """Predeclared rule for stopping expensive candidates after seed 2."""

    checkpoint_seeds: tuple[int, int] = (1, 2)
    minimum_oriented_mean_gap: float = 0.005
    minimum_time_ratio: float = 3.0
    require_worse_every_seed: bool = True
    policy_version: str = "seed2-futility-v1"

    def __post_init__(self) -> None:
        if len(set(self.checkpoint_seeds)) != 2 or any(
            seed < 0 for seed in self.checkpoint_seeds
        ):
            raise ValueError(
                "checkpoint_seeds must contain two unique non-negative seeds"
            )
        if not math.isfinite(self.minimum_oriented_mean_gap) or (
            self.minimum_oriented_mean_gap <= 0
        ):
            raise ValueError("minimum_oriented_mean_gap must be finite and positive")
        if not math.isfinite(self.minimum_time_ratio) or self.minimum_time_ratio <= 1:
            raise ValueError("minimum_time_ratio must be finite and greater than one")
        if not self.policy_version:
            raise ValueError("policy_version is required")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class FutilityDecision:
    """Complete, hash-addressed evidence for one candidate decision."""

    candidate_id: str
    incumbent_id: str | None
    decision: str
    reason: str
    checkpoint_seeds: tuple[int, ...]
    maximize: bool
    candidate_scores: tuple[float | None, ...]
    incumbent_scores: tuple[float | None, ...]
    oriented_gaps: tuple[float, ...]
    mean_oriented_gap: float | None
    candidate_seconds: float | None
    incumbent_seconds: float | None
    time_ratio: float | None
    worse_every_seed: bool | None
    gap_threshold_met: bool | None
    time_threshold_met: bool | None
    protected: bool
    policy: RobustFutilityPolicy
    decision_sha256: str = field(init=False, default="")

    def __post_init__(self) -> None:
        if self.decision not in {"continue", "stop", "defer"}:
            raise ValueError("decision must be continue, stop, or defer")
        payload = asdict(self)
        payload.pop("decision_sha256", None)
        digest = hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        object.__setattr__(self, "decision_sha256", digest)

    @property
    def decision_id(self) -> str:
        """Stable identifier suitable for idempotent evidence-ledger writes."""

        return self.decision_sha256

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


SeedRuns = Mapping[str, Mapping[int, RunResult]]


@dataclass(frozen=True)
class _CheckpointEvidence:
    scores: tuple[float, ...]
    seconds: float


def _checkpoint_evidence(
    config_id: str,
    seed_runs: SeedRuns,
    policy: RobustFutilityPolicy,
    *,
    role: str,
) -> tuple[_CheckpointEvidence | None, str | None]:
    runs = seed_runs.get(config_id)
    if runs is None:
        return None, f"{role}_missing"
    scores = []
    elapsed = []
    for seed in policy.checkpoint_seeds:
        result = runs.get(seed)
        if result is None:
            return None, f"{role}_missing_checkpoint_seed"
        if result.status != "completed":
            return None, f"{role}_failed_checkpoint_fit"
        score = float(result.validation_score)
        seconds = float(result.elapsed_seconds)
        if not math.isfinite(score):
            return None, f"{role}_nonfinite_score"
        if not math.isfinite(seconds) or seconds <= 0:
            return None, f"{role}_invalid_elapsed"
        scores.append(score)
        elapsed.append(seconds)
    return _CheckpointEvidence(tuple(scores), float(sum(elapsed))), None


def choose_incumbent(
    seed_runs: SeedRuns,
    *,
    maximize: bool,
    policy: RobustFutilityPolicy | None = None,
) -> str | None:
    """Choose the best valid two-seed candidate with deterministic tie-breaks.

    Invalid candidates are not silently declared weak; their own decisions
    remain ``defer``.  Among valid candidates, the order is oriented mean
    score, lower cumulative time, then lexical configuration ID.
    """

    policy = policy or RobustFutilityPolicy()
    direction = 1.0 if maximize else -1.0
    valid: list[tuple[str, _CheckpointEvidence]] = []
    for config_id in sorted(seed_runs):
        runs = seed_runs[config_id]
        if any(seed not in runs for seed in policy.checkpoint_seeds):
            # A seed barrier is global: do not freeze an incumbent while any
            # candidate is still waiting to reach the checkpoint.
            return None
        evidence, _ = _checkpoint_evidence(
            config_id, seed_runs, policy, role="candidate"
        )
        if evidence is not None:
            valid.append((config_id, evidence))
    if not valid:
        return None
    return min(
        valid,
        key=lambda item: (
            -direction * (sum(item[1].scores) / len(item[1].scores)),
            item[1].seconds,
            item[0],
        ),
    )[0]


def _deferred_decision(
    candidate_id: str,
    incumbent_id: str | None,
    *,
    maximize: bool,
    protected: bool,
    policy: RobustFutilityPolicy,
    reason: str,
    candidate: _CheckpointEvidence | None = None,
    incumbent: _CheckpointEvidence | None = None,
) -> FutilityDecision:
    width = len(policy.checkpoint_seeds)
    return FutilityDecision(
        candidate_id=candidate_id,
        incumbent_id=incumbent_id,
        decision="defer",
        reason=reason,
        checkpoint_seeds=policy.checkpoint_seeds,
        maximize=maximize,
        candidate_scores=(
            candidate.scores if candidate is not None else (None,) * width
        ),
        incumbent_scores=(
            incumbent.scores if incumbent is not None else (None,) * width
        ),
        oriented_gaps=(),
        mean_oriented_gap=None,
        candidate_seconds=candidate.seconds if candidate is not None else None,
        incumbent_seconds=incumbent.seconds if incumbent is not None else None,
        time_ratio=None,
        worse_every_seed=None,
        gap_threshold_met=None,
        time_threshold_met=None,
        protected=protected,
        policy=policy,
    )


def evaluate_futility(
    candidate_id: str,
    incumbent_id: str | None,
    seed_runs: SeedRuns,
    *,
    maximize: bool,
    protected_ids: frozenset[str] = frozenset(),
    policy: RobustFutilityPolicy | None = None,
) -> FutilityDecision:
    """Evaluate one candidate against the frozen seed-2 incumbent."""

    policy = policy or RobustFutilityPolicy()
    protected = candidate_id in protected_ids
    candidate, candidate_error = _checkpoint_evidence(
        candidate_id, seed_runs, policy, role="candidate"
    )
    if candidate_error is not None:
        return _deferred_decision(
            candidate_id,
            incumbent_id,
            maximize=maximize,
            protected=protected,
            policy=policy,
            reason=candidate_error,
        )
    if incumbent_id is None:
        return _deferred_decision(
            candidate_id,
            None,
            maximize=maximize,
            protected=protected,
            policy=policy,
            reason="no_valid_incumbent",
            candidate=candidate,
        )
    incumbent, incumbent_error = _checkpoint_evidence(
        incumbent_id, seed_runs, policy, role="incumbent"
    )
    if incumbent_error is not None:
        return _deferred_decision(
            candidate_id,
            incumbent_id,
            maximize=maximize,
            protected=protected,
            policy=policy,
            reason=incumbent_error,
            candidate=candidate,
        )

    direction = 1.0 if maximize else -1.0
    oriented_gaps = tuple(
        direction * (incumbent_score - candidate_score)
        for candidate_score, incumbent_score in zip(candidate.scores, incumbent.scores)
    )
    mean_gap = float(sum(oriented_gaps) / len(oriented_gaps))
    time_ratio = float(candidate.seconds / incumbent.seconds)
    worse_every_seed = all(gap > 0 for gap in oriented_gaps)
    gap_threshold_met = mean_gap >= policy.minimum_oriented_mean_gap
    time_threshold_met = time_ratio >= policy.minimum_time_ratio
    criteria_met = (
        (worse_every_seed or not policy.require_worse_every_seed)
        and gap_threshold_met
        and time_threshold_met
    )
    if candidate_id == incumbent_id:
        decision = "continue"
        reason = "incumbent"
    elif protected:
        decision = "continue"
        reason = "protected_selected_action_pair"
    elif criteria_met:
        decision = "stop"
        reason = "futility_thresholds_met"
    else:
        decision = "continue"
        reason = "futility_thresholds_not_met"
    return FutilityDecision(
        candidate_id=candidate_id,
        incumbent_id=incumbent_id,
        decision=decision,
        reason=reason,
        checkpoint_seeds=policy.checkpoint_seeds,
        maximize=maximize,
        candidate_scores=candidate.scores,
        incumbent_scores=incumbent.scores,
        oriented_gaps=oriented_gaps,
        mean_oriented_gap=mean_gap,
        candidate_seconds=candidate.seconds,
        incumbent_seconds=incumbent.seconds,
        time_ratio=time_ratio,
        worse_every_seed=worse_every_seed,
        gap_threshold_met=gap_threshold_met,
        time_threshold_met=time_threshold_met,
        protected=protected,
        policy=policy,
    )


def evaluate_race(
    seed_runs: SeedRuns,
    *,
    maximize: bool,
    protected_ids: frozenset[str] = frozenset(),
    policy: RobustFutilityPolicy | None = None,
) -> tuple[FutilityDecision, ...]:
    """Evaluate all configurations at one deterministic seed-2 barrier."""

    policy = policy or RobustFutilityPolicy()
    incumbent_id = choose_incumbent(seed_runs, maximize=maximize, policy=policy)
    return tuple(
        evaluate_futility(
            candidate_id,
            incumbent_id,
            seed_runs,
            maximize=maximize,
            protected_ids=protected_ids,
            policy=policy,
        )
        for candidate_id in sorted(seed_runs)
    )
