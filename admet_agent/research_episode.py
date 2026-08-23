"""Fail-closed sequential episodes for production metric-valued EC-VOI v2.

Decisions are emitted before execution, a whole matched pair is reserved
atomically, and only complete same-seed pairs may update the posterior.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from typing import Any

from .research_policy_v2 import (
    ActionPair,
    MatchedEffectObservation,
    MetricValuedECVOIPolicy,
)


def _digest(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode()).hexdigest()


@dataclass(frozen=True)
class EpisodeBudget:
    max_pairs: int
    max_candidate_fits: int
    max_fit_seconds: float

    def __post_init__(self) -> None:
        if self.max_pairs <= 0 or self.max_candidate_fits <= 0:
            raise ValueError("Episode pair and candidate budgets must be positive")
        if not math.isfinite(self.max_fit_seconds) or self.max_fit_seconds <= 0:
            raise ValueError("Episode fit-time budget must be positive")


@dataclass(frozen=True)
class PairExecution:
    pair_id: str
    action_id: str
    treatment_run_ids: tuple[str, ...]
    control_run_ids: tuple[str, ...]
    ordered_seeds: tuple[int, ...]
    actual_fit_seconds: float
    observation: MatchedEffectObservation | None
    failure_reason: str | None = None

    def __post_init__(self) -> None:
        if not math.isfinite(self.actual_fit_seconds) or self.actual_fit_seconds < 0:
            raise ValueError("Actual pair fit time must be finite and non-negative")
        if self.observation is not None and self.observation.action_id != self.action_id:
            raise ValueError("Execution observation action ID does not match the pair")

    def complete_for(self, action: ActionPair) -> bool:
        return bool(
            self.failure_reason is None
            and self.observation is not None
            and self.ordered_seeds == action.ordered_seeds
            and len(self.treatment_run_ids) == len(action.ordered_seeds)
            and len(self.control_run_ids) == len(action.ordered_seeds)
            and len(set(self.treatment_run_ids)) == len(self.treatment_run_ids)
            and len(set(self.control_run_ids)) == len(self.control_run_ids)
            and all(self.treatment_run_ids)
            and all(self.control_run_ids)
            and self.observation.single_axis_verified
            and self.observation.complete_pair
        )


@dataclass(frozen=True)
class EpisodeResult:
    decision: str
    stop_reason: str
    executed_pair_ids: tuple[str, ...]
    executed_action_ids: tuple[str, ...]
    observations: tuple[MatchedEffectObservation, ...]
    used_pairs: int
    used_candidate_fits: int
    reserved_fit_seconds: float
    actual_fit_seconds: float
    ledger: tuple[dict[str, Any], ...]

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["episode_sha256"] = _digest(payload)
        return payload


PairExecutor = Callable[[ActionPair, dict[str, Any]], PairExecution]


def _fits_budget(
    action: ActionPair,
    budget: EpisodeBudget,
    *,
    used_pairs: int,
    used_candidates: int,
    reserved_seconds: float,
) -> bool:
    return bool(
        used_pairs + 1 <= budget.max_pairs
        and used_candidates + action.candidate_charge <= budget.max_candidate_fits
        and reserved_seconds + action.predicted_pair_fit_seconds
        <= budget.max_fit_seconds
    )


def run_episode(
    *,
    policy: MetricValuedECVOIPolicy,
    actions: Sequence[ActionPair],
    prior_observations: Sequence[MatchedEffectObservation],
    budget: EpisodeBudget,
    executor: PairExecutor,
) -> EpisodeResult:
    """Run each predeclared pair at most once under atomic budget reservation."""

    pair_ids = [action.pair_id for action in actions]
    if not actions or len(pair_ids) != len(set(pair_ids)):
        raise ValueError("Episode requires unique predeclared pair IDs")
    pending = {action.pair_id: action for action in actions}
    posterior_observations = list(prior_observations)
    executed_pairs: list[str] = []
    executed_actions: list[str] = []
    ledger: list[dict[str, Any]] = []
    used_pairs = 0
    used_candidates = 0
    reserved_seconds = 0.0
    actual_seconds = 0.0
    stop_reason = "no_pending_pairs"

    while pending:
        admissible = [
            action
            for action in pending.values()
            if _fits_budget(
                action,
                budget,
                used_pairs=used_pairs,
                used_candidates=used_candidates,
                reserved_seconds=reserved_seconds,
            )
        ]
        if not admissible:
            stop_reason = "budget_exhausted_before_complete_pair"
            break
        decision = policy.rank(
            admissible,
            posterior_observations,
            current_episode_local_evidence=bool(
                any(
                    item.endpoint_local
                    for item in posterior_observations[len(prior_observations) :]
                )
            ),
        )
        decision_event = {
            "event_index": len(ledger),
            "event_type": "decision_emitted",
            "decision": decision,
            "used_pairs_before": used_pairs,
            "used_candidate_fits_before": used_candidates,
            "reserved_fit_seconds_before": reserved_seconds,
        }
        decision_event["event_sha256"] = _digest(decision_event)
        ledger.append(decision_event)
        if decision["decision"] == "stop":
            stop_reason = "nonpositive_robust_utility"
            break

        pair_id = str(decision["selected_pair_id"])
        action = pending.pop(pair_id)
        action_id = action.action_id
        used_pairs += 1
        used_candidates += action.candidate_charge
        reserved_seconds += action.predicted_pair_fit_seconds
        reservation = {
            "event_index": len(ledger),
            "event_type": "pair_budget_reserved",
            "pair_id": pair_id,
            "action_id": action_id,
            "pair_sha256": action.to_dict()["pair_sha256"],
            "candidate_charge": action.candidate_charge,
            "reserved_fit_seconds": action.predicted_pair_fit_seconds,
            "decision_event_sha256": decision_event["event_sha256"],
        }
        reservation["event_sha256"] = _digest(reservation)
        ledger.append(reservation)

        execution = executor(action, decision)
        if execution.pair_id != pair_id or execution.action_id != action_id:
            raise RuntimeError("Executor returned a different action than was reserved")
        actual_seconds += execution.actual_fit_seconds
        if execution.actual_fit_seconds > action.predicted_pair_fit_seconds:
            raise RuntimeError("Pair exceeded its pre-reserved hard fit-time ceiling")
        complete = execution.complete_for(action)
        if complete:
            posterior_observations.append(execution.observation)  # type: ignore[arg-type]
        executed_pairs.append(pair_id)
        executed_actions.append(action_id)
        execution_event = {
            "event_index": len(ledger),
            "event_type": "pair_execution_complete" if complete else "pair_execution_failed",
            "pair_id": pair_id,
            "action_id": action_id,
            "ordered_seeds": list(execution.ordered_seeds),
            "treatment_run_ids": list(execution.treatment_run_ids),
            "control_run_ids": list(execution.control_run_ids),
            "actual_fit_seconds": execution.actual_fit_seconds,
            "posterior_updated": complete,
            "failure_reason": execution.failure_reason if not complete else None,
            "reservation_event_sha256": reservation["event_sha256"],
        }
        execution_event["event_sha256"] = _digest(execution_event)
        ledger.append(execution_event)
    else:
        stop_reason = "no_pending_pairs"

    terminal = {
        "event_index": len(ledger),
        "event_type": "episode_stopped",
        "stop_reason": stop_reason,
        "used_pairs": used_pairs,
        "used_candidate_fits": used_candidates,
        "reserved_fit_seconds": reserved_seconds,
        "actual_fit_seconds": actual_seconds,
    }
    terminal["event_sha256"] = _digest(terminal)
    ledger.append(terminal)
    return EpisodeResult(
        decision="stop",
        stop_reason=stop_reason,
        executed_pair_ids=tuple(executed_pairs),
        executed_action_ids=tuple(executed_actions),
        observations=tuple(posterior_observations),
        used_pairs=used_pairs,
        used_candidate_fits=used_candidates,
        reserved_fit_seconds=reserved_seconds,
        actual_fit_seconds=actual_seconds,
        ledger=tuple(ledger),
    )


__all__ = [
    "EpisodeBudget",
    "EpisodeResult",
    "PairExecution",
    "run_episode",
]
