"""Production metric-valued EC-VOI v2 acquisition contracts.

The source-bound v1 controls were frozen before this module entered the
production package.  All policy inputs are discovery-only matched effects;
test and leaderboard evidence fail closed at object construction.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from typing import Any, Literal

import numpy as np
from scipy.stats import norm

Axis = Literal["feature", "model", "data", "replicate"]
ECVOI_V2_ROUTING_POLICIES = (
    "ecvoi-v2",
    "ecvoi-v2-no-context",
    "ecvoi-v2-no-memory",
    "ecvoi-v2-no-voi",
    "ecvoi-v2-no-cost",
    "ecvoi-v2-no-risk",
    "ecvoi-v2-shuffled-context",
    "ecvoi-v2-shuffled-memory",
    "ecvoi-v3",
    "ecvoi-v3-no-initial-exploration",
)
AXIS_FIELDS: dict[Axis, frozenset[str]] = {
    "feature": frozenset({"representations", "feature_parameters"}),
    "model": frozenset(
        {"model", "hyperparameters", "imbalance", "loss", "ensemble"}
    ),
    "data": frozenset({"data_policy", "external_sources"}),
    "replicate": frozenset({"replication_policy"}),
}
PROTECTED_FIELDS = frozenset(
    {"task", "task_type", "metric", "split", "target", "test_loader"}
)


def _valid_sha256(value: str) -> bool:
    return len(value) == 64 and all(character in "0123456789abcdef" for character in value)


def _canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)


@dataclass(frozen=True)
class ActionPair:
    pair_id: str
    action_id: str
    axis: Axis
    treatment: Mapping[str, Any]
    control: Mapping[str, Any]
    ordered_seeds: tuple[int, ...]
    predicted_pair_fit_seconds: float
    candidate_charge: int
    task_fit_budget_seconds: float
    task_candidate_budget: int
    practical_effect_threshold: float
    effect_standardization_scale: float
    predicted_native_observation_se: float
    scientific_risk: float
    context_sha256: str

    def __post_init__(self) -> None:
        if not self.pair_id or not self.action_id:
            raise ValueError("Action pair requires non-empty pair and action IDs")
        if self.axis not in AXIS_FIELDS:
            raise ValueError(f"Unknown action axis {self.axis!r}")
        if len(set(self.ordered_seeds)) != len(self.ordered_seeds) or len(self.ordered_seeds) < 3:
            raise ValueError("Action pairs require at least three distinct ordered seeds")
        if any(seed < 0 for seed in self.ordered_seeds):
            raise ValueError("Action-pair seeds must be non-negative")
        positive = {
            "predicted_pair_fit_seconds": self.predicted_pair_fit_seconds,
            "candidate_charge": self.candidate_charge,
            "task_fit_budget_seconds": self.task_fit_budget_seconds,
            "task_candidate_budget": self.task_candidate_budget,
            "effect_standardization_scale": self.effect_standardization_scale,
            "predicted_native_observation_se": self.predicted_native_observation_se,
        }
        if any(not math.isfinite(float(value)) or float(value) <= 0 for value in positive.values()):
            raise ValueError("Action-pair costs, budgets and observation SE must be positive")
        if self.predicted_pair_fit_seconds > self.task_fit_budget_seconds:
            raise ValueError("Predicted pair fit time exceeds the task budget")
        if self.candidate_charge > self.task_candidate_budget:
            raise ValueError("Pair candidate charge exceeds the task budget")
        if not math.isfinite(self.practical_effect_threshold) or self.practical_effect_threshold < 0:
            raise ValueError("Practical effect threshold must be finite and non-negative")
        if not math.isfinite(self.scientific_risk) or not 0 <= self.scientific_risk <= 1:
            raise ValueError("Scientific risk must be within [0, 1]")
        if not _valid_sha256(self.context_sha256):
            raise ValueError("Action pair requires a lowercase SHA-256 task context")
        changed = self.changed_fields()
        if not changed:
            raise ValueError("Treatment and control must differ on the declared axis")
        if changed & PROTECTED_FIELDS:
            raise ValueError(f"Action pair changes protected fields: {sorted(changed & PROTECTED_FIELDS)}")
        illegal = changed - AXIS_FIELDS[self.axis]
        if illegal:
            raise ValueError(
                f"Action pair declared {self.axis} but also changes {sorted(illegal)}"
            )

    def changed_fields(self) -> frozenset[str]:
        keys = set(self.treatment) | set(self.control)
        return frozenset(
            key
            for key in keys
            if _canonical(self.treatment.get(key)) != _canonical(self.control.get(key))
        )

    def normalized_pair_cost(self) -> float:
        fit_fraction = self.predicted_pair_fit_seconds / self.task_fit_budget_seconds
        candidate_fraction = self.candidate_charge / self.task_candidate_budget
        return float(min(1.0, 0.5 * fit_fraction + 0.5 * candidate_fraction))

    def standardized_effect_threshold(self) -> float:
        return float(self.practical_effect_threshold / self.effect_standardization_scale)

    def predicted_standardized_observation_se(self) -> float:
        return float(
            self.predicted_native_observation_se / self.effect_standardization_scale
        )

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["changed_fields"] = sorted(self.changed_fields())
        payload["normalized_pair_cost"] = self.normalized_pair_cost()
        payload["standardized_effect_threshold"] = self.standardized_effect_threshold()
        payload["predicted_standardized_observation_se"] = (
            self.predicted_standardized_observation_se()
        )
        payload["pair_sha256"] = hashlib.sha256(_canonical(payload).encode()).hexdigest()
        return payload


@dataclass(frozen=True)
class MatchedEffectObservation:
    endpoint_id: str
    pair_id: str
    action_id: str
    native_oriented_delta: float
    native_standard_error: float
    effect_standardization_scale: float
    context_similarity: float
    endpoint_local: bool
    transport_eligible: bool
    test_derived: bool
    leaderboard_derived: bool
    single_axis_verified: bool
    complete_pair: bool
    provenance_sha256: str

    def __post_init__(self) -> None:
        if not self.endpoint_id or not self.pair_id or not self.action_id:
            raise ValueError("Matched effect requires endpoint, pair and action IDs")
        if not math.isfinite(self.native_oriented_delta):
            raise ValueError("Matched effect delta must be finite")
        if not math.isfinite(self.native_standard_error) or self.native_standard_error <= 0:
            raise ValueError("Matched effect standard error must be positive")
        if (
            not math.isfinite(self.effect_standardization_scale)
            or self.effect_standardization_scale <= 0
        ):
            raise ValueError("Matched effect standardization scale must be positive")
        if not math.isfinite(self.context_similarity) or not 0 <= self.context_similarity <= 1:
            raise ValueError("Context similarity must be within [0, 1]")
        if not _valid_sha256(self.provenance_sha256):
            raise ValueError("Matched effect requires a lowercase SHA-256 provenance hash")
        if self.test_derived or self.leaderboard_derived:
            raise RuntimeError("Test- or leaderboard-derived evidence is forbidden")

    def standardized_oriented_delta(self) -> float:
        return float(self.native_oriented_delta / self.effect_standardization_scale)

    def standardized_standard_error(self) -> float:
        return float(self.native_standard_error / self.effect_standardization_scale)


@dataclass(frozen=True)
class GaussianEffectPosterior:
    mean: float
    standard_deviation: float
    effective_endpoints: float
    admitted_endpoints: int
    excluded_observations: int
    evidence_source: str


@dataclass(frozen=True)
class MetricValuedECVOIPolicy:
    prior_mean: float = 0.0
    prior_standard_deviation: float = 1.5
    between_endpoint_standard_deviation: float = 1.0
    robust_z: float = 1.0
    information_weight: float = 0.04
    compute_cost_weight: float = 0.08
    scientific_risk_weight: float = 0.08
    transport_scope: Literal["contextual", "global", "none"] = "contextual"
    allow_one_endpoint_local_exploration: bool = False
    maximum_initial_exploration_risk: float = 0.10
    policy_version: str = "metric-valued-ecvoi-v2"

    def __post_init__(self) -> None:
        positive = {
            "prior_standard_deviation": self.prior_standard_deviation,
            "between_endpoint_standard_deviation": self.between_endpoint_standard_deviation,
            "robust_z": self.robust_z,
        }
        if any(not math.isfinite(value) or value <= 0 for value in positive.values()):
            raise ValueError("Posterior scales and robust_z must be finite and positive")
        nonnegative = {
            "information_weight": self.information_weight,
            "compute_cost_weight": self.compute_cost_weight,
            "scientific_risk_weight": self.scientific_risk_weight,
        }
        if any(not math.isfinite(value) or value < 0 for value in nonnegative.values()):
            raise ValueError("Acquisition weights must be finite and non-negative")
        if self.transport_scope not in {"contextual", "global", "none"}:
            raise ValueError("Unknown transport scope")
        if not 0 <= self.maximum_initial_exploration_risk <= 1:
            raise ValueError("Initial exploration risk must be within [0, 1]")

    def posterior(
        self,
        action_id: str,
        observations: Sequence[MatchedEffectObservation],
    ) -> GaussianEffectPosterior:
        action_observations = [item for item in observations if item.action_id == action_id]
        endpoint_pairs = [
            (item.endpoint_id, item.pair_id) for item in action_observations
        ]
        if len(endpoint_pairs) != len(set(endpoint_pairs)):
            raise ValueError(
                "Posterior input contains a duplicate endpoint/pair observation"
            )
        precision = 1.0 / self.prior_standard_deviation**2
        weighted_mean = self.prior_mean * precision
        admitted = 0
        excluded = 0
        effective = 0.0
        sources = set()
        by_endpoint: dict[str, list[MatchedEffectObservation]] = {}
        for observation in action_observations:
            by_endpoint.setdefault(observation.endpoint_id, []).append(observation)
        for endpoint_observations in by_endpoint.values():
            eligible: list[MatchedEffectObservation] = []
            endpoint_weights = set()
            endpoint_sources = set()
            for observation in endpoint_observations:
                if not observation.single_axis_verified or not observation.complete_pair:
                    excluded += 1
                    continue
                if observation.endpoint_local:
                    weight = 1.0
                    evidence_source = "endpoint_local"
                elif self.transport_scope == "none" or not observation.transport_eligible:
                    excluded += 1
                    continue
                elif self.transport_scope == "contextual":
                    weight = observation.context_similarity
                    evidence_source = "context_matched_transport"
                else:
                    weight = 1.0
                    evidence_source = "global_transport"
                if weight <= 0:
                    excluded += 1
                    continue
                eligible.append(observation)
                endpoint_weights.add(weight)
                endpoint_sources.add(evidence_source)
            if not eligible:
                continue
            if len(endpoint_weights) != 1 or len(endpoint_sources) != 1:
                raise ValueError(
                    "Observations from one endpoint/action require a consistent evidence scope"
                )
            weight = endpoint_weights.pop()
            sources.add(endpoint_sources.pop())
            deltas = np.asarray(
                [item.standardized_oriented_delta() for item in eligible], dtype=float
            )
            standard_errors = np.asarray(
                [item.standardized_standard_error() for item in eligible], dtype=float
            )
            # Multiple concrete pairs on one endpoint are family-level evidence,
            # not independent endpoint replicates. Preserve pair heterogeneity
            # instead of shrinking uncertainty by sqrt(number_of_pairs).
            within_endpoint_variance = float(np.mean(np.square(standard_errors)))
            if len(deltas) > 1:
                within_endpoint_variance += float(np.var(deltas, ddof=1))
            endpoint_delta = float(np.mean(deltas))
            variance = within_endpoint_variance + self.between_endpoint_standard_deviation**2
            item_precision = weight / variance
            precision += item_precision
            weighted_mean += item_precision * endpoint_delta
            admitted += 1
            effective += weight
        posterior_variance = 1.0 / precision
        source = "+".join(sorted(sources)) if sources else "prior_only"
        return GaussianEffectPosterior(
            mean=float(weighted_mean / precision),
            standard_deviation=float(math.sqrt(posterior_variance)),
            effective_endpoints=float(effective),
            admitted_endpoints=admitted,
            excluded_observations=excluded,
            evidence_source=source,
        )

    @staticmethod
    def _expected_improvement(mean: float, sd: float, threshold: float) -> float:
        if sd <= 0:
            return max(0.0, mean - threshold)
        z = (mean - threshold) / sd
        return float((mean - threshold) * norm.cdf(z) + sd * norm.pdf(z))

    @staticmethod
    def _information_gain(posterior_sd: float, observation_se: float) -> float:
        return float(0.5 * math.log1p((posterior_sd / observation_se) ** 2))

    def evaluate(
        self,
        action: ActionPair,
        observations: Sequence[MatchedEffectObservation],
    ) -> dict[str, Any]:
        posterior = self.posterior(action.action_id, observations)
        expected_improvement = self._expected_improvement(
            posterior.mean,
            posterior.standard_deviation,
            action.standardized_effect_threshold(),
        )
        robust_improvement = max(
            0.0,
            posterior.mean
            - self.robust_z * posterior.standard_deviation
            - action.standardized_effect_threshold(),
        )
        information_gain = self._information_gain(
            posterior.standard_deviation,
            action.predicted_standardized_observation_se(),
        )
        normalized_cost = action.normalized_pair_cost()
        acquisition_utility = (
            expected_improvement
            + self.information_weight * information_gain
            - self.compute_cost_weight * normalized_cost
            - self.scientific_risk_weight * action.scientific_risk
        )
        robust_utility = (
            robust_improvement
            + self.information_weight * information_gain
            - self.compute_cost_weight * normalized_cost
            - self.scientific_risk_weight * action.scientific_risk
        )
        return {
            **action.to_dict(),
            "routing_policy": self.policy_version,
            "effect_unit": "discovery_standardized_oriented_paired_delta",
            "posterior": asdict(posterior),
            "expected_improvement_above_threshold": expected_improvement,
            "robust_improvement_lcb": robust_improvement,
            "expected_information_gain_nats": information_gain,
            "compute_penalty": self.compute_cost_weight * normalized_cost,
            "scientific_risk_penalty": self.scientific_risk_weight * action.scientific_risk,
            "acquisition_utility": float(acquisition_utility),
            "robust_utility": float(robust_utility),
            "utility": float(robust_utility),
            "eligible_for_execution": robust_utility > 0,
            "emitted_before_execution": True,
        }

    def rank(
        self,
        actions: Sequence[ActionPair],
        observations: Sequence[MatchedEffectObservation],
        *,
        current_episode_local_evidence: bool = False,
    ) -> dict[str, Any]:
        if not actions:
            raise ValueError("At least one resolved action pair is required")
        pair_ids = [action.pair_id for action in actions]
        if len(pair_ids) != len(set(pair_ids)):
            raise ValueError("Pair IDs must be unique within an episode step")
        rows = [self.evaluate(action, observations) for action in actions]
        endpoint_ids = {str(action.treatment.get("task", "")) for action in actions}
        has_historical_endpoint_local_evidence = any(
            item.endpoint_local and item.endpoint_id in endpoint_ids
            for item in observations
        )
        initial_exploration = bool(
            self.allow_one_endpoint_local_exploration
            and not current_episode_local_evidence
        )
        if initial_exploration:
            for row in rows:
                row["eligible_for_initial_exploration"] = bool(
                    row["acquisition_utility"] > 0
                    and row["scientific_risk"]
                    <= self.maximum_initial_exploration_risk
                )
            rows.sort(
                key=lambda row: (-row["acquisition_utility"], row["pair_id"])
            )
            selected = (
                rows[0]
                if rows[0]["eligible_for_initial_exploration"]
                else None
            )
            decision_basis = "one_safe_endpoint_local_exploration"
        else:
            for row in rows:
                row["eligible_for_initial_exploration"] = False
            rows.sort(key=lambda row: (-row["utility"], row["pair_id"]))
            selected = rows[0] if rows[0]["eligible_for_execution"] else None
            decision_basis = "robust_utility"
        return {
            "policy_version": self.policy_version,
            "ranked_pairs": rows,
            "decision": "execute_pair" if selected is not None else "stop",
            "selected_action_id": (
                selected["action_id"] if selected is not None else None
            ),
            "selected_pair_id": (
                selected["pair_id"] if selected is not None else None
            ),
            "selected_pair_sha256": (
                selected["pair_sha256"] if selected is not None else None
            ),
            "decision_basis": decision_basis,
            "historical_endpoint_local_evidence_available": (
                has_historical_endpoint_local_evidence
            ),
            "current_episode_local_evidence_available": (
                current_episode_local_evidence
            ),
            "emitted_before_execution": True,
        }


def metric_valued_policy_for_variant(
    routing_policy: str,
) -> MetricValuedECVOIPolicy:
    """Construct only preregistered v2 policy ablations."""

    if routing_policy not in ECVOI_V2_ROUTING_POLICIES:
        raise ValueError(f"Unknown EC-VOI v2 routing policy {routing_policy!r}")
    overrides: dict[str, Any] = {
        "policy_version": routing_policy,
    }
    if routing_policy == "ecvoi-v2-no-context":
        overrides["transport_scope"] = "global"
    elif routing_policy == "ecvoi-v2-no-memory":
        overrides["transport_scope"] = "none"
    elif routing_policy == "ecvoi-v2-no-voi":
        overrides["information_weight"] = 0.0
    elif routing_policy == "ecvoi-v2-no-cost":
        overrides["compute_cost_weight"] = 0.0
    elif routing_policy == "ecvoi-v2-no-risk":
        overrides["scientific_risk_weight"] = 0.0
    elif routing_policy == "ecvoi-v3":
        overrides["allow_one_endpoint_local_exploration"] = True
    return MetricValuedECVOIPolicy(**overrides)

__all__ = [
    "AXIS_FIELDS",
    "ECVOI_V2_ROUTING_POLICIES",
    "ActionPair",
    "GaussianEffectPosterior",
    "MatchedEffectObservation",
    "MetricValuedECVOIPolicy",
    "metric_valued_policy_for_variant",
]
