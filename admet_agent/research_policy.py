"""Evidence-conditioned value-of-information routing for research actions.

This module is deliberately independent of model fitting.  It ranks bounded,
predeclared actions using discovery-only endpoint evidence and only those
cross-task outcomes that passed the prospective transport firewall.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Literal


@dataclass(frozen=True)
class EvidenceConditionedVOIPolicy:
    """Risk-aware Beta-Bernoulli action policy with an exploration term.

    ``evidence_confidence`` defines an endpoint-local Beta prior.  Prospective
    transport outcomes contribute fractional supported/refuted counts only
    after the memory layer marks them routing-eligible.  The exploitation term
    is a lower confidence bound on normalized effect; the exploration term is
    the expected one-observation reduction in posterior standard deviation.
    """

    local_prior_strength: float = 4.0
    credible_z: float = 1.0
    information_bonus: float = 0.25
    cost_weight: float = 0.55
    risk_weight: float = 0.45
    transport_scope: Literal["contextual", "global", "none"] = "contextual"
    policy_version: str = "ec-voi-v1"

    def __post_init__(self) -> None:
        positive = {
            "local_prior_strength": self.local_prior_strength,
            "credible_z": self.credible_z,
        }
        for name, value in positive.items():
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        bounded = {
            "information_bonus": self.information_bonus,
            "cost_weight": self.cost_weight,
            "risk_weight": self.risk_weight,
        }
        for name, value in bounded.items():
            if not math.isfinite(value) or not 0 <= value <= 1:
                raise ValueError(f"{name} must be finite and within [0, 1]")
        if not math.isclose(self.cost_weight + self.risk_weight, 1.0):
            raise ValueError("cost_weight and risk_weight must sum to one")
        if self.transport_scope not in {"contextual", "global", "none"}:
            raise ValueError("transport_scope must be contextual, global, or none")

    def _transport_counts(
        self, prior: dict[str, Any] | None
    ) -> tuple[float, float, str]:
        if self.transport_scope == "none":
            return 0.0, 0.0, "endpoint_local_only"
        if not prior:
            return 0.0, 0.0, "endpoint_local_only"
        if (
            self.transport_scope == "contextual"
            and prior.get("contextual_routing_eligible") is True
        ):
            supported = float(prior.get("contextual_supported", 0.0))
            refuted = float(prior.get("contextual_refuted", 0.0))
            inconclusive = float(prior.get("contextual_inconclusive", 0.0))
            return (
                supported + 0.5 * inconclusive,
                refuted + 0.5 * inconclusive,
                "context_matched_transport",
            )
        if self.transport_scope == "contextual":
            return 0.0, 0.0, "endpoint_local_context_gate"
        if prior.get("routing_eligible") is not True:
            return 0.0, 0.0, "endpoint_local_only"
        supported = float(prior.get("supported", 0.0))
        refuted = float(prior.get("refuted", 0.0))
        inconclusive = float(prior.get("inconclusive", 0.0))
        return (
            supported + 0.5 * inconclusive,
            refuted + 0.5 * inconclusive,
            "global_transport_fallback",
        )

    def _negative_gate(self, prior: dict[str, Any] | None) -> bool:
        if self.transport_scope == "none":
            return False
        if not prior:
            return False
        contextual = bool(
            self.transport_scope == "contextual"
            and prior.get("contextual_routing_eligible") is True
        )
        if self.transport_scope == "contextual" and not contextual:
            return False
        if not contextual and prior.get("routing_eligible") is not True:
            return False
        prefix = "contextual_" if contextual else ""
        observations = int(
            prior.get(f"{prefix}transport_eligible_observations", 0)
        )
        refuted_rate = (
            float(prior.get(f"{prefix}refuted", 0)) / observations
            if observations
            else 0.0
        )
        return bool(
            prior.get(f"{prefix}negative_gate_confirmed") is True
            and observations >= 5
            and refuted_rate >= 0.80
            and float(prior.get(f"{prefix}posterior_support_probability", 0.5))
            < 0.35
        )

    def rank(
        self,
        actions: Sequence[Any],
        *,
        effect_priority: float,
        action_priors: dict[str, dict[str, Any]] | None = None,
    ) -> list[dict[str, Any]]:
        if not actions:
            raise ValueError("At least one research action is required")
        if not 0.5 <= effect_priority <= 1.0:
            raise ValueError("effect_priority must be within [0.5, 1.0]")
        action_priors = action_priors or {}
        rows = []
        for action in actions:
            prior = action_priors.get(action.action_id)
            successes, failures, evidence_source = self._transport_counts(prior)
            admitted = not self._negative_gate(prior)
            confidence = float(action.evidence_confidence)
            alpha = 1.0 + self.local_prior_strength * confidence + successes
            beta = 1.0 + self.local_prior_strength * (1.0 - confidence) + failures
            total = alpha + beta
            support = alpha / total
            variance = alpha * beta / (total * total * (total + 1.0))
            support_sd = math.sqrt(variance)
            posterior_effect = float(action.expected_effect) * support
            posterior_effect_sd = float(action.expected_effect) * support_sd
            robust_effect = max(
                0.0,
                posterior_effect - self.credible_z * posterior_effect_sd,
            )
            expected_next_variance = variance * total / (total + 1.0)
            raw_information_value = float(action.expected_effect) * (
                support_sd - math.sqrt(expected_next_variance)
            )
            rows.append(
                {
                    **action.to_dict(),
                    "routing_policy": self.policy_version,
                    "prior_evidence": (
                        prior if self.transport_scope != "none" else None
                    ),
                    "prior_evidence_ignored": bool(
                        prior and self.transport_scope == "none"
                    ),
                    "transport_scope": self.transport_scope,
                    "evidence_source": evidence_source,
                    "posterior_alpha": alpha,
                    "posterior_beta": beta,
                    "posterior_support_probability": support,
                    "posterior_support_sd": support_sd,
                    "posterior_expected_effect": posterior_effect,
                    "posterior_effect_sd": posterior_effect_sd,
                    "robust_effect_lcb": robust_effect,
                    "raw_information_value": raw_information_value,
                    "routing_admitted": admitted,
                    "routing_block_reason": (
                        "prospective_transport_refutation_gate"
                        if not admitted
                        else None
                    ),
                }
            )
        max_information = max(row["raw_information_value"] for row in rows)
        for row in rows:
            normalized_information = (
                row["raw_information_value"] / max_information
                if max_information > 0
                else 0.0
            )
            penalty = (1.0 - effect_priority) * (
                self.cost_weight * float(row["compute_cost"])
                + self.risk_weight * float(row["scientific_risk"])
            )
            exploitation = effect_priority * float(row["robust_effect_lcb"])
            exploration = (
                (1.0 - effect_priority)
                * self.information_bonus
                * normalized_information
            )
            row["normalized_information_value"] = normalized_information
            row["exploitation_utility"] = exploitation
            row["exploration_utility"] = exploration
            row["cost_risk_penalty"] = penalty
            row["utility"] = (
                exploitation + exploration - penalty
                if row["routing_admitted"]
                else -1.0
            )
        rows.sort(key=lambda item: (-item["utility"], item["action_id"]))
        return rows


__all__ = ["EvidenceConditionedVOIPolicy"]
