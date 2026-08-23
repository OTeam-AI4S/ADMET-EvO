"""Harm-calibrated EC-VOI v5 eligibility and cohort allocation."""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class EligibilityContract:
    practical_gain: float = 0.005
    harm_margin: float = 0.005
    harm_probability_limit: float = 0.20
    cost_penalty: float = 0.00025
    information_gain_weight: float = 0.0

    def __post_init__(self) -> None:
        if self.practical_gain < 0 or self.harm_margin < 0:
            raise ValueError("Gain and harm margins must be non-negative")
        if not 0 <= self.harm_probability_limit <= 1:
            raise ValueError("Harm probability limit must be in [0, 1]")
        if self.cost_penalty < 0 or self.information_gain_weight < 0:
            raise ValueError("Penalty and information weight must be non-negative")


def gate_action(
    posterior: dict[str, Any],
    *,
    contract: EligibilityContract,
    stop: bool = False,
) -> dict[str, Any]:
    """Apply prospective practical-effect and harm eligibility constraints."""
    mean = float(posterior.get("mean", 0.0))
    harm_probability = float(posterior.get("harm_probability", 0.0))
    cost = float(posterior.get("cost_ratio", 0.0))
    information_gain = float(posterior.get("information_gain", 0.0))
    if not all(math.isfinite(value) for value in (mean, harm_probability, cost, information_gain)):
        raise ValueError("Action posterior fields must be finite")
    if cost < 0 or information_gain < 0 or not 0 <= harm_probability <= 1:
        raise ValueError("Invalid posterior cost, information gain, or harm probability")
    if stop:
        if abs(mean) > 1e-12 or abs(cost) > 1e-12 or abs(harm_probability) > 1e-12:
            raise ValueError("Stop must have zero predicted gain, cost, and harm")
        eligible = True
        reasons: list[str] = []
    else:
        reasons = []
        if mean < contract.practical_gain:
            reasons.append("below_practical_gain")
        if harm_probability > contract.harm_probability_limit:
            reasons.append("exceeds_harm_probability")
        eligible = not reasons
    utility = (
        mean
        - contract.cost_penalty * cost
        + contract.information_gain_weight * information_gain
        if eligible
        else -math.inf
    )
    return {
        **posterior,
        "mean": mean,
        "harm_probability": harm_probability,
        "cost_ratio": cost,
        "information_gain": information_gain,
        "eligible": eligible,
        "ineligibility_reasons": reasons,
        "utility": float(utility),
    }


def prepare_state_actions(
    raw_predictions: dict[str, dict[str, dict[str, Any]]],
    *,
    contract: EligibilityContract,
) -> dict[str, dict[str, dict[str, Any]]]:
    """Gate every action and require an explicit zero-cost Stop option."""
    prepared = {}
    for state, actions in raw_predictions.items():
        if "stop" not in actions:
            raise ValueError(f"State {state!r} lacks explicit stop action")
        prepared[state] = {
            action: gate_action(
                posterior,
                contract=contract,
                stop=action == "stop",
            )
            for action, posterior in actions.items()
        }
    return prepared


def allocate_actions(
    predictions: dict[str, dict[str, dict[str, Any]]],
    *,
    budget: float,
) -> dict[str, Any]:
    """Exact variable-action multiple-choice knapsack; budget need not be filled."""
    if budget < 0 or not math.isfinite(budget):
        raise ValueError("Budget must be finite and non-negative")
    states = sorted(predictions)
    action_spaces = [tuple(sorted(predictions[state])) for state in states]
    best: tuple[float, float, tuple[str, ...]] | None = None
    for choices in itertools.product(*action_spaces):
        selected = [
            predictions[state][action]
            for state, action in zip(states, choices, strict=True)
        ]
        if not all(bool(item["eligible"]) for item in selected):
            continue
        cost = float(sum(float(item["cost_ratio"]) for item in selected))
        if cost > budget + 1e-12:
            continue
        utility = float(sum(float(item["utility"]) for item in selected))
        # Higher utility, then lower cost, then deterministic action tuple.
        candidate = (utility, -cost, choices)
        if best is None or candidate > best:
            best = candidate
    if best is None:
        raise RuntimeError("No feasible allocation; explicit Stop should make one available")
    utility, negative_cost, choices = best
    selected_actions = dict(zip(states, choices, strict=True))
    return {
        "actions": selected_actions,
        "predicted_utility": float(utility),
        "predicted_effect": float(
            sum(
                predictions[state][action]["mean"]
                for state, action in selected_actions.items()
            )
        ),
        "predicted_cost": float(-negative_cost),
        "unused_budget": float(budget + negative_cost),
        "budget": float(budget),
    }


def score_allocation(
    allocation: dict[str, Any],
    actual_effects: dict[str, dict[str, float]],
    predictions: dict[str, dict[str, dict[str, Any]]],
    *,
    harm_margin: float,
) -> dict[str, Any]:
    effects = {
        state: float(actual_effects[state][action])
        for state, action in allocation["actions"].items()
    }
    return {
        **allocation,
        "effects": effects,
        "total_oriented_effect": float(sum(effects.values())),
        "mean_oriented_effect": float(np.mean(list(effects.values()))),
        "harmful_interventions": int(
            sum(effect < -harm_margin for effect in effects.values())
        ),
        "realized_normalized_cost": float(
            sum(
                predictions[state][action]["cost_ratio"]
                for state, action in allocation["actions"].items()
            )
        ),
    }


__all__ = [
    "EligibilityContract",
    "allocate_actions",
    "gate_action",
    "prepare_state_actions",
    "score_allocation",
]
