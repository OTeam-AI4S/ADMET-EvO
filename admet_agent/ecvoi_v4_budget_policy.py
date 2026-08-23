"""Cohort-level cost-constrained allocation for EC-VOI v4 actions."""

from __future__ import annotations

import itertools
from typing import Any

import numpy as np


ACTIONS = ("baseline", "feature_rich", "imbalance_aware")


def allocate_actions(
    state_predictions: dict[str, dict[str, dict[str, Any]]],
    *,
    budget: float,
) -> dict[str, Any]:
    """Solve a deterministic multiple-choice knapsack by exact enumeration."""

    states = sorted(state_predictions)
    best: tuple[float, float, tuple[str, ...]] | None = None
    for choices in itertools.product(ACTIONS, repeat=len(states)):
        cost = sum(
            float(state_predictions[state][action]["cost_ratio"])
            for state, action in zip(states, choices, strict=True)
        )
        if cost > budget + 1e-12:
            continue
        if not all(
            bool(state_predictions[state][action]["eligible"])
            for state, action in zip(states, choices, strict=True)
        ):
            continue
        value = sum(
            float(state_predictions[state][action]["mean"])
            for state, action in zip(states, choices, strict=True)
        )
        candidate = (value, -cost, choices)
        if best is None or candidate > best:
            best = candidate
    if best is None:
        raise RuntimeError("No feasible cohort allocation")
    value, negative_cost, choices = best
    return {
        "actions": dict(zip(states, choices, strict=True)),
        "predicted_effect": float(value),
        "predicted_cost": float(-negative_cost),
        "budget": float(budget),
    }


def oracle_allocation(
    state_predictions: dict[str, dict[str, dict[str, Any]]],
    actual_effects: dict[str, dict[str, float]],
    *,
    budget: float,
) -> dict[str, Any]:
    oracle = {
        state: {
            action: {
                **state_predictions[state][action],
                "mean": actual_effects[state][action],
            }
            for action in ACTIONS
        }
        for state in state_predictions
    }
    return allocate_actions(oracle, budget=budget)


def score_allocation(
    allocation: dict[str, Any],
    actual_effects: dict[str, dict[str, float]],
    state_predictions: dict[str, dict[str, dict[str, Any]]],
    *,
    harm_margin: float,
) -> dict[str, Any]:
    effects = {
        state: float(actual_effects[state][action])
        for state, action in allocation["actions"].items()
    }
    realized_cost = sum(
        float(state_predictions[state][action]["cost_ratio"])
        for state, action in allocation["actions"].items()
    )
    return {
        **allocation,
        "effects": effects,
        "total_oriented_effect": float(sum(effects.values())),
        "mean_oriented_effect": float(np.mean(list(effects.values()))),
        "harmful_interventions": sum(value < -harm_margin for value in effects.values()),
        "realized_normalized_cost": float(realized_cost),
    }


__all__ = ["ACTIONS", "allocate_actions", "oracle_allocation", "score_allocation"]
