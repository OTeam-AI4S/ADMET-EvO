"""Identifiability-aware, source-balanced EC-VOI policy.

This module is deliberately separate from the frozen v5 analysis.  It may be
developed on the v5 negative-control cohort, but promotion requires a new,
prospectively frozen multi-source cohort.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

from .ecvoi_v5_policy import EligibilityContract, gate_action
from .public_policy_analysis import STATE_FEATURES


@dataclass(frozen=True)
class IdentifiabilityContract:
    """Pre-outcome conditions for allowing task diagnostics to affect actions."""

    min_training_sources: int = 5
    min_source_win_fraction: float = 0.60
    min_median_loss_gain: float = 0.0
    ridge_alpha: float = 4.0
    full_weight_sources: int = 9

    def __post_init__(self) -> None:
        if self.min_training_sources < 4:
            raise ValueError("At least four training sources are required")
        if not 0.5 <= self.min_source_win_fraction <= 1.0:
            raise ValueError("Source win fraction must be in [0.5, 1]")
        if self.ridge_alpha <= 0:
            raise ValueError("ridge_alpha must be positive")
        if self.full_weight_sources < self.min_training_sources:
            raise ValueError("full_weight_sources cannot be below min_training_sources")


def _source_weights(frame: pd.DataFrame) -> np.ndarray:
    counts = frame["source_id"].value_counts()
    return frame["source_id"].map(lambda source: 1.0 / counts[source]).to_numpy(dtype=float)


def _balanced_mean(frame: pd.DataFrame, effect_column: str) -> float:
    by_source = frame.groupby("source_id", sort=True)[effect_column].mean()
    return float(by_source.mean())


def _state_fit_predict(
    train: pd.DataFrame,
    target: pd.DataFrame,
    effect_column: str,
    *,
    alpha: float,
    extra_source_weights: dict[str, float] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    x = train[list(STATE_FEATURES)].to_numpy(dtype=float)
    target_x = target[list(STATE_FEATURES)].to_numpy(dtype=float)
    y = train[effect_column].to_numpy(dtype=float)
    weights = _source_weights(train)
    if extra_source_weights is not None:
        weights = weights * train["source_id"].map(extra_source_weights).to_numpy(dtype=float)
    mean = np.average(x, axis=0, weights=weights)
    variance = np.average((x - mean) ** 2, axis=0, weights=weights)
    scale = np.sqrt(np.maximum(variance, 0.0))
    scale[scale < 1e-12] = 1.0
    x_scaled = (x - mean) / scale
    target_scaled = (target_x - mean) / scale
    model = Ridge(alpha=alpha).fit(x_scaled, y, sample_weight=weights)
    return model.predict(target_scaled), model.predict(x_scaled)


def diagnostic_identifiability(
    train: pd.DataFrame,
    *,
    action: str,
    contract: IdentifiabilityContract,
) -> dict[str, Any]:
    """Compare state-conditioned and source-balanced-prior loss by held source."""
    effect_column = f"effect_{action}"
    work = train.loc[np.isfinite(train[effect_column])].copy()
    sources = sorted(work["source_id"].unique())
    if len(sources) < contract.min_training_sources:
        return {
            "identifiable": False,
            "training_sources": len(sources),
            "source_win_fraction": 0.0,
            "median_loss_gain": 0.0,
            "normalized_loss_gain": 0.0,
            "shrinkage_weight": 0.0,
            "reason": "insufficient_training_sources",
        }

    gains = []
    prior_losses = []
    for held_source in sources:
        fit = work.loc[work["source_id"] != held_source]
        held = work.loc[work["source_id"] == held_source]
        if fit["source_id"].nunique() < 2:
            continue
        prior = _balanced_mean(fit, effect_column)
        state_prediction, _ = _state_fit_predict(
            fit, held, effect_column, alpha=contract.ridge_alpha
        )
        actual = held[effect_column].to_numpy(dtype=float)
        prior_loss = float(np.mean((actual - prior) ** 2))
        state_loss = float(np.mean((actual - state_prediction) ** 2))
        gains.append(prior_loss - state_loss)
        prior_losses.append(prior_loss)

    median_gain = float(np.median(gains)) if gains else 0.0
    win_fraction = float(np.mean(np.asarray(gains) > 0.0)) if gains else 0.0
    scale = max(float(np.median(prior_losses)) if prior_losses else 0.0, 1e-6)
    normalized_gain = float(np.clip(median_gain / scale, 0.0, 1.0))
    identifiable = bool(
        win_fraction >= contract.min_source_win_fraction
        and median_gain > contract.min_median_loss_gain
    )
    source_reliability = float(
        np.clip(
            (len(sources) - contract.min_training_sources + 1)
            / (contract.full_weight_sources - contract.min_training_sources + 1),
            0.0,
            1.0,
        )
    )
    shrinkage = normalized_gain * source_reliability if identifiable else 0.0
    return {
        "identifiable": identifiable,
        "training_sources": len(sources),
        "source_win_fraction": win_fraction,
        "median_loss_gain": median_gain,
        "normalized_loss_gain": normalized_gain,
        "shrinkage_weight": float(shrinkage),
        "reason": "identified" if identifiable else "diagnostics_do_not_transfer",
    }


def cross_source_posterior_v6(
    table: pd.DataFrame,
    held_index: int,
    *,
    action: str,
    identifiability_contract: IdentifiabilityContract,
    bootstraps: int,
    seed: int,
    harm_margin: float,
) -> dict[str, Any]:
    """Predict after removing the whole held source and shrink unsafe diagnostics."""
    held = table.iloc[[held_index]]
    held_source = str(held.iloc[0]["source_id"])
    effect_column = f"effect_{action}"
    train = table.loc[table["source_id"] != held_source].copy()
    train = train.loc[np.isfinite(train[effect_column])].reset_index(drop=True)
    diagnosis = diagnostic_identifiability(
        train, action=action, contract=identifiability_contract
    )
    sources = sorted(train["source_id"].unique())
    cost = 6.0 if action == "nested_imbalance" else 1.0
    if len(sources) < 2:
        return {
            "mean": 0.0, "harm_probability": 1.0, "cost_ratio": cost,
            "information_gain": 0.0, **diagnosis,
        }

    rng = np.random.default_rng(seed + held_index * 1009 + sum(map(ord, action)))
    draws = []
    shrinkage = float(diagnosis["shrinkage_weight"])
    for _ in range(bootstraps):
        source_draw = rng.dirichlet(np.ones(len(sources)))
        source_weights = dict(zip(sources, source_draw, strict=True))
        source_means = train.groupby("source_id", sort=True)[effect_column].mean()
        prior = float(sum(source_weights[s] * source_means[s] for s in sources))
        state_prediction, fitted = _state_fit_predict(
            train,
            held,
            effect_column,
            alpha=identifiability_contract.ridge_alpha,
            extra_source_weights=source_weights,
        )
        prediction = prior + shrinkage * (float(state_prediction[0]) - prior)
        fitted_shrunk = prior + shrinkage * (fitted - prior)
        residual = train[effect_column].to_numpy(dtype=float) - fitted_shrunk
        residual_scale = max(0.01, float(np.sqrt(np.average(residual**2, weights=_source_weights(train)))))
        draws.append(prediction + float(rng.normal(0.0, residual_scale)))
    values = np.asarray(draws, dtype=float)
    return {
        "mean": float(values.mean()),
        "harm_probability": float(np.mean(values < -harm_margin)),
        "cost_ratio": cost,
        "information_gain": 0.0,
        **diagnosis,
    }


def select_policy_v6(
    table: pd.DataFrame,
    *,
    eligibility_contract: EligibilityContract,
    identifiability_contract: IdentifiabilityContract,
    bootstraps: int,
    seed: int,
) -> pd.DataFrame:
    rows = []
    for index, held in table.iterrows():
        candidates = {
            "stop": gate_action(
                {"mean": 0.0, "harm_probability": 0.0, "cost_ratio": 0.0, "information_gain": 0.0},
                contract=eligibility_contract,
                stop=True,
            )
        }
        for action in ("feature_rich", "nested_imbalance"):
            if action == "nested_imbalance" and held["task_type"] != "classification":
                continue
            posterior = cross_source_posterior_v6(
                table,
                index,
                action=action,
                identifiability_contract=identifiability_contract,
                bootstraps=bootstraps,
                seed=seed,
                harm_margin=eligibility_contract.harm_margin,
            )
            candidates[action] = gate_action(posterior, contract=eligibility_contract)
        eligible = [
            (value["utility"], -value["cost_ratio"], action)
            for action, value in candidates.items()
            if value["eligible"]
        ]
        _, _, selected = max(eligible)
        actual = 0.0 if selected == "stop" else float(held[f"effect_{selected}"])
        oracle = max(
            0.0,
            float(held["effect_feature_rich"]),
            float(held["effect_nested_imbalance"])
            if held["task_type"] == "classification" else -np.inf,
        )
        chosen = candidates[selected]
        rows.append({
            "endpoint_id": held["endpoint_id"], "source_id": held["source_id"],
            "task_type": held["task_type"], "policy": "v6_identifiability_shrinkage",
            "selected_action": selected, "actual_effect": actual,
            "oracle_effect": oracle, "oracle_regret": oracle - actual,
            "harmful_intervention": bool(actual < -eligibility_contract.harm_margin),
            "realized_fit_cost": float(chosen["cost_ratio"]),
            "predicted_mean": float(chosen["mean"]),
            "predicted_harm_probability": float(chosen["harm_probability"]),
            "diagnostics_identifiable": bool(chosen.get("identifiable", False)),
            "diagnostic_shrinkage_weight": float(chosen.get("shrinkage_weight", 0.0)),
            "training_sources": int(chosen.get("training_sources", 0)),
        })
    return pd.DataFrame(rows)


__all__ = [
    "IdentifiabilityContract", "cross_source_posterior_v6",
    "diagnostic_identifiability", "select_policy_v6",
]
