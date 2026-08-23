"""Resolve scientific action families to executable EC-VOI v2 pairs."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from typing import Any

from .registry import is_model_available, is_representation_available
from .research_policy_v2 import ActionPair
from .schemas import Budget, ExperimentSpec, TaskProfile

RICH_2D = ("morgan", "avalon", "erg", "rdkit2d")


def _digest(value: Any) -> str:
    rendered = json.dumps(
        value, sort_keys=True, separators=(",", ":"), default=str
    )
    return hashlib.sha256(rendered.encode()).hexdigest()


@dataclass(frozen=True)
class ResolvedActionPair:
    action_pair: ActionPair
    treatment: ExperimentSpec
    control: ExperimentSpec
    per_fit_timeout_seconds: float

    def __post_init__(self) -> None:
        if (
            not math.isfinite(self.per_fit_timeout_seconds)
            or self.per_fit_timeout_seconds <= 0
        ):
            raise ValueError("Per-fit timeout must be finite and positive")
        expected = self.per_fit_timeout_seconds * 2 * len(
            self.action_pair.ordered_seeds
        )
        if not math.isclose(
            expected, self.action_pair.predicted_pair_fit_seconds
        ):
            raise ValueError("Pair reservation must cover every treatment/control fit")


def _base_spec(
    profile: TaskProfile,
    *,
    seed: int,
    model: str,
    representations: tuple[str, ...],
    imbalance: str,
    data_policy: str,
    max_minutes: int,
    hypothesis: str,
) -> ExperimentSpec:
    return ExperimentSpec(
        task=profile.task,
        seed=seed,
        representations=representations,
        model=model,
        imbalance=imbalance,
        data_policy=data_policy,
        budget=Budget(max_candidates=2, max_minutes=max_minutes),
        hypothesis=hypothesis,
    )


def resolve_action_pair(
    action: dict[str, Any],
    profile: TaskProfile,
    *,
    ordered_seeds: tuple[int, ...],
    context_sha256: str,
    task_fit_budget_seconds: float,
    task_candidate_budget: int,
    per_fit_timeout_seconds: float = 300.0,
    practical_effect_threshold: float = 0.005,
    effect_standardization_scale: float = 0.005,
    predicted_native_observation_se: float = 0.005,
) -> ResolvedActionPair | None:
    """Resolve a supported action family or reject it without silent fallback."""

    action_id = str(action.get("action_id", ""))
    axis = action.get("axis")
    if action_id in {"", "data-leakage-safe", "data-external-quarantine"}:
        return None
    if axis not in {"feature", "model", "data"}:
        # Replicate and stop are episode decisions, not synthetic score pairs.
        return None
    if len(ordered_seeds) < 3:
        raise ValueError("EC-VOI v2 action pairs require at least three seeds")
    pair_seconds = 2.0 * len(ordered_seeds) * per_fit_timeout_seconds
    if pair_seconds > task_fit_budget_seconds:
        return None
    max_minutes = max(1, math.ceil(per_fit_timeout_seconds / 60.0))
    hypothesis = f"EC-VOI v2 matched pair for {action_id}"
    primary_seed = ordered_seeds[0]

    treatment: ExperimentSpec
    control: ExperimentSpec
    if action_id == "feature-domain-shift":
        if not is_model_available("extra_trees") or not all(
            is_representation_available(name) for name in RICH_2D
        ):
            return None
        treatment = _base_spec(
            profile,
            seed=primary_seed,
            model="extra_trees",
            representations=RICH_2D,
            imbalance="none",
            data_policy="identity_safe",
            max_minutes=max_minutes,
            hypothesis=hypothesis,
        )
        control = _base_spec(
            profile,
            seed=primary_seed,
            model="extra_trees",
            representations=("morgan",),
            imbalance="none",
            data_policy="identity_safe",
            max_minutes=max_minutes,
            hypothesis=hypothesis,
        )
    elif action_id == "model-estimator-family":
        if not all(
            is_model_available(name) for name in ("random_forest", "extra_trees")
        ) or not all(is_representation_available(name) for name in RICH_2D):
            return None
        common = {
            "profile": profile,
            "seed": primary_seed,
            "representations": RICH_2D,
            "imbalance": "none",
            "data_policy": "identity_safe",
            "max_minutes": max_minutes,
            "hypothesis": hypothesis,
        }
        treatment = _base_spec(model="random_forest", **common)
        control = _base_spec(model="extra_trees", **common)
    elif action_id == "model-imbalance-aware":
        if profile.task_type != "classification" or not is_model_available(
            "extra_trees"
        ):
            return None
        common = {
            "profile": profile,
            "seed": primary_seed,
            "model": "extra_trees",
            "representations": RICH_2D,
            "data_policy": "identity_safe",
            "max_minutes": max_minutes,
            "hypothesis": hypothesis,
        }
        treatment = _base_spec(imbalance="class_weight", **common)
        control = _base_spec(imbalance="none", **common)
    elif action_id == "data-quality-filter":
        model = "lightgbm" if is_model_available("lightgbm") else "extra_trees"
        if not is_model_available(model):
            return None
        imbalance = (
            "class_weight"
            if profile.task_type == "classification"
            and profile.positive_rate_train is not None
            and not 0.3 <= profile.positive_rate_train <= 0.7
            else "none"
        )
        common = {
            "profile": profile,
            "seed": primary_seed,
            "model": model,
            "representations": RICH_2D,
            "imbalance": imbalance,
            "max_minutes": max_minutes,
            "hypothesis": hypothesis,
        }
        treatment = _base_spec(data_policy="leakage_safe", **common)
        control = _base_spec(data_policy="identity_safe", **common)
    else:
        return None

    pair_payload = {
        "action_id": action_id,
        "context_sha256": context_sha256,
        "ordered_seeds": ordered_seeds,
        "treatment": treatment.to_dict(),
        "control": control.to_dict(),
    }
    pair = ActionPair(
        pair_id=f"{action_id}:{_digest(pair_payload)}",
        action_id=action_id,
        axis=axis,
        treatment=treatment.to_dict(),
        control=control.to_dict(),
        ordered_seeds=ordered_seeds,
        predicted_pair_fit_seconds=pair_seconds,
        candidate_charge=2,
        task_fit_budget_seconds=task_fit_budget_seconds,
        task_candidate_budget=task_candidate_budget,
        practical_effect_threshold=practical_effect_threshold,
        effect_standardization_scale=effect_standardization_scale,
        predicted_native_observation_se=predicted_native_observation_se,
        scientific_risk=float(action.get("scientific_risk", 0.0)),
        context_sha256=context_sha256,
    )
    return ResolvedActionPair(
        action_pair=pair,
        treatment=treatment,
        control=control,
        per_fit_timeout_seconds=per_fit_timeout_seconds,
    )


__all__ = ["RICH_2D", "ResolvedActionPair", "resolve_action_pair"]
