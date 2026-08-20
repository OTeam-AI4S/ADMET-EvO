"""Matched-control evidence for diagnosis-driven research actions.

The planner may suggest many scientifically plausible actions, but an action is
credited only when discovery runs contain a treatment/control pair that differs
on the declared axis and is repeated on the same validation seeds.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from .schemas import ExperimentSpec, RunResult
from .tasks import TaskSpec


@dataclass(frozen=True)
class MatchedActionEvidence:
    action_id: str
    axis: str
    status: str
    treatment_run_id: str | None
    control_run_id: str | None
    validation_seeds: tuple[int, ...]
    oriented_seed_deltas: tuple[float, ...]
    mean_oriented_delta: float | None
    delta_std: float | None
    minimum_effect: float
    single_axis_verified: bool
    explanation: str
    positive_delta_rate: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _action(plan: dict[str, Any]) -> dict[str, Any] | None:
    selected = plan.get("selected_action_id")
    return next(
        (item for item in plan.get("actions", []) if item.get("action_id") == selected),
        None,
    )


def _same_model_conditions(left: ExperimentSpec, right: ExperimentSpec) -> bool:
    return (
        left.model == right.model
        and left.hyperparameters == right.hyperparameters
        and left.imbalance == right.imbalance
        and left.data_policy == right.data_policy
    )


def _same_data_conditions(left: ExperimentSpec, right: ExperimentSpec) -> bool:
    return (
        left.model == right.model
        and left.representations == right.representations
        and left.hyperparameters == right.hyperparameters
        and left.imbalance == right.imbalance
    )


def _find_pair(
    action: dict[str, Any],
    experiments: dict[str, ExperimentSpec],
) -> tuple[str, str] | None:
    axis = action["axis"]
    payload = action.get("payload", {})
    rows = list(experiments.items())
    if axis == "feature":
        treatments = [
            (run_id, spec)
            for run_id, spec in rows
            if "feature intervention" in spec.hypothesis
        ]
        controls = [
            (run_id, spec)
            for run_id, spec in rows
            if "matched feature control" in spec.hypothesis
        ]
        for treatment_id, treatment in treatments:
            for control_id, control in controls:
                if (
                    _same_model_conditions(treatment, control)
                    and treatment.representations != control.representations
                ):
                    return treatment_id, control_id
    elif axis == "data":
        target_policy = payload.get("data_policy")
        for treatment_id, treatment in rows:
            if treatment.data_policy != target_policy:
                continue
            for control_id, control in rows:
                if control.data_policy == "official" and _same_data_conditions(
                    treatment, control
                ):
                    return treatment_id, control_id
    elif axis == "model":
        target_models = set(payload.get("candidate_models", []))
        if payload.get("candidate_model"):
            target_models.add(payload["candidate_model"])
        fallback = {
            "balanced_extra_trees": "extra_trees",
            "random_forest": "extra_trees",
            "svm_rbf": "extra_trees",
        }
        for treatment_id, treatment in rows:
            if treatment.model not in target_models:
                continue
            wanted_control = fallback.get(treatment.model)
            for control_id, control in rows:
                if (
                    control.model == wanted_control
                    and control.representations == treatment.representations
                    and control.data_policy == treatment.data_policy
                ):
                    return treatment_id, control_id
    return None


def selected_action_pair(
    plan: dict[str, Any],
    experiments: dict[str, ExperimentSpec],
) -> tuple[str, str] | None:
    """Resolve the preregistered treatment/control IDs for structural protection."""

    action = _action(plan)
    if action is None or action.get("axis") not in {"feature", "model", "data"}:
        return None
    return _find_pair(action, experiments)


def evaluate_action_evidence(
    plan: dict[str, Any],
    task: TaskSpec,
    experiments: dict[str, ExperimentSpec],
    seed_runs: dict[str, dict[int, RunResult]],
    *,
    minimum_effect: float = 0.005,
) -> MatchedActionEvidence:
    """Evaluate the selected action without using test data.

    ``supported`` requires a positive mean paired effect at or above the
    preregistered native-metric threshold and at least 80% of paired seeds to
    point in the same direction.  ``refuted`` applies the symmetric rule.
    Missing pairs, failed repeats, unstable directions, or smaller effects
    remain ``inconclusive`` and cannot count as autonomous scientific success.
    """

    action = _action(plan)
    action_id = str(plan.get("selected_action_id", "unknown"))
    if action is None:
        return MatchedActionEvidence(
            action_id,
            "unknown",
            "not_executed",
            None,
            None,
            (),
            (),
            None,
            None,
            minimum_effect,
            False,
            "Selected action is missing from the preregistered action list.",
        )
    axis = action["axis"]
    if axis == "replicate":
        repeated = [
            (run_id, runs)
            for run_id, runs in seed_runs.items()
            if len([item for item in runs.values() if item.status == "completed"]) >= 5
        ]
        if len(repeated) < 2:
            return MatchedActionEvidence(
                action_id,
                axis,
                "inconclusive",
                None,
                None,
                (),
                (),
                None,
                None,
                minimum_effect,
                False,
                "Replication action requires at least two candidates over five completed discovery seeds.",
            )
        direction = 1.0 if task.maximize else -1.0
        ordered = sorted(
            repeated,
            key=lambda item: (
                direction
                * float(
                    np.mean(
                        [
                            run.validation_score
                            for run in item[1].values()
                            if run.status == "completed"
                        ]
                    )
                )
            ),
            reverse=True,
        )
        treatment_id, treatment_runs = ordered[0]
        control_id, control_runs = ordered[1]
        shared = sorted(set(treatment_runs) & set(control_runs))
        deltas = tuple(
            direction
            * (
                treatment_runs[seed].validation_score
                - control_runs[seed].validation_score
            )
            for seed in shared
            if treatment_runs[seed].status == "completed"
            and control_runs[seed].status == "completed"
        )
        win_rate = float(np.mean(np.asarray(deltas) >= 0.0)) if deltas else 0.0
        mean_delta = float(np.mean(deltas)) if deltas else None
        delta_std = float(np.std(deltas)) if deltas else None
        if (
            len(deltas) >= 5
            and win_rate >= 0.8
            and mean_delta is not None
            and mean_delta >= 0
        ):
            status = "supported"
            explanation = "The promoted candidate beat the runner-up on at least 80% of paired seeds."
        elif len(deltas) >= 5 and win_rate <= 0.5:
            status = "refuted"
            explanation = "The apparent winner did not remain superior across discovery resamples."
        else:
            status = "inconclusive"
            explanation = (
                "Replication did not establish stable superiority over the runner-up."
            )
        return MatchedActionEvidence(
            action_id,
            axis,
            status,
            treatment_id,
            control_id,
            tuple(shared),
            deltas,
            mean_delta,
            delta_std,
            minimum_effect,
            True,
            explanation,
            win_rate,
        )
    if axis == "stop":
        return MatchedActionEvidence(
            action_id,
            axis,
            "not_executed",
            None,
            None,
            (),
            (),
            None,
            None,
            minimum_effect,
            False,
            "Stopping is evaluated at the episode level, not as a score intervention.",
        )
    pair = _find_pair(action, experiments)
    if pair is None:
        return MatchedActionEvidence(
            action_id,
            axis,
            "inconclusive",
            None,
            None,
            (),
            (),
            None,
            None,
            minimum_effect,
            False,
            "No treatment/control pair differing only on the declared action axis was executed.",
        )
    treatment_id, control_id = pair
    treatment_runs = seed_runs.get(treatment_id, {})
    control_runs = seed_runs.get(control_id, {})
    shared = sorted(set(treatment_runs) & set(control_runs))
    direction = 1.0 if task.maximize else -1.0
    deltas = tuple(
        direction
        * (treatment_runs[seed].validation_score - control_runs[seed].validation_score)
        for seed in shared
        if treatment_runs[seed].status == "completed"
        and control_runs[seed].status == "completed"
    )
    if len(deltas) < 3:
        status = "inconclusive"
        explanation = (
            "Matched action evidence requires at least three completed paired seeds."
        )
        mean_delta = float(np.mean(deltas)) if deltas else None
        delta_std = float(np.std(deltas)) if deltas else None
        positive_rate = float(np.mean(np.asarray(deltas) > 0.0)) if deltas else None
    else:
        mean_delta = float(np.mean(deltas))
        delta_std = float(np.std(deltas))
        positive_rate = float(np.mean(np.asarray(deltas) > 0.0))
        negative_rate = float(np.mean(np.asarray(deltas) < 0.0))
        if mean_delta >= minimum_effect and positive_rate >= 0.8:
            status = "supported"
            explanation = (
                "The preregistered intervention exceeded the minimum paired "
                "effect with at least 80% directional consistency."
            )
        elif mean_delta <= -minimum_effect and negative_rate >= 0.8:
            status = "refuted"
            explanation = (
                "The preregistered intervention was worse than its matched "
                "control on at least 80% of paired seeds."
            )
        else:
            status = "inconclusive"
            explanation = (
                "The paired effect did not satisfy both the minimum practical "
                "effect and 80% directional-consistency gates."
            )
    return MatchedActionEvidence(
        action_id,
        axis,
        status,
        treatment_id,
        control_id,
        tuple(shared),
        deltas,
        mean_delta,
        delta_std,
        minimum_effect,
        True,
        explanation,
        positive_rate,
    )
