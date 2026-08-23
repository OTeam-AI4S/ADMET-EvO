"""Two-lane category x data-regime policy for the fixed TDC ADMET-22 suite.

Production reuses the qualified frozen finalist and stops.  Research is an
exception path: a task receives one matched experiment only when a compact,
test-blind triage signal identifies a concrete problem.  This keeps scientific
autonomy measurable without turning every production task into a new search.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from typing import Any, Literal

from .tasks import resolve_task

DataRegime = Literal["low_data", "higher_data"]
TaskFamily = Literal[
    "binary_auroc",
    "imbalanced_binary_auprc",
    "continuous_mae",
    "rank_pk_spearman",
]


def data_regime(n_train: int) -> DataRegime:
    """Use two actionable regimes; the former medium bucket had no unique action."""

    if isinstance(n_train, bool) or not isinstance(n_train, int) or n_train <= 0:
        raise ValueError("n_train must be a positive integer")
    return "low_data" if n_train < 1000 else "higher_data"


def task_family(task_name: str) -> TaskFamily:
    task = resolve_task(task_name)
    if task.metric == "auroc":
        return "binary_auroc"
    if task.metric == "auprc":
        return "imbalanced_binary_auprc"
    if task.metric == "mae":
        return "continuous_mae"
    if task.metric == "spearman":
        return "rank_pk_spearman"
    raise ValueError(f"Unsupported metric {task.metric!r}")


_FAMILY_QUESTION = {
    "binary_auroc": "representation_or_scaffold_generalization",
    "imbalanced_binary_auprc": "minority_recall_without_precision_collapse",
    "continuous_mae": "robust_continuous_error_under_scaffold_shift",
    "rank_pk_spearman": "ordering_consistency_under_scaffold_shift",
}


@dataclass(frozen=True)
class LeanTaskRoute:
    task: str
    admet_category: str
    task_family: TaskFamily
    n_train: int
    data_regime: DataRegime
    production_action: str
    research_action: str
    scientific_question: str
    confirmation_seeds: tuple[int, ...]
    maximum_research_hypotheses: int
    maximum_research_configurations: int
    maximum_research_fits: int
    reviewer_mode: str
    reason: str
    test_loaded: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _forbidden_test_paths(value: Any, path: str = "signals") -> tuple[str, ...]:
    forbidden: list[str] = []
    if isinstance(value, Mapping):
        for raw_key, child in value.items():
            text = str(raw_key)
            normalized = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", text).lower()
            tokens = tuple(token for token in re.split(r"[^a-z0-9]+", normalized) if token)
            safe = normalized == "test_loaded" and child is False
            if not safe and any(token in {"test", "testing", "heldout", "holdout"} for token in tokens):
                forbidden.append(f"{path}.{text}")
            forbidden.extend(_forbidden_test_paths(child, f"{path}.{text}"))
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for index, child in enumerate(value):
            forbidden.extend(_forbidden_test_paths(child, f"{path}[{index}]"))
    return tuple(sorted(set(forbidden)))


def route_task(
    task_name: str,
    n_train: int,
    signals: Mapping[str, Any] | None = None,
) -> LeanTaskRoute:
    """Route a task without scoring a candidate or inspecting the frozen test.

    Accepted signals are deliberately few: endpoint/data contract, frozen
    artifact availability, qualification status, uncertainty, and data drift.
    No scalar hardness score or multi-stage search tree is used.
    """

    state = {
        "contract_ok": True,
        "frozen_artifact_available": True,
        "qualification_ok": True,
        "uncertainty_high": False,
        "data_drift_or_gap": False,
        **dict(signals or {}),
    }
    forbidden = _forbidden_test_paths(state)
    if forbidden:
        raise ValueError(f"Test-like fields are forbidden: {list(forbidden)}")
    unknown = set(state) - {
        "contract_ok",
        "frozen_artifact_available",
        "qualification_ok",
        "uncertainty_high",
        "data_drift_or_gap",
        "test_loaded",
    }
    if unknown:
        raise ValueError(f"Unknown triage signals: {sorted(unknown)}")
    non_boolean = [key for key, value in state.items() if not isinstance(value, bool)]
    if non_boolean:
        raise ValueError(f"Triage signals must be boolean: {sorted(non_boolean)}")

    task = resolve_task(task_name)
    family = task_family(task_name)
    regime = data_regime(n_train)
    seeds = (1, 2, 3, 4, 5) if regime == "low_data" else (1, 2, 3)
    common = {
        "task": task.name,
        "admet_category": task.category,
        "task_family": family,
        "n_train": n_train,
        "data_regime": regime,
        "scientific_question": _FAMILY_QUESTION[family],
        "confirmation_seeds": seeds,
        "maximum_research_hypotheses": 1,
        "maximum_research_configurations": 2,
        "test_loaded": False,
    }

    if not state["contract_ok"]:
        return LeanTaskRoute(
            **common,
            production_action="quarantine",
            research_action="repair_data_contract",
            maximum_research_fits=0,
            reviewer_mode="targeted_data_review",
            reason="endpoint_or_data_contract_failed_before_modeling",
        )
    if not state["frozen_artifact_available"]:
        return LeanTaskRoute(
            **common,
            production_action="restore_frozen_artifact",
            research_action="none",
            maximum_research_fits=0,
            reviewer_mode="single_agent",
            reason="qualified_recipe_exists_but_its_artifact_is_unavailable",
        )
    if state["data_drift_or_gap"]:
        return LeanTaskRoute(
            **common,
            production_action="reuse_frozen_recipe_without_mutation",
            research_action="open_data_stewardship_ticket",
            maximum_research_fits=0,
            reviewer_mode="targeted_data_review",
            reason="provenance_or_distribution_signal_requires_data_work_first",
        )
    if not state["qualification_ok"] or state["uncertainty_high"]:
        return LeanTaskRoute(
            **common,
            production_action="reuse_frozen_recipe_without_mutation",
            research_action="run_one_single_axis_matched_experiment",
            maximum_research_fits=2 * len(seeds),
            reviewer_mode="single_agent",
            reason="a_specific_validation_side_problem_justifies_one_research_ticket",
        )
    return LeanTaskRoute(
        **common,
        production_action="reuse_qualified_frozen_recipe_and_stop",
        research_action="none",
        maximum_research_fits=0,
        reviewer_mode="single_agent",
        reason="prediction_is_qualified_and_no_actionable_problem_is_present",
    )


__all__ = ["DataRegime", "LeanTaskRoute", "TaskFamily", "data_regime", "route_task", "task_family"]
