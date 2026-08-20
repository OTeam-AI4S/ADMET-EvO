"""Strict parsing and prediction aggregation for preregistered expert recipes."""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np
from scipy.stats import rankdata

from .tasks import TaskSpec

_IDENTIFIER = re.compile(r"^[a-z0-9_]+$")
_TERM = re.compile(r"^(?P<name>[a-z0-9_]+)@(?P<weight>[0-9]+(?:\.[0-9]+)?)$")


@dataclass(frozen=True)
class RecipeMember:
    name: str
    weight: float


@dataclass(frozen=True)
class FrozenRecipe:
    aggregation: str
    members: tuple[RecipeMember, ...]

    @property
    def expert_names(self) -> tuple[str, ...]:
        return tuple(member.name for member in self.members)

    def to_dict(self) -> dict:
        return {
            "aggregation": self.aggregation,
            "members": [
                {"name": member.name, "weight": member.weight}
                for member in self.members
            ],
        }


@dataclass(frozen=True)
class ExpertRequirement:
    backend: str
    representations: tuple[str, ...] = ()
    model: str | None = None
    target_transform: str = "identity"
    requires_inner_oof: bool = False


_RICH_2D = ("morgan", "avalon", "erg", "rdkit2d")
EXPERTS = {
    "2d_lightgbm": ExpertRequirement("tabular", _RICH_2D, "lightgbm"),
    "2d_extra_trees": ExpertRequirement("tabular", _RICH_2D, "extra_trees"),
    "2d_svm": ExpertRequirement("tabular", _RICH_2D, "svm_rbf"),
    "finetuned_unimol2": ExpertRequirement(
        "unimol2_finetune", requires_inner_oof=True
    ),
    "fixed_rank_top3": ExpertRequirement(
        "dynamic_oof_ensemble", target_transform="percentile_rank", requires_inner_oof=True
    ),
    "rank_target_xgboost": ExpertRequirement(
        "tabular", _RICH_2D, "xgboost", target_transform="percentile_rank"
    ),
    "minimol_morgan_kernel": ExpertRequirement(
        "precomputed_kernel",
        ("morgan", "morgan_r3", "minimol"),
        "svc_c10_balanced",
    ),
    "focal_minimol_16": ExpertRequirement(
        "isolated_torch_head",
        ("minimol",),
        "focal_mlp_16x100",
    ),
    "maplight_gnn": ExpertRequirement(
        "isolated_maplight_gnn",
        ("morgan_count", "avalon_count", "erg", "rdkit2d", "gin_supervised_masking"),
        "catboost_fixed",
    ),
}


def parse_recipe(value: str) -> FrozenRecipe:
    value = value.strip()
    if _IDENTIFIER.fullmatch(value):
        return FrozenRecipe("raw", (RecipeMember(value, 1.0),))
    match = re.fullmatch(r"(raw|rank)\((.+)\)", value)
    if match is None:
        raise ValueError(f"Invalid frozen recipe: {value!r}")
    aggregation, body = match.groups()
    terms = body.split("+")
    members = []
    for term in terms:
        parsed = _TERM.fullmatch(term)
        if parsed is None:
            raise ValueError(f"Invalid frozen recipe term: {term!r}")
        members.append(
            RecipeMember(parsed.group("name"), float(parsed.group("weight")))
        )
    names = [member.name for member in members]
    if len(set(names)) != len(names):
        raise ValueError("Frozen recipes cannot repeat an expert")
    total = sum(member.weight for member in members)
    if total <= 0:
        raise ValueError("Frozen recipe weights must have a positive sum")
    return FrozenRecipe(
        aggregation,
        tuple(RecipeMember(member.name, member.weight / total) for member in members),
    )


def build_execution_plan(task: TaskSpec, value: str) -> dict:
    recipe = parse_recipe(value)
    if recipe.aggregation == "rank" and task.metric == "mae":
        raise ValueError("Rank aggregation is not valid for MAE tasks")
    unknown = sorted(set(recipe.expert_names) - EXPERTS.keys())
    if unknown:
        raise ValueError(f"Frozen recipe contains unknown experts: {unknown}")
    return {
        "task": task.name,
        "metric": task.metric,
        "recipe": value,
        **recipe.to_dict(),
        "experts": {
            name: {
                "backend": EXPERTS[name].backend,
                "representations": list(EXPERTS[name].representations),
                "model": EXPERTS[name].model,
                "target_transform": EXPERTS[name].target_transform,
                "requires_inner_oof": EXPERTS[name].requires_inner_oof,
            }
            for name in recipe.expert_names
        },
    }


def aggregate_predictions(
    recipe: FrozenRecipe,
    predictions: Mapping[str, np.ndarray],
) -> np.ndarray:
    missing = sorted(set(recipe.expert_names) - predictions.keys())
    if missing:
        raise ValueError(f"Missing expert predictions: {missing}")
    values = []
    expected_length = None
    for member in recipe.members:
        prediction = np.asarray(predictions[member.name], dtype=float)
        if prediction.ndim != 1 or len(prediction) == 0:
            raise ValueError("Expert predictions must be non-empty one-dimensional arrays")
        if not np.isfinite(prediction).all():
            raise ValueError("Expert predictions must be finite")
        if expected_length is None:
            expected_length = len(prediction)
        elif len(prediction) != expected_length:
            raise ValueError("Expert prediction lengths do not match")
        if recipe.aggregation == "rank":
            prediction = rankdata(prediction, method="average") / len(prediction)
        values.append(member.weight * prediction)
    return np.sum(values, axis=0)
