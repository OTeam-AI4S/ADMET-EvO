"""Serializable contracts between the agent brain and deterministic executor."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from .registry import MODELS, REPRESENTATIONS


ALLOWED_REPRESENTATIONS = frozenset(REPRESENTATIONS)
ALLOWED_MODELS = frozenset(MODELS)
ALLOWED_IMBALANCE = frozenset({"none", "class_weight"})
ALLOWED_DEVICES = frozenset({"cpu", "cuda"})


@dataclass(frozen=True)
class Budget:
    max_candidates: int = 3
    max_minutes: int = 60

    @classmethod
    def from_dict(cls, value: dict[str, Any] | None) -> "Budget":
        return cls(**(value or {}))


@dataclass(frozen=True)
class ExperimentSpec:
    task: str
    seed: int
    representations: tuple[str, ...] = ("morgan", "rdkit2d")
    model: str = "extra_trees"
    imbalance: str = "none"
    device: str = "cpu"
    n_jobs: int = 8
    hyperparameters: dict[str, Any] = field(default_factory=dict)
    budget: Budget = field(default_factory=Budget)
    hypothesis: str = "protocol baseline"

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "ExperimentSpec":
        payload = dict(value)
        payload["representations"] = tuple(payload.get("representations", ("morgan", "rdkit2d")))
        payload["budget"] = Budget.from_dict(payload.get("budget"))
        return cls(**payload)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class TaskProfile:
    task: str
    task_type: str
    metric: str
    n_train: int
    n_valid: int
    positive_rate_train: float | None
    target_mean: float
    target_std: float
    invalid_smiles_train: int
    unique_scaffolds_train: int
    unseen_valid_scaffold_rate: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class RunResult:
    run_id: str
    task: str
    seed: int
    model: str
    metric: str
    validation_score: float
    elapsed_seconds: float
    validation_std: float | None = None
    representations: tuple[str, ...] = ()
    hyperparameters: dict[str, Any] = field(default_factory=dict)
    hypothesis: str = ""
    status: str = "completed"
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
