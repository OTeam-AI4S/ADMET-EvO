"""GPT-5.6 Responses adapter for validation-only experiment decisions."""
from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from .registry import planner_catalog
from .schemas import Budget, ExperimentSpec, RunResult, TaskProfile


class Hyperparameters(BaseModel):
    model_config = ConfigDict(extra="forbid")
    n_estimators: int | None
    n_members: int | None
    majority_ratio: float | None
    min_samples_leaf: int | None
    max_features: float | Literal["sqrt", "log2"] | None
    criterion: Literal["squared_error", "absolute_error", "gini", "entropy", "log_loss"] | None
    bootstrap: bool | None
    max_samples: float | None
    iterations: int | None
    depth: int | None
    learning_rate: float | None
    l2_leaf_reg: float | None
    random_strength: float | None
    max_depth: int | None
    subsample: float | None
    colsample_bytree: float | None
    min_child_weight: float | None
    reg_alpha: float | None
    reg_lambda: float | None
    num_leaves: int | None
    min_child_samples: int | None
    C: float | None
    gamma: float | Literal["scale", "auto"] | None
    epsilon: float | None

    def compact(self, model: str) -> dict:
        values = {key: value for key, value in self.model_dump().items() if value is not None}
        allowed = {
            "extra_trees": {"n_estimators", "min_samples_leaf", "max_features", "criterion", "bootstrap", "max_samples"},
            "random_forest": {"n_estimators", "min_samples_leaf", "max_features", "criterion", "bootstrap", "max_samples"},
            "balanced_extra_trees": {"n_estimators", "n_members", "majority_ratio", "min_samples_leaf", "max_features", "criterion"},
            "catboost": {"iterations", "depth", "learning_rate", "l2_leaf_reg", "random_strength"},
            "xgboost": {"n_estimators", "max_depth", "learning_rate", "subsample", "colsample_bytree",
                        "min_child_weight", "reg_alpha", "reg_lambda"},
            "lightgbm": {"n_estimators", "num_leaves", "max_depth", "learning_rate", "subsample",
                         "colsample_bytree", "min_child_samples", "reg_alpha", "reg_lambda"},
            "svm_rbf": {"C", "gamma", "epsilon"},
        }
        return {key: value for key, value in values.items() if key in allowed[model]}


class Candidate(BaseModel):
    model_config = ConfigDict(extra="forbid")
    representations: list[Literal["morgan", "morgan_r3", "morgan_count", "avalon", "erg", "maccs", "rdkit_fp", "atom_pair", "torsion", "rdkit2d", "unimol2"]] = Field(min_length=1, max_length=4)
    model: Literal["extra_trees", "random_forest", "balanced_extra_trees", "catboost", "xgboost", "lightgbm", "svm_rbf"]
    imbalance: Literal["none", "class_weight"]
    hyperparameters: Hyperparameters
    hypothesis: str


class ExperimentPlan(BaseModel):
    model_config = ConfigDict(extra="forbid")
    analysis: str
    candidates: list[Candidate] = Field(min_length=1, max_length=3)


class FinalistDecision(BaseModel):
    model_config = ConfigDict(extra="forbid")
    run_id: str
    rationale: str


class OpenAIPlanner:
    def __init__(self, config_path: Path, model: str = "gpt-5.6-sol", audit_path: Path | None = None):
        from openai import OpenAI
        config = self._read_config(config_path)
        self.model = model
        self.client = OpenAI(api_key=config["OPENAI_API_KEY"], base_url=config.get("OPENAI_BASE_URL"))
        self.audit_path = audit_path
        self.cross_task_evidence: list[dict] = []
        self.hardness_card: dict = {}
        self.research_plan: dict = {}

    def set_context(
        self,
        *,
        cross_task_evidence: list[dict],
        hardness_card: dict | None = None,
        research_plan: dict | None = None,
    ) -> None:
        self.cross_task_evidence = cross_task_evidence
        self.hardness_card = hardness_card or {}
        self.research_plan = research_plan or {}

    @staticmethod
    def _read_config(path: Path) -> dict[str, str]:
        config = {}
        for line in path.read_text(encoding="utf-8").splitlines():
            name, separator, value = line.partition("=")
            if separator and value.strip():
                config[name.strip()] = value.strip()
        if "OPENAI_API_KEY" not in config:
            raise ValueError(f"OPENAI_API_KEY is missing from {path}")
        return config

    def _audit(self, response, phase: str, task: str) -> None:
        if self.audit_path is None:
            return
        self.audit_path.parent.mkdir(parents=True, exist_ok=True)
        usage = response.usage
        record = {"phase": phase, "task": task, "model": response.model, "response_id": response.id,
                  "input_tokens": usage.input_tokens, "output_tokens": usage.output_tokens,
                  "total_tokens": usage.total_tokens}
        with self.audit_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, sort_keys=True) + "\n")

    def propose(self, profile: TaskProfile, history: Sequence[RunResult]) -> list[ExperimentSpec]:
        prompt = {
            "task_profile": profile.to_dict(),
            "prior_validation_runs": [item.to_dict() for item in history],
            "tool_catalog": planner_catalog(),
            "cross_task_evidence": self.cross_task_evidence,
            "hardness_card": self.hardness_card,
            "research_plan": self.research_plan,
            "allowed": {"imbalance": ["none", "class_weight"]},
            "constraints": {"n_estimators": [100, 1200], "min_samples_leaf": [1, 20],
                            "n_members": [3, 25], "majority_ratio": [0.5, 10.0],
                            "max_features_float": [0.1, 1.0], "max_samples": [0.3, 1.0],
                            "iterations": [100, 2000], "depth": [3, 12],
                            "learning_rate": [0.005, 0.3], "l2_leaf_reg": [0.0, 30.0],
                            "random_strength": [0.0, 10.0], "max_depth": [2, 16],
                            "subsample": [0.3, 1.0], "colsample_bytree": [0.1, 1.0],
                            "min_child_weight": [0.0, 30.0], "reg_alpha": [0.0, 50.0],
                            "reg_lambda": [0.0, 50.0], "num_leaves": [7, 255],
                            "min_child_samples": [2, 100], "C": [0.01, 1000.0],
                            "gamma_numeric": [1e-6, 10.0], "epsilon": [0.001, 2.0]},
            "rules": ["Return 1-3 diverse experiments", "Use class_weight only for classification",
                      "Never use test information", "Fill every hyperparameter field; use null when irrelevant",
                      "Use only parameters valid for the selected model", "Do not repeat a prior configuration",
                      "External-data skills may propose evidence work but cannot directly merge data into these model candidates"],
        }
        response = self.client.responses.parse(
            model=self.model, reasoning={"effort": "high"},
            instructions=("Plan molecular ADMET experiments under scaffold shift. Optimize only the supplied "
                          "validation metric. Use prior runs as evidence and favor falsifiable hypotheses."),
            input=json.dumps(prompt), text_format=ExperimentPlan)
        self._audit(response, "plan", profile.task)
        plan = response.output_parsed
        if plan is None:
            raise RuntimeError("GPT-5.6 returned no parsed experiment plan")
        count = len(plan.candidates)
        return [ExperimentSpec(task=profile.task, seed=1, representations=tuple(item.representations),
                               model=item.model, imbalance=item.imbalance,
                               hyperparameters=item.hyperparameters.compact(item.model),
                               budget=Budget(max_candidates=count, max_minutes=60),
                               hypothesis=item.hypothesis) for item in plan.candidates]

    def select_finalist(self, profile: TaskProfile, results: Sequence[RunResult]) -> str:
        response = self.client.responses.parse(
            model=self.model, reasoning={"effort": "medium"},
            instructions=("Select exactly one completed validation run by run_id. MAE is minimized; AUROC, AUPRC, "
                          "and Spearman are maximized. Never infer test performance."),
            input=json.dumps({"task_profile": profile.to_dict(),
                              "completed_validation_runs": [item.to_dict() for item in results]}),
            text_format=FinalistDecision)
        self._audit(response, "select_finalist", profile.task)
        decision = response.output_parsed
        available = {item.run_id for item in results}
        if decision is None or decision.run_id not in available:
            raise RuntimeError("GPT-5.6 selected an unavailable finalist run")
        return decision.run_id
