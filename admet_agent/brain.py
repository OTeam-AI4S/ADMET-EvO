"""Planner interface. GPT-5.6 will implement this contract once credentials are available."""

from __future__ import annotations

from typing import Protocol, Sequence

from .registry import is_model_available
from .schemas import Budget, ExperimentSpec, RunResult, TaskProfile
from .tuning import OptunaRefiner


class AgentBrain(Protocol):
    def propose(self, profile: TaskProfile, history: Sequence[RunResult]) -> list[ExperimentSpec]: ...

    def select_finalist(self, profile: TaskProfile, results: Sequence[RunResult]) -> str: ...


class RuleBasedBrain:
    """Deterministic evolutionary planner with task-local and cross-task memory."""

    def __init__(self, max_candidates: int = 6):
        self.max_candidates = max_candidates
        self.cross_task_evidence: list[dict] = []
        self.refiner = OptunaRefiner()

    def set_context(self, *, cross_task_evidence: list[dict]) -> None:
        self.cross_task_evidence = cross_task_evidence

    @staticmethod
    def _signature(model: str, representations, hyperparameters) -> tuple:
        return (
            model,
            tuple(representations),
            tuple(sorted(hyperparameters.items())),
        )

    def propose(self, profile: TaskProfile, history: Sequence[RunResult]) -> list[ExperimentSpec]:
        imbalance = (
            "class_weight"
            if profile.task_type == "classification"
            and profile.positive_rate_train is not None
            and not 0.3 <= profile.positive_rate_train <= 0.7
            else "none"
        )
        rich = ("morgan", "avalon", "erg", "rdkit2d")
        candidates: list[tuple[str, tuple[str, ...], dict, str]] = []
        if not history:
            baseline_models = [
                "extra_trees", "random_forest", "catboost", "xgboost", "lightgbm"
            ]
            if profile.n_train <= 5000:
                baseline_models.append("svm_rbf")
            candidates.extend(
                (model, rich, {}, f"strong multi-view baseline with {model}")
                for model in baseline_models
                if is_model_available(model)
            )
        else:
            direction = 1 if profile.metric != "mae" else -1
            completed = [item for item in history if item.status == "completed"]
            winner = max(completed, key=lambda item: direction * item.validation_score)
            model = winner.model
            for parameters in self.refiner.suggest(
                profile,
                history,
                model,
                tuple(winner.representations),
                count=min(3, self.max_candidates),
            ):
                candidates.append(
                    (
                        model,
                        tuple(winner.representations),
                        parameters,
                        "Optuna TPE refinement informed by completed validation trials",
                    )
                )
            candidates.extend([
                (
                    model,
                    ("morgan_r3", "avalon", "erg", "rdkit2d"),
                    self._tuned(model, 0, profile.task_type),
                    "increase circular radius while retaining transferable descriptors",
                ),
                (
                    model,
                    ("morgan_count", "atom_pair", "maccs", "rdkit2d"),
                    self._tuned(model, 1, profile.task_type),
                    "test count and topological features complementary to binary ECFP",
                ),
                (
                    model,
                    ("morgan", "rdkit_fp", "torsion", "rdkit2d"),
                    self._tuned(model, 2, profile.task_type),
                    "test path and torsion information under scaffold shift",
                ),
            ])
            for evidence in self.cross_task_evidence:
                candidates.insert(
                    0,
                    (
                        evidence["model"],
                        tuple(evidence["representations"]),
                        evidence["hyperparameters"],
                        "transfer a high-percentile recipe from comparable ADMET tasks",
                    ),
                )
        seen = {
            self._signature(item.model, item.representations, item.hyperparameters)
            for item in history
        }
        proposals = []
        for model, representations, hyperparameters, hypothesis in candidates:
            if not is_model_available(model):
                continue
            signature = self._signature(model, representations, hyperparameters)
            if signature in seen:
                continue
            proposals.append(
                ExperimentSpec(
                    task=profile.task,
                    seed=1,
                    representations=representations,
                    model=model,
                    imbalance=imbalance,
                    hyperparameters=hyperparameters,
                    budget=Budget(max_candidates=self.max_candidates),
                    hypothesis=hypothesis,
                )
            )
            if len(proposals) == self.max_candidates:
                break
        return proposals

    @staticmethod
    def _tuned(model: str, variant: int, task_type: str) -> dict:
        criterion = (
            ("gini", "entropy", "log_loss")
            if task_type == "classification"
            else ("squared_error", "absolute_error", "squared_error")
        )[variant]
        if model == "catboost":
            return (
                {"iterations": 900, "depth": 6, "learning_rate": 0.04, "l2_leaf_reg": 5.0},
                {"iterations": 1200, "depth": 5, "learning_rate": 0.025, "l2_leaf_reg": 8.0},
                {"iterations": 700, "depth": 8, "learning_rate": 0.05, "random_strength": 1.5},
            )[variant]
        if model == "xgboost":
            return (
                {"n_estimators": 800, "max_depth": 4, "learning_rate": 0.035,
                 "subsample": 0.9, "colsample_bytree": 0.6, "min_child_weight": 2.0,
                 "reg_lambda": 3.0},
                {"n_estimators": 1000, "max_depth": 3, "learning_rate": 0.025,
                 "subsample": 0.8, "colsample_bytree": 0.8, "min_child_weight": 4.0,
                 "reg_alpha": 0.1, "reg_lambda": 5.0},
                {"n_estimators": 600, "max_depth": 7, "learning_rate": 0.05,
                 "subsample": 0.75, "colsample_bytree": 0.5, "min_child_weight": 6.0,
                 "reg_alpha": 0.5, "reg_lambda": 8.0},
            )[variant]
        if model == "lightgbm":
            return (
                {"n_estimators": 800, "num_leaves": 31, "max_depth": 7,
                 "learning_rate": 0.035, "subsample": 0.9, "colsample_bytree": 0.7,
                 "min_child_samples": 12, "reg_lambda": 3.0},
                {"n_estimators": 1000, "num_leaves": 15, "max_depth": 5,
                 "learning_rate": 0.025, "subsample": 0.8, "colsample_bytree": 0.8,
                 "min_child_samples": 20, "reg_alpha": 0.1, "reg_lambda": 5.0},
                {"n_estimators": 600, "num_leaves": 63, "max_depth": 9,
                 "learning_rate": 0.05, "subsample": 0.75, "colsample_bytree": 0.6,
                 "min_child_samples": 25, "reg_alpha": 0.5, "reg_lambda": 8.0},
            )[variant]
        if model == "svm_rbf":
            return (
                {"C": 3.0, "gamma": "scale", **({"epsilon": 0.1} if task_type == "regression" else {})},
                {"C": 10.0, "gamma": 0.001, **({"epsilon": 0.05} if task_type == "regression" else {})},
                {"C": 30.0, "gamma": 0.0001, **({"epsilon": 0.2} if task_type == "regression" else {})},
            )[variant]
        return (
            {"n_estimators": 700, "min_samples_leaf": 1, "max_features": 0.5, "criterion": criterion},
            {"n_estimators": 800, "min_samples_leaf": 3, "max_features": "sqrt", "criterion": criterion},
            {"n_estimators": 600, "min_samples_leaf": 2, "max_features": 0.8, "criterion": criterion},
        )[variant]

    def select_finalist(self, profile: TaskProfile, results: Sequence[RunResult]) -> str:
        if not results:
            raise ValueError("Cannot select a finalist without completed runs")
        direction = 1 if profile.metric != "mae" else -1
        return max(results, key=lambda item: direction * item.validation_score).run_id


class HybridBrain:
    """Broad deterministic baselines followed by GPT and Optuna refinement."""

    def __init__(self, planner: AgentBrain, max_candidates: int = 6):
        self.planner = planner
        self.rules = RuleBasedBrain(max_candidates=max_candidates)
        self.max_candidates = max_candidates
        self.errors: list[str] = []

    def set_context(self, *, cross_task_evidence: list[dict]) -> None:
        self.rules.set_context(cross_task_evidence=cross_task_evidence)
        setter = getattr(self.planner, "set_context", None)
        if callable(setter):
            setter(cross_task_evidence=cross_task_evidence)

    def drain_errors(self) -> list[str]:
        errors, self.errors = self.errors, []
        return errors

    @staticmethod
    def _signature(spec: ExperimentSpec) -> tuple:
        return (
            spec.model,
            tuple(spec.representations),
            tuple(sorted(spec.hyperparameters.items())),
        )

    def propose(
        self,
        profile: TaskProfile,
        history: Sequence[RunResult],
    ) -> list[ExperimentSpec]:
        if not history:
            return self.rules.propose(profile, history)
        try:
            combined = list(self.planner.propose(profile, history))
        except Exception as exc:
            self.errors.append(f"GPT planner fallback: {type(exc).__name__}: {exc}")
            combined = []
        combined.extend(self.rules.propose(profile, history))
        seen = set()
        proposals = []
        for item in combined:
            signature = self._signature(item)
            if signature in seen:
                continue
            seen.add(signature)
            payload = item.to_dict()
            payload["budget"] = {
                "max_candidates": self.max_candidates,
                "max_minutes": item.budget.max_minutes,
            }
            proposals.append(ExperimentSpec.from_dict(payload))
            if len(proposals) == self.max_candidates:
                break
        return proposals

    def select_finalist(
        self,
        profile: TaskProfile,
        results: Sequence[RunResult],
    ) -> str:
        return self.rules.select_finalist(profile, results)


class OpenAIPlanner:
    """Credential-free placeholder documenting the GPT-5.6 integration boundary."""

    def __init__(self, model: str = "gpt-5.6-sol"):
        self.model = model

    def propose(self, profile: TaskProfile, history: Sequence[RunResult]) -> list[ExperimentSpec]:
        raise RuntimeError(
            "OpenAI planner is not configured yet. Set up the dedicated API credential, "
            "then implement this adapter with the Responses API and strict JSON schema output."
        )

    def select_finalist(self, profile: TaskProfile, results: Sequence[RunResult]) -> str:
        raise RuntimeError("OpenAI planner is not configured yet")
