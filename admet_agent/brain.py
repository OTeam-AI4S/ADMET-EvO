"""Planner interface. GPT-5.6 will implement this contract once credentials are available."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

from .registry import is_model_available, is_representation_available
from .schemas import Budget, ExperimentSpec, RunResult, TaskProfile
from .tuning import OptunaRefiner


class AgentBrain(Protocol):
    def propose(
        self, profile: TaskProfile, history: Sequence[RunResult]
    ) -> list[ExperimentSpec]: ...

    def select_finalist(
        self, profile: TaskProfile, results: Sequence[RunResult]
    ) -> str: ...


class RuleBasedBrain:
    """Deterministic evolutionary planner with task-local and cross-task memory."""

    def __init__(self, max_candidates: int = 6):
        self.max_candidates = max_candidates
        self.cross_task_evidence: list[dict] = []
        self.refiner = OptunaRefiner()
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
    def _signature(
        model: str, representations, hyperparameters, data_policy: str = "official"
    ) -> tuple:
        return (
            model,
            tuple(representations),
            tuple(sorted(hyperparameters.items())),
            data_policy,
        )

    def _allow_unimol2(self) -> bool:
        """Require discovery evidence before proposing an expensive 3D view."""
        if self.research_plan.get("selected_action_id") != "feature-domain-shift":
            return False
        domain_shift = self.hardness_card.get("dimensions", {}).get("domain_shift", {})
        if float(domain_shift.get("severity", 0.0)) < 0.65:
            return False
        if float(domain_shift.get("confidence", 0.0)) < 0.50:
            return False
        selected = next(
            (
                action
                for action in self.research_plan.get("actions", [])
                if action.get("action_id") == "feature-domain-shift"
            ),
            {},
        )
        prior = selected.get("prior_evidence") or {}
        if not prior or prior.get("routing_eligible") is not True:
            return False
        if float(prior.get("mean_oriented_delta", 0.0)) < 0.0:
            return False
        return float(prior.get("posterior_support_probability", 0.5)) >= 0.5

    def propose(
        self, profile: TaskProfile, history: Sequence[RunResult]
    ) -> list[ExperimentSpec]:
        imbalance = (
            "class_weight"
            if profile.task_type == "classification"
            and profile.positive_rate_train is not None
            and not 0.3 <= profile.positive_rate_train <= 0.7
            else "none"
        )
        rich = ("morgan", "avalon", "erg", "rdkit2d")
        candidates: list[tuple[str, tuple[str, ...], dict, str]] = []
        selected_action = self.research_plan.get("selected_action_id")
        if not history:
            baseline_models = [
                "lightgbm",
                "xgboost",
                "extra_trees",
                "random_forest",
                "catboost",
            ]
            imbalance_severity = (
                self.hardness_card.get("dimensions", {})
                .get("imbalance", {})
                .get("severity", 0.0)
            )
            if imbalance_severity >= 0.25:
                baseline_models.insert(0, "balanced_extra_trees")
            if profile.n_train <= 5000:
                baseline_models.append("svm_rbf")
            for model in baseline_models:
                if not is_model_available(model):
                    continue
                hypothesis = f"strong multi-view baseline with {model}"
                if (
                    selected_action == "model-imbalance-aware"
                    and model == "balanced_extra_trees"
                ):
                    hypothesis = (
                        "imbalance-aware model intervention: balanced ExtraTrees"
                    )
                elif (
                    selected_action == "model-imbalance-aware"
                    and model == "extra_trees"
                ):
                    hypothesis = "matched model control: unbalanced ExtraTrees"
                candidates.append((model, rich, {}, hypothesis))
            # Every diagnosis-driven intervention needs a matched control.  These
            # candidates hold model and hyperparameters fixed and vary only the
            # feature axis, so a later gain is scientifically attributable.
            if selected_action == "feature-domain-shift" and is_model_available(
                "extra_trees"
            ):
                rich_extra_trees = (
                    "extra_trees",
                    rich,
                    {},
                    "domain-shift feature intervention: transferable multi-view features",
                )
                feature_control = (
                    "extra_trees",
                    ("morgan",),
                    {},
                    "matched feature control: Morgan fingerprint only",
                )
                candidates = [rich_extra_trees, feature_control, *candidates]
            if selected_action == "model-estimator-family" and is_model_available(
                "random_forest"
            ):
                candidates = [
                    (
                        "random_forest",
                        rich,
                        {},
                        "estimator-family model intervention: RandomForest",
                    ),
                    (
                        "extra_trees",
                        rich,
                        {},
                        "matched model control: ExtraTrees",
                    ),
                    *candidates,
                ]
            if self._allow_unimol2() and is_representation_available("unimol2"):
                unimol_treatment = (
                    "extra_trees",
                    ("unimol2", "morgan", "rdkit2d"),
                    {},
                    "3D feature intervention: add frozen Uni-Mol2 to matched 2D features",
                )
                unimol_control = (
                    "extra_trees",
                    ("morgan", "rdkit2d"),
                    {},
                    "matched feature control: 2D features for Uni-Mol2 intervention",
                )
                candidates = [unimol_treatment, unimol_control, *candidates]
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
            controlled_parameters = (
                dict(winner.hyperparameters)
                if selected_action == "feature-domain-shift"
                else None
            )
            feature_label = (
                "controlled feature test"
                if selected_action == "feature-domain-shift"
                else "secondary feature exploration"
            )
            candidates.extend(
                [
                    (
                        model,
                        ("morgan_r3", "avalon", "erg", "rdkit2d"),
                        controlled_parameters
                        if controlled_parameters is not None
                        else self._tuned(model, 0, profile.task_type),
                        f"{feature_label}: increase circular radius while retaining descriptors",
                    ),
                    (
                        model,
                        ("morgan_count", "atom_pair", "maccs", "rdkit2d"),
                        controlled_parameters
                        if controlled_parameters is not None
                        else self._tuned(model, 1, profile.task_type),
                        f"{feature_label}: count and topological features",
                    ),
                    (
                        model,
                        ("morgan", "rdkit_fp", "torsion", "rdkit2d"),
                        controlled_parameters
                        if controlled_parameters is not None
                        else self._tuned(model, 2, profile.task_type),
                        f"{feature_label}: path and torsion information under scaffold shift",
                    ),
                ]
            )
            for evidence in self.cross_task_evidence:
                if (
                    evidence.get("evidence_tasks", 0) < 2
                    or evidence.get("mean_within_task_percentile", 0.0) < 0.75
                ):
                    continue
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
            self._signature(
                item.model,
                item.representations,
                item.hyperparameters,
                item.data_policy,
            )
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
            seen.add(signature)
            if len(proposals) == self.max_candidates:
                break
        dimensions = self.hardness_card.get("dimensions", {})
        policy = None
        if (
            selected_action == "data-leakage-safe"
            and dimensions.get("split_leakage_risk", {}).get("severity", 0) > 0
        ):
            policy = "leakage_safe"
        elif (
            selected_action == "data-quality-filter"
            and dimensions.get("data_quality", {}).get("severity", 0) >= 0.03
        ):
            policy = "quality_filtered"
        if policy and proposals:
            baseline = proposals[0]
            data_candidate = ExperimentSpec.from_dict(
                {
                    **baseline.to_dict(),
                    "data_policy": policy,
                    "hypothesis": (
                        f"hold model and features fixed while testing the {policy} "
                        "training-data intervention"
                    ),
                }
            )
            proposals = [data_candidate, *proposals][: self.max_candidates]
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
                {
                    "iterations": 900,
                    "depth": 6,
                    "learning_rate": 0.04,
                    "l2_leaf_reg": 5.0,
                },
                {
                    "iterations": 1200,
                    "depth": 5,
                    "learning_rate": 0.025,
                    "l2_leaf_reg": 8.0,
                },
                {
                    "iterations": 700,
                    "depth": 8,
                    "learning_rate": 0.05,
                    "random_strength": 1.5,
                },
            )[variant]
        if model == "xgboost":
            return (
                {
                    "n_estimators": 800,
                    "max_depth": 4,
                    "learning_rate": 0.035,
                    "subsample": 0.9,
                    "colsample_bytree": 0.6,
                    "min_child_weight": 2.0,
                    "reg_lambda": 3.0,
                },
                {
                    "n_estimators": 1000,
                    "max_depth": 3,
                    "learning_rate": 0.025,
                    "subsample": 0.8,
                    "colsample_bytree": 0.8,
                    "min_child_weight": 4.0,
                    "reg_alpha": 0.1,
                    "reg_lambda": 5.0,
                },
                {
                    "n_estimators": 600,
                    "max_depth": 7,
                    "learning_rate": 0.05,
                    "subsample": 0.75,
                    "colsample_bytree": 0.5,
                    "min_child_weight": 6.0,
                    "reg_alpha": 0.5,
                    "reg_lambda": 8.0,
                },
            )[variant]
        if model == "lightgbm":
            return (
                {
                    "n_estimators": 800,
                    "num_leaves": 31,
                    "max_depth": 7,
                    "learning_rate": 0.035,
                    "subsample": 0.9,
                    "colsample_bytree": 0.7,
                    "min_child_samples": 12,
                    "reg_lambda": 3.0,
                },
                {
                    "n_estimators": 1000,
                    "num_leaves": 15,
                    "max_depth": 5,
                    "learning_rate": 0.025,
                    "subsample": 0.8,
                    "colsample_bytree": 0.8,
                    "min_child_samples": 20,
                    "reg_alpha": 0.1,
                    "reg_lambda": 5.0,
                },
                {
                    "n_estimators": 600,
                    "num_leaves": 63,
                    "max_depth": 9,
                    "learning_rate": 0.05,
                    "subsample": 0.75,
                    "colsample_bytree": 0.6,
                    "min_child_samples": 25,
                    "reg_alpha": 0.5,
                    "reg_lambda": 8.0,
                },
            )[variant]
        if model == "svm_rbf":
            return (
                {
                    "C": 3.0,
                    "gamma": "scale",
                    **({"epsilon": 0.1} if task_type == "regression" else {}),
                },
                {
                    "C": 10.0,
                    "gamma": 0.001,
                    **({"epsilon": 0.05} if task_type == "regression" else {}),
                },
                {
                    "C": 30.0,
                    "gamma": 0.0001,
                    **({"epsilon": 0.2} if task_type == "regression" else {}),
                },
            )[variant]
        if model == "balanced_extra_trees":
            return (
                {
                    "n_estimators": 300,
                    "n_members": 9,
                    "majority_ratio": 1.0,
                    "min_samples_leaf": 1,
                    "max_features": 0.5,
                    "criterion": criterion,
                },
                {
                    "n_estimators": 300,
                    "n_members": 7,
                    "majority_ratio": 2.0,
                    "min_samples_leaf": 2,
                    "max_features": "sqrt",
                    "criterion": criterion,
                },
                {
                    "n_estimators": 400,
                    "n_members": 5,
                    "majority_ratio": 3.0,
                    "min_samples_leaf": 3,
                    "max_features": 0.8,
                    "criterion": criterion,
                },
            )[variant]
        return (
            {
                "n_estimators": 700,
                "min_samples_leaf": 1,
                "max_features": 0.5,
                "criterion": criterion,
            },
            {
                "n_estimators": 800,
                "min_samples_leaf": 3,
                "max_features": "sqrt",
                "criterion": criterion,
            },
            {
                "n_estimators": 600,
                "min_samples_leaf": 2,
                "max_features": 0.8,
                "criterion": criterion,
            },
        )[variant]

    def select_finalist(
        self, profile: TaskProfile, results: Sequence[RunResult]
    ) -> str:
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

    def set_context(
        self,
        *,
        cross_task_evidence: list[dict],
        hardness_card: dict | None = None,
        research_plan: dict | None = None,
    ) -> None:
        self.rules.set_context(
            cross_task_evidence=cross_task_evidence,
            hardness_card=hardness_card,
            research_plan=research_plan,
        )
        setter = getattr(self.planner, "set_context", None)
        if callable(setter):
            setter(
                cross_task_evidence=cross_task_evidence,
                hardness_card=hardness_card,
                research_plan=research_plan,
            )

    def drain_errors(self) -> list[str]:
        errors, self.errors = self.errors, []
        return errors

    @staticmethod
    def _signature(spec: ExperimentSpec) -> tuple:
        return (
            spec.model,
            tuple(spec.representations),
            tuple(sorted(spec.hyperparameters.items())),
            spec.data_policy,
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

    def propose(
        self, profile: TaskProfile, history: Sequence[RunResult]
    ) -> list[ExperimentSpec]:
        raise RuntimeError(
            "OpenAI planner is not configured yet. Set up the dedicated API credential, "
            "then implement this adapter with the Responses API and strict JSON schema output."
        )

    def select_finalist(
        self, profile: TaskProfile, results: Sequence[RunResult]
    ) -> str:
        raise RuntimeError("OpenAI planner is not configured yet")
