"""Executable control policies for same-budget autonomous-research comparisons."""

from __future__ import annotations

import hashlib
import json
import math
import random
from collections.abc import Sequence
from copy import deepcopy
from dataclasses import asdict, dataclass
from typing import Any, ClassVar

from .brain import RuleBasedBrain
from .committee import (
    CommitteeBrain,
    DataStewardProposer,
    ModelResearcherProposer,
    ProposalAssessment,
    SkepticProposer,
    proposal_fingerprint,
)
from .registry import is_model_available, is_representation_available
from .schemas import Budget, ExperimentSpec, RunResult, TaskProfile

RICH = ("morgan", "avalon", "erg", "rdkit2d")
FEATURE_VIEWS = (
    RICH,
    ("morgan",),
    ("morgan_count", "atom_pair", "maccs", "rdkit2d"),
    ("morgan", "rdkit_fp", "torsion", "rdkit2d"),
)
EXPANDED_FEATURE_VIEWS = (
    *FEATURE_VIEWS,
    ("morgan", "rdkit2d"),
    ("unimol2", "morgan", "rdkit2d"),
)
EXPANDED_DATA_POLICIES = (
    "official",
    "canonical_deduplicate",
    "quality_filtered",
    "leakage_safe",
)


def _imbalance(profile: TaskProfile) -> str:
    return (
        "class_weight"
        if profile.task_type == "classification"
        and profile.positive_rate_train is not None
        and not 0.3 <= profile.positive_rate_train <= 0.7
        else "none"
    )


def _models(profile: TaskProfile) -> list[str]:
    names = ["lightgbm", "xgboost", "extra_trees", "random_forest", "catboost"]
    if profile.n_train <= 5000:
        names.append("svm_rbf")
    return [name for name in names if is_model_available(name)]


def _signature(spec: ExperimentSpec) -> tuple:
    return (
        spec.model,
        spec.representations,
        tuple(sorted(spec.hyperparameters.items())),
        spec.data_policy,
    )


class FixedRulesControlBrain(RuleBasedBrain):
    """Deterministic search order without Hardness or cross-task memory."""

    benchmark_arm = "rules"

    def set_context(
        self,
        *,
        hardness_card: dict | None = None,
        **_: object,
    ) -> None:
        super().set_context(
            cross_task_evidence=[],
            hardness_card=hardness_card,
            research_plan=None,
        )


class AgentNoMemoryBrain(RuleBasedBrain):
    """Agent ablation that receives no cross-task/action-memory evidence."""

    benchmark_arm = "agent_no_memory"

    def set_context(
        self,
        *,
        cross_task_evidence: list[dict],
        hardness_card: dict | None = None,
        research_plan: dict | None = None,
    ) -> None:
        del cross_task_evidence
        self.effective_research_plan = deepcopy(research_plan or {})
        super().set_context(
            cross_task_evidence=[],
            hardness_card=hardness_card,
            research_plan=self.effective_research_plan,
        )


class AgentNoHardnessBrain(RuleBasedBrain):
    """Agent ablation with memory but no diagnosis-conditioned routing."""

    benchmark_arm = "agent_no_hardness"

    def set_context(
        self,
        *,
        cross_task_evidence: list[dict],
        hardness_card: dict | None = None,
        research_plan: dict | None = None,
    ) -> None:
        del hardness_card, research_plan
        self.effective_research_plan = {
            "selected_action_id": "ablation-no-hardness",
            "actions": [
                {
                    "action_id": "ablation-no-hardness",
                    "axis": "replicate",
                    "falsification_test": (
                        "same-budget model search without endpoint diagnosis"
                    ),
                }
            ],
            "ablation": "no_hardness",
        }
        super().set_context(
            cross_task_evidence=cross_task_evidence,
            hardness_card={},
            research_plan={},
        )


class AgentNoDataBrain(RuleBasedBrain):
    """Agent ablation whose effective plan cannot select a data action."""

    benchmark_arm = "agent_no_data"

    def set_context(
        self,
        *,
        cross_task_evidence: list[dict],
        hardness_card: dict | None = None,
        research_plan: dict | None = None,
    ) -> None:
        plan = deepcopy(research_plan or {})
        actions = [
            action for action in plan.get("actions", []) if action.get("axis") != "data"
        ]
        selected = next(
            (
                action.get("action_id")
                for action in actions
                if action.get("axis") != "stop" and action.get("action_id")
            ),
            actions[0].get("action_id") if actions else "ablation-no-data",
        )
        if not actions:
            actions = [
                {
                    "action_id": selected,
                    "axis": "replicate",
                    "falsification_test": "same-budget search with data actions removed",
                }
            ]
        plan["actions"] = actions
        plan["selected_action_id"] = selected
        plan["ablation"] = "no_data_actions"
        self.effective_research_plan = plan
        super().set_context(
            cross_task_evidence=cross_task_evidence,
            hardness_card=hardness_card,
            research_plan=plan,
        )


class AutoMLControlBrain:
    """Model/hyperparameter search with data and representation held fixed."""

    benchmark_arm = "automl"

    def __init__(self, max_candidates: int = 6):
        self.max_candidates = max_candidates

    def set_context(self, **_: object) -> None:
        return None

    def propose(
        self,
        profile: TaskProfile,
        history: Sequence[RunResult],
    ) -> list[ExperimentSpec]:
        seen = {
            (
                item.model,
                tuple(item.representations),
                tuple(sorted(item.hyperparameters.items())),
                item.data_policy,
            )
            for item in history
        }
        pool = []
        for model in _models(profile):
            pool.append((model, {}))
        for variant in range(3):
            for model in _models(profile):
                pool.append(
                    (model, RuleBasedBrain._tuned(model, variant, profile.task_type))
                )
        proposals = []
        for model, hyperparameters in pool:
            spec = ExperimentSpec(
                task=profile.task,
                seed=1,
                representations=RICH,
                model=model,
                imbalance=_imbalance(profile),
                hyperparameters=hyperparameters,
                budget=Budget(max_candidates=self.max_candidates),
                hypothesis="AutoML control: model/hyperparameter search with fixed features and data",
            )
            if _signature(spec) in seen:
                continue
            proposals.append(spec)
            if len(proposals) == self.max_candidates:
                break
        return proposals

    def select_finalist(
        self, profile: TaskProfile, results: Sequence[RunResult]
    ) -> str:
        direction = -1.0 if profile.metric == "mae" else 1.0
        return max(results, key=lambda item: direction * item.validation_score).run_id


@dataclass(frozen=True)
class ExpandedAutoMLDecision:
    """Serializable audit for scalar, non-scientific AutoML acquisition."""

    policy: str
    policy_seed: int
    round_index: int
    max_candidates: int
    candidate_pool_size: int
    eligible_candidates: int
    duplicate_candidates_rejected: int
    pool_sha256: str
    selected: tuple[dict[str, Any], ...]
    uses_hardness: bool = False
    uses_research_plan: bool = False
    uses_action_memory: bool = False
    uses_matched_pair_priority: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class ExpandedSpaceAutoMLControlBrain(AutoMLControlBrain):
    """AutoML over the Agent's primitive space without scientific routing.

    This control receives the same model, representation, data-policy and
    imbalance primitives as the research agent.  It has no HardnessCard,
    selected research action, cross-task memory or causal-pair priority.  A
    fixed-seed greedy acquisition balances validation-history similarity,
    primitive coverage and estimated cost under the same candidate count.
    """

    benchmark_arm = "expanded_automl"
    policy_name = "expanded_space_scalar_validation_acquisition"

    _MODEL_COST: ClassVar[dict[str, float]] = {
        "lightgbm": 0.16,
        "xgboost": 0.24,
        "extra_trees": 0.48,
        "random_forest": 0.58,
        "catboost": 0.42,
        "svm_rbf": 0.34,
        "balanced_extra_trees": 0.72,
    }

    def __init__(
        self,
        max_candidates: int = 6,
        policy_seed: int = 20260819,
    ) -> None:
        super().__init__(max_candidates=max_candidates)
        self.policy_seed = policy_seed
        self.last_decision: ExpandedAutoMLDecision | None = None
        self._task: str | None = None
        self._round_index = 0
        self._issued: dict[str, set[str]] = {}

    def set_context(self, **_: object) -> None:
        """Intentionally ignore all scientific routing and memory context."""

    @staticmethod
    def _available_views() -> tuple[tuple[str, ...], ...]:
        return tuple(
            view
            for view in EXPANDED_FEATURE_VIEWS
            if all(is_representation_available(name) for name in view)
        )

    @staticmethod
    def _available_models(profile: TaskProfile) -> tuple[str, ...]:
        names = list(_models(profile))
        if profile.task_type == "classification" and is_model_available(
            "balanced_extra_trees"
        ):
            names.append("balanced_extra_trees")
        return tuple(dict.fromkeys(names))

    @staticmethod
    def _parameter_pool(model: str, task_type: str) -> tuple[dict[str, Any], ...]:
        values = [{}]
        values.extend(
            RuleBasedBrain._tuned(model, variant, task_type) for variant in range(3)
        )
        unique: dict[str, dict[str, Any]] = {}
        for parameters in values:
            key = json.dumps(parameters, sort_keys=True, separators=(",", ":"))
            unique[key] = parameters
        return tuple(unique[key] for key in sorted(unique))

    def candidate_pool(self, profile: TaskProfile) -> list[ExperimentSpec]:
        """Build the full primitive union without consulting scientific context."""

        imbalances = (
            ("none", "class_weight")
            if profile.task_type == "classification"
            else ("none",)
        )
        pool = []
        for model in self._available_models(profile):
            for representations in self._available_views():
                for data_policy in EXPANDED_DATA_POLICIES:
                    for imbalance in imbalances:
                        for hyperparameters in self._parameter_pool(
                            model, profile.task_type
                        ):
                            pool.append(
                                ExperimentSpec(
                                    task=profile.task,
                                    seed=1,
                                    representations=representations,
                                    model=model,
                                    imbalance=imbalance,
                                    data_policy=data_policy,
                                    hyperparameters=hyperparameters,
                                    budget=Budget(max_candidates=self.max_candidates),
                                    hypothesis=(
                                        "expanded-space AutoML scalar acquisition; "
                                        "no scientific hypothesis or causal attribution"
                                    ),
                                )
                            )
        return sorted(pool, key=proposal_fingerprint)

    @staticmethod
    def _history_key(item: RunResult) -> tuple:
        return (
            item.task,
            item.model,
            tuple(item.representations),
            json.dumps(item.hyperparameters, sort_keys=True, default=str),
            item.data_policy,
            item.run_id,
        )

    @staticmethod
    def _result_signature(item: RunResult) -> tuple:
        # RunResult does not retain imbalance, so resume-time de-duplication is
        # deliberately conservative across otherwise identical configurations.
        return (
            item.model,
            tuple(item.representations),
            tuple(sorted(item.hyperparameters.items())),
            item.data_policy,
        )

    @staticmethod
    def _tuning_family(hyperparameters: dict[str, Any]) -> str:
        return "default" if not hyperparameters else "tuned"

    @classmethod
    def _attributes(cls, spec: ExperimentSpec) -> tuple[Any, ...]:
        return (
            spec.model,
            tuple(spec.representations),
            spec.data_policy,
            spec.imbalance,
            cls._tuning_family(spec.hyperparameters),
        )

    @classmethod
    def _normalized_cost(cls, spec: ExperimentSpec) -> float:
        model_cost = cls._MODEL_COST.get(spec.model, 0.50)
        representations = set(spec.representations)
        if "unimol2" in representations:
            feature_cost = 1.0
        elif representations & {"atom_pair", "torsion", "morgan_count"}:
            feature_cost = 0.55
        elif len(representations) >= 4:
            feature_cost = 0.38
        elif len(representations) == 1:
            feature_cost = 0.08
        else:
            feature_cost = 0.20
        parameter = spec.hyperparameters.get(
            "n_estimators", spec.hyperparameters.get("iterations", 0)
        )
        scale_cost = min(0.20, max(0.0, float(parameter) / 6000.0))
        data_cost = 0.04 if spec.data_policy != "official" else 0.0
        return min(
            1.0,
            0.55 * model_cost + 0.35 * feature_cost + scale_cost + data_cost,
        )

    def _seed_fraction(self, fingerprint: str) -> float:
        digest = hashlib.sha256(f"{self.policy_seed}:{fingerprint}".encode()).digest()
        return int.from_bytes(digest[:8], "big") / float(2**64 - 1)

    @classmethod
    def _similarity(cls, spec: ExperimentSpec, result: RunResult) -> float:
        return (
            0.35 * float(spec.model == result.model)
            + 0.30 * float(tuple(spec.representations) == tuple(result.representations))
            + 0.15 * float(spec.data_policy == result.data_policy)
            + 0.20
            * float(
                cls._tuning_family(spec.hyperparameters)
                == cls._tuning_family(result.hyperparameters)
            )
        )

    @classmethod
    def _history_quality(
        cls,
        profile: TaskProfile,
        history: Sequence[RunResult],
    ) -> dict[str, float]:
        completed = [
            item
            for item in history
            if item.status == "completed" and math.isfinite(item.validation_score)
        ]
        if not completed:
            return {}
        direction = -1.0 if profile.metric == "mae" else 1.0
        oriented = {
            item.run_id: direction * float(item.validation_score) for item in completed
        }
        low, high = min(oriented.values()), max(oriented.values())
        if math.isclose(low, high):
            return {run_id: 0.5 for run_id in oriented}
        return {
            run_id: (value - low) / (high - low) for run_id, value in oriented.items()
        }

    @classmethod
    def _novelty(
        cls,
        spec: ExperimentSpec,
        selected: Sequence[ExperimentSpec],
        history: Sequence[RunResult],
    ) -> float:
        attributes = cls._attributes(spec)
        selected_attributes = [cls._attributes(item) for item in selected]
        round_novelty = sum(
            all(value != prior[index] for prior in selected_attributes)
            for index, value in enumerate(attributes)
        ) / len(attributes)
        if not history:
            return float(round_novelty)
        history_values = (
            {item.model for item in history},
            {tuple(item.representations) for item in history},
            {item.data_policy for item in history},
            set(),  # imbalance is intentionally absent from RunResult
            {cls._tuning_family(item.hyperparameters) for item in history},
        )
        history_novelty = sum(
            1.0 if not values or value not in values else 0.0
            for value, values in zip(attributes, history_values, strict=True)
        ) / len(attributes)
        return float(0.5 * round_novelty + 0.5 * history_novelty)

    def _acquisition(
        self,
        spec: ExperimentSpec,
        profile: TaskProfile,
        history: Sequence[RunResult],
        quality: dict[str, float],
        selected: Sequence[ExperimentSpec],
    ) -> dict[str, float]:
        completed = [
            item
            for item in history
            if item.status == "completed" and item.run_id in quality
        ]
        exploitation = (
            max(
                quality[item.run_id] * (0.20 + 0.80 * self._similarity(spec, item))
                for item in completed
            )
            if completed
            else 0.50
        )
        novelty = self._novelty(spec, selected, history)
        cost = self._normalized_cost(spec)
        jitter = 1e-6 * self._seed_fraction(proposal_fingerprint(spec))
        value = 0.55 * exploitation + 0.30 * novelty - 0.15 * cost + jitter
        return {
            "value": float(value),
            "exploitation": float(exploitation),
            "novelty": float(novelty),
            "normalized_cost": float(cost),
        }

    def propose(
        self,
        profile: TaskProfile,
        history: Sequence[RunResult],
    ) -> list[ExperimentSpec]:
        if profile.task != self._task or (not history and self._round_index > 0):
            self._task = profile.task
            self._round_index = 0
            self._issued[profile.task] = set()
        self._round_index += 1
        issued = self._issued.setdefault(profile.task, set())
        ordered_history = sorted(history, key=self._history_key)
        history_signatures = {self._result_signature(item) for item in ordered_history}
        pool = self.candidate_pool(profile)
        pool_fingerprints = [proposal_fingerprint(item) for item in pool]
        eligible = [
            item
            for item in pool
            if proposal_fingerprint(item) not in issued
            and _signature(item) not in history_signatures
        ]
        quality = self._history_quality(profile, ordered_history)
        selected: list[ExperimentSpec] = []
        selected_audit: list[dict[str, Any]] = []
        remaining = list(eligible)
        while remaining and len(selected) < self.max_candidates:
            ranked = []
            for candidate in remaining:
                acquisition = self._acquisition(
                    candidate,
                    profile,
                    ordered_history,
                    quality,
                    selected,
                )
                ranked.append(
                    (
                        -acquisition["value"],
                        proposal_fingerprint(candidate),
                        candidate,
                        acquisition,
                    )
                )
            _, fingerprint, winner, acquisition = min(
                ranked, key=lambda item: (item[0], item[1])
            )
            selected.append(winner)
            selected_audit.append(
                {
                    "fingerprint": fingerprint,
                    "acquisition": acquisition,
                    "model": winner.model,
                    "representations": list(winner.representations),
                    "data_policy": winner.data_policy,
                    "imbalance": winner.imbalance,
                    "tuning_family": self._tuning_family(winner.hyperparameters),
                }
            )
            issued.add(fingerprint)
            remaining = [
                item for item in remaining if proposal_fingerprint(item) != fingerprint
            ]

        self.last_decision = ExpandedAutoMLDecision(
            policy=self.policy_name,
            policy_seed=self.policy_seed,
            round_index=self._round_index,
            max_candidates=self.max_candidates,
            candidate_pool_size=len(pool),
            eligible_candidates=len(eligible),
            duplicate_candidates_rejected=len(pool) - len(eligible),
            pool_sha256=hashlib.sha256(
                "\n".join(pool_fingerprints).encode("utf-8")
            ).hexdigest(),
            selected=tuple(selected_audit),
        )
        return selected


class SeededRandomControlBrain(AutoMLControlBrain):
    """Seeded random actions over the same model/feature search primitives."""

    benchmark_arm = "random"

    def __init__(self, max_candidates: int = 6, policy_seed: int = 20260819):
        super().__init__(max_candidates=max_candidates)
        self.policy_seed = policy_seed

    def propose(
        self,
        profile: TaskProfile,
        history: Sequence[RunResult],
    ) -> list[ExperimentSpec]:
        pool = []
        models = _models(profile)
        for model_index, model in enumerate(models):
            for feature_index, representations in enumerate(FEATURE_VIEWS):
                variant = (model_index + feature_index) % 3
                pool.append(
                    ExperimentSpec(
                        task=profile.task,
                        seed=1,
                        representations=representations,
                        model=model,
                        imbalance=_imbalance(profile),
                        hyperparameters=RuleBasedBrain._tuned(
                            model, variant, profile.task_type
                        ),
                        budget=Budget(max_candidates=self.max_candidates),
                        hypothesis="same-budget seeded random research action control",
                    )
                )
        digest = hashlib.sha256(f"{profile.task}:{self.policy_seed}".encode()).digest()
        random.Random(int.from_bytes(digest[:8], "big")).shuffle(pool)
        seen = {
            (
                item.model,
                tuple(item.representations),
                tuple(sorted(item.hyperparameters.items())),
                item.data_policy,
            )
            for item in history
        }
        return [item for item in pool if _signature(item) not in seen][
            : self.max_candidates
        ]


class StrongFixedControlBrain(AutoMLControlBrain):
    """One preregistered strong recipe, with no adaptive search."""

    benchmark_arm = "strong_fixed"

    def propose(
        self,
        profile: TaskProfile,
        history: Sequence[RunResult],
    ) -> list[ExperimentSpec]:
        if history:
            return []
        available = _models(profile)
        if not available:
            return []
        model = "lightgbm" if "lightgbm" in available else available[0]
        return [
            ExperimentSpec(
                task=profile.task,
                seed=1,
                representations=RICH,
                model=model,
                imbalance=_imbalance(profile),
                budget=Budget(max_candidates=1),
                hypothesis="preregistered strong fixed multi-view endpoint baseline",
            )
        ]


class _DataStewardActionBrain:
    """Emit only a discovery-justified, one-axis data treatment/control pair."""

    def __init__(self, max_candidates: int):
        self.max_candidates = max_candidates
        self.hardness_card: dict = {}
        self.research_plan: dict = {}

    def set_context(
        self,
        *,
        hardness_card: dict | None = None,
        research_plan: dict | None = None,
        **_: object,
    ) -> None:
        self.hardness_card = hardness_card or {}
        self.research_plan = research_plan or {}

    def propose(
        self, profile: TaskProfile, history: Sequence[RunResult]
    ) -> list[ExperimentSpec]:
        if history or self.max_candidates < 2:
            return []
        dimensions = self.hardness_card.get("dimensions", {})
        selected_action = self.research_plan.get("selected_action_id")
        if (
            selected_action == "data-leakage-safe"
            and dimensions.get("split_leakage_risk", {}).get("severity", 0.0) > 0
        ):
            policy = "leakage_safe"
        elif (
            selected_action == "data-quality-filter"
            and dimensions.get("data_quality", {}).get("severity", 0.0) >= 0.03
        ):
            policy = "quality_filtered"
        else:
            return []
        available = _models(profile)
        if not available:
            return []
        model = "lightgbm" if "lightgbm" in available else available[0]
        control = ExperimentSpec(
            task=profile.task,
            seed=1,
            representations=RICH,
            model=model,
            imbalance=_imbalance(profile),
            data_policy="official",
            budget=Budget(max_candidates=self.max_candidates),
            hypothesis=f"matched official-data control for {policy}",
        )
        treatment = ExperimentSpec.from_dict(
            {
                **control.to_dict(),
                "data_policy": policy,
                "hypothesis": f"data intervention: {policy}",
            }
        )
        return [treatment, control]

    def select_finalist(
        self, profile: TaskProfile, results: Sequence[RunResult]
    ) -> str:
        direction = -1.0 if profile.metric == "mae" else 1.0
        return max(results, key=lambda item: direction * item.validation_score).run_id


class _SkepticControlBrain:
    """Propose a compact low-cost baseline set for falsification."""

    def __init__(self, max_candidates: int):
        self.max_candidates = max_candidates

    def set_context(self, **_: object) -> None:
        return None

    def propose(
        self, profile: TaskProfile, history: Sequence[RunResult]
    ) -> list[ExperimentSpec]:
        seen = {_signature(item) for item in history}
        available = _models(profile)
        candidates = []
        for model, representations in (
            ("lightgbm", RICH),
            ("extra_trees", ("morgan",)),
        ):
            if model not in available:
                continue
            candidate = ExperimentSpec(
                task=profile.task,
                seed=1,
                representations=representations,
                model=model,
                imbalance=_imbalance(profile),
                budget=Budget(max_candidates=self.max_candidates),
                hypothesis="skeptical low-cost falsification baseline",
            )
            if _signature(candidate) not in seen:
                candidates.append(candidate)
        return candidates[: self.max_candidates]

    def select_finalist(
        self, profile: TaskProfile, results: Sequence[RunResult]
    ) -> str:
        direction = -1.0 if profile.metric == "mae" else 1.0
        return max(results, key=lambda item: direction * item.validation_score).run_id


def _data_assessment(
    spec: ExperimentSpec, _profile: TaskProfile, _history: Sequence[RunResult]
) -> ProposalAssessment:
    identity = {
        "task": spec.task,
        "model": spec.model,
        "representations": spec.representations,
        "imbalance": spec.imbalance,
        "hyperparameters": spec.hyperparameters,
    }
    group = hashlib.sha256(
        json.dumps(identity, sort_keys=True, default=str).encode()
    ).hexdigest()[:16]
    return ProposalAssessment(
        expected_gain=0.55,
        confidence=0.80,
        compute_cost=0.10,
        scientific_risk=0.08,
        matched_group=f"data:{group}",
        pair_role="control" if spec.data_policy == "official" else "treatment",
        matched_axis="data",
        rationale=("one-axis data treatment/control",),
    )


def _model_assessment(
    spec: ExperimentSpec, _profile: TaskProfile, _history: Sequence[RunResult]
) -> ProposalAssessment:
    hypothesis = spec.hypothesis.lower()
    pair = {}
    if "3d feature intervention" in hypothesis:
        pair = {
            "matched_group": "feature:unimol2",
            "pair_role": "treatment",
            "matched_axis": "feature",
        }
    elif "2d features for uni-mol2" in hypothesis:
        pair = {
            "matched_group": "feature:unimol2",
            "pair_role": "control",
            "matched_axis": "feature",
        }
    elif "domain-shift feature intervention" in hypothesis:
        pair = {
            "matched_group": "feature:domain-shift",
            "pair_role": "treatment",
            "matched_axis": "feature",
        }
    elif "morgan fingerprint only" in hypothesis:
        pair = {
            "matched_group": "feature:domain-shift",
            "pair_role": "control",
            "matched_axis": "feature",
        }
    elif "imbalance-aware model intervention" in hypothesis:
        pair = {
            "matched_group": "model:imbalance-aware",
            "pair_role": "treatment",
            "matched_axis": "model",
        }
    elif "matched model control: unbalanced extratrees" in hypothesis:
        pair = {
            "matched_group": "model:imbalance-aware",
            "pair_role": "control",
            "matched_axis": "model",
        }
    compute_cost = {
        "lightgbm": 0.18,
        "xgboost": 0.25,
        "catboost": 0.35,
        "svm_rbf": 0.30,
        "extra_trees": 0.60,
        "random_forest": 0.65,
        "balanced_extra_trees": 0.70,
    }.get(spec.model, 0.50)
    if "unimol2" in spec.representations:
        compute_cost = 1.0
    return ProposalAssessment(
        expected_gain=0.55 if "intervention" in hypothesis else 0.42,
        confidence=0.65,
        compute_cost=compute_cost,
        scientific_risk=0.12,
        rationale=("model/representation evidence proposal",),
        **pair,
    )


def _skeptic_assessment(
    _spec: ExperimentSpec, _profile: TaskProfile, _history: Sequence[RunResult]
) -> ProposalAssessment:
    return ProposalAssessment(
        expected_gain=0.20,
        confidence=0.85,
        compute_cost=0.10,
        scientific_risk=0.05,
        rationale=("low-cost falsification baseline",),
    )


def committee_brain(max_candidates: int = 4) -> CommitteeBrain:
    """Build the deterministic, budget-preserving discovery committee."""
    committee = CommitteeBrain(
        (
            DataStewardProposer(
                _DataStewardActionBrain(max_candidates),
                proposer_id="data-steward",
                assessor=_data_assessment,
            ),
            ModelResearcherProposer(
                RuleBasedBrain(max_candidates=max_candidates),
                proposer_id="model-researcher",
                assessor=_model_assessment,
            ),
            SkepticProposer(
                _SkepticControlBrain(max_candidates),
                proposer_id="skeptic",
                assessor=_skeptic_assessment,
            ),
        ),
        max_candidates=max_candidates,
    )
    committee.benchmark_arm = "committee"
    return committee


def benchmark_brain(
    arm: str,
    *,
    max_candidates: int = 6,
    policy_seed: int = 20260819,
):
    if arm == "agent":
        brain = RuleBasedBrain(max_candidates=max_candidates)
        brain.benchmark_arm = "agent"
        return brain
    if arm == "rules":
        return FixedRulesControlBrain(max_candidates=max_candidates)
    if arm == "random":
        return SeededRandomControlBrain(
            max_candidates=max_candidates, policy_seed=policy_seed
        )
    if arm == "automl":
        return AutoMLControlBrain(max_candidates=max_candidates)
    if arm == "expanded_automl":
        return ExpandedSpaceAutoMLControlBrain(
            max_candidates=max_candidates,
            policy_seed=policy_seed,
        )
    if arm == "agent_no_memory":
        return AgentNoMemoryBrain(max_candidates=max_candidates)
    if arm == "agent_no_hardness":
        return AgentNoHardnessBrain(max_candidates=max_candidates)
    if arm == "agent_no_data":
        return AgentNoDataBrain(max_candidates=max_candidates)
    if arm == "strong_fixed":
        return StrongFixedControlBrain(max_candidates=1)
    if arm == "committee":
        return committee_brain(max_candidates=max_candidates)
    raise ValueError(f"Unknown benchmark arm: {arm}")
