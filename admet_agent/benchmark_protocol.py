"""Preregistered, same-budget comparisons for TDC-22 research policies."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import binomtest

from .tasks import TASKS, resolve_task


@dataclass(frozen=True)
class BenchmarkArm:
    """A preregistered policy arm and its discovery budget.

    ``max_actions_per_task`` limits distinct primary-seed configurations that
    are actually executed.  A configuration's additional validation-seed
    runs are replicate fits, not additional actions.  The legacy
    ``max_candidates_per_task`` field limits primary-seed candidate fits; it
    deliberately does not include replicate fits or derived ensemble
    evaluations.
    """

    name: str
    policy: str
    comparison_class: str
    max_actions_per_task: int
    max_candidates_per_task: int
    max_compute_minutes_per_task: int
    uses_hardness: bool
    uses_cross_task_memory: bool
    allowed_axes: tuple[str, ...]
    notes: str = ""


@dataclass(frozen=True)
class TDC22BenchmarkProtocol:
    tasks: tuple[str, ...]
    discovery_seeds: tuple[int, ...]
    arms: tuple[BenchmarkArm, ...]
    primary_selection: str = "effect_first_score_after_multiseed"
    minimum_action_effect: float = 0.005
    required_validation_seeds: int = 5
    stability_review_threshold: float = 0.05
    robust_promotion_policy: str = "matched_pair_plus_pareto_score_plus_model_diversity"
    candidate_timeout_policy: str = "hard_wall_time_per_candidate"
    robust_futility_policy: str = "seed2_gap0.005_time3x_worse_both_protect_action_pair"
    robust_futility_applies_to_all_arms: bool = True
    test_policy: str = "one_shot_after_global_freeze"
    test_feedback_to_discovery: bool = False
    protocol_version: str = "tdc22-policy-benchmark-v5"

    def validate(self) -> None:
        expected = {item.name for item in TASKS.values()}
        if set(self.tasks) != expected or len(self.tasks) != 22:
            raise ValueError(
                "Benchmark protocol must contain exactly the canonical 22 tasks"
            )
        if len(set(self.discovery_seeds)) < 3 or any(
            seed < 0 for seed in self.discovery_seeds
        ):
            raise ValueError(
                "At least three unique non-negative discovery seeds are required"
            )
        if self.test_feedback_to_discovery:
            raise ValueError("Test feedback to discovery is forbidden")
        if self.required_validation_seeds < 5:
            raise ValueError("At least five validation seeds are required")
        if not 0 < self.minimum_action_effect < 1:
            raise ValueError("minimum_action_effect must be within (0, 1)")
        if not 0 < self.stability_review_threshold < 1:
            raise ValueError("stability_review_threshold must be within (0, 1)")
        if self.robust_promotion_policy != (
            "matched_pair_plus_pareto_score_plus_model_diversity"
        ):
            raise ValueError(
                "Benchmark protocol requires the audited v3 promotion policy"
            )
        if self.candidate_timeout_policy != "hard_wall_time_per_candidate":
            raise ValueError("Benchmark protocol requires hard per-candidate timeouts")
        if self.robust_futility_policy != (
            "seed2_gap0.005_time3x_worse_both_protect_action_pair"
        ):
            raise ValueError(
                "Benchmark protocol requires the preregistered seed-2 race"
            )
        if self.robust_futility_applies_to_all_arms is not True:
            raise ValueError(
                "The same futility policy must apply to every benchmark arm"
            )
        names = [arm.name for arm in self.arms]
        if len(names) != len(set(names)):
            raise ValueError("Benchmark arm names must be unique")
        for arm in self.arms:
            if arm.max_actions_per_task < 1:
                raise ValueError(f"{arm.name} must allow at least one action")
            if arm.max_candidates_per_task < arm.max_actions_per_task:
                raise ValueError(
                    f"{arm.name} candidate budget cannot be smaller than its "
                    "unique-configuration action budget"
                )
            if arm.max_compute_minutes_per_task < 1:
                raise ValueError(f"{arm.name} must have a positive compute budget")
        sequential = [
            arm for arm in self.arms if arm.comparison_class == "same_budget_sequential"
        ]
        if len(sequential) < 3:
            raise ValueError("Agent, random and rule sequential controls are required")
        budgets = {
            (
                arm.max_actions_per_task,
                arm.max_candidates_per_task,
                arm.max_compute_minutes_per_task,
            )
            for arm in sequential
        }
        if len(budgets) != 1:
            raise ValueError("Sequential policy controls must use identical budgets")
        required = {
            "agent",
            "committee",
            "random",
            "rules",
            "automl",
            "expanded_automl",
            "strong_fixed",
        }
        if not required.issubset(names):
            raise ValueError(
                f"Missing required controls: {sorted(required - set(names))}"
            )

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        digest = hashlib.sha256(
            json.dumps(payload, sort_keys=True).encode("utf-8")
        ).hexdigest()
        return {**payload, "sha256": digest}


def default_protocol() -> TDC22BenchmarkProtocol:
    full_axes = ("feature", "model", "data", "replicate", "stop")
    protocol = TDC22BenchmarkProtocol(
        tasks=tuple(item.name for item in TASKS.values()),
        discovery_seeds=(1, 2, 3, 4, 5),
        arms=(
            BenchmarkArm(
                "agent",
                "hardness_effect_first",
                "same_budget_sequential",
                8,
                24,
                90,
                True,
                True,
                full_axes,
            ),
            BenchmarkArm(
                "committee",
                "deterministic_three_role_arbitration",
                "same_budget_sequential",
                8,
                24,
                90,
                True,
                True,
                full_axes,
                (
                    "Three logical proposers share one deterministic arbiter and "
                    "one execution queue; they do not multiply the action budget."
                ),
            ),
            BenchmarkArm(
                "random",
                "seeded_random_action",
                "same_budget_sequential",
                8,
                24,
                90,
                False,
                False,
                full_axes,
            ),
            BenchmarkArm(
                "rules",
                "fixed_action_precedence",
                "same_budget_sequential",
                8,
                24,
                90,
                True,
                False,
                full_axes,
            ),
            BenchmarkArm(
                "automl",
                "model_and_hyperparameter_search",
                "same_compute_reference",
                8,
                24,
                90,
                False,
                False,
                ("model", "replicate"),
            ),
            BenchmarkArm(
                "expanded_automl",
                "expanded_space_scalar_validation_acquisition",
                "same_budget_sequential",
                8,
                24,
                90,
                False,
                False,
                ("feature", "model", "data", "replicate"),
                (
                    "Same primitive search space as the Agent, but no hardness, "
                    "research plan, matched-pair priority, action memory or "
                    "scientific stopping decision."
                ),
            ),
            BenchmarkArm(
                "strong_fixed",
                "preregistered_endpoint_recipe",
                "fixed_reference",
                1,
                1,
                90,
                False,
                False,
                ("replicate",),
            ),
            BenchmarkArm(
                "agent_no_memory",
                "hardness_effect_first",
                "ablation",
                8,
                24,
                90,
                True,
                False,
                full_axes,
            ),
            BenchmarkArm(
                "agent_no_hardness",
                "effect_first_without_diagnosis",
                "ablation",
                8,
                24,
                90,
                False,
                True,
                full_axes,
            ),
            BenchmarkArm(
                "agent_no_data",
                "hardness_effect_first",
                "ablation",
                8,
                24,
                90,
                True,
                True,
                ("feature", "model", "replicate", "stop"),
            ),
        ),
    )
    protocol.validate()
    return protocol


@dataclass(frozen=True)
class PolicyTaskResult:
    task: str
    arm: str
    validation_score: float
    validation_std: float
    compute_seconds: float
    actions_attempted: int
    protocol_sha256: str
    # Explicit v4 accounting.  The two fields above remain the serialized
    # aliases used by older reports: actions_attempted == unique_configurations
    # and compute_seconds == fit_seconds.
    unique_configurations: int | None = None
    replicate_fits: int | None = None
    total_fits: int | None = None
    completed_fits: int | None = None
    failed_fits: int | None = None
    fit_seconds: float | None = None
    completed_fit_seconds: float | None = None
    failed_fit_seconds: float | None = None
    ensemble_evaluations: int = 0

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> PolicyTaskResult:
        return cls(**value)

    @property
    def accounted_unique_configurations(self) -> int:
        return int(
            self.actions_attempted
            if self.unique_configurations is None
            else self.unique_configurations
        )

    @property
    def accounted_fit_seconds(self) -> float:
        return float(
            self.compute_seconds if self.fit_seconds is None else self.fit_seconds
        )


def validate_search_budget_request(
    arm: dict[str, Any],
    *,
    rounds: int,
    candidates_per_round: int,
    one_shot: bool = False,
) -> dict[str, int]:
    """Reject a benchmark search that can exceed its action budget pre-fit.

    This is intentionally an upper-bound check.  Runtime de-duplication or
    early stopping cannot be used to preregister an over-budget command.
    Replicate fits are accounted separately by the result collector.
    """

    if rounds < 1:
        raise ValueError("rounds must be positive")
    if candidates_per_round < 1:
        raise ValueError("candidates_per_round must be positive")
    requested = candidates_per_round if one_shot else rounds * candidates_per_round
    action_limit = int(arm["max_actions_per_task"])
    candidate_limit = int(arm["max_candidates_per_task"])
    if requested > action_limit:
        raise ValueError(
            "Requested search can execute "
            f"{requested} unique configurations per task, exceeding the "
            f"protocol max_actions_per_task={action_limit}"
        )
    if requested > candidate_limit:
        raise ValueError(
            "Requested search can execute "
            f"{requested} primary candidate fits per task, exceeding the "
            f"protocol max_candidates_per_task={candidate_limit}"
        )
    return {
        "requested_unique_configurations": requested,
        "requested_primary_candidate_fits": requested,
        "max_actions_per_task": action_limit,
        "max_candidates_per_task": candidate_limit,
    }


def _bootstrap_interval(
    values: np.ndarray, seed: int = 20260819
) -> tuple[float, float]:
    if len(values) == 0:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(values), size=(10000, len(values)))
    means = values[indices].mean(axis=1)
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def compare_policy_results(
    results: list[PolicyTaskResult],
    *,
    agent_arm: str = "agent",
) -> dict[str, Any]:
    if not results:
        raise ValueError("Policy results are required")
    expected = {item.name for item in TASKS.values()}
    protocol_hashes = {item.protocol_sha256 for item in results}
    if len(protocol_hashes) != 1:
        raise ValueError("All policy results must reference the same protocol hash")
    by_arm: dict[str, dict[str, PolicyTaskResult]] = {}
    for item in results:
        task = resolve_task(item.task).name
        if task in by_arm.setdefault(item.arm, {}):
            raise ValueError(f"Duplicate result for {item.arm}/{task}")
        by_arm[item.arm][task] = item
    if set(by_arm.get(agent_arm, {})) != expected:
        raise ValueError("Agent arm must cover all 22 tasks")
    comparisons = {}
    for arm, rows in sorted(by_arm.items()):
        if arm == agent_arm:
            continue
        shared = sorted(expected & set(rows))
        deltas = []
        standardized_deltas = []
        cost_ratios = []
        action_ratios = []
        action_matches = []
        for task_name in shared:
            task = resolve_task(task_name)
            agent = by_arm[agent_arm][task_name]
            control = rows[task_name]
            direction = 1.0 if task.maximize else -1.0
            deltas.append(
                direction * (agent.validation_score - control.validation_score)
            )
            uncertainty = max(
                0.02,
                float(np.hypot(agent.validation_std, control.validation_std)),
            )
            standardized_deltas.append(deltas[-1] / uncertainty)
            cost_ratios.append(
                agent.accounted_fit_seconds / max(1e-9, control.accounted_fit_seconds)
            )
            action_ratios.append(
                agent.accounted_unique_configurations
                / max(1, control.accounted_unique_configurations)
            )
            action_matches.append(
                agent.accounted_unique_configurations
                == control.accounted_unique_configurations
            )
        values = np.asarray(deltas, dtype=float)
        standardized = np.asarray(standardized_deltas, dtype=float)
        low, high = _bootstrap_interval(values)
        standardized_low, standardized_high = _bootstrap_interval(standardized)
        wins = int(np.sum(values > 1e-12))
        ties = int(np.sum(np.abs(values) <= 1e-12))
        losses = int(np.sum(values < -1e-12))
        non_ties = wins + losses
        sign_p = (
            float(binomtest(wins, non_ties, 0.5, alternative="greater").pvalue)
            if non_ties
            else 1.0
        )
        complete = len(shared) == 22
        action_budget_parity = bool(action_matches and all(action_matches))
        comparisons[arm] = {
            "tasks": len(shared),
            "complete": complete,
            "mean_oriented_delta": float(np.mean(values)) if len(values) else None,
            "median_oriented_delta": float(np.median(values)) if len(values) else None,
            "bootstrap_95_ci": [low, high],
            "mean_uncertainty_standardized_delta": (
                float(np.mean(standardized)) if len(standardized) else None
            ),
            "standardized_bootstrap_95_ci": [standardized_low, standardized_high],
            "wins": wins,
            "ties": ties,
            "losses": losses,
            "one_sided_paired_sign_p": sign_p,
            "mean_compute_ratio": float(np.mean(cost_ratios)) if cost_ratios else None,
            "mean_action_ratio": float(np.mean(action_ratios))
            if action_ratios
            else None,
            "action_budget_parity": action_budget_parity,
            "superiority_supported": bool(
                complete and low > 0 and sign_p < 0.05 and action_budget_parity
            ),
        }
    return {
        "protocol_sha256": next(iter(protocol_hashes)),
        "agent_arm": agent_arm,
        "comparisons": comparisons,
        "test_loaded": False,
    }


def write_default_protocol(path: Path) -> dict[str, Any]:
    payload = default_protocol().to_dict()
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if existing.get("sha256") != payload["sha256"]:
            raise RuntimeError(
                "Refusing to overwrite a different benchmark preregistration"
            )
    else:
        path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return payload


def load_protocol(path: Path) -> dict[str, Any]:
    """Load and cryptographically verify a preregistered benchmark protocol."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    claimed = payload.get("sha256")
    base = {key: value for key, value in payload.items() if key != "sha256"}
    actual = hashlib.sha256(json.dumps(base, sort_keys=True).encode()).hexdigest()
    if claimed != actual:
        raise ValueError("Benchmark protocol SHA-256 is invalid")
    if payload.get("test_feedback_to_discovery") is not False:
        raise ValueError("Benchmark protocol must forbid test feedback")
    return payload
