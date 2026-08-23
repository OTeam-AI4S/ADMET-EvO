"""TDC orchestration for validation-only evolution and frozen test evaluation."""

from __future__ import annotations

import hashlib
import itertools
import json
import platform
import shutil
import signal
import subprocess
import sys
import threading
import time
import uuid
from copy import deepcopy
from dataclasses import asdict, replace
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from scipy.stats import rankdata
from tdc.benchmark_group import admet_group

from .action_evidence import evaluate_action_evidence, selected_action_pair
from .action_memory import ResearchOutcomeMemory
from .brain import AgentBrain, RuleBasedBrain
from .data_governance import MolecularDataCurator, canonical_smiles
from .data_skill import ADMETDataCollectorSkill
from .ecvoi_v2_memory import load_matched_effect_memory
from .endpoint_contracts import endpoint_contract
from .evidence_router import EvidenceRouter
from .experiment_db import ExperimentDatabase
from .fault_detection import ScientificFaultDetector
from .features import FeatureCache
from .frozen_executor import FrozenRecipeExecutor
from .frozen_recipe import build_execution_plan
from .guard import validate_experiment
from .hardness import profile_hardness
from .memory import ExperimentMemory
from .profiling import profile_task
from .research_episode import EpisodeBudget, PairExecution, run_episode
from .research_loop import EffectFirstResearchPlanner, EvidenceLedger
from .research_pairing import ResolvedActionPair, resolve_action_pair
from .research_policy_v2 import (
    ECVOI_V2_ROUTING_POLICIES,
    MatchedEffectObservation,
    metric_valued_policy_for_variant,
)
from .robust_racing import RobustFutilityPolicy, evaluate_race
from .schemas import ExperimentSpec, RunResult, TaskProfile
from .selection import build_model, predict, score
from .self_correction import ValidationSelfCorrectionPlanner
from .source_snapshot import snapshot_source
from .task_context import load_context_registry, task_context_signature
from .tasks import TASKS, TaskSpec, resolve_task


def _columns(frame) -> tuple[str, str]:
    smiles = "Drug" if "Drug" in frame.columns else "SMILES"
    target = "Y"
    if smiles not in frame.columns or target not in frame.columns:
        raise ValueError(f"Unexpected TDC columns: {list(frame.columns)}")
    return smiles, target


def _json_default(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"Cannot serialize {type(value).__name__}")


def _write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, default=_json_default),
        encoding="utf-8",
    )


def _experiment_signature(experiment: ExperimentSpec) -> str:
    return json.dumps(
        {
            "representations": experiment.representations,
            "model": experiment.model,
            "imbalance": experiment.imbalance,
            "data_policy": experiment.data_policy,
            "hyperparameters": experiment.hyperparameters,
        },
        sort_keys=True,
    )


def _percentile_ranks(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values)
    if values.ndim != 1 or len(values) == 0:
        raise ValueError("Rank ensembles require a non-empty prediction vector")
    return np.asarray(rankdata(values, method="average") / len(values), dtype=float)


def _ensemble_evidence(
    ensemble_scores: list[float],
    left_scores: list[float],
    right_scores: list[float],
    *,
    maximize: bool,
    minimum_effect: float = 0.005,
    minimum_positive_rate: float = 0.8,
) -> dict[str, Any]:
    """Test a blend against its stronger member over matched validation seeds."""
    if not (
        len(ensemble_scores) == len(left_scores) == len(right_scores)
        and ensemble_scores
    ):
        raise ValueError(
            "Ensemble/member score vectors must have equal non-zero length"
        )
    direction = 1.0 if maximize else -1.0
    left_mean = float(np.mean(left_scores))
    right_mean = float(np.mean(right_scores))
    baseline_name = (
        "left" if direction * left_mean >= direction * right_mean else "right"
    )
    baseline_scores = left_scores if baseline_name == "left" else right_scores
    deltas = [
        direction * (float(value) - float(baseline))
        for value, baseline in zip(ensemble_scores, baseline_scores, strict=True)
    ]
    mean_delta = float(np.mean(deltas))
    positive_rate = float(np.mean([value > 0.0 for value in deltas]))
    return {
        "baseline_member": baseline_name,
        "oriented_deltas": deltas,
        "mean_oriented_delta": mean_delta,
        "positive_delta_rate": positive_rate,
        "minimum_effect": float(minimum_effect),
        "minimum_positive_rate": float(minimum_positive_rate),
        "supported": (
            mean_delta >= minimum_effect and positive_rate >= minimum_positive_rate
        ),
    }


def _select_distinct_ensemble_winners(
    candidates: list[tuple[RunResult, dict[str, Any], dict[int, np.ndarray]]],
    *,
    maximize: bool,
    limit: int = 3,
) -> list[tuple[RunResult, dict[str, Any], dict[int, np.ndarray]]]:
    """Keep only the best supported blend for each underlying member pair."""
    if limit < 1:
        raise ValueError("Ensemble persistence limit must be positive")
    direction = 1.0 if maximize else -1.0
    ordered = sorted(
        candidates,
        key=lambda item: (
            -direction * item[0].validation_score,
            item[0].run_id,
        ),
    )
    winners = []
    seen_pairs: set[tuple[str, ...]] = set()
    for item in ordered:
        pair = tuple(sorted(str(value) for value in item[1]["members"]))
        if pair in seen_pairs:
            continue
        seen_pairs.add(pair)
        winners.append(item)
        if len(winners) == limit:
            break
    return winners


class CandidateTimeoutError(TimeoutError):
    """Raised when one experiment exceeds its preregistered wall-time budget."""


def _arm_candidate_timeout(seconds: float | None):
    """Arm a process-local hard timeout when POSIX main-thread signals exist."""
    if (
        seconds is None
        or seconds <= 0
        or threading.current_thread() is not threading.main_thread()
        or not hasattr(signal, "SIGALRM")
        or not hasattr(signal, "setitimer")
    ):
        return None
    previous_timer = signal.getitimer(signal.ITIMER_REAL)
    if previous_timer[0] > 0:
        return None
    previous_handler = signal.getsignal(signal.SIGALRM)

    def handler(_signum, _frame):
        raise CandidateTimeoutError(
            f"candidate exceeded hard wall-time limit of {seconds:.3f} seconds"
        )

    signal.signal(signal.SIGALRM, handler)
    signal.setitimer(signal.ITIMER_REAL, float(seconds))
    return previous_handler


def _disarm_candidate_timeout(previous_handler) -> None:
    if previous_handler is None:
        return
    signal.setitimer(signal.ITIMER_REAL, 0.0)
    signal.signal(signal.SIGALRM, previous_handler)


def _select_robust_candidates(
    task: TaskSpec,
    completed: list[RunResult],
    experiments: dict[str, ExperimentSpec],
    limit: int,
) -> list[RunResult]:
    """Promote causal controls, strong candidates and model diversity.

    A diversity-only fill can discard the best tuned recipe merely because a
    weaker mandatory treatment/control uses the same model family. Reserve
    score-first slots for candidates on the score/runtime Pareto frontier,
    then spend the remaining capacity on model diversity.
    """
    direction = 1.0 if task.maximize else -1.0
    ordered = sorted(
        completed,
        key=lambda item: direction * item.validation_score,
        reverse=True,
    )
    if not ordered:
        return []
    limit = max(1, min(int(limit), len(ordered)))

    mandatory_ids: set[str] = set()
    model_controls = {
        "balanced_extra_trees": "extra_trees",
        "svm_rbf": "extra_trees",
    }
    for item in ordered:
        spec = experiments[item.run_id]
        if (
            "feature intervention" in spec.hypothesis
            or "matched feature control" in spec.hypothesis
        ):
            mandatory_ids.add(item.run_id)
        if spec.data_policy != "official":
            mandatory_ids.add(item.run_id)
            for control in ordered:
                control_spec = experiments[control.run_id]
                if (
                    control_spec.data_policy == "official"
                    and control_spec.model == spec.model
                    and control_spec.representations == spec.representations
                    and control_spec.hyperparameters == spec.hyperparameters
                    and control_spec.imbalance == spec.imbalance
                ):
                    mandatory_ids.add(control.run_id)
                    break
        control_model = model_controls.get(spec.model)
        if control_model is not None:
            mandatory_ids.add(item.run_id)
            for control in ordered:
                control_spec = experiments[control.run_id]
                if (
                    control_spec.model == control_model
                    and control_spec.representations == spec.representations
                    and control_spec.data_policy == spec.data_policy
                ):
                    mandatory_ids.add(control.run_id)
                    break
        if spec.imbalance != "none":
            for control in ordered:
                control_spec = experiments[control.run_id]
                if (
                    control_spec.model == spec.model
                    and control_spec.representations == spec.representations
                    and control_spec.data_policy == spec.data_policy
                    and control_spec.imbalance == "none"
                    and control_spec.hyperparameters == spec.hyperparameters
                ):
                    mandatory_ids.add(item.run_id)
                    mandatory_ids.add(control.run_id)
                    break

    selected = [item for item in ordered if item.run_id in mandatory_ids][:limit]
    selected_ids = {item.run_id for item in selected}
    capacity = limit - len(selected)

    def dominated(candidate: RunResult) -> bool:
        candidate_score = direction * candidate.validation_score
        for other in ordered:
            if other.run_id == candidate.run_id:
                continue
            other_score = direction * other.validation_score
            no_worse_score = other_score >= candidate_score
            no_slower = other.elapsed_seconds <= candidate.elapsed_seconds
            strictly_better = (
                other_score > candidate_score
                or other.elapsed_seconds < candidate.elapsed_seconds
            )
            if no_worse_score and no_slower and strictly_better:
                return True
        return False

    performance_slots = min(capacity, max(1, limit // 3))
    for item in ordered:
        if performance_slots == 0:
            break
        if item.run_id in selected_ids or dominated(item):
            continue
        selected.append(item)
        selected_ids.add(item.run_id)
        performance_slots -= 1

    seen_models = {item.model for item in selected}
    for item in ordered:
        if len(selected) == limit:
            break
        if item.run_id in selected_ids or item.model in seen_models:
            continue
        selected.append(item)
        selected_ids.add(item.run_id)
        seen_models.add(item.model)

    for item in ordered:
        if len(selected) == limit:
            break
        if item.run_id in selected_ids:
            continue
        selected.append(item)
        selected_ids.add(item.run_id)
    return selected


def _scaleup_stop_reason(
    candidate: ExperimentSpec,
    history: list[RunResult],
    experiments: dict[str, ExperimentSpec],
    remaining_seconds: float,
) -> str | None:
    """Reject tree-count scale-ups that cannot plausibly finish in-budget.

    The estimate is deliberately conservative: it requires an already
    completed pilot on the same model, representations, data policy,
    imbalance setting and training criterion.  A scale-up is stopped only
    when linear runtime extrapolation exceeds the remaining round budget by
    at least 15%.
    """
    candidate_trees = candidate.hyperparameters.get("n_estimators")
    if (
        isinstance(candidate_trees, bool)
        or not isinstance(candidate_trees, (int, float))
        or candidate_trees < 2
        or remaining_seconds <= 0
    ):
        return None
    candidate_criterion = candidate.hyperparameters.get("criterion")
    estimates: list[tuple[float, RunResult, float]] = []
    for prior in history:
        if prior.status != "completed" or prior.elapsed_seconds <= 0:
            continue
        prior_spec = experiments.get(prior.run_id)
        if prior_spec is None:
            continue
        prior_trees = prior_spec.hyperparameters.get("n_estimators")
        if (
            isinstance(prior_trees, bool)
            or not isinstance(prior_trees, (int, float))
            or prior_trees <= 0
            or prior_trees >= candidate_trees
        ):
            continue
        if (
            prior_spec.model != candidate.model
            or prior_spec.representations != candidate.representations
            or prior_spec.data_policy != candidate.data_policy
            or prior_spec.imbalance != candidate.imbalance
            or prior_spec.hyperparameters.get("criterion") != candidate_criterion
        ):
            continue
        predicted = prior.elapsed_seconds * float(candidate_trees) / float(prior_trees)
        estimates.append((predicted, prior, float(prior_trees)))
    if not estimates:
        return None
    predicted, pilot, pilot_trees = min(estimates, key=lambda item: item[0])
    if predicted <= 1.15 * remaining_seconds:
        return None
    return (
        f"soft-stopped scale-up from {pilot_trees:g} to {float(candidate_trees):g} "
        f"trees: pilot {pilot.run_id} took {pilot.elapsed_seconds:.3f}s; "
        f"linear estimate {predicted:.3f}s exceeds remaining budget "
        f"{remaining_seconds:.3f}s by the preregistered 15% safety margin"
    )


class ADMETAgent:
    def __init__(
        self,
        data_dir: Path,
        run_dir: Path,
        brain: AgentBrain | None = None,
        device: str = "auto",
        n_jobs: int = 8,
        selection_policy: str = "score",
        uncertainty_penalty: float = 0.5,
        cost_penalty: float = 0.005,
        effect_priority: float = 0.80,
        research_routing_policy: str = "legacy",
        action_memory_snapshot: Path | None = None,
        ecvoi_v2_memory_snapshot: Path | None = None,
        action_context_registry: Path | None = None,
        benchmark_protocol: dict[str, Any] | None = None,
        feature_cache_dir: Path | None = None,
        persist_discovery_models: bool = False,
    ):
        self.group = admet_group(path=str(data_dir))
        self.device = self._resolve_device(device)
        if (
            isinstance(n_jobs, bool)
            or not isinstance(n_jobs, int)
            or not 1 <= n_jobs <= 256
        ):
            raise ValueError("n_jobs must be an integer between 1 and 256")
        self.n_jobs = n_jobs
        self.run_dir = run_dir
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.source_manifest = snapshot_source(self.run_dir)
        self.feature_cache_dir = feature_cache_dir or (run_dir / "feature_cache")
        self.cache = FeatureCache(self.feature_cache_dir, device=self.device)
        self.persist_discovery_models = bool(persist_discovery_models)
        self.memory = ExperimentMemory(run_dir / "memory.jsonl")
        self.database = ExperimentDatabase(run_dir / "experiments.sqlite")
        self.evidence_ledger = EvidenceLedger(run_dir / "evidence_ledger.jsonl")
        self.action_memory = ResearchOutcomeMemory(run_dir / "action_outcomes.jsonl")
        if action_memory_snapshot is not None:
            action_memory_snapshot = Path(action_memory_snapshot)
            if not action_memory_snapshot.is_file():
                raise FileNotFoundError(
                    f"Action-memory snapshot does not exist: {action_memory_snapshot}"
                )
            self.action_memory_snapshot_path = action_memory_snapshot
            self.action_memory_snapshot_sha256 = hashlib.sha256(
                action_memory_snapshot.read_bytes()
            ).hexdigest()
            self.routing_action_memory = ResearchOutcomeMemory(action_memory_snapshot)
        else:
            self.action_memory_snapshot_path = None
            self.action_memory_snapshot_sha256 = None
            self.routing_action_memory = self.action_memory
        if ecvoi_v2_memory_snapshot is not None:
            ecvoi_v2_memory_snapshot = Path(ecvoi_v2_memory_snapshot)
            if not ecvoi_v2_memory_snapshot.is_file():
                raise FileNotFoundError(
                    "EC-VOI v2 matched-effect memory does not exist: "
                    f"{ecvoi_v2_memory_snapshot}"
                )
            self.ecvoi_v2_memory_snapshot_path = ecvoi_v2_memory_snapshot
            self.ecvoi_v2_memory_snapshot_sha256 = hashlib.sha256(
                ecvoi_v2_memory_snapshot.read_bytes()
            ).hexdigest()
        else:
            self.ecvoi_v2_memory_snapshot_path = None
            self.ecvoi_v2_memory_snapshot_sha256 = None
        if action_context_registry is not None:
            action_context_registry = Path(action_context_registry)
            registry = load_context_registry(action_context_registry)
            self.action_context_registry_path = action_context_registry
            self.action_context_registry_sha256 = registry["sha256"]
            self.action_context_registry = registry["contexts"]
        else:
            self.action_context_registry_path = None
            self.action_context_registry_sha256 = None
            self.action_context_registry = None
        self.research_planner = EffectFirstResearchPlanner(
            routing_policy=research_routing_policy
        )
        self.research_routing_policy = research_routing_policy
        self.data_skill = ADMETDataCollectorSkill()
        self.self_correction_planner = ValidationSelfCorrectionPlanner()
        self.fault_detector = ScientificFaultDetector()
        self.brain = brain or RuleBasedBrain()
        if selection_policy not in {"score", "lcb", "cost-aware-lcb"}:
            raise ValueError("selection_policy must be score, lcb, or cost-aware-lcb")
        self.selection_policy = selection_policy
        if not 0.5 <= effect_priority <= 1.0:
            raise ValueError("effect_priority must be within [0.5, 1.0]")
        self.effect_priority = float(effect_priority)
        self.benchmark_protocol = benchmark_protocol
        self.evidence_router = EvidenceRouter(
            uncertainty_penalty=uncertainty_penalty,
            cost_penalty=cost_penalty if selection_policy == "cost-aware-lcb" else 0.0,
        )

    def _route_data_skill(
        self, task_name: str, research_plan: dict[str, Any]
    ) -> dict[str, Any]:
        skill = getattr(self, "data_skill", None)
        if skill is None:
            skill = ADMETDataCollectorSkill()
            self.data_skill = skill
        return skill.route(task_name, research_plan)

    @staticmethod
    def _resolve_device(requested: str) -> str:
        if requested not in {"auto", "cpu", "cuda"}:
            raise ValueError("device must be auto, cpu, or cuda")
        available = False
        executable = shutil.which("nvidia-smi")
        if executable:
            probe = subprocess.run(
                [executable, "--query-gpu=name", "--format=csv,noheader"],
                capture_output=True,
                text=True,
                check=False,
            )
            available = probe.returncode == 0 and bool(probe.stdout.strip())
        if requested == "cuda" and not available:
            raise RuntimeError(
                "CUDA was requested but this node exposes no NVIDIA GPU; "
                "run on an allocated GPU compute node"
            )
        return (
            "cuda"
            if requested == "cuda" or (requested == "auto" and available)
            else "cpu"
        )

    def _gpu_manifest(self) -> list[str]:
        executable = shutil.which("nvidia-smi")
        if not executable:
            return []
        probe = subprocess.run(
            [executable, "--query-gpu=name,memory.total", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            check=False,
        )
        return (
            [line.strip() for line in probe.stdout.splitlines() if line.strip()]
            if probe.returncode == 0
            else []
        )

    def _execute(
        self,
        experiment: ExperimentSpec,
        train,
        valid,
        smiles_col: str,
        y_col: str,
        profile: TaskProfile,
        timeout_seconds: float | None = None,
    ) -> RunResult:
        validate_experiment(experiment)
        started = time.monotonic()
        run_id = f"{experiment.task.lower()}-{experiment.seed}-{uuid.uuid4().hex[:8]}"
        artifact_dir = self.run_dir / "artifacts" / run_id
        artifact_dir.mkdir(parents=True)
        _write_json(artifact_dir / "experiment.json", experiment.to_dict())
        previous_alarm_handler = _arm_candidate_timeout(timeout_seconds)
        try:
            fit_train = train
            curator = MolecularDataCurator(smiles_col, y_col)
            valid_keys = {
                value
                for value in (
                    canonical_smiles(item) for item in valid[smiles_col].tolist()
                )
                if value is not None
            }
            fit_train, data_audit = curator.curate(
                train,
                task_type=profile.task_type,
                policy=experiment.data_policy,
                validation_smiles=valid_keys,
            )
            if len(fit_train) == 0:
                raise ValueError(
                    f"Data policy {experiment.data_policy} removed all training rows"
                )
            _write_json(artifact_dir / "data_audit.json", data_audit.to_dict())
            combined_smiles = (
                fit_train[smiles_col].tolist() + valid[smiles_col].tolist()
            )
            x_combined = self.cache.get_or_compute(
                combined_smiles, experiment.representations
            )
            x_train = x_combined[: len(fit_train)]
            x_valid = x_combined[len(fit_train) :]
            y_train = fit_train[y_col].to_numpy()
            y_valid = valid[y_col].to_numpy()
            task = resolve_task(experiment.task)
            model = build_model(experiment, task, y_train)
            model.fit(x_train, y_train)
            predictions = predict(model, x_valid, task)
            validation_score = score(y_valid, predictions, task.metric)
            if self.persist_discovery_models:
                joblib.dump(model, artifact_dir / "model.joblib")
            np.save(
                artifact_dir / "validation_predictions.npy",
                predictions,
                allow_pickle=False,
            )
            result = RunResult(
                run_id=run_id,
                task=task.name,
                seed=experiment.seed,
                model=experiment.model,
                metric=task.metric,
                validation_score=validation_score,
                elapsed_seconds=time.monotonic() - started,
                representations=experiment.representations,
                hyperparameters=experiment.hyperparameters,
                hypothesis=experiment.hypothesis,
                data_policy=experiment.data_policy,
            )
        except Exception as exc:
            result = RunResult(
                run_id=run_id,
                task=experiment.task,
                seed=experiment.seed,
                model=experiment.model,
                metric=resolve_task(experiment.task).metric,
                validation_score=float("nan"),
                elapsed_seconds=time.monotonic() - started,
                representations=experiment.representations,
                hyperparameters=experiment.hyperparameters,
                hypothesis=experiment.hypothesis,
                status="failed",
                error=f"{type(exc).__name__}: {exc}",
                data_policy=experiment.data_policy,
            )
        finally:
            _disarm_candidate_timeout(previous_alarm_handler)
        _write_json(artifact_dir / "result.json", result.to_dict())
        self.memory.append({**experiment.to_dict(), **result.to_dict()})
        self.database.record(profile, experiment, result)
        self.evidence_ledger.append(
            "experiment_completed"
            if result.status == "completed"
            else "experiment_failed",
            {"experiment": experiment.to_dict(), "result": result.to_dict()},
        )
        return result

    def _shared_anchor_recipe(
        self, task: TaskSpec
    ) -> tuple[dict[str, Any], dict[str, Any]] | None:
        """Load a protocol-bound endpoint anchor without consulting test data."""

        if self.benchmark_protocol is None:
            return None
        invariants = self.benchmark_protocol.get("frozen_invariants", {})
        raw_path = invariants.get("shared_anchor_mapping_path")
        expected_sha256 = invariants.get("shared_anchor_mapping_sha256")
        if raw_path is None and expected_sha256 is None:
            return None
        if not isinstance(raw_path, str) or not isinstance(expected_sha256, str):
            raise RuntimeError("Shared anchor requires a path and SHA-256")
        path = Path(raw_path)
        if not path.is_file():
            raise FileNotFoundError(f"Shared anchor mapping does not exist: {path}")
        actual_sha256 = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual_sha256 != expected_sha256:
            raise RuntimeError("Shared anchor mapping SHA-256 does not match protocol")
        mapping = json.loads(path.read_text(encoding="utf-8"))
        if task.name not in mapping:
            raise RuntimeError(f"Shared anchor mapping has no recipe for {task.name}")
        frozen = self._normalize_frozen(task, mapping[task.name])
        data_policy = invariants.get("shared_anchor_data_policy", "identity_safe")
        if data_policy != "identity_safe":
            raise RuntimeError("Shared discovery anchors require identity_safe data")
        if frozen["kind"] == "single":
            frozen["data_policy"] = data_policy
        elif frozen["kind"] == "ensemble":
            frozen["members"] = [
                {**member, "data_policy": data_policy}
                for member in frozen["members"]
            ]
        else:
            raise RuntimeError(
                "Shared discovery anchors currently require serializable model specs"
            )
        recipe_sha256 = hashlib.sha256(
            json.dumps(frozen, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        return frozen, {
            "mapping_path": str(path),
            "mapping_sha256": actual_sha256,
            "normalized_recipe_sha256": recipe_sha256,
            "data_policy": data_policy,
            "test_loaded": False,
            "leaderboard_loaded": False,
        }

    def _protocol_discovery_data_policy(self) -> str | None:
        if self.benchmark_protocol is None:
            return None
        value = self.benchmark_protocol.get("frozen_invariants", {}).get(
            "mandatory_discovery_data_policy"
        )
        if value is None:
            return None
        if value != "identity_safe":
            raise RuntimeError(
                "Formal shared-anchor benchmarks require identity_safe discovery data"
            )
        return value

    def _execute_shared_anchor(
        self,
        *,
        task: TaskSpec,
        benchmark_name: str,
        validation_seeds: list[int] | tuple[int, ...],
        task_dir: Path,
    ) -> dict[str, Any] | None:
        loaded = self._shared_anchor_recipe(task)
        if loaded is None:
            return None
        task_dir.mkdir(parents=True, exist_ok=True)
        frozen, contract = loaded
        if frozen["kind"] == "single":
            members = [
                {key: value for key, value in frozen.items() if key != "kind"}
            ]
            weights = [1.0]
        else:
            members = frozen["members"]
            weights = frozen["weights"]
        aggregation = frozen.get("aggregation", "probability")
        member_results: list[RunResult] = []
        seed_results: list[RunResult] = []
        for seed in validation_seeds:
            train, valid = self.group.get_train_valid_split(
                benchmark=benchmark_name,
                split_type="default",
                seed=seed,
            )
            smiles_col, y_col = _columns(train)
            seed_profile = profile_task(train, valid, task, smiles_col, y_col)
            combined = None
            seed_elapsed = 0.0
            seed_run_ids = []
            seed_representations: list[str] = []
            failed = False
            for member, weight in zip(members, weights, strict=True):
                experiment = ExperimentSpec.from_dict(
                    {
                        **member,
                        "task": task.name,
                        "seed": seed,
                        "device": self.device,
                        "n_jobs": self.n_jobs,
                        "hypothesis": member.get(
                            "hypothesis", "protocol-bound shared strong anchor"
                        ),
                    }
                )
                result = self._execute(
                    experiment,
                    train,
                    valid,
                    smiles_col,
                    y_col,
                    seed_profile,
                    timeout_seconds=min(300.0, experiment.budget.max_minutes * 60),
                )
                member_results.append(result)
                seed_elapsed += result.elapsed_seconds
                seed_run_ids.append(result.run_id)
                seed_representations.extend(experiment.representations)
                if result.status != "completed":
                    failed = True
                    continue
                prediction = np.load(
                    self.run_dir
                    / "artifacts"
                    / result.run_id
                    / "validation_predictions.npy",
                    allow_pickle=False,
                )
                if aggregation == "rank":
                    prediction = _percentile_ranks(prediction)
                weighted = float(weight) * prediction
                combined = weighted if combined is None else combined + weighted
            if failed or combined is None:
                continue
            anchor_run_id = (
                f"{task.name.lower()}-shared-anchor-{seed}-{uuid.uuid4().hex[:8]}"
            )
            result = RunResult(
                run_id=anchor_run_id,
                task=task.name,
                seed=seed,
                model="ensemble" if len(members) > 1 else members[0]["model"],
                metric=task.metric,
                validation_score=score(
                    valid[y_col].to_numpy(), combined, task.metric
                ),
                elapsed_seconds=seed_elapsed,
                representations=tuple(dict.fromkeys(seed_representations)),
                hyperparameters={
                    "shared_anchor_recipe_sha256": contract[
                        "normalized_recipe_sha256"
                    ],
                    "member_count": len(seed_run_ids),
                    "aggregation": aggregation,
                },
                hypothesis="protocol-bound endpoint-specific shared strong anchor",
                data_policy=contract["data_policy"],
            )
            artifact_dir = self.run_dir / "artifacts" / anchor_run_id
            artifact_dir.mkdir(parents=True, exist_ok=False)
            _write_json(artifact_dir / "result.json", result.to_dict())
            _write_json(artifact_dir / "frozen_anchor.json", frozen)
            np.save(
                artifact_dir / "validation_predictions.npy",
                combined,
                allow_pickle=False,
            )
            seed_results.append(result)
        complete = len(seed_results) == len(validation_seeds)
        aggregate = None
        if complete:
            aggregate = replace(
                seed_results[0],
                validation_score=float(
                    np.mean([item.validation_score for item in seed_results])
                ),
                validation_std=float(
                    np.std([item.validation_score for item in seed_results])
                ),
                elapsed_seconds=float(
                    sum(item.elapsed_seconds for item in seed_results)
                ),
            )
        summary = {
            **contract,
            "required": True,
            "complete": complete,
            "ordered_seeds": list(validation_seeds),
            "member_count": len(members),
            "member_run_ids": [item.run_id for item in member_results],
            "seed_results": [item.to_dict() for item in seed_results],
            "aggregate_result": aggregate.to_dict() if aggregate else None,
            "frozen_recipe": frozen,
            "reserved_fit_seconds": float(
                330.0 * len(members) * len(validation_seeds)
            ),
            "actual_fit_seconds": float(
                sum(item.elapsed_seconds for item in member_results)
            ),
        }
        _write_json(task_dir / "shared_strong_anchor.json", summary)
        self.evidence_ledger.append(
            "shared_strong_anchor_evaluated",
            {"task": task.name, **summary, "test_loaded": False},
        )
        return {
            "summary": summary,
            "aggregate": aggregate,
            "member_results": member_results,
            "frozen_recipe": frozen,
        }

    def _run_ecvoi_v2_search(
        self,
        *,
        task: TaskSpec,
        benchmark_name: str,
        profile: TaskProfile,
        hardness,
        contract,
        research_plan: dict[str, Any],
        task_dir: Path,
        validation_seeds: list[int],
        task_compute_minutes: int | None,
        benchmark_arm: str,
    ) -> dict[str, Any]:
        """Execute a v2 episode as sequential, atomically reserved pairs."""

        if self.research_routing_policy not in ECVOI_V2_ROUTING_POLICIES:
            raise RuntimeError("v2 episode called for a non-v2 routing policy")
        if len(validation_seeds) < 3:
            raise ValueError("EC-VOI v2 requires at least three validation seeds")
        if self.action_context_registry is None:
            raise RuntimeError("EC-VOI v2 requires a frozen task-context registry")
        if self.ecvoi_v2_memory_snapshot_path is None:
            raise RuntimeError("EC-VOI v2 requires a matched-effect memory snapshot")
        context = self.action_context_registry.get(task.name)
        if context is None:
            raise RuntimeError(f"No frozen EC-VOI v2 context for {task.name}")
        context_sha256 = hashlib.sha256(
            json.dumps(context, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        task_fit_budget_seconds = float(
            60 * task_compute_minutes if task_compute_minutes is not None else 5400
        )
        task_candidate_budget = 8
        if self.benchmark_protocol is not None:
            arm = next(
                item
                for item in self.benchmark_protocol["arms"]
                if item["name"] == benchmark_arm
            )
            task_candidate_budget = int(arm["max_actions_per_task"])

        memory_mode = (
            "none"
            if self.research_routing_policy == "ecvoi-v2-no-memory"
            else "shuffled"
            if self.research_routing_policy == "ecvoi-v2-shuffled-memory"
            else "correct"
        )
        context_mode = (
            "global"
            if self.research_routing_policy == "ecvoi-v2-no-context"
            else "shuffled"
            if self.research_routing_policy == "ecvoi-v2-shuffled-context"
            else "correct"
        )
        prior_observations, memory_audit = load_matched_effect_memory(
            self.ecvoi_v2_memory_snapshot_path,
            target_endpoint=task.name,
            contexts=self.action_context_registry,
            expected_sha256=self.ecvoi_v2_memory_snapshot_sha256,
            memory_mode=memory_mode,
            context_mode=context_mode,
        )
        ordered_seeds = tuple(validation_seeds[:3])
        resolved: list[ResolvedActionPair] = []
        unresolved = []
        for action in research_plan.get("actions", []):
            candidate = resolve_action_pair(
                action,
                profile,
                ordered_seeds=ordered_seeds,
                context_sha256=context_sha256,
                task_fit_budget_seconds=task_fit_budget_seconds,
                task_candidate_budget=task_candidate_budget,
                per_fit_timeout_seconds=330.0,
            )
            if candidate is None:
                unresolved.append(
                    {
                        "action_id": action.get("action_id"),
                        "axis": action.get("axis"),
                        "reason": "unresolved_or_episode_level_action",
                    }
                )
            else:
                resolved.append(candidate)
        resolved_by_id = {
            item.action_pair.pair_id: item for item in resolved
        }
        history: list[RunResult] = []
        experiments: dict[str, ExperimentSpec] = {}
        pair_role_runs: dict[tuple[str, str], list[RunResult]] = {}
        anchor_runs: list[RunResult] = []
        shared_anchor_execution: dict[str, Any] | None = None
        shared_anchor_aggregate: RunResult | None = None
        floor_cache: dict[int, tuple[Any, Any, str, str, TaskProfile]] = {}

        def discovery_split(seed: int):
            if seed in floor_cache:
                return floor_cache[seed]
            seed_train, seed_valid = self.group.get_train_valid_split(
                benchmark=benchmark_name,
                split_type="default",
                seed=seed,
            )
            smiles_col, y_col = _columns(seed_train)
            valid_keys = {
                value
                for value in (
                    canonical_smiles(item)
                    for item in seed_valid[smiles_col].tolist()
                )
                if value is not None
            }
            floor_train, floor_audit = MolecularDataCurator(
                smiles_col, y_col
            ).curate(
                seed_train,
                task_type=profile.task_type,
                policy="identity_safe",
                validation_smiles=valid_keys,
            )
            if not len(floor_train):
                raise RuntimeError("Mandatory identity safety floor removed all rows")
            _write_json(
                task_dir / f"identity_safety_floor_seed_{seed}.json",
                floor_audit.to_dict(),
            )
            floor_profile = profile_task(
                floor_train, seed_valid, task, smiles_col, y_col
            )
            floor_cache[seed] = (
                seed_train,
                seed_valid,
                smiles_col,
                y_col,
                floor_profile,
            )
            return floor_cache[seed]

        anchor_timeout_seconds = 300.0
        anchor_reserved_seconds = 330.0 * len(ordered_seeds)
        anchor_imbalance = (
            "class_weight"
            if profile.task_type == "classification"
            and profile.positive_rate_train is not None
            and not 0.3 <= profile.positive_rate_train <= 0.7
            else "none"
        )
        if self.research_routing_policy in {
            "ecvoi-v3",
            "ecvoi-v3-no-initial-exploration",
        }:
            shared_anchor_execution = self._execute_shared_anchor(
                task=task,
                benchmark_name=benchmark_name,
                validation_seeds=ordered_seeds,
                task_dir=task_dir,
            )
            if shared_anchor_execution is None:
                raise RuntimeError("EC-VOI v3 requires a protocol-bound shared anchor")
            history.extend(shared_anchor_execution["member_results"])
            shared_anchor_aggregate = shared_anchor_execution["aggregate"]
            anchor_reserved_seconds = float(
                shared_anchor_execution["summary"]["reserved_fit_seconds"]
            )
        else:
            for seed in ordered_seeds:
                floor_train, seed_valid, smiles_col, y_col, floor_profile = (
                    discovery_split(seed)
                )
                anchor = ExperimentSpec(
                    task=task.name,
                    seed=seed,
                    representations=("morgan", "avalon", "erg", "rdkit2d"),
                    model="extra_trees",
                    imbalance=anchor_imbalance,
                    data_policy="identity_safe",
                    device=self.device,
                    n_jobs=self.n_jobs,
                    hypothesis="EC-VOI v2 mandatory identity-safe baseline anchor",
                )
                result = self._execute(
                    anchor,
                    floor_train,
                    seed_valid,
                    smiles_col,
                    y_col,
                    floor_profile,
                    timeout_seconds=anchor_timeout_seconds,
                )
                history.append(result)
                experiments[result.run_id] = anchor
                anchor_runs.append(result)

        def executor(action_pair, decision):
            resolved_pair = resolved_by_id[action_pair.pair_id]
            treatment_ids = []
            control_ids = []
            actual_seconds = 0.0
            failures = []
            role_results: dict[str, list[RunResult]] = {
                "treatment": [],
                "control": [],
            }
            execution_timeout = min(
                300.0, 0.9 * resolved_pair.per_fit_timeout_seconds
            )
            for seed in action_pair.ordered_seeds:
                floor_train, seed_valid, smiles_col, y_col, floor_profile = (
                    discovery_split(seed)
                )
                for role, template in (
                    ("treatment", resolved_pair.treatment),
                    ("control", resolved_pair.control),
                ):
                    experiment = replace(
                        template,
                        seed=seed,
                        device=self.device,
                        n_jobs=self.n_jobs,
                    )
                    result = self._execute(
                        experiment,
                        floor_train,
                        seed_valid,
                        smiles_col,
                        y_col,
                        floor_profile,
                        timeout_seconds=execution_timeout,
                    )
                    history.append(result)
                    experiments[result.run_id] = experiment
                    role_results[role].append(result)
                    pair_role_runs.setdefault(
                        (action_pair.pair_id, role), []
                    ).append(result)
                    actual_seconds += result.elapsed_seconds
                    if role == "treatment":
                        treatment_ids.append(result.run_id)
                    else:
                        control_ids.append(result.run_id)
                    if result.status != "completed":
                        failures.append(f"{role}/seed={seed}:{result.error}")
            observation = None
            if not failures:
                direction = 1.0 if task.maximize else -1.0
                deltas = np.asarray(
                    [
                        direction
                        * (
                            treatment.validation_score
                            - control.validation_score
                        )
                        for treatment, control in zip(
                            role_results["treatment"],
                            role_results["control"],
                            strict=True,
                        )
                    ],
                    dtype=float,
                )
                native_se = max(
                    0.0005,
                    float(np.std(deltas, ddof=1) / np.sqrt(len(deltas))),
                )
                provenance = {
                    "pair_id": action_pair.pair_id,
                    "decision": decision,
                    "treatment_run_ids": treatment_ids,
                    "control_run_ids": control_ids,
                    "native_oriented_deltas": deltas.tolist(),
                    "test_loaded": False,
                }
                provenance_sha256 = hashlib.sha256(
                    json.dumps(
                        provenance,
                        sort_keys=True,
                        separators=(",", ":"),
                    ).encode()
                ).hexdigest()
                observation = MatchedEffectObservation(
                    endpoint_id=task.name,
                    pair_id=action_pair.pair_id,
                    action_id=action_pair.action_id,
                    native_oriented_delta=float(np.mean(deltas)),
                    native_standard_error=native_se,
                    effect_standardization_scale=(
                        action_pair.effect_standardization_scale
                    ),
                    context_similarity=1.0,
                    endpoint_local=True,
                    transport_eligible=True,
                    test_derived=False,
                    leaderboard_derived=False,
                    single_axis_verified=True,
                    complete_pair=True,
                    provenance_sha256=provenance_sha256,
                )
            return PairExecution(
                pair_id=action_pair.pair_id,
                action_id=action_pair.action_id,
                treatment_run_ids=tuple(treatment_ids),
                control_run_ids=tuple(control_ids),
                ordered_seeds=action_pair.ordered_seeds,
                actual_fit_seconds=actual_seconds,
                observation=observation,
                failure_reason="; ".join(failures) if failures else None,
            )

        if resolved:
            policy = metric_valued_policy_for_variant(
                self.research_routing_policy
            )
            episode = run_episode(
                policy=policy,
                actions=[item.action_pair for item in resolved],
                prior_observations=prior_observations,
                budget=EpisodeBudget(
                    max_pairs=max(
                        1,
                        task_candidate_budget // 2
                        if self.research_routing_policy.startswith("ecvoi-v3")
                        else (task_candidate_budget - 1) // 2,
                    ),
                    max_candidate_fits=max(
                        1,
                        task_candidate_budget
                        if self.research_routing_policy.startswith("ecvoi-v3")
                        else task_candidate_budget - 1,
                    ),
                    max_fit_seconds=max(
                        1.0, task_fit_budget_seconds - anchor_reserved_seconds
                    ),
                ),
                executor=executor,
            )
            episode_payload = episode.to_dict()
        else:
            episode_payload = {
                "decision": "stop",
                "stop_reason": "no_resolved_pairs",
                "executed_pair_ids": [],
                "executed_action_ids": [],
                "observations": [asdict(item) for item in prior_observations],
                "used_pairs": 0,
                "used_candidate_fits": 0,
                "reserved_fit_seconds": 0.0,
                "actual_fit_seconds": 0.0,
                "ledger": [],
            }
            episode_payload["episode_sha256"] = hashlib.sha256(
                json.dumps(episode_payload, sort_keys=True).encode()
            ).hexdigest()
        for event in episode_payload.get("ledger", []):
            self.evidence_ledger.append(
                f"ecvoi_v2_{event['event_type']}",
                {"task": task.name, **event, "test_loaded": False},
            )

        robust_results = []
        finalist_specs: dict[str, dict[str, Any]] = {}
        if shared_anchor_aggregate is not None and shared_anchor_execution is not None:
            robust_results.append(shared_anchor_aggregate)
            finalist_specs[shared_anchor_aggregate.run_id] = (
                shared_anchor_execution["frozen_recipe"]
            )
        elif len(anchor_runs) == len(ordered_seeds) and all(
            item.status == "completed" for item in anchor_runs
        ):
            anchor_values = [item.validation_score for item in anchor_runs]
            anchor_aggregate = replace(
                anchor_runs[0],
                validation_score=float(np.mean(anchor_values)),
                validation_std=float(np.std(anchor_values)),
                elapsed_seconds=float(
                    sum(item.elapsed_seconds for item in anchor_runs)
                ),
            )
            robust_results.append(anchor_aggregate)
            finalist_specs[anchor_aggregate.run_id] = experiments[
                anchor_aggregate.run_id
            ].to_dict()
        for (pair_id, role), runs in sorted(pair_role_runs.items()):
            if len(runs) != len(ordered_seeds) or any(
                item.status != "completed" for item in runs
            ):
                continue
            values = [item.validation_score for item in runs]
            aggregated = replace(
                runs[0],
                validation_score=float(np.mean(values)),
                validation_std=float(np.std(values)),
                elapsed_seconds=float(sum(item.elapsed_seconds for item in runs)),
                hypothesis=f"{runs[0].hypothesis}; {pair_id}; role={role}",
            )
            robust_results.append(aggregated)
            finalist_specs[aggregated.run_id] = experiments[
                aggregated.run_id
            ].to_dict()
        direction = 1.0 if task.maximize else -1.0
        finalist = (
            max(
                robust_results,
                key=lambda item: direction * item.validation_score,
            )
            if robust_results
            else None
        )
        finalist_run_id = finalist.run_id if finalist is not None else None
        finalist_spec = (
            finalist_specs[finalist_run_id]
            if finalist_run_id is not None
            else None
        )
        local_observations = [
            item
            for item in episode_payload.get("observations", [])
            if item.get("endpoint_local") is True
            and item.get("pair_id") in resolved_by_id
        ]
        summary = {
            "version": "ecvoi-v2-sequential-pair-search-v1",
            "profile": profile.to_dict(),
            "endpoint_contract": contract.to_dict(),
            "hardness": hardness.to_dict(),
            "research_plan": research_plan,
            "resolved_action_pairs": [
                item.action_pair.to_dict() for item in resolved
            ],
            "unresolved_actions": unresolved,
            "ecvoi_v2_memory_audit": memory_audit,
            "ecvoi_v2_episode": episode_payload,
            "baseline_anchor": {
                **(
                    shared_anchor_execution["summary"]
                    if shared_anchor_execution is not None
                    else {
                        "required": True,
                        "candidate_charge": 1,
                        "ordered_seeds": list(ordered_seeds),
                        "reserved_fit_seconds": anchor_reserved_seconds,
                        "actual_fit_seconds": float(
                            sum(item.elapsed_seconds for item in anchor_runs)
                        ),
                        "run_ids": [item.run_id for item in anchor_runs],
                        "complete": len(anchor_runs) == len(ordered_seeds)
                        and all(item.status == "completed" for item in anchor_runs),
                    }
                ),
            },
            "action_evidence": local_observations,
            "validation_seeds": list(ordered_seeds),
            "results": [item.to_dict() for item in history],
            "robust_results": [item.to_dict() for item in robust_results],
            "ensemble_results": [],
            "selection_policy": self.selection_policy,
            "research_routing_policy": self.research_routing_policy,
            "action_memory_snapshot_sha256": self.action_memory_snapshot_sha256,
            "ecvoi_v2_memory_snapshot_sha256": (
                self.ecvoi_v2_memory_snapshot_sha256
            ),
            "action_context_registry_sha256": self.action_context_registry_sha256,
            "benchmark_arm": benchmark_arm,
            "benchmark_protocol_sha256": (
                self.benchmark_protocol.get("sha256")
                if self.benchmark_protocol
                else None
            ),
            "benchmark_protocol_version": (
                self.benchmark_protocol.get("protocol_version")
                if self.benchmark_protocol
                else None
            ),
            "source_sha256": self.source_manifest["source_sha256"],
            "mandatory_identity_safety_floor": True,
            "test_loaded": False,
            "leaderboard_loaded": False,
            "finalist_run_id": finalist_run_id,
            "finalist_spec": finalist_spec,
            "database": self.database.summary(),
        }
        _write_json(task_dir / f"search_seed_{ordered_seeds[0]}.json", summary)
        if finalist_spec is not None:
            _write_json(
                task_dir / f"finalist_seed_{ordered_seeds[0]}.json",
                {task.name: finalist_spec},
            )
        return summary

    def _robust_results(
        self,
        task: TaskSpec,
        benchmark_name: str,
        primary_seed: int,
        validation_seeds: list[int],
        completed: list[RunResult],
        experiments: dict[str, ExperimentSpec],
        history: list[RunResult],
        robust_top_k: int,
        research_plan: dict[str, Any],
        task_deadline: float | None,
    ) -> tuple[
        list[RunResult],
        dict[str, dict[int, RunResult]],
        dict[str, Any],
    ]:
        eligible = completed
        if len(validation_seeds) > 1:
            eligible = _select_robust_candidates(
                task,
                completed,
                experiments,
                robust_top_k,
            )
        seed_runs: dict[str, dict[int, RunResult]] = {
            item.run_id: {primary_seed: item} for item in eligible
        }
        racing_policy = RobustFutilityPolicy()
        racing_declared = bool(
            self.benchmark_protocol
            and self.benchmark_protocol.get("robust_futility_policy")
            == "seed2_gap0.005_time3x_worse_both_protect_action_pair"
            and self.benchmark_protocol.get("robust_futility_applies_to_all_arms")
            is True
        )
        racing_eligible = (
            racing_declared
            and set(racing_policy.checkpoint_seeds).issubset(validation_seeds)
            and primary_seed in racing_policy.checkpoint_seeds
        )
        ordered_seeds = list(validation_seeds)
        if racing_eligible:
            ordered_seeds = [
                *(
                    seed
                    for seed in racing_policy.checkpoint_seeds
                    if seed != primary_seed
                ),
                *(
                    seed
                    for seed in validation_seeds
                    if seed not in racing_policy.checkpoint_seeds
                    and seed != primary_seed
                ),
            ]
        stopped_ids: set[str] = set()
        decisions = ()
        protected_ids: frozenset[str] = frozenset()
        protected_action_pair: tuple[str, str] | None = None
        task_budget_exhausted = False
        for repeat_seed in ordered_seeds:
            if repeat_seed == primary_seed:
                continue
            train, valid = self.group.get_train_valid_split(
                benchmark=benchmark_name,
                split_type="default",
                seed=repeat_seed,
            )
            smiles_col, y_col = _columns(train)
            profile = profile_task(train, valid, task, smiles_col, y_col)
            for original in eligible:
                if original.run_id in stopped_ids:
                    continue
                if task_deadline is not None and time.monotonic() >= task_deadline:
                    task_budget_exhausted = True
                    break
                experiment = ExperimentSpec.from_dict(
                    {
                        **experiments[original.run_id].to_dict(),
                        "seed": repeat_seed,
                        "device": self.device,
                        "n_jobs": self.n_jobs,
                    }
                )
                repeated = self._execute(
                    experiment,
                    train,
                    valid,
                    smiles_col,
                    y_col,
                    profile,
                    timeout_seconds=min(
                        experiment.budget.max_minutes * 60,
                        max(
                            1.0,
                            task_deadline - time.monotonic(),
                        )
                        if task_deadline is not None
                        else experiment.budget.max_minutes * 60,
                    ),
                )
                history.append(repeated)
                seed_runs[original.run_id][repeat_seed] = repeated
            if task_budget_exhausted:
                break
            if racing_eligible and repeat_seed == racing_policy.checkpoint_seeds[-1]:
                pair = selected_action_pair(
                    research_plan,
                    {
                        run_id: experiments[run_id]
                        for run_id in seed_runs
                        if run_id in experiments
                    },
                )
                protected_action_pair = pair
                protected = set(pair or ())
                action = next(
                    (
                        item
                        for item in research_plan.get("actions", [])
                        if item.get("action_id")
                        == research_plan.get("selected_action_id")
                    ),
                    None,
                )
                if action and action.get("axis") == "replicate":
                    direction = 1.0 if task.maximize else -1.0
                    scored = []
                    for run_id, runs in seed_runs.items():
                        checkpoint = [
                            runs.get(seed) for seed in racing_policy.checkpoint_seeds
                        ]
                        if any(
                            item is None or item.status != "completed"
                            for item in checkpoint
                        ):
                            continue
                        scored.append(
                            (
                                direction
                                * float(
                                    np.mean(
                                        [
                                            item.validation_score
                                            for item in checkpoint
                                            if item is not None
                                        ]
                                    )
                                ),
                                run_id,
                            )
                        )
                    protected.update(
                        run_id
                        for _, run_id in sorted(
                            scored, key=lambda item: (-item[0], item[1])
                        )[:2]
                    )
                protected_ids = frozenset(protected)
                decisions = evaluate_race(
                    seed_runs,
                    maximize=task.maximize,
                    protected_ids=protected_ids,
                    policy=racing_policy,
                )
                stopped_ids = {
                    item.candidate_id for item in decisions if item.decision == "stop"
                }
        aggregated = []
        for original in eligible:
            repeats = [
                seed_runs[original.run_id].get(seed) for seed in validation_seeds
            ]
            if any(item is None or item.status != "completed" for item in repeats):
                continue
            values = [item.validation_score for item in repeats if item is not None]
            aggregated.append(
                replace(
                    original,
                    validation_score=float(np.mean(values)),
                    validation_std=float(np.std(values)),
                    elapsed_seconds=float(
                        sum(
                            item.elapsed_seconds for item in repeats if item is not None
                        )
                    ),
                    hypothesis=(
                        f"{original.hypothesis}; robust validation over "
                        f"{len(values)} seed(s)"
                    ),
                )
            )
        planned_fits_avoided = sum(
            1
            for run_id in stopped_ids
            for repeat_seed in validation_seeds
            if repeat_seed not in racing_policy.checkpoint_seeds
            and repeat_seed not in seed_runs[run_id]
        )
        racing = {
            "protocol": racing_policy.policy_version,
            "enabled": racing_eligible,
            "checkpoint_seeds": list(racing_policy.checkpoint_seeds),
            "protected_ids": sorted(protected_ids),
            "protected_action_pair": list(protected_action_pair or ()),
            "decisions": [item.to_dict() for item in decisions],
            "stopped_candidate_ids": sorted(stopped_ids),
            "planned_fits_avoided": planned_fits_avoided,
            "task_budget_exhausted": task_budget_exhausted,
            "test_loaded": False,
        }
        return aggregated, seed_runs, racing

    def _ensemble_candidates(
        self,
        task: TaskSpec,
        benchmark_name: str,
        validation_seeds: list[int],
        base_results: list[RunResult],
        seed_runs: dict[str, dict[int, RunResult]],
        experiments: dict[str, ExperimentSpec],
        max_base_models: int = 6,
    ) -> tuple[list[RunResult], dict[str, dict[str, Any]]]:
        direction = 1 if task.maximize else -1
        ordered = sorted(
            base_results,
            key=lambda item: direction * item.validation_score,
            reverse=True,
        )[:max_base_models]
        if len(ordered) < 2:
            return [], {}
        targets = {}
        for seed in validation_seeds:
            train, valid = self.group.get_train_valid_split(
                benchmark=benchmark_name,
                split_type="default",
                seed=seed,
            )
            _, y_col = _columns(train)
            targets[seed] = valid[y_col].to_numpy()
        candidates: list[tuple[RunResult, dict[str, Any], dict[int, np.ndarray]]] = []
        rejected_evidence: list[dict[str, Any]] = []
        for left, right in itertools.combinations(ordered, 2):
            blend_recipes = [("probability", weight) for weight in (0.25, 0.5, 0.75)]
            if task.metric in {"auroc", "auprc", "spearman"}:
                blend_recipes.append(("rank", 0.5))
            for aggregation, left_weight in blend_recipes:
                blend_started = time.monotonic()
                scores = []
                left_scores = []
                right_scores = []
                predictions_by_seed = {}
                valid = True
                for seed in validation_seeds:
                    left_run = seed_runs[left.run_id].get(seed)
                    right_run = seed_runs[right.run_id].get(seed)
                    if (
                        left_run is None
                        or right_run is None
                        or left_run.status != "completed"
                        or right_run.status != "completed"
                    ):
                        valid = False
                        break
                    left_pred = np.load(
                        self.run_dir
                        / "artifacts"
                        / left_run.run_id
                        / "validation_predictions.npy",
                        allow_pickle=False,
                    )
                    right_pred = np.load(
                        self.run_dir
                        / "artifacts"
                        / right_run.run_id
                        / "validation_predictions.npy",
                        allow_pickle=False,
                    )
                    if aggregation == "rank":
                        left_pred = _percentile_ranks(left_pred)
                        right_pred = _percentile_ranks(right_pred)
                    blended = left_weight * left_pred + (1.0 - left_weight) * right_pred
                    predictions_by_seed[seed] = blended
                    left_scores.append(score(targets[seed], left_pred, task.metric))
                    right_scores.append(score(targets[seed], right_pred, task.metric))
                    scores.append(score(targets[seed], blended, task.metric))
                if not valid:
                    continue
                evidence = _ensemble_evidence(
                    scores,
                    left_scores,
                    right_scores,
                    maximize=task.maximize,
                )
                if not evidence["supported"]:
                    rejected_evidence.append(
                        {
                            "members": [left.run_id, right.run_id],
                            "weights": [left_weight, 1.0 - left_weight],
                            "aggregation": aggregation,
                            **evidence,
                        }
                    )
                    continue
                run_id = f"{task.name.lower()}-ensemble-{uuid.uuid4().hex[:8]}"
                member_specs = [
                    experiments[left.run_id].to_dict(),
                    experiments[right.run_id].to_dict(),
                ]
                frozen = {
                    "kind": "ensemble",
                    "task": task.name,
                    "members": member_specs,
                    "weights": [left_weight, 1.0 - left_weight],
                    "aggregation": aggregation,
                    "evidence": evidence,
                    "hypothesis": (
                        "equal-weight percentile-rank blend of complementary candidates"
                        if aggregation == "rank"
                        else "validation-selected probability blend of complementary candidates"
                    ),
                }
                result = RunResult(
                    run_id=run_id,
                    task=task.name,
                    seed=validation_seeds[0],
                    model="ensemble",
                    metric=task.metric,
                    validation_score=float(np.mean(scores)),
                    validation_std=float(np.std(scores)),
                    elapsed_seconds=time.monotonic() - blend_started,
                    representations=tuple(
                        dict.fromkeys(
                            experiments[left.run_id].representations
                            + experiments[right.run_id].representations
                        )
                    ),
                    hyperparameters={
                        "members": [left.run_id, right.run_id],
                        "weights": frozen["weights"],
                        "aggregation": aggregation,
                        "evidence": evidence,
                    },
                    hypothesis=frozen["hypothesis"],
                )
                candidates.append((result, frozen, predictions_by_seed))
        winners = _select_distinct_ensemble_winners(
            candidates,
            maximize=task.maximize,
            limit=3,
        )
        for result, frozen, predictions_by_seed in winners:
            artifact_dir = self.run_dir / "artifacts" / result.run_id
            artifact_dir.mkdir(parents=True)
            _write_json(artifact_dir / "ensemble.json", frozen)
            _write_json(artifact_dir / "result.json", result.to_dict())
            np.savez_compressed(
                artifact_dir / "validation_predictions.npz",
                **{
                    f"seed_{seed}": value for seed, value in predictions_by_seed.items()
                },
            )
            self.memory.append({**frozen, **result.to_dict()})
        self.evidence_ledger.append(
            "ensemble_screened",
            {
                "task": task.name,
                "accepted": len(candidates),
                "distinct_member_pairs": len(
                    {
                        tuple(sorted(str(value) for value in item[1]["members"]))
                        for item in candidates
                    }
                ),
                "persisted": len(winners),
                "rejected": len(rejected_evidence),
                "rejected_evidence": rejected_evidence,
                "minimum_effect": 0.005,
                "minimum_positive_rate": 0.8,
                "test_loaded": False,
            },
        )
        return [item[0] for item in winners], {
            item[0].run_id: item[1] for item in winners
        }

    def search(
        self,
        task_name: str,
        seed: int = 1,
        rounds: int = 1,
        validation_seeds: list[int] | None = None,
        robust_top_k: int = 6,
        ensembles: bool = True,
    ) -> dict:
        if rounds < 1:
            raise ValueError("rounds must be at least 1")
        validation_seeds = list(dict.fromkeys(validation_seeds or [seed]))
        if seed not in validation_seeds:
            validation_seeds.insert(0, seed)
        if any(value < 0 for value in validation_seeds):
            raise ValueError("validation seeds must be non-negative")
        task = resolve_task(task_name)
        forced_discovery_data_policy = self._protocol_discovery_data_policy()
        # Discovery deliberately uses only the split API.  group.get(), which
        # materialises the test frame in PyTDC, is certification-only below.
        name = task.name
        train, valid = self.group.get_train_valid_split(
            benchmark=name, split_type="default", seed=seed
        )
        smiles_col, y_col = _columns(train)
        profile = profile_task(train, valid, task, smiles_col, y_col)
        hardness = profile_hardness(train, valid, task, smiles_col, y_col, seed=seed)
        contract = endpoint_contract(task.name)
        benchmark_arm = getattr(self.brain, "benchmark_arm", "agent")
        task_compute_minutes = None
        if self.benchmark_protocol:
            arm = next(
                (
                    item
                    for item in self.benchmark_protocol.get("arms", [])
                    if item.get("name") == benchmark_arm
                ),
                None,
            )
            if arm is not None:
                task_compute_minutes = int(arm["max_compute_minutes_per_task"])
        task_deadline = (
            time.monotonic() + task_compute_minutes * 60
            if task_compute_minutes is not None
            else None
        )
        if benchmark_arm in {"agent_no_memory", "ecvoi_no_memory"}:
            action_priors = {}
        else:
            context = task_context_signature(hardness.to_dict(), contract.category)
            action_priors = self.routing_action_memory.priors(
                contract.category,
                context_signature=(
                    context if self.action_context_registry is not None else None
                ),
                context_registry=self.action_context_registry,
            )
        research_plan = self.research_planner.plan(
            hardness,
            effect_priority=self.effect_priority,
            action_priors=action_priors,
        )
        fault_report = self.fault_detector.detect(hardness, contract)
        task_dir = self.run_dir / "tasks" / task.name
        task_dir.mkdir(parents=True, exist_ok=True)
        _write_json(task_dir / f"profile_seed_{seed}.json", profile.to_dict())
        _write_json(task_dir / "endpoint_contract.json", contract.to_dict())
        _write_json(task_dir / f"hardness_seed_{seed}.json", hardness.to_dict())
        _write_json(task_dir / f"research_plan_seed_{seed}.json", research_plan)
        _write_json(task_dir / f"fault_report_seed_{seed}.json", fault_report.to_dict())
        self.evidence_ledger.append(
            "research_plan_created",
            {
                "task": task.name,
                "endpoint_contract": contract.to_dict(),
                "hardness": hardness.to_dict(),
                "research_plan": research_plan,
                "fault_report": fault_report.to_dict(),
            },
        )
        cross_task_evidence = self.database.cross_task_insights(
            profile.task_type,
            profile.metric,
            exclude_task=task.name,
        )
        set_context = getattr(self.brain, "set_context", None)
        if callable(set_context):
            try:
                set_context(
                    cross_task_evidence=cross_task_evidence,
                    hardness_card=hardness.to_dict(),
                    research_plan=research_plan,
                )
            except TypeError:
                set_context(cross_task_evidence=cross_task_evidence)
        effective_plan = getattr(self.brain, "effective_research_plan", None)
        if isinstance(effective_plan, dict) and effective_plan != research_plan:
            research_plan = deepcopy(effective_plan)
            _write_json(
                task_dir / f"effective_research_plan_seed_{seed}.json",
                research_plan,
            )
            self.evidence_ledger.append(
                "ablation_research_plan_created",
                {
                    "task": task.name,
                    "benchmark_arm": benchmark_arm,
                    "research_plan": research_plan,
                    "test_loaded": False,
                },
            )

        data_skill_route = self._route_data_skill(task.name, research_plan)
        _write_json(task_dir / f"data_skill_route_seed_{seed}.json", data_skill_route)
        self.evidence_ledger.append(
            "data_skill_routed",
            {
                "task": task.name,
                "benchmark_arm": benchmark_arm,
                "data_skill_route": data_skill_route,
                "test_loaded": False,
            },
        )

        if self.research_routing_policy in ECVOI_V2_ROUTING_POLICIES:
            return self._run_ecvoi_v2_search(
                task=task,
                benchmark_name=name,
                profile=profile,
                hardness=hardness,
                contract=contract,
                research_plan=research_plan,
                task_dir=task_dir,
                validation_seeds=validation_seeds,
                task_compute_minutes=task_compute_minutes,
                benchmark_arm=benchmark_arm,
            )

        shared_anchor_execution = self._execute_shared_anchor(
            task=task,
            benchmark_name=name,
            validation_seeds=validation_seeds,
            task_dir=task_dir,
        )
        shared_anchor_aggregate = (
            shared_anchor_execution["aggregate"]
            if shared_anchor_execution is not None
            else None
        )
        if shared_anchor_execution is not None and shared_anchor_aggregate is None:
            raise RuntimeError("Protocol-required shared strong anchor did not complete")
        # The common anchor is a finalist fallback, not an adaptive proposal.
        # Keeping it outside planner history prevents single-model planners
        # from treating a frozen ensemble as an Optuna-searchable estimator.
        history: list[RunResult] = []
        experiments: dict[str, ExperimentSpec] = {}
        signatures: set[str] = set()
        planner_errors: list[str] = []
        stopping_decisions: list[dict[str, Any]] = []
        planning_decisions: list[dict[str, Any]] = []
        for round_index in range(rounds):
            try:
                proposals = self.brain.propose(profile, history)
            except Exception as exc:
                planner_errors.append(
                    f"round {round_index + 1}: {type(exc).__name__}: {exc}"
                )
                if history:
                    break
                raise
            drain_errors = getattr(self.brain, "drain_errors", None)
            if callable(drain_errors):
                planner_errors.extend(drain_errors())
            arbitration = getattr(self.brain, "last_decision", None)
            if arbitration is not None and callable(
                getattr(arbitration, "to_dict", None)
            ):
                decision = {
                    "round": round_index + 1,
                    **arbitration.to_dict(),
                    "test_loaded": False,
                }
                planning_decisions.append(decision)
                self.evidence_ledger.append("committee_arbitrated", decision)
            if not proposals:
                planner_errors.append(
                    f"round {round_index + 1}: planner returned no new experiments"
                )
                break
            candidate_limit = min(item.budget.max_candidates for item in proposals)
            if len(proposals) > candidate_limit:
                planner_errors.append(
                    f"round {round_index + 1}: {len(proposals)} candidates "
                    f"exceeded budget {candidate_limit}"
                )
                proposals = proposals[:candidate_limit]
            deadline = (
                time.monotonic()
                + min(item.budget.max_minutes for item in proposals) * 60
            )
            if task_deadline is not None:
                deadline = min(deadline, task_deadline)
            executed = 0
            for proposed in proposals:
                if time.monotonic() >= deadline:
                    break
                experiment = ExperimentSpec.from_dict(
                    {
                        **proposed.to_dict(),
                        "task": task.name,
                        "seed": seed,
                        "data_policy": (
                            forced_discovery_data_policy
                            or proposed.data_policy
                        ),
                        "device": self.device,
                        "n_jobs": self.n_jobs,
                    }
                )
                signature = _experiment_signature(experiment)
                if signature in signatures:
                    continue
                signatures.add(signature)
                try:
                    validate_experiment(experiment)
                except Exception as exc:
                    planner_errors.append(
                        f"round {round_index + 1} rejected candidate: "
                        f"{type(exc).__name__}: {exc}"
                    )
                    continue
                remaining_seconds = max(0.0, deadline - time.monotonic())
                stop_reason = _scaleup_stop_reason(
                    experiment,
                    history,
                    experiments,
                    remaining_seconds,
                )
                if stop_reason is not None:
                    decision = {
                        "round": round_index + 1,
                        "experiment": experiment.to_dict(),
                        "reason": stop_reason,
                        "test_loaded": False,
                    }
                    stopping_decisions.append(decision)
                    self.evidence_ledger.append("candidate_soft_stopped", decision)
                    continue
                result = self._execute(
                    experiment,
                    train,
                    valid,
                    smiles_col,
                    y_col,
                    profile,
                    timeout_seconds=max(1.0, deadline - time.monotonic()),
                )
                history.append(result)
                experiments[result.run_id] = experiment
                executed += 1
            if executed == 0:
                break

        completed = [
            item
            for item in history
            if item.status == "completed" and item.run_id in experiments
        ]
        robust_results, seed_runs, robust_racing = self._robust_results(
            task,
            name,
            seed,
            validation_seeds,
            completed,
            experiments,
            history,
            robust_top_k,
            research_plan,
            task_deadline,
        )
        self.evidence_ledger.append("robust_racing_decided", robust_racing)
        action_evidence = evaluate_action_evidence(
            research_plan,
            task,
            experiments,
            seed_runs,
            preferred_pair=(
                tuple(robust_racing["protected_action_pair"])
                if len(robust_racing.get("protected_action_pair", [])) == 2
                else None
            ),
        )
        self.evidence_ledger.append(
            "research_action_evaluated",
            {
                "task": task.name,
                "seed": seed,
                "evidence": action_evidence.to_dict(),
                "test_loaded": False,
            },
        )
        self.action_memory.append(
            task=task.name,
            category=contract.category,
            evidence=action_evidence.to_dict(),
        )
        max_actions = 2
        if self.benchmark_protocol:
            arm = next(
                (
                    item
                    for item in self.benchmark_protocol.get("arms", [])
                    if item.get("name") == benchmark_arm
                ),
                None,
            )
            if arm is not None:
                max_actions = int(arm.get("max_actions_per_task", max_actions))
        corrective_decision = self.self_correction_planner.decide(
            research_plan,
            action_evidence,
            attempted_action_ids=(action_evidence.action_id,),
            remaining_action_budget=max(0, max_actions - 1),
            sequence_index=2,
        )
        _write_json(
            task_dir / f"corrective_decision_seed_{seed}.json",
            corrective_decision.to_dict(),
        )
        self.evidence_ledger.append(
            (
                "corrective_action_scheduled"
                if corrective_decision.requires_execution
                else "research_episode_stopped"
            ),
            {
                "task": task.name,
                "seed": seed,
                "decision": corrective_decision.to_dict(),
                "test_loaded": False,
            },
        )
        finalist_specs: dict[str, dict[str, Any]] = {
            item.run_id: experiments[item.run_id].to_dict() for item in robust_results
        }
        if shared_anchor_aggregate is not None and shared_anchor_execution is not None:
            finalist_specs[shared_anchor_aggregate.run_id] = (
                shared_anchor_execution["frozen_recipe"]
            )
        ensemble_results: list[RunResult] = []
        if ensembles:
            ensemble_results, ensemble_specs = self._ensemble_candidates(
                task,
                name,
                validation_seeds,
                robust_results,
                seed_runs,
                experiments,
            )
            finalist_specs.update(ensemble_specs)
        selectable = (
            ([shared_anchor_aggregate] if shared_anchor_aggregate is not None else [])
            + robust_results
            + ensemble_results
        )
        finalist_run_id, routing_decision = self._select_finalist(profile, selectable)
        finalist_spec = (
            finalist_specs[finalist_run_id] if finalist_run_id is not None else None
        )
        summary = {
            "version": ("2.4-routing" if self.selection_policy != "score" else "2.1"),
            "profile": profile.to_dict(),
            "endpoint_contract": contract.to_dict(),
            "hardness": hardness.to_dict(),
            "research_plan": research_plan,
            "data_skill_route": data_skill_route,
            "action_evidence": action_evidence.to_dict(),
            "corrective_decision": corrective_decision.to_dict(),
            "fault_report": fault_report.to_dict(),
            "rounds_requested": rounds,
            "task_compute_budget_minutes": task_compute_minutes,
            "validation_seeds": validation_seeds,
            "cross_task_evidence": cross_task_evidence,
            "results": [item.to_dict() for item in history],
            "robust_results": [item.to_dict() for item in robust_results],
            "robust_racing": robust_racing,
            "ensemble_results": [item.to_dict() for item in ensemble_results],
            "shared_strong_anchor": (
                shared_anchor_execution["summary"]
                if shared_anchor_execution is not None
                else None
            ),
            "selection_policy": self.selection_policy,
            "research_routing_policy": self.research_routing_policy,
            "action_memory_snapshot_sha256": self.action_memory_snapshot_sha256,
            "ecvoi_v2_memory_snapshot_sha256": (
                self.ecvoi_v2_memory_snapshot_sha256
            ),
            "action_context_registry_sha256": self.action_context_registry_sha256,
            "benchmark_arm": benchmark_arm,
            "benchmark_protocol_sha256": (
                self.benchmark_protocol.get("sha256")
                if self.benchmark_protocol
                else None
            ),
            "benchmark_protocol_version": (
                self.benchmark_protocol.get("protocol_version")
                if self.benchmark_protocol
                else None
            ),
            "source_sha256": self.source_manifest["source_sha256"],
            "artifact_policy": {
                "discovery_models_persisted": self.persist_discovery_models,
                "feature_cache_dir": str(self.feature_cache_dir),
                "validation_predictions_persisted": True,
            },
            "routing_decision": routing_decision,
            "planner_errors": planner_errors,
            "planning_decisions": planning_decisions,
            "stopping_decisions": stopping_decisions,
            "finalist_run_id": finalist_run_id,
            "finalist_spec": finalist_spec,
            "database": self.database.summary(),
        }
        summary_path = task_dir / f"search_seed_{seed}.json"
        _write_json(summary_path, summary)
        if finalist_spec is not None:
            _write_json(
                task_dir / f"finalist_seed_{seed}.json",
                {task.name: finalist_spec},
            )
        return summary

    def diagnose(self, task_name: str, seed: int = 1) -> dict[str, Any]:
        """Build a discovery-only endpoint diagnosis without training models."""
        if seed < 0:
            raise ValueError("seed must be non-negative")
        task = resolve_task(task_name)
        train, valid = self.group.get_train_valid_split(
            benchmark=task.name, split_type="default", seed=seed
        )
        smiles_col, y_col = _columns(train)
        profile = profile_task(train, valid, task, smiles_col, y_col)
        hardness = profile_hardness(train, valid, task, smiles_col, y_col, seed=seed)
        contract = endpoint_contract(task.name)
        action_memory = getattr(
            self,
            "routing_action_memory",
            getattr(self, "action_memory", None),
        )
        context_registry = getattr(self, "action_context_registry", None)
        action_priors = (
            action_memory.priors(
                contract.category,
                context_signature=(
                    task_context_signature(hardness.to_dict(), contract.category)
                    if context_registry is not None
                    else None
                ),
                context_registry=context_registry,
            )
            if action_memory is not None
            else {}
        )
        plan = self.research_planner.plan(
            hardness,
            effect_priority=self.effect_priority,
            action_priors=action_priors,
        )
        data_skill_route = self._route_data_skill(task.name, plan)
        fault_report = self.fault_detector.detect(hardness, contract)
        packet = {
            "protocol": "discovery_only_no_test_access",
            "task": task.name,
            "seed": seed,
            "endpoint_contract": contract.to_dict(),
            "profile": profile.to_dict(),
            "hardness": hardness.to_dict(),
            "research_plan": plan,
            "data_skill_route": data_skill_route,
            "fault_report": fault_report.to_dict(),
        }
        path = self.run_dir / "tasks" / task.name / f"diagnosis_seed_{seed}.json"
        _write_json(path, packet)
        self.evidence_ledger.append("endpoint_diagnosed", packet)
        return packet

    def _select_finalist(
        self,
        profile: TaskProfile,
        selectable: list[RunResult],
    ) -> tuple[str | None, dict[str, Any] | None]:
        if not selectable:
            return None, None
        if self.selection_policy == "score":
            return self.brain.select_finalist(profile, selectable), None
        decision = self.evidence_router.select(profile, selectable)
        return decision["selected_run_id"], decision

    @staticmethod
    def _normalize_frozen(task: TaskSpec, frozen: object) -> dict[str, Any]:
        if isinstance(frozen, str):
            frozen = {
                "model": frozen,
                "representations": ["morgan", "avalon", "erg", "rdkit2d"],
            }
        if not isinstance(frozen, dict):
            raise ValueError(
                f"Finalist for {task.name} must be a model name or experiment object"
            )
        if frozen.get("kind") == "recipe":
            recipe = frozen.get("recipe")
            if not isinstance(recipe, str):
                raise ValueError(
                    f"Recipe finalist for {task.name} requires a recipe string"
                )
            build_execution_plan(task, recipe)
            protocol = {
                "model_size": "84m",
                "learning_rate": 1e-4,
                "epochs": 20,
                "batch_size": 16,
                "inner_folds": 5,
                **frozen.get("protocol", {}),
            }
            if protocol["model_size"] != "84m":
                raise ValueError("Frozen v2.4 recipes require Uni-Mol2 84m")
            for key in ("epochs", "batch_size", "inner_folds"):
                if isinstance(protocol[key], bool) or not isinstance(
                    protocol[key], int
                ):
                    raise ValueError(f"Frozen recipe protocol {key} must be an integer")
                if protocol[key] <= 0:
                    raise ValueError(f"Frozen recipe protocol {key} must be positive")
            if not isinstance(protocol["learning_rate"], (int, float)):
                raise ValueError("Frozen recipe learning_rate must be numeric")
            if not 0 < float(protocol["learning_rate"]) <= 1:
                raise ValueError("Frozen recipe learning_rate must be in (0, 1]")
            return {
                "kind": "recipe",
                "task": task.name,
                "recipe": recipe,
                "protocol": protocol,
                "hypothesis": frozen.get(
                    "hypothesis", "preregistered heterogeneous frozen recipe"
                ),
            }
        if frozen.get("kind") == "ensemble":
            members = frozen.get("members")
            weights = frozen.get("weights")
            aggregation = frozen.get("aggregation", "probability")
            if (
                not isinstance(members, list)
                or len(members) < 2
                or not isinstance(weights, list)
                or len(weights) != len(members)
            ):
                raise ValueError(f"Invalid ensemble finalist for {task.name}")
            if any(
                not isinstance(value, (int, float)) or value < 0 for value in weights
            ):
                raise ValueError("Ensemble weights must be non-negative numbers")
            total = float(sum(weights))
            if total <= 0:
                raise ValueError("Ensemble weights must have a positive sum")
            if aggregation not in {"probability", "rank"}:
                raise ValueError("Ensemble aggregation must be probability or rank")
            if aggregation == "rank" and task.metric == "mae":
                raise ValueError("Rank aggregation is not valid for MAE tasks")
            normalized_members = []
            for member in members:
                experiment = ExperimentSpec.from_dict(
                    {**member, "task": task.name, "seed": 0}
                )
                validate_experiment(experiment)
                normalized_members.append(experiment.to_dict())
            return {
                "kind": "ensemble",
                "task": task.name,
                "members": normalized_members,
                "weights": [float(value) / total for value in weights],
                "aggregation": aggregation,
                "hypothesis": frozen.get("hypothesis", "frozen ensemble"),
            }
        experiment = ExperimentSpec.from_dict(
            {
                **frozen,
                "task": task.name,
                "seed": 0,
                "hypothesis": frozen.get("hypothesis", "frozen finalist"),
            }
        )
        validate_experiment(experiment)
        return {"kind": "single", **experiment.to_dict()}

    def _predict_frozen(
        self,
        task: TaskSpec,
        frozen: dict[str, Any],
        seed: int,
        train,
        valid,
        test,
        smiles_col: str,
        y_col: str,
    ) -> np.ndarray:
        joined = pd.concat([train, valid], ignore_index=True)
        if frozen["kind"] == "recipe":
            executor = FrozenRecipeExecutor(
                cache=self.cache,
                artifact_dir=(
                    self.run_dir / "final" / "artifacts" / task.name / f"seed_{seed}"
                ),
                device=self.device,
                n_jobs=self.n_jobs,
            )
            return executor.predict(
                task=task,
                recipe_value=frozen["recipe"],
                seed=seed,
                fit_frame=joined,
                predict_smiles=test[smiles_col].astype(str).tolist(),
                smiles_col=smiles_col,
                y_col=y_col,
                protocol=frozen["protocol"],
            )
        if frozen["kind"] == "single":
            members = [{key: value for key, value in frozen.items() if key != "kind"}]
            weights = [1.0]
        else:
            members = frozen["members"]
            weights = frozen["weights"]
        aggregation = frozen.get("aggregation", "probability")
        combined = None
        for member, weight in zip(members, weights):
            experiment = ExperimentSpec.from_dict(
                {
                    **member,
                    "task": task.name,
                    "seed": seed,
                    "device": self.device,
                    "n_jobs": self.n_jobs,
                    "hypothesis": member.get("hypothesis", "frozen ensemble member"),
                }
            )
            validate_experiment(experiment)
            fit_frame = joined
            if experiment.data_policy != "official":
                fit_frame, audit = MolecularDataCurator(smiles_col, y_col).curate(
                    joined,
                    task_type=task.task_type,
                    policy=experiment.data_policy,
                )
                audit_path = (
                    self.run_dir
                    / "final"
                    / "artifacts"
                    / task.name
                    / f"seed_{seed}"
                    / f"{experiment.model}_data_audit.json"
                )
                _write_json(audit_path, audit.to_dict())
            x_fit = self.cache.get_or_compute(
                fit_frame[smiles_col].tolist(), experiment.representations
            )
            x_test = self.cache.get_or_compute(
                test[smiles_col].tolist(), experiment.representations
            )
            model = build_model(experiment, task, fit_frame[y_col].to_numpy())
            model.fit(x_fit, fit_frame[y_col].to_numpy())
            member_predictions = predict(model, x_test, task)
            if aggregation == "rank":
                member_predictions = _percentile_ranks(member_predictions)
            weighted = float(weight) * member_predictions
            combined = weighted if combined is None else combined + weighted
        if combined is None:
            raise RuntimeError("Frozen finalist produced no predictions")
        return np.asarray(combined)

    def _lock_finalists(
        self,
        task_names: list[str],
        normalized: dict[str, dict[str, Any]],
    ) -> dict[str, Any]:
        payload = {
            "protocol": "tdc-admet-test-v1",
            "tasks": sorted(task_names),
            "finalists": {name: normalized[name] for name in sorted(normalized)},
        }
        digest = hashlib.sha256(
            json.dumps(payload, sort_keys=True).encode()
        ).hexdigest()
        lock = {"sha256": digest, **payload}
        lock_path = self.run_dir / "final" / "freeze.json"
        if lock_path.exists():
            existing = json.loads(lock_path.read_text(encoding="utf-8"))
            if existing.get("sha256") != digest:
                raise RuntimeError(
                    "Test firewall refused a different finalist configuration in "
                    f"{self.run_dir}; use a new run directory for a new frozen version"
                )
        else:
            _write_json(lock_path, lock)
        return lock

    def finalize(
        self,
        task_names: list[str],
        seeds: list[int],
        finalists: dict[str, object],
    ) -> dict:
        if len(set(seeds)) < 5:
            raise ValueError(
                "Official TDC finalization requires at least five unique seeds"
            )
        if any(seed < 0 for seed in seeds):
            raise ValueError("Finalization seeds must be non-negative")
        normalized = {
            resolve_task(name).name: self._normalize_frozen(
                resolve_task(name), finalists[resolve_task(name).name]
            )
            for name in task_names
        }
        freeze = self._lock_finalists(task_names, normalized)
        self.evidence_ledger.append(
            "frozen_test_certification_started",
            {
                "tasks": sorted(resolve_task(name).name for name in task_names),
                "seeds": sorted(set(seeds)),
                "freeze_sha256": freeze["sha256"],
                "feedback_policy": "no_test_feedback_to_discovery",
            },
        )
        all_predictions = []
        audits = []
        for seed in seeds:
            seed_predictions = {}
            for requested in task_names:
                task = resolve_task(requested)
                benchmark = self.group.get(task.name)
                name = benchmark["name"]
                train, valid = self.group.get_train_valid_split(
                    benchmark=name, split_type="default", seed=seed
                )
                test = benchmark["test"]
                smiles_col, y_col = _columns(train)
                frozen = normalized[task.name]
                seed_predictions[name] = self._predict_frozen(
                    task,
                    frozen,
                    seed,
                    train,
                    valid,
                    test,
                    smiles_col,
                    y_col,
                )
                audits.append(
                    {
                        "task": name,
                        "seed": seed,
                        "frozen_finalist": frozen,
                        "freeze_sha256": freeze["sha256"],
                    }
                )
            all_predictions.append(seed_predictions)
        results = self.group.evaluate_many(all_predictions)
        payload = {
            "version": "2.1",
            "seeds": seeds,
            "results": results,
            "audit": audits,
            "finalists": normalized,
            "freeze_sha256": freeze["sha256"],
        }
        final_dir = self.run_dir / "final"
        _write_json(final_dir / "results.json", payload)
        for seed, predictions in zip(seeds, all_predictions):
            np.savez_compressed(
                final_dir / f"predictions_seed_{seed}.npz", **predictions
            )
        self.evidence_ledger.append(
            "frozen_test_certification_completed",
            {
                "freeze_sha256": freeze["sha256"],
                "results_path": str(final_dir / "results.json"),
            },
        )
        return payload

    def environment_manifest(self) -> None:
        try:
            unimol_tools_version = version("unimol-tools")
        except PackageNotFoundError:
            unimol_tools_version = None
        _write_json(
            self.run_dir / "environment.json",
            {
                "agent_version": "3.0-research-loop",
                "python": sys.version,
                "platform": platform.platform(),
                "device": self.device,
                "n_jobs": self.n_jobs,
                "gpus": self._gpu_manifest(),
                "unimol_tools": unimol_tools_version,
                "selection": {
                    "policy": self.selection_policy,
                    "research_routing_policy": self.research_routing_policy,
                    "uncertainty_penalty": self.evidence_router.uncertainty_penalty,
                    "cost_penalty": self.evidence_router.cost_penalty,
                    "effect_priority": self.effect_priority,
                },
                "action_memory_snapshot": (
                    {
                        "path": str(self.action_memory_snapshot_path),
                        "sha256": self.action_memory_snapshot_sha256,
                        "read_only_for_routing": True,
                    }
                    if self.action_memory_snapshot_path is not None
                    else None
                ),
                "action_context_registry": (
                    {
                        "path": str(self.action_context_registry_path),
                        "sha256": self.action_context_registry_sha256,
                        "test_loaded": False,
                    }
                    if self.action_context_registry_path is not None
                    else None
                ),
                "benchmark_protocol_sha256": (
                    self.benchmark_protocol.get("sha256")
                    if self.benchmark_protocol
                    else None
                ),
                "benchmark_protocol_version": (
                    self.benchmark_protocol.get("protocol_version")
                    if self.benchmark_protocol
                    else None
                ),
                "source_sha256": self.source_manifest["source_sha256"],
                "source_snapshot": "source_snapshot/manifest.json",
            },
        )


def all_task_names() -> list[str]:
    return [spec.name for spec in TASKS.values()]
