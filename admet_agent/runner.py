"""TDC orchestration for validation-only evolution and frozen test evaluation."""

from __future__ import annotations

import hashlib
import itertools
import json
import platform
import shutil
import subprocess
import sys
import time
import uuid
from dataclasses import replace
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from tdc.benchmark_group import admet_group

from .brain import AgentBrain, RuleBasedBrain
from .experiment_db import ExperimentDatabase
from .features import FeatureCache
from .guard import validate_experiment
from .memory import ExperimentMemory
from .profiling import profile_task
from .schemas import ExperimentSpec, RunResult, TaskProfile
from .selection import build_model, predict, score
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
            "hyperparameters": experiment.hyperparameters,
        },
        sort_keys=True,
    )


class ADMETAgent:
    def __init__(
        self,
        data_dir: Path,
        run_dir: Path,
        brain: AgentBrain | None = None,
        device: str = "auto",
        n_jobs: int = 8,
    ):
        self.group = admet_group(path=str(data_dir))
        self.device = self._resolve_device(device)
        if isinstance(n_jobs, bool) or not isinstance(n_jobs, int) or not 1 <= n_jobs <= 256:
            raise ValueError("n_jobs must be an integer between 1 and 256")
        self.n_jobs = n_jobs
        self.run_dir = run_dir
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.cache = FeatureCache(run_dir / "feature_cache")
        self.memory = ExperimentMemory(run_dir / "memory.jsonl")
        self.database = ExperimentDatabase(run_dir / "experiments.sqlite")
        self.brain = brain or RuleBasedBrain()

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
        return "cuda" if requested == "cuda" or (requested == "auto" and available) else "cpu"

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
        return [
            line.strip() for line in probe.stdout.splitlines() if line.strip()
        ] if probe.returncode == 0 else []

    def _execute(
        self,
        experiment: ExperimentSpec,
        train,
        valid,
        smiles_col: str,
        y_col: str,
        profile: TaskProfile,
    ) -> RunResult:
        validate_experiment(experiment)
        started = time.monotonic()
        run_id = f"{experiment.task.lower()}-{experiment.seed}-{uuid.uuid4().hex[:8]}"
        artifact_dir = self.run_dir / "artifacts" / run_id
        artifact_dir.mkdir(parents=True)
        _write_json(artifact_dir / "experiment.json", experiment.to_dict())
        try:
            combined_smiles = (
                train[smiles_col].tolist() + valid[smiles_col].tolist()
            )
            x_combined = self.cache.get_or_compute(
                combined_smiles, experiment.representations
            )
            x_train = x_combined[: len(train)]
            x_valid = x_combined[len(train):]
            y_train = train[y_col].to_numpy()
            y_valid = valid[y_col].to_numpy()
            task = resolve_task(experiment.task)
            model = build_model(experiment, task, y_train)
            model.fit(x_train, y_train)
            predictions = predict(model, x_valid, task)
            validation_score = score(y_valid, predictions, task.metric)
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
            )
        _write_json(artifact_dir / "result.json", result.to_dict())
        self.memory.append({**experiment.to_dict(), **result.to_dict()})
        self.database.record(profile, experiment, result)
        return result

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
    ) -> tuple[list[RunResult], dict[str, dict[int, RunResult]]]:
        direction = 1 if task.maximize else -1
        eligible = sorted(
            completed,
            key=lambda item: direction * item.validation_score,
            reverse=True,
        )
        if len(validation_seeds) > 1:
            eligible = eligible[: max(1, robust_top_k)]
        seed_runs: dict[str, dict[int, RunResult]] = {
            item.run_id: {primary_seed: item} for item in eligible
        }
        for seed in validation_seeds:
            if seed == primary_seed:
                continue
            train, valid = self.group.get_train_valid_split(
                benchmark=benchmark_name,
                split_type="default",
                seed=seed,
            )
            smiles_col, y_col = _columns(train)
            profile = profile_task(train, valid, task, smiles_col, y_col)
            for original in eligible:
                experiment = ExperimentSpec.from_dict(
                    {**experiments[original.run_id].to_dict(), "seed": seed, "device": self.device, "n_jobs": self.n_jobs}
                )
                repeated = self._execute(
                    experiment, train, valid, smiles_col, y_col, profile
                )
                history.append(repeated)
                seed_runs[original.run_id][seed] = repeated
        aggregated = []
        for original in eligible:
            repeats = [
                seed_runs[original.run_id].get(seed)
                for seed in validation_seeds
            ]
            if any(item is None or item.status != "completed" for item in repeats):
                continue
            values = [item.validation_score for item in repeats if item is not None]
            aggregated.append(
                replace(
                    original,
                    validation_score=float(np.mean(values)),
                    validation_std=float(np.std(values)),
                    hypothesis=(
                        f"{original.hypothesis}; robust validation over "
                        f"{len(values)} seed(s)"
                    ),
                )
            )
        return aggregated, seed_runs

    def _ensemble_candidates(
        self,
        task: TaskSpec,
        benchmark_name: str,
        validation_seeds: list[int],
        base_results: list[RunResult],
        seed_runs: dict[str, dict[int, RunResult]],
        experiments: dict[str, ExperimentSpec],
        max_base_models: int = 4,
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
        candidates: list[tuple[RunResult, dict[str, Any]]] = []
        for left, right in itertools.combinations(ordered, 2):
            for left_weight in (0.25, 0.5, 0.75):
                scores = []
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
                    blended = left_weight * left_pred + (1.0 - left_weight) * right_pred
                    predictions_by_seed[seed] = blended
                    scores.append(score(targets[seed], blended, task.metric))
                if not valid:
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
                    "hypothesis": "validation-selected blend of complementary candidates",
                }
                result = RunResult(
                    run_id=run_id,
                    task=task.name,
                    seed=validation_seeds[0],
                    model="ensemble",
                    metric=task.metric,
                    validation_score=float(np.mean(scores)),
                    validation_std=float(np.std(scores)),
                    elapsed_seconds=0.0,
                    representations=tuple(
                        dict.fromkeys(
                            experiments[left.run_id].representations
                            + experiments[right.run_id].representations
                        )
                    ),
                    hyperparameters={
                        "members": [left.run_id, right.run_id],
                        "weights": frozen["weights"],
                    },
                    hypothesis=frozen["hypothesis"],
                )
                artifact_dir = self.run_dir / "artifacts" / run_id
                artifact_dir.mkdir(parents=True)
                _write_json(artifact_dir / "ensemble.json", frozen)
                _write_json(artifact_dir / "result.json", result.to_dict())
                np.savez_compressed(
                    artifact_dir / "validation_predictions.npz",
                    **{f"seed_{seed}": value for seed, value in predictions_by_seed.items()},
                )
                candidates.append((result, frozen))
        candidates.sort(
            key=lambda item: direction * item[0].validation_score,
            reverse=True,
        )
        winners = candidates[:3]
        for result, frozen in winners:
            self.memory.append({**frozen, **result.to_dict()})
        return [item[0] for item in winners], {
            item[0].run_id: item[1] for item in winners
        }

    def search(
        self,
        task_name: str,
        seed: int = 1,
        rounds: int = 1,
        validation_seeds: list[int] | None = None,
        robust_top_k: int = 3,
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
        benchmark = self.group.get(task.name)
        name = benchmark["name"]
        train, valid = self.group.get_train_valid_split(
            benchmark=name, split_type="default", seed=seed
        )
        smiles_col, y_col = _columns(train)
        profile = profile_task(train, valid, task, smiles_col, y_col)
        task_dir = self.run_dir / "tasks" / task.name
        task_dir.mkdir(parents=True, exist_ok=True)
        _write_json(task_dir / f"profile_seed_{seed}.json", profile.to_dict())

        cross_task_evidence = self.database.cross_task_insights(
            profile.task_type,
            profile.metric,
            exclude_task=task.name,
        )
        set_context = getattr(self.brain, "set_context", None)
        if callable(set_context):
            set_context(cross_task_evidence=cross_task_evidence)

        history: list[RunResult] = []
        experiments: dict[str, ExperimentSpec] = {}
        signatures: set[str] = set()
        planner_errors: list[str] = []
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
            executed = 0
            for proposed in proposals:
                if time.monotonic() >= deadline:
                    break
                experiment = ExperimentSpec.from_dict(
                    {**proposed.to_dict(), "task": task.name, "seed": seed, "device": self.device, "n_jobs": self.n_jobs}
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
                result = self._execute(
                    experiment, train, valid, smiles_col, y_col, profile
                )
                history.append(result)
                experiments[result.run_id] = experiment
                executed += 1
            if executed == 0:
                break

        completed = [item for item in history if item.status == "completed"]
        robust_results, seed_runs = self._robust_results(
            task,
            name,
            seed,
            validation_seeds,
            completed,
            experiments,
            history,
            robust_top_k,
        )
        finalist_specs: dict[str, dict[str, Any]] = {
            item.run_id: experiments[item.run_id].to_dict()
            for item in robust_results
        }
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
        selectable = robust_results + ensemble_results
        finalist_run_id = (
            self.brain.select_finalist(profile, selectable) if selectable else None
        )
        finalist_spec = (
            finalist_specs[finalist_run_id] if finalist_run_id is not None else None
        )
        summary = {
            "version": "2.1",
            "profile": profile.to_dict(),
            "rounds_requested": rounds,
            "validation_seeds": validation_seeds,
            "cross_task_evidence": cross_task_evidence,
            "results": [item.to_dict() for item in history],
            "robust_results": [item.to_dict() for item in robust_results],
            "ensemble_results": [item.to_dict() for item in ensemble_results],
            "planner_errors": planner_errors,
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
        if frozen.get("kind") == "ensemble":
            members = frozen.get("members")
            weights = frozen.get("weights")
            if (
                not isinstance(members, list)
                or len(members) < 2
                or not isinstance(weights, list)
                or len(weights) != len(members)
            ):
                raise ValueError(f"Invalid ensemble finalist for {task.name}")
            if any(not isinstance(value, (int, float)) or value < 0 for value in weights):
                raise ValueError("Ensemble weights must be non-negative numbers")
            total = float(sum(weights))
            if total <= 0:
                raise ValueError("Ensemble weights must have a positive sum")
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
        if frozen["kind"] == "single":
            members = [
                {key: value for key, value in frozen.items() if key != "kind"}
            ]
            weights = [1.0]
        else:
            members = frozen["members"]
            weights = frozen["weights"]
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
            x_fit = self.cache.get_or_compute(
                joined[smiles_col].tolist(), experiment.representations
            )
            x_test = self.cache.get_or_compute(
                test[smiles_col].tolist(), experiment.representations
            )
            model = build_model(experiment, task, joined[y_col].to_numpy())
            model.fit(x_fit, joined[y_col].to_numpy())
            member_predictions = predict(model, x_test, task)
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
            raise ValueError("Official TDC finalization requires at least five unique seeds")
        if any(seed < 0 for seed in seeds):
            raise ValueError("Finalization seeds must be non-negative")
        normalized = {
            resolve_task(name).name: self._normalize_frozen(
                resolve_task(name), finalists[resolve_task(name).name]
            )
            for name in task_names
        }
        freeze = self._lock_finalists(task_names, normalized)
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
        return payload

    def environment_manifest(self) -> None:
        _write_json(
            self.run_dir / "environment.json",
            {
                "agent_version": "2.1",
                "python": sys.version,
                "platform": platform.platform(),
                "device": self.device,
                "n_jobs": self.n_jobs,
                "gpus": self._gpu_manifest(),
            },
        )


def all_task_names() -> list[str]:
    return [spec.name for spec in TASKS.values()]
