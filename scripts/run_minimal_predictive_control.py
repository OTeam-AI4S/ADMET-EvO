"""Run the frozen TDC-22 three-arm predictive control with exact-fit reuse."""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from admet_agent.profiling import profile_task
from admet_agent.runner import ADMETAgent
from admet_agent.schemas import Budget, ExperimentSpec, TaskProfile
from admet_agent.source_snapshot import production_source_manifest
from admet_agent.streamlined_task_policy import build_minimal_pair_specs
from admet_agent.tasks import resolve_task

RICH = ("morgan", "avalon", "erg", "rdkit2d")


def _digest(payload: Any) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode()
    ).hexdigest()


def _atomic_json(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def load_protocol(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    claimed = payload.pop("sha256", None)
    if claimed != _digest(payload):
        raise ValueError("Predictive-control protocol SHA-256 is invalid")
    if payload.get("test_loaded") is not False or payload.get("test_feedback_to_discovery") is not False:
        raise ValueError("Predictive control must not load or use test feedback")
    if len(payload.get("tasks", [])) != 22 or len(set(payload["tasks"])) != 22:
        raise ValueError("Predictive control requires exactly 22 unique tasks")
    if payload.get("discovery_seed") == payload.get("evaluation_seed"):
        raise ValueError("Discovery and evaluation seeds must differ")
    return {**payload, "sha256": claimed}


def _imbalance(profile: TaskProfile) -> str:
    rate = profile.positive_rate_train
    return (
        "class_weight"
        if profile.task_type == "classification" and rate is not None and not 0.3 <= rate <= 0.7
        else "none"
    )


def build_arm_specs(profile: TaskProfile, arm: str, seed: int, workers: int) -> dict[str, ExperimentSpec]:
    common = {
        "task": profile.task,
        "seed": seed,
        "data_policy": "official",
        "device": "cpu",
        "n_jobs": workers,
        "budget": Budget(max_candidates=2, max_minutes=30),
    }
    if arm == "streamlined_agent":
        control, treatment = build_minimal_pair_specs(profile)
        return {
            "control": replace(control, seed=seed, n_jobs=workers, budget=common["budget"]),
            "treatment": replace(treatment, seed=seed, n_jobs=workers, budget=common["budget"]),
        }
    if arm == "expanded_automl":
        return {
            "lightgbm": ExperimentSpec(
                **common, representations=RICH, model="lightgbm",
                imbalance=_imbalance(profile),
                hypothesis="scalar validation acquisition; no scientific hypothesis or causal attribution",
            ),
            "extra_trees": ExperimentSpec(
                **common, representations=RICH, model="extra_trees",
                imbalance=_imbalance(profile),
                hypothesis="scalar validation acquisition; no scientific hypothesis or causal attribution",
            ),
        }
    if arm == "strong_fixed":
        return {
            "fixed": ExperimentSpec(
                **{**common, "budget": Budget(max_candidates=1, max_minutes=30)},
                representations=RICH,
                model="lightgbm",
                imbalance=_imbalance(profile),
                hypothesis="preregistered strong fixed rich-2D LightGBM anchor",
            )
        }
    raise ValueError(f"Unknown predictive-control arm {arm!r}")


def scientific_fingerprint(spec: ExperimentSpec) -> str:
    return _digest(
        {
            "task": spec.task,
            "seed": spec.seed,
            "representations": spec.representations,
            "model": spec.model,
            "imbalance": spec.imbalance,
            "data_policy": spec.data_policy,
            "device": spec.device,
            "hyperparameters": spec.hyperparameters,
        }
    )


def choose_screen_winner(assignments: dict[str, dict], arm: str, task: str) -> str:
    rows = [
        row for row in assignments.values()
        if row["task"] == task and row["arm"] == arm and row["phase"] == "screen"
    ]
    if len(rows) != 2 or any(row["status"] != "completed" for row in rows):
        raise RuntimeError(f"Incomplete {arm} screen for {task}")
    direction = 1.0 if resolve_task(task).maximize else -1.0
    winner = max(
        rows,
        key=lambda row: (direction * float(row["result"]["validation_score"]), row["role"]),
    )
    return str(winner["role"])


def _columns(frame) -> tuple[str, str]:
    smiles = "Drug" if "Drug" in frame.columns else str(frame.columns[0])
    target = "Y" if "Y" in frame.columns else str(frame.columns[-1])
    return smiles, target


def _new_registry(protocol: dict, source_sha: str, runner_sha: str) -> dict:
    return {
        "version": "minimal-predictive-control-runner-v1",
        "protocol_sha256": protocol["sha256"],
        "source_sha256": source_sha,
        "runner_sha256": runner_sha,
        "test_loaded": False,
        "task_profiles": {},
        "physical_records": {},
        "assignments": {},
        "selections": {},
    }


def _run_assignment(
    *, registry: dict, registry_path: Path, agent: ADMETAgent, spec: ExperimentSpec,
    task: str, arm: str, phase: str, role: str, train, valid, smiles_col: str,
    target_col: str, profile: TaskProfile, timeout_seconds: float,
) -> bool:
    assignment_id = f"{task}|{arm}|{phase}|{role}"
    if assignment_id in registry["assignments"]:
        return registry["assignments"][assignment_id]["status"] == "completed"
    fingerprint = scientific_fingerprint(spec)
    physical = registry["physical_records"].get(fingerprint)
    if physical is None:
        result = agent._execute(
            spec, train, valid, smiles_col, target_col, profile,
            timeout_seconds=timeout_seconds,
        )
        physical = {
            "fingerprint": fingerprint,
            "spec": spec.to_dict(),
            "result": result.to_dict(),
            "test_loaded": False,
        }
        registry["physical_records"][fingerprint] = physical
    result = physical["result"]
    registry["assignments"][assignment_id] = {
        "assignment_id": assignment_id,
        "task": task,
        "arm": arm,
        "phase": phase,
        "role": role,
        "fingerprint": fingerprint,
        "status": result["status"],
        "result": result,
        "charged_fits": 1,
        "physical_fit_reused": sum(
            row["fingerprint"] == fingerprint for row in registry["assignments"].values()
        ) > 0,
        "test_loaded": False,
    }
    registry["updated_at"] = datetime.now(timezone.utc).isoformat()
    _atomic_json(registry_path, registry)
    print(
        f"{assignment_id}: {result['status']} "
        f"reused={registry['assignments'][assignment_id]['physical_fit_reused']}",
        flush=True,
    )
    return result["status"] == "completed"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, default=Path("runs/minimal-predictive-control-v1/protocol.json"))
    parser.add_argument("--run-dir", type=Path, default=Path("runs/minimal-predictive-control-v1"))
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--feature-cache-dir", type=Path, default=Path("cache/features"))
    parser.add_argument("--phase", choices=("screen", "evaluation", "all"), default="all")
    parser.add_argument("--task", action="append")
    parser.add_argument("--workers", type=int, default=8)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    protocol = load_protocol(args.protocol)
    source_sha = production_source_manifest()["source_sha256"]
    runner_sha = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    if source_sha != protocol["source_sha256"] or runner_sha != protocol["runner_sha256"]:
        raise RuntimeError("Frozen protocol source or runner hash mismatch")
    tasks = [resolve_task(task).name for task in (args.task or protocol["tasks"])]
    if not set(tasks).issubset(protocol["tasks"]):
        raise ValueError("Requested task is outside the protocol")

    args.run_dir.mkdir(parents=True, exist_ok=True)
    registry_path = args.run_dir / "registry.json"
    registry = (
        json.loads(registry_path.read_text(encoding="utf-8"))
        if registry_path.exists() else _new_registry(protocol, source_sha, runner_sha)
    )
    if registry["protocol_sha256"] != protocol["sha256"]:
        raise RuntimeError("Registry belongs to a different protocol")
    agent = ADMETAgent(
        data_dir=args.data_dir, run_dir=args.run_dir, device="cpu", n_jobs=args.workers,
        feature_cache_dir=args.feature_cache_dir, persist_discovery_models=False,
    )
    phases = ("screen", "evaluation") if args.phase == "all" else (args.phase,)
    failures = 0
    for task in tasks:
        task_failures = 0
        discovery_train, discovery_valid = agent.group.get_train_valid_split(
            benchmark=task, split_type="default", seed=protocol["discovery_seed"]
        )
        smiles_col, target_col = _columns(discovery_train)
        discovery_profile = profile_task(
            discovery_train, discovery_valid, resolve_task(task), smiles_col, target_col
        )
        registry["task_profiles"].setdefault(task, discovery_profile.to_dict())
        _atomic_json(registry_path, registry)
        if "screen" in phases:
            for arm in ("streamlined_agent", "expanded_automl"):
                for role, spec in build_arm_specs(
                    discovery_profile, arm, protocol["discovery_seed"], args.workers
                ).items():
                    ok = _run_assignment(
                        registry=registry, registry_path=registry_path, agent=agent,
                        spec=spec, task=task, arm=arm, phase="screen", role=role,
                        train=discovery_train, valid=discovery_valid,
                        smiles_col=smiles_col, target_col=target_col,
                        profile=discovery_profile,
                        timeout_seconds=protocol["max_minutes_per_physical_fit"] * 60,
                    )
                    failures += not ok
                    task_failures += not ok
            if task_failures == 0:
                for arm in ("streamlined_agent", "expanded_automl"):
                    selected_role = choose_screen_winner(registry["assignments"], arm, task)
                    screen_rows = sorted(
                        (
                            row for row in registry["assignments"].values()
                            if row["task"] == task and row["arm"] == arm and row["phase"] == "screen"
                        ),
                        key=lambda row: row["role"],
                    )
                    core = {
                        "task": task,
                        "arm": arm,
                        "selected_role": selected_role,
                        "selection_seed": protocol["discovery_seed"],
                        "screen_assignments_sha256": _digest(screen_rows),
                        "test_loaded": False,
                    }
                    registry["selections"][f"{task}|{arm}"] = {
                        **core, "decision_sha256": _digest(core)
                    }
                _atomic_json(registry_path, registry)

        if "evaluation" in phases:
            for arm in ("streamlined_agent", "expanded_automl"):
                if f"{task}|{arm}" not in registry["selections"]:
                    raise RuntimeError(f"Missing frozen screen selection for {task}|{arm}")
            evaluation_train, evaluation_valid = agent.group.get_train_valid_split(
                benchmark=task, split_type="default", seed=protocol["evaluation_seed"]
            )
            eval_smiles, eval_target = _columns(evaluation_train)
            evaluation_profile = profile_task(
                evaluation_train, evaluation_valid, resolve_task(task), eval_smiles, eval_target
            )
            for arm in ("streamlined_agent", "expanded_automl"):
                role = registry["selections"][f"{task}|{arm}"]["selected_role"]
                spec = build_arm_specs(
                    evaluation_profile, arm, protocol["evaluation_seed"], args.workers
                )[role]
                ok = _run_assignment(
                    registry=registry, registry_path=registry_path, agent=agent,
                    spec=spec, task=task, arm=arm, phase="evaluation", role=role,
                    train=evaluation_train, valid=evaluation_valid,
                    smiles_col=eval_smiles, target_col=eval_target,
                    profile=evaluation_profile,
                    timeout_seconds=protocol["max_minutes_per_physical_fit"] * 60,
                )
                failures += not ok
                task_failures += not ok
            fixed_role, fixed_spec = next(iter(build_arm_specs(
                evaluation_profile, "strong_fixed", protocol["evaluation_seed"], args.workers
            ).items()))
            ok = _run_assignment(
                registry=registry, registry_path=registry_path, agent=agent,
                spec=fixed_spec, task=task, arm="strong_fixed", phase="evaluation", role=fixed_role,
                train=evaluation_train, valid=evaluation_valid,
                smiles_col=eval_smiles, target_col=eval_target,
                profile=evaluation_profile,
                timeout_seconds=protocol["max_minutes_per_physical_fit"] * 60,
            )
            failures += not ok
            task_failures += not ok
    print(
        f"logical={len(registry['assignments'])}; physical={len(registry['physical_records'])}; failures={failures}",
        flush=True,
    )
    return int(failures > 0)


if __name__ == "__main__":
    raise SystemExit(main())
