"""Execute the frozen post-memory pruning confirmation with safe resume."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

from admet_agent.profiling import profile_task
from admet_agent.runner import ADMETAgent
from admet_agent.schemas import Budget, ExperimentSpec
from admet_agent.source_snapshot import production_source_manifest
from admet_agent.tasks import resolve_task


def _digest(payload: dict) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _load_protocol(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    claimed = payload.pop("sha256", None)
    if claimed != _digest(payload):
        raise ValueError("Confirmation protocol SHA-256 is invalid")
    if payload.get("test_feedback_to_discovery") is not False:
        raise ValueError("Test feedback is forbidden")
    if len(payload.get("endpoints", [])) != 5:
        raise ValueError("Exactly five confirmation endpoints are required")
    return {**payload, "sha256": claimed}


def _columns(frame) -> tuple[str, str]:
    smiles = "Drug" if "Drug" in frame.columns else str(frame.columns[0])
    target = "Y" if "Y" in frame.columns else str(frame.columns[-1])
    return smiles, target


def _specs(protocol: dict, task: str, seed: int, workers: int) -> dict:
    common = {
        "task": task,
        "seed": seed,
        "representations": tuple(protocol["representations"]),
        "data_policy": "official",
        "device": "cpu",
        "n_jobs": workers,
        "budget": Budget(max_candidates=2, max_minutes=30),
    }
    return {
        "control": ExperimentSpec(
            **common,
            model="extra_trees",
            hyperparameters=dict(protocol["hyperparameters"]["extra_trees"]),
            hypothesis="post-memory matched control: ExtraTrees",
        ),
        "treatment": ExperimentSpec(
            **common,
            model="random_forest",
            hyperparameters=dict(protocol["hyperparameters"]["random_forest"]),
            hypothesis="post-memory evaluated pruned action: RandomForest",
        ),
    }


def _atomic_json(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--protocol",
        type=Path,
        default=Path("runs/v5-post-memory-confirmation/protocol.json"),
    )
    parser.add_argument(
        "--run-dir", type=Path, default=Path("runs/v5-post-memory-confirmation")
    )
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument(
        "--feature-cache-dir", type=Path, default=Path("cache/features")
    )
    parser.add_argument("--task", action="append")
    parser.add_argument("--workers", type=int, default=8)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    protocol = _load_protocol(args.protocol)
    current_source = production_source_manifest()["source_sha256"]
    current_runner = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    if current_source != protocol["source_sha256"]:
        raise RuntimeError("Production source does not match frozen protocol")
    if current_runner != protocol["runner_sha256"]:
        raise RuntimeError("Runner does not match frozen protocol")
    requested = [
        resolve_task(task).name for task in (args.task or protocol["endpoints"])
    ]
    if not set(requested).issubset(protocol["endpoints"]):
        raise ValueError("Requested endpoint is outside the confirmation cohort")

    args.run_dir.mkdir(parents=True, exist_ok=True)
    registry_path = args.run_dir / "confirmation_registry.json"
    if registry_path.exists():
        registry = json.loads(registry_path.read_text(encoding="utf-8"))
        if registry.get("protocol_sha256") != protocol["sha256"]:
            raise RuntimeError("Registry belongs to a different protocol")
    else:
        registry = {
            "version": "post-memory-confirmation-runner-v1",
            "protocol_sha256": protocol["sha256"],
            "source_sha256": current_source,
            "runner_sha256": current_runner,
            "test_loaded": False,
            "records": {},
        }
    agent = ADMETAgent(
        data_dir=args.data_dir,
        run_dir=args.run_dir,
        device="cpu",
        n_jobs=args.workers,
        feature_cache_dir=args.feature_cache_dir,
        persist_discovery_models=False,
    )
    failures = 0
    for task in requested:
        spent = sum(
            float(row["result"]["elapsed_seconds"])
            for key, row in registry["records"].items()
            if key.startswith(f"{task}|")
        )
        for seed in protocol["validation_seeds"]:
            train, valid = agent.group.get_train_valid_split(
                benchmark=task, split_type="default", seed=seed
            )
            smiles_col, target_col = _columns(train)
            profile = profile_task(
                train, valid, resolve_task(task), smiles_col, target_col
            )
            for role, spec in _specs(protocol, task, seed, args.workers).items():
                key = f"{task}|{seed}|{role}"
                if key in registry["records"]:
                    continue
                remaining = protocol["max_compute_minutes_per_endpoint"] * 60 - spent
                if remaining <= 0:
                    raise RuntimeError(f"Endpoint budget exhausted for {task}")
                result = agent._execute(
                    spec,
                    train,
                    valid,
                    smiles_col,
                    target_col,
                    profile,
                    timeout_seconds=min(1800, remaining),
                )
                spent += result.elapsed_seconds
                registry["records"][key] = {
                    "task": task,
                    "seed": seed,
                    "role": role,
                    "spec": spec.to_dict(),
                    "result": result.to_dict(),
                    "test_loaded": False,
                }
                registry["updated_at"] = datetime.now(timezone.utc).isoformat()
                _atomic_json(registry_path, registry)
                print(
                    f"{key}: {result.status} ({result.elapsed_seconds:.3f}s)",
                    flush=True,
                )
                failures += result.status != "completed"
    expected = len(requested) * len(protocol["validation_seeds"]) * 2
    completed = sum(
        row["task"] in requested and row["result"]["status"] == "completed"
        for row in registry["records"].values()
    )
    print(f"completed={completed}/{expected}; failures={failures}", flush=True)
    return int(failures > 0)


if __name__ == "__main__":
    raise SystemExit(main())
