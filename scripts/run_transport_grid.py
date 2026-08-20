"""Execute the preregistered validation-only transport grid with safe resume."""

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
from admet_agent.transport_protocol import load_transport_protocol


def _columns(frame) -> tuple[str, str]:
    smiles = "Drug" if "Drug" in frame.columns else str(frame.columns[0])
    target = "Y" if "Y" in frame.columns else str(frame.columns[-1])
    return smiles, target


def _atomic_json(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def _specs(
    protocol: dict, task: str, seed: int, n_jobs: int
) -> dict[str, ExperimentSpec]:
    hyperparameters = protocol["model_hyperparameters"]
    conditions = {
        "shared-control": protocol["actions"][0]["control"],
        **{
            f"treatment:{action['action_id']}": action["treatment"]
            for action in protocol["actions"]
        },
    }
    specs = {}
    for role, condition in conditions.items():
        model = condition["model"]
        specs[role] = ExperimentSpec(
            task=task,
            seed=seed,
            model=model,
            representations=tuple(condition["representations"]),
            data_policy=condition["data_policy"],
            device="cpu",
            n_jobs=n_jobs,
            hyperparameters=dict(hyperparameters[model]),
            budget=Budget(max_candidates=4, max_minutes=30),
            hypothesis=(
                "preregistered shared matched control"
                if role == "shared-control"
                else f"preregistered transport intervention {role.split(':', 1)[1]}"
            ),
        )
    return specs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--protocol", type=Path, default=Path("runs/v5-transport-grid/protocol.json")
    )
    parser.add_argument("--run-dir", type=Path, default=Path("runs/v5-transport-grid"))
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument(
        "--feature-cache-dir", type=Path, default=Path("cache/features")
    )
    parser.add_argument("--task", action="append")
    parser.add_argument("--workers", type=int, default=8)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    protocol = load_transport_protocol(args.protocol)
    current_source = production_source_manifest()["source_sha256"]
    if current_source != protocol["source_sha256"]:
        raise RuntimeError("Production source does not match the frozen protocol")
    current_runner = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    if current_runner != protocol["runner_sha256"]:
        raise RuntimeError("Transport runner does not match the frozen protocol")
    requested = [
        resolve_task(task).name
        for task in (args.task or protocol["sentinel_endpoints"])
    ]
    if not set(requested).issubset(protocol["sentinel_endpoints"]):
        raise ValueError("Requested endpoint is outside the frozen sentinel cohort")

    args.run_dir.mkdir(parents=True, exist_ok=True)
    registry_path = args.run_dir / "transport_registry.json"
    if registry_path.exists():
        registry = json.loads(registry_path.read_text(encoding="utf-8"))
        if registry.get("protocol_sha256") != protocol["sha256"]:
            raise RuntimeError("Registry belongs to a different protocol")
    else:
        registry = {
            "version": "transport-grid-runner-v1",
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
    for task_name in requested:
        spent = sum(
            float(row["result"]["elapsed_seconds"])
            for key, row in registry["records"].items()
            if key.startswith(f"{task_name}|")
        )
        endpoint_budget = protocol["max_compute_minutes_per_endpoint"] * 60
        for seed in protocol["validation_seeds"]:
            train, valid = agent.group.get_train_valid_split(
                benchmark=task_name, split_type="default", seed=seed
            )
            smiles_col, target_col = _columns(train)
            profile = profile_task(
                train,
                valid,
                resolve_task(task_name),
                smiles_col,
                target_col,
            )
            for role, spec in _specs(protocol, task_name, seed, args.workers).items():
                key = f"{task_name}|{seed}|{role}"
                if key in registry["records"]:
                    continue
                remaining = endpoint_budget - spent
                if remaining <= 0:
                    raise RuntimeError(
                        f"Endpoint compute budget exhausted for {task_name}"
                    )
                result = agent._execute(
                    spec,
                    train,
                    valid,
                    smiles_col,
                    target_col,
                    profile,
                    timeout_seconds=min(spec.budget.max_minutes * 60, remaining),
                )
                spent += result.elapsed_seconds
                registry["records"][key] = {
                    "task": task_name,
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
    expected = len(requested) * len(protocol["validation_seeds"]) * 4
    completed = sum(
        row["task"] in requested and row["result"]["status"] == "completed"
        for row in registry["records"].values()
    )
    print(f"completed={completed}/{expected}; failures={failures}", flush=True)
    return int(failures > 0)


if __name__ == "__main__":
    raise SystemExit(main())
