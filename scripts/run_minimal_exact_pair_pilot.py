"""Run the two missing category-policy matched pairs with gated confirmation."""

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


def _digest(payload: object) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def load_protocol(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    claimed = payload.pop("sha256", None)
    if claimed != _digest(payload):
        raise ValueError("Pilot protocol SHA-256 is invalid")
    if payload.get("test_loaded") is not False:
        raise ValueError("The pilot must not load test data")
    if payload.get("test_feedback_to_discovery") is not False:
        raise ValueError("Test feedback is forbidden")
    if set(payload.get("tasks", {})) != {"CYP2C9_Veith", "LD50_Zhu"}:
        raise ValueError("The frozen pilot must contain exactly two tasks")
    return {**payload, "sha256": claimed}


def _columns(frame) -> tuple[str, str]:
    smiles = "Drug" if "Drug" in frame.columns else str(frame.columns[0])
    target = "Y" if "Y" in frame.columns else str(frame.columns[-1])
    return smiles, target


def build_specs(protocol: dict, task: str, seed: int, workers: int) -> dict[str, ExperimentSpec]:
    task_protocol = protocol["tasks"][task]
    common = {
        "task": task,
        "seed": seed,
        "representations": tuple(task_protocol["representations"]),
        "data_policy": "official",
        "device": "cpu",
        "n_jobs": workers,
        "budget": Budget(max_candidates=2, max_minutes=30),
    }
    specs = {}
    for role in ("control", "treatment"):
        recipe = task_protocol[role]
        specs[role] = ExperimentSpec(
            **common,
            model=recipe["model"],
            imbalance=recipe["imbalance"],
            hyperparameters=dict(recipe.get("hyperparameters", {})),
            hypothesis=recipe["hypothesis"],
        )
    return specs


def _screen_records(registry: dict, protocol: dict, task: str) -> list[dict]:
    rows = []
    for seed in protocol["screen_seeds"]:
        for role in ("control", "treatment"):
            key = f"{task}|{seed}|{role}"
            row = registry["records"].get(key)
            if row is None or row["result"]["status"] != "completed":
                raise RuntimeError(f"Screen is incomplete: {key}")
            rows.append(row)
    return rows


def promotion_decision(registry: dict, protocol: dict) -> dict:
    tasks = {}
    screen_payload = {}
    threshold = float(protocol["promotion_gate"]["minimum_mean_oriented_gain"])
    for task in protocol["tasks"]:
        rows = _screen_records(registry, protocol, task)
        screen_payload[task] = rows
        by_key = {(row["seed"], row["role"]): row for row in rows}
        direction = 1.0 if resolve_task(task).maximize else -1.0
        deltas = [
            direction
            * (
                float(by_key[(seed, "treatment")]["result"]["validation_score"])
                - float(by_key[(seed, "control")]["result"]["validation_score"])
            )
            for seed in protocol["screen_seeds"]
        ]
        mean_delta = sum(deltas) / len(deltas)
        promoted = all(delta > 0 for delta in deltas) and mean_delta >= threshold
        tasks[task] = {
            "oriented_deltas": deltas,
            "mean_oriented_delta": mean_delta,
            "all_directions_positive": all(delta > 0 for delta in deltas),
            "promoted": promoted,
            "reason": (
                "screen_direction_and_effect_gate_passed"
                if promoted
                else "stop_without_confirmation"
            ),
        }
    body = {
        "version": "minimal-exact-pair-promotion-v1",
        "protocol_sha256": protocol["sha256"],
        "screen_records_sha256": _digest(screen_payload),
        "tasks": tasks,
        "test_loaded": False,
    }
    return {**body, "sha256": _digest(body)}


def _atomic_json(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def _load_or_create_registry(path: Path, protocol: dict, source: str, runner: str) -> dict:
    if path.exists():
        registry = json.loads(path.read_text(encoding="utf-8"))
        if registry.get("protocol_sha256") != protocol["sha256"]:
            raise RuntimeError("Registry belongs to a different protocol")
        return registry
    return {
        "version": "minimal-exact-pair-runner-v1",
        "protocol_sha256": protocol["sha256"],
        "source_sha256": source,
        "runner_sha256": runner,
        "test_loaded": False,
        "records": {},
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--protocol", type=Path,
        default=Path("runs/minimal-exact-pair-pilot-v1/protocol.json"),
    )
    parser.add_argument(
        "--run-dir", type=Path, default=Path("runs/minimal-exact-pair-pilot-v1")
    )
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--feature-cache-dir", type=Path, default=Path("cache/features"))
    parser.add_argument("--phase", choices=("screen", "confirmation", "all"), default="screen")
    parser.add_argument("--workers", type=int, default=8)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    protocol = load_protocol(args.protocol)
    source_sha = production_source_manifest()["source_sha256"]
    runner_sha = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    if source_sha != protocol["source_sha256"]:
        raise RuntimeError("Production source does not match frozen protocol")
    if runner_sha != protocol["runner_sha256"]:
        raise RuntimeError("Runner does not match frozen protocol")

    args.run_dir.mkdir(parents=True, exist_ok=True)
    registry_path = args.run_dir / "registry.json"
    registry = _load_or_create_registry(registry_path, protocol, source_sha, runner_sha)
    agent = ADMETAgent(
        data_dir=args.data_dir,
        run_dir=args.run_dir,
        device="cpu",
        n_jobs=args.workers,
        feature_cache_dir=args.feature_cache_dir,
        persist_discovery_models=False,
    )

    phases = ("screen", "confirmation") if args.phase == "all" else (args.phase,)
    failures = 0
    for phase in phases:
        if phase == "screen":
            seeds = protocol["screen_seeds"]
            tasks = list(protocol["tasks"])
        else:
            decision_path = args.run_dir / "promotion_decision.json"
            current = promotion_decision(registry, protocol)
            if decision_path.exists():
                frozen = json.loads(decision_path.read_text(encoding="utf-8"))
                if frozen != current:
                    raise RuntimeError("Frozen promotion decision or screen records changed")
            else:
                _atomic_json(decision_path, current)
            seeds = protocol["confirmation_seeds"]
            tasks = [task for task, row in current["tasks"].items() if row["promoted"]]

        for task in tasks:
            for seed in seeds:
                train, valid = agent.group.get_train_valid_split(
                    benchmark=task, split_type="default", seed=seed
                )
                smiles_col, target_col = _columns(train)
                profile = profile_task(train, valid, resolve_task(task), smiles_col, target_col)
                for role, spec in build_specs(protocol, task, seed, args.workers).items():
                    key = f"{task}|{seed}|{role}"
                    if key in registry["records"]:
                        continue
                    result = agent._execute(
                        spec, train, valid, smiles_col, target_col, profile,
                        timeout_seconds=protocol["max_compute_minutes_per_fit"] * 60,
                    )
                    registry["records"][key] = {
                        "task": task,
                        "seed": seed,
                        "role": role,
                        "phase": phase,
                        "spec": spec.to_dict(),
                        "result": result.to_dict(),
                        "test_loaded": False,
                    }
                    registry["updated_at"] = datetime.now(timezone.utc).isoformat()
                    _atomic_json(registry_path, registry)
                    print(f"{key}: {result.status} ({result.elapsed_seconds:.3f}s)", flush=True)
                    failures += result.status != "completed"
        if phase == "screen" and failures == 0:
            _atomic_json(args.run_dir / "promotion_decision.json", promotion_decision(registry, protocol))

    print(f"records={len(registry['records'])}; failures={failures}", flush=True)
    return int(failures > 0)


if __name__ == "__main__":
    raise SystemExit(main())
