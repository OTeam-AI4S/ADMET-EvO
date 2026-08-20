"""Run a preregistered policy arm one TDC task at a time with safe resume.

The wrapper does not alter search semantics.  It skips a task only when its
summary is complete under the current production source hash, protocol hash,
arm name and discovery seeds.  Test data are never requested here.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from admet_agent.benchmark_protocol import load_protocol
from admet_agent.source_snapshot import production_source_manifest
from admet_agent.tasks import resolve_task

ARM_BRAINS = {
    "agent": "agent",
    "committee": "committee",
    "random": "random-control",
    "rules": "fixed-rules-control",
    "automl": "automl-control",
    "expanded_automl": "expanded-automl-control",
    "strong_fixed": "strong-fixed-control",
    "agent_no_memory": "agent-no-memory",
    "agent_no_hardness": "agent-no-hardness",
    "agent_no_data": "agent-no-data",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", choices=tuple(ARM_BRAINS), required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument(
        "--feature-cache-dir", type=Path, default=Path("cache/features")
    )
    parser.add_argument("--task", action="append")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--device", choices=("cpu", "cuda", "auto"), default="cpu")
    parser.add_argument("--robust-top-k", type=int, default=6)
    parser.add_argument("--policy-seed", type=int, default=20260819)
    parser.add_argument("--continue-on-error", action="store_true")
    return parser.parse_args()


def _summary_path(run_dir: Path, task: str, seed: int) -> Path:
    return run_dir / "tasks" / task / f"search_seed_{seed}.json"


def _is_complete(
    path: Path,
    *,
    task: str,
    arm: str,
    protocol: dict,
    source_sha256: str,
) -> bool:
    if not path.exists():
        return False
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return bool(
        value.get("profile", {}).get("task") == task
        and value.get("benchmark_arm") == arm
        and value.get("benchmark_protocol_sha256") == protocol["sha256"]
        and value.get("source_sha256") == source_sha256
        and value.get("validation_seeds") == protocol["discovery_seeds"]
        and value.get("finalist_run_id")
        and value.get("finalist_spec")
        and value.get("robust_results")
    )


def _write_registry(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "protocol": "resumable-policy-arm-runner-v1",
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "test_loaded": False,
        "tasks": rows,
    }
    temporary = path.with_suffix(".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
    )
    temporary.replace(path)


def main() -> int:
    args = parse_args()
    protocol = load_protocol(args.protocol)
    source_sha256 = production_source_manifest()["source_sha256"]
    arm = next(item for item in protocol["arms"] if item["name"] == args.arm)
    requested = (
        [resolve_task(name).name for name in args.task]
        if args.task
        else list(protocol["tasks"])
    )
    if not set(requested).issubset(set(protocol["tasks"])):
        raise SystemExit("Requested task is outside the preregistered protocol")

    rounds = 1 if args.arm == "strong_fixed" else 2
    candidates = 1 if args.arm == "strong_fixed" else 4
    if rounds * candidates > int(arm["max_actions_per_task"]):
        raise SystemExit("Runner request exceeds the preregistered action budget")

    args.run_dir.mkdir(parents=True, exist_ok=True)
    log_dir = args.run_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    registry_path = args.run_dir / "runner_registry.json"
    registry_rows: list[dict] = []
    failed = False

    for task in requested:
        summary = _summary_path(args.run_dir, task, int(protocol["discovery_seeds"][0]))
        if _is_complete(
            summary,
            task=task,
            arm=args.arm,
            protocol=protocol,
            source_sha256=source_sha256,
        ):
            row = {"task": task, "status": "skipped_complete"}
            registry_rows.append(row)
            _write_registry(registry_path, registry_rows)
            print(f"{task}: skipped_complete", flush=True)
            continue

        command = [
            sys.executable,
            "-m",
            "admet_agent.cli",
            "search",
            "--task",
            task,
            "--seed",
            str(protocol["discovery_seeds"][0]),
            "--rounds",
            str(rounds),
            "--candidates-per-round",
            str(candidates),
            "--validation-seeds",
            ",".join(str(value) for value in protocol["discovery_seeds"]),
            "--robust-top-k",
            str(args.robust_top_k),
            "--brain",
            ARM_BRAINS[args.arm],
            "--policy-seed",
            str(args.policy_seed),
            "--benchmark-protocol",
            str(args.protocol),
            "--selection-policy",
            "score",
            "--workers",
            str(args.workers),
            "--device",
            args.device,
            "--data-dir",
            str(args.data_dir),
            "--run-dir",
            str(args.run_dir),
            "--feature-cache-dir",
            str(args.feature_cache_dir),
        ]
        log_path = log_dir / f"{task}.log"
        print(f"{task}: running", flush=True)
        with log_path.open("w", encoding="utf-8") as log:
            completed = subprocess.run(
                command,
                cwd=Path(__file__).resolve().parents[1],
                stdout=log,
                stderr=subprocess.STDOUT,
                text=True,
                check=False,
            )
        status = "completed" if completed.returncode == 0 else "failed"
        row = {
            "task": task,
            "status": status,
            "returncode": completed.returncode,
            "log": str(log_path),
        }
        registry_rows.append(row)
        _write_registry(registry_path, registry_rows)
        print(f"{task}: {status}", flush=True)
        if completed.returncode != 0:
            failed = True
            if not args.continue_on_error:
                break
    return int(failed)


if __name__ == "__main__":
    raise SystemExit(main())
