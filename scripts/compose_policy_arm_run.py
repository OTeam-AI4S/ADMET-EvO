"""Compose complete task shards into one auditable policy-arm run directory.

This is intentionally a packaging operation, not a search operation.  It is
used when a process interruption leaves one task without a finalist and that
task is rerun from scratch in a clean directory.  Every copied task summary
and referenced artifact is hash-bound in ``composition_manifest.json``.  An
aborted source can be recorded for operational-cost disclosure, but none of
its candidates are admitted to the canonical arm.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from admet_agent.benchmark_protocol import load_protocol
from admet_agent.tasks import resolve_task


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _tree_manifest(root: Path) -> list[dict[str, Any]]:
    return [
        {
            "path": str(path.relative_to(root)),
            "bytes": path.stat().st_size,
            "sha256": _sha256(path),
        }
        for path in sorted(item for item in root.rglob("*") if item.is_file())
    ]


def _load_complete_summary(
    run_dir: Path,
    *,
    task: str,
    arm: str,
    protocol: dict[str, Any],
) -> tuple[Path, dict[str, Any]]:
    seed = int(protocol["discovery_seeds"][0])
    path = run_dir / "tasks" / task / f"search_seed_{seed}.json"
    if not path.is_file():
        raise ValueError(f"Missing complete task summary: {path}")
    summary = json.loads(path.read_text(encoding="utf-8"))
    checks = {
        "task": summary.get("profile", {}).get("task") == task,
        "arm": summary.get("benchmark_arm") == arm,
        "protocol": summary.get("benchmark_protocol_sha256") == protocol["sha256"],
        "source": summary.get("source_sha256")
        == protocol.get("implementation_source_sha256"),
        "seeds": summary.get("validation_seeds") == protocol["discovery_seeds"],
        "finalist": bool(summary.get("finalist_run_id")),
        "finalist_spec": bool(summary.get("finalist_spec")),
        "robust_results": bool(summary.get("robust_results")),
    }
    failed = sorted(key for key, passed in checks.items() if not passed)
    if failed:
        raise ValueError(f"Incomplete or incompatible {task} shard: {failed}")
    return path, summary


def _copy_shard(
    source: Path,
    target: Path,
    *,
    task: str,
    summary: dict[str, Any],
) -> list[str]:
    shutil.copytree(source / "tasks" / task, target / "tasks" / task)
    run_ids = sorted(
        {
            str(item["run_id"])
            for section in ("results",)
            for item in summary.get(section, [])
            if item.get("run_id")
        }
    )
    for run_id in run_ids:
        artifact = source / "artifacts" / run_id
        if not artifact.is_dir():
            raise ValueError(f"Missing referenced artifact directory: {artifact}")
        shutil.copytree(artifact, target / "artifacts" / run_id)
    return run_ids


def _aborted_accounting(run_dir: Path, *, task_filter: str) -> dict[str, Any]:
    database = run_dir / "experiments.sqlite"
    result: dict[str, Any] = {
        "run_dir": str(run_dir.resolve()),
        "excluded_task": task_filter,
        "experiments_sqlite_sha256": _sha256(database) if database.is_file() else None,
        "completed_fits": 0,
        "failed_fits": 0,
        "fit_seconds": 0.0,
        "tasks": {},
    }
    if not database.is_file():
        return result
    connection = sqlite3.connect(f"file:{database.resolve()}?mode=ro", uri=True)
    try:
        rows = connection.execute(
            "SELECT task, status, COUNT(*), COALESCE(SUM(elapsed_seconds), 0) "
            "FROM runs WHERE task = ? GROUP BY task, status ORDER BY task, status",
            (task_filter,),
        ).fetchall()
    finally:
        connection.close()
    for task, status, count, seconds in rows:
        result["tasks"].setdefault(task, {})[status] = {
            "fits": int(count),
            "fit_seconds": float(seconds),
        }
        if status == "completed":
            result["completed_fits"] += int(count)
        elif status == "failed":
            result["failed_fits"] += int(count)
        result["fit_seconds"] += float(seconds)
    return result


def compose_policy_arm_run(
    *,
    protocol_path: Path,
    arm: str,
    task_sources: dict[str, Path],
    output_dir: Path,
    aborted_task_sources: dict[str, Path] | None = None,
    expected_tasks: list[str] | None = None,
) -> dict[str, Any]:
    protocol = load_protocol(protocol_path)
    allowed_arms = {item["name"] for item in protocol["arms"]}
    if arm not in allowed_arms:
        raise ValueError(f"Unknown protocol arm: {arm}")
    protocol_tasks = set(protocol["tasks"])
    requested_tasks = (
        {resolve_task(task).name for task in expected_tasks}
        if expected_tasks is not None
        else protocol_tasks
    )
    if not requested_tasks or not requested_tasks.issubset(protocol_tasks):
        raise ValueError("Expected tasks must be a non-empty protocol-task subset")
    normalized = {resolve_task(task).name: path for task, path in task_sources.items()}
    if set(normalized) != requested_tasks:
        raise ValueError(
            "Task shards must exactly cover expected tasks; "
            f"missing={sorted(requested_tasks - set(normalized))}, "
            f"extra={sorted(set(normalized) - requested_tasks)}"
        )
    if output_dir.exists():
        raise FileExistsError(f"Output directory already exists: {output_dir}")
    temporary = output_dir.with_name(f".{output_dir.name}.composing")
    if temporary.exists():
        raise FileExistsError(f"Temporary output already exists: {temporary}")
    temporary.mkdir(parents=True)
    (temporary / "artifacts").mkdir()
    copied: list[dict[str, Any]] = []
    try:
        for task in protocol["tasks"]:
            if task not in requested_tasks:
                continue
            source = normalized[task].resolve()
            summary_path, summary = _load_complete_summary(
                source, task=task, arm=arm, protocol=protocol
            )
            run_ids = _copy_shard(
                source, temporary, task=task, summary=summary
            )
            copied.append(
                {
                    "task": task,
                    "source_run_dir": str(source),
                    "source_summary": str(summary_path.resolve()),
                    "source_summary_sha256": _sha256(summary_path),
                    "referenced_artifact_run_ids": run_ids,
                }
            )
        manifest: dict[str, Any] = {
            "protocol": "policy-arm-composition-v1",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "arm": arm,
            "benchmark_protocol": str(protocol_path.resolve()),
            "benchmark_protocol_sha256": protocol["sha256"],
            "implementation_source_sha256": protocol.get(
                "implementation_source_sha256"
            ),
            "protocol_task_count": len(protocol_tasks),
            "composed_task_count": len(requested_tasks),
            "composed_tasks": [
                task for task in protocol["tasks"] if task in requested_tasks
            ],
            "canonical_candidates_from_aborted_runs": False,
            "operational_overhead_excluded_from_matched_inference": True,
            "operational_overhead_reason": (
                "process interruption before finalist; recovery task was rerun "
                "from scratch under the frozen protocol"
            ),
            "task_shards": copied,
            "aborted_runs": [
                _aborted_accounting(path.resolve(), task_filter=resolve_task(task).name)
                for task, path in sorted((aborted_task_sources or {}).items())
            ],
        }
        manifest["composed_files"] = _tree_manifest(temporary)
        manifest_path = temporary / "composition_manifest.json"
        manifest_path.write_text(
            json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
        )
        temporary.replace(output_dir)
        manifest["composition_manifest_sha256"] = _sha256(
            output_dir / "composition_manifest.json"
        )
        return manifest
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--arm", required=True)
    parser.add_argument(
        "--task",
        action="append",
        help="Expected protocol-task subset; omit to require the full protocol",
    )
    parser.add_argument(
        "--task-source",
        action="append",
        required=True,
        help="TASK=RUN_DIR; repeat once for every protocol task",
    )
    parser.add_argument(
        "--aborted-task-source",
        action="append",
        default=[],
        help="TASK=RUN_DIR for an interrupted shard excluded from the canonical arm",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    task_sources: dict[str, Path] = {}
    for value in args.task_source:
        if "=" not in value:
            raise SystemExit("--task-source must use TASK=RUN_DIR")
        task, raw_path = value.split("=", 1)
        if task in task_sources:
            raise SystemExit(f"Duplicate --task-source for {task}")
        task_sources[task] = Path(raw_path)
    aborted_task_sources: dict[str, Path] = {}
    for value in args.aborted_task_source:
        if "=" not in value:
            raise SystemExit("--aborted-task-source must use TASK=RUN_DIR")
        task, raw_path = value.split("=", 1)
        if task in aborted_task_sources:
            raise SystemExit(f"Duplicate --aborted-task-source for {task}")
        aborted_task_sources[task] = Path(raw_path)
    payload = compose_policy_arm_run(
        protocol_path=args.protocol,
        arm=args.arm,
        task_sources=task_sources,
        output_dir=args.output_dir,
        aborted_task_sources=aborted_task_sources,
        expected_tasks=args.task,
    )
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
