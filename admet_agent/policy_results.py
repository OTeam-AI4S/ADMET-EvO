"""Collect auditable policy-arm outcomes from discovery search directories."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

from .benchmark_protocol import PolicyTaskResult
from .tasks import resolve_task


def _configuration_signature(experiment: dict[str, Any]) -> str:
    """Return a seed-independent signature for an executed configuration."""

    fields = {
        "representations": experiment.get("representations", ()),
        "model": experiment.get("model"),
        "imbalance": experiment.get("imbalance", "none"),
        "data_policy": experiment.get("data_policy", "official"),
        "hyperparameters": experiment.get("hyperparameters", {}),
    }
    return json.dumps(fields, sort_keys=True, separators=(",", ":"))


def _fit_accounting(
    run_dir: Path,
    summary: dict[str, Any],
) -> dict[str, Any]:
    """Account only executor fits; robust/ensemble rows are derived views."""

    results = list(summary.get("results", []))
    validation_seeds = summary.get("validation_seeds") or [1]
    primary_seed = validation_seeds[0]
    statuses = [item.get("status", "completed") for item in results]
    unknown = sorted(set(statuses) - {"completed", "failed"})
    if unknown:
        raise ValueError(f"Unknown fit statuses in search summary: {unknown}")
    run_ids = [item.get("run_id") for item in results if item.get("run_id")]
    if len(run_ids) != len(set(run_ids)):
        raise ValueError("Search summary contains duplicate fit run_ids")

    primary = [item for item in results if item.get("seed") == primary_seed]
    signatures: set[str] = set()
    artifact_specs = 0
    for index, item in enumerate(primary):
        run_id = item.get("run_id")
        experiment_path = run_dir / "artifacts" / str(run_id) / "experiment.json"
        if run_id and experiment_path.exists():
            experiment = json.loads(experiment_path.read_text(encoding="utf-8"))
            signatures.add(_configuration_signature(experiment))
            artifact_specs += 1
        else:
            # Legacy summaries did not embed imbalance or the full experiment.
            # Count each primary fit conservatively rather than undercounting a
            # potentially distinct action.
            signatures.add(f"legacy-primary-fit:{run_id or index}")

    elapsed_values = [float(item.get("elapsed_seconds", 0.0)) for item in results]
    if any(not math.isfinite(value) or value < 0 for value in elapsed_values):
        raise ValueError("Fit elapsed time cannot be negative")
    completed_seconds = sum(
        value
        for value, status in zip(elapsed_values, statuses)
        if status == "completed"
    )
    failed_seconds = sum(
        value
        for value, status in zip(elapsed_values, statuses)
        if status == "failed"
    )
    completed = statuses.count("completed")
    failed = statuses.count("failed")
    total = len(results)
    if completed + failed != total:
        raise ValueError("Completed and failed fit counts must sum to total fits")
    return {
        "unique_configurations": len(signatures),
        "primary_candidate_fits": len(primary),
        "replicate_fits": total - len(primary),
        "total_fits": total,
        "completed_fits": completed,
        "failed_fits": failed,
        "fit_seconds": completed_seconds + failed_seconds,
        "completed_fit_seconds": completed_seconds,
        "failed_fit_seconds": failed_seconds,
        "ensemble_evaluations": len(summary.get("ensemble_results", [])),
        "configuration_count_source": (
            "artifact_experiment_specs"
            if artifact_specs == len(primary)
            else "conservative_primary_fit_records"
        ),
    }


def _protocol(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    claimed = value.get("sha256")
    payload = {key: item for key, item in value.items() if key != "sha256"}
    actual = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
    if claimed != actual:
        raise ValueError("Policy protocol SHA-256 is invalid")
    return value


def collect_policy_results(
    protocol_path: Path,
    arm_run_dirs: dict[str, Path],
    output: Path | None = None,
) -> dict[str, Any]:
    protocol = _protocol(protocol_path)
    allowed = {item["name"]: item for item in protocol["arms"]}
    unknown = set(arm_run_dirs) - set(allowed)
    if unknown:
        raise ValueError(f"Unknown policy arms: {sorted(unknown)}")
    rows: list[PolicyTaskResult] = []
    arm_summaries = {}
    for arm, run_dir in sorted(arm_run_dirs.items()):
        task_names = []
        accounted_task_names = []
        totals = {
            "unique_configurations": 0,
            "primary_candidate_fits": 0,
            "replicate_fits": 0,
            "total_fits": 0,
            "completed_fits": 0,
            "failed_fits": 0,
            "fit_seconds": 0.0,
            "completed_fit_seconds": 0.0,
            "failed_fit_seconds": 0.0,
            "ensemble_evaluations": 0,
        }
        for path in sorted(run_dir.glob("tasks/*/search_seed_*.json")):
            summary = json.loads(path.read_text(encoding="utf-8"))
            recorded_arm = summary.get("benchmark_arm", "agent")
            if recorded_arm != arm:
                raise ValueError(
                    f"{path} records benchmark arm {recorded_arm}, expected {arm}"
                )
            if summary.get("benchmark_protocol_sha256") != protocol["sha256"]:
                raise ValueError(f"{path} is not bound to the requested protocol")
            expected_source_sha256 = protocol.get("implementation_source_sha256")
            if (
                expected_source_sha256
                and summary.get("source_sha256") != expected_source_sha256
            ):
                raise ValueError(
                    f"{path} is not bound to the frozen implementation source"
                )
            expected_router = allowed[arm].get(
                "research_routing_policy", "legacy"
            )
            recorded_router = summary.get("research_routing_policy", "legacy")
            if recorded_router != expected_router:
                raise ValueError(
                    f"{path} records research router {recorded_router}, "
                    f"expected {expected_router} for arm {arm}"
                )
            expected_memory_sha256 = protocol.get("frozen_invariants", {}).get(
                "action_memory_snapshot_sha256"
            )
            if (
                expected_memory_sha256
                and summary.get("action_memory_snapshot_sha256")
                != expected_memory_sha256
            ):
                raise ValueError(
                    f"{path} is not bound to the frozen action-memory snapshot"
                )
            expected_context_sha256 = protocol.get("frozen_invariants", {}).get(
                "action_context_registry_sha256"
            )
            if (
                expected_context_sha256
                and summary.get("action_context_registry_sha256")
                != expected_context_sha256
            ):
                raise ValueError(
                    f"{path} is not bound to the frozen task-context registry"
                )
            expected_v2_memory_sha256 = protocol.get("frozen_invariants", {}).get(
                "ecvoi_v2_memory_snapshot_sha256"
            )
            if (
                expected_v2_memory_sha256
                and summary.get("ecvoi_v2_memory_snapshot_sha256")
                != expected_v2_memory_sha256
            ):
                raise ValueError(
                    f"{path} is not bound to the frozen EC-VOI v2 memory snapshot"
                )
            if summary.get("test_loaded", False) is not False or summary.get(
                "leaderboard_loaded", False
            ) is not False:
                raise ValueError(f"{path} contains forbidden evaluation feedback")
            accounting = _fit_accounting(run_dir, summary)
            accounting_task = summary.get("profile", {}).get("task", path.parent.name)
            if accounting_task not in set(protocol["tasks"]):
                raise ValueError(
                    f"{arm}/{accounting_task} is outside the protocol task cohort"
                )
            if accounting["unique_configurations"] > int(
                allowed[arm]["max_actions_per_task"]
            ):
                raise ValueError(
                    f"{arm}/{accounting_task} exceeded unique-configuration "
                    "action budget"
                )
            if accounting["primary_candidate_fits"] > int(
                allowed[arm]["max_candidates_per_task"]
            ):
                raise ValueError(f"{arm}/{accounting_task} exceeded candidate budget")
            if accounting["fit_seconds"] > 60.0 * float(
                allowed[arm]["max_compute_minutes_per_task"]
            ):
                raise ValueError(f"{arm}/{accounting_task} exceeded compute budget")
            accounted_task_names.append(accounting_task)
            for key in totals:
                totals[key] += accounting[key]

            finalist_id = summary.get("finalist_run_id")
            shared_anchor = summary.get("shared_strong_anchor") or {}
            shared_anchor_result = shared_anchor.get("aggregate_result")
            finalist = next(
                (
                    item
                    for item in summary.get("robust_results", [])
                    + summary.get("ensemble_results", [])
                    if item.get("run_id") == finalist_id
                ),
                None,
            )
            if (
                finalist is None
                and isinstance(shared_anchor_result, dict)
                and shared_anchor_result.get("run_id") == finalist_id
            ):
                finalist = shared_anchor_result
            if finalist is None:
                continue
            result = PolicyTaskResult(
                task=resolve_task(finalist["task"]).name,
                arm=arm,
                validation_score=float(finalist["validation_score"]),
                validation_std=float(finalist.get("validation_std") or 0.0),
                compute_seconds=accounting["fit_seconds"],
                actions_attempted=accounting["unique_configurations"],
                protocol_sha256=protocol["sha256"],
                unique_configurations=accounting["unique_configurations"],
                replicate_fits=accounting["replicate_fits"],
                total_fits=accounting["total_fits"],
                completed_fits=accounting["completed_fits"],
                failed_fits=accounting["failed_fits"],
                fit_seconds=accounting["fit_seconds"],
                completed_fit_seconds=accounting["completed_fit_seconds"],
                failed_fit_seconds=accounting["failed_fit_seconds"],
                ensemble_evaluations=accounting["ensemble_evaluations"],
            )
            rows.append(result)
            task_names.append(result.task)
        arm_summaries[arm] = {
            "tasks": len(task_names),
            "task_names": sorted(task_names),
            "accounted_searches": len(accounted_task_names),
            "accounted_task_names": sorted(accounted_task_names),
            # Backward-compatible aliases.
            "actions_attempted": totals["unique_configurations"],
            "compute_seconds": totals["fit_seconds"],
            **totals,
        }
    payload = {
        "protocol_sha256": protocol["sha256"],
        "protocol_version": protocol["protocol_version"],
        "arms": arm_summaries,
        "results": [item.__dict__ for item in rows],
        "accounting_semantics": {
            "unique_configurations": (
                "distinct seed-independent primary configurations; each is one action"
            ),
            "replicate_fits": "non-primary validation-seed model fits",
            "total_fits": "primary candidate fits plus replicate fits",
            "fit_seconds": "completed plus failed executor fit elapsed time",
            "ensemble_evaluations": (
                "derived prediction evaluations; excluded from total_fits and fit_seconds"
            ),
            "legacy_aliases": {
                "actions_attempted": "unique_configurations",
                "compute_seconds": "fit_seconds",
            },
        },
        "test_loaded": False,
    }
    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return payload
