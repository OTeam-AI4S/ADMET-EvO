"""Conservative audit of legacy Agent and AutoML discovery runs.

The historical runs are useful negative evidence, but they are not promoted to
a formal same-budget benchmark when their observed action counts violate the
bound protocol or their source snapshots differ.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from .metric_normalization import MetricNormalizationProfile
from .tasks import TASKS

LEGACY_POLICY_AUDIT_VERSION = "legacy-agent-automl-audit-v1"
EXECUTION_KERNEL_FILES = (
    "admet_agent/data_governance.py",
    "admet_agent/experiment_db.py",
    "admet_agent/features.py",
    "admet_agent/guard.py",
    "admet_agent/profiling.py",
    "admet_agent/registry.py",
    "admet_agent/schemas.py",
    "admet_agent/selection.py",
    "admet_agent/tasks.py",
    "admet_agent/tuning.py",
)


def _protocol(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    claimed = payload.get("sha256")
    base = {key: value for key, value in payload.items() if key != "sha256"}
    actual = hashlib.sha256(json.dumps(base, sort_keys=True).encode()).hexdigest()
    if claimed != actual:
        raise ValueError("Protocol SHA-256 is invalid")
    if payload.get("test_feedback_to_discovery") is not False:
        raise ValueError("Protocol must forbid test feedback")
    return payload


def _summaries(run_dir: Path, expected_arm: str) -> dict[str, dict[str, Any]]:
    rows = {}
    for path in sorted(run_dir.glob("tasks/*/search_seed_*.json")):
        summary = json.loads(path.read_text(encoding="utf-8"))
        if summary.get("benchmark_arm") != expected_arm:
            raise ValueError(f"{path} does not belong to arm {expected_arm}")
        task = summary.get("profile", {}).get("task")
        if not task or task in rows:
            raise ValueError(f"Missing or duplicate task in {path}")
        rows[task] = summary
    expected = {task.name for task in TASKS.values()}
    if set(rows) != expected:
        raise ValueError(f"{expected_arm} must cover exactly the TDC 22 tasks")
    return rows


def _finalist(summary: dict[str, Any]) -> dict[str, Any]:
    finalist_id = summary.get("finalist_run_id")
    finalist = next(
        (
            row
            for row in summary.get("robust_results", [])
            + summary.get("ensemble_results", [])
            if row.get("run_id") == finalist_id
        ),
        None,
    )
    if finalist is None:
        raise ValueError(f"Finalist {finalist_id!r} is missing")
    return finalist


def _account(summary: dict[str, Any]) -> dict[str, Any]:
    results = summary.get("results", [])
    seeds = summary.get("validation_seeds", [])
    if not seeds:
        raise ValueError("Validation seeds are required")
    elapsed = [float(row.get("elapsed_seconds", float("nan"))) for row in results]
    if any(not math.isfinite(value) or value < 0 for value in elapsed):
        raise ValueError("Fit times must be finite and non-negative")
    return {
        "primary_configurations": sum(row.get("seed") == seeds[0] for row in results),
        "total_fits": len(results),
        "fit_seconds": sum(elapsed),
    }


def _source_audit(agent_dir: Path, control_dir: Path) -> dict[str, Any]:
    agent_path = agent_dir / "source_snapshot" / "manifest.json"
    control_path = control_dir / "source_snapshot" / "manifest.json"
    agent = json.loads(agent_path.read_text(encoding="utf-8"))
    control = json.loads(control_path.read_text(encoding="utf-8"))
    files = sorted(set(agent["files"]) | set(control["files"]))
    differing = [
        path for path in files if agent["files"].get(path) != control["files"].get(path)
    ]
    kernel_differences = [path for path in EXECUTION_KERNEL_FILES if path in differing]
    return {
        "same_source_snapshot": agent["source_sha256"] == control["source_sha256"],
        "agent_source_sha256": agent["source_sha256"],
        "control_source_sha256": control["source_sha256"],
        "differing_files": differing,
        "execution_kernel_files": list(EXECUTION_KERNEL_FILES),
        "execution_kernel_identical": not kernel_differences,
        "execution_kernel_differences": kernel_differences,
        "interpretation": (
            "Feature generation, estimators, scoring, task contracts and data execution "
            "are hash-identical; policy promotion/evidence code differs. Source mismatch "
            "still blocks a formal causal benchmark claim."
        ),
    }


def _bootstrap(values: np.ndarray, seed: int = 20260820) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(values), size=(10000, len(values)))
    means = values[indices].mean(axis=1)
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def audit_legacy_agent_vs_automl(
    *,
    agent_dir: Path,
    automl_dir: Path,
    protocol_path: Path,
    profiles: dict[str, MetricNormalizationProfile],
) -> dict[str, Any]:
    protocol = _protocol(protocol_path)
    agent = _summaries(agent_dir, "agent")
    control = _summaries(automl_dir, "automl")
    expected = set(agent)
    if set(profiles) != expected:
        raise ValueError("One train-only normalization profile is required per task")
    arm_limits = {row["name"]: row for row in protocol["arms"]}
    task_rows = []
    for task in sorted(expected):
        left = agent[task]
        right = control[task]
        if (
            left.get("benchmark_protocol_sha256") != protocol["sha256"]
            or right.get("benchmark_protocol_sha256") != protocol["sha256"]
        ):
            raise ValueError("Every summary must bind to the audited protocol")
        if left.get("validation_seeds") != right.get("validation_seeds"):
            raise ValueError("Agent and AutoML validation seeds must match")
        left_account = _account(left)
        right_account = _account(right)
        left_finalist = _finalist(left)
        right_finalist = _finalist(right)
        profile = profiles[task]
        delta = profile.utility(left_finalist["validation_score"]) - profile.utility(
            right_finalist["validation_score"]
        )
        task_rows.append(
            {
                "task": task,
                "normalized_oriented_delta": delta,
                "agent_validation_score": float(left_finalist["validation_score"]),
                "automl_validation_score": float(right_finalist["validation_score"]),
                "agent_primary_configurations": left_account["primary_configurations"],
                "automl_primary_configurations": right_account[
                    "primary_configurations"
                ],
                "agent_total_fits": left_account["total_fits"],
                "automl_total_fits": right_account["total_fits"],
                "agent_fit_seconds": left_account["fit_seconds"],
                "automl_fit_seconds": right_account["fit_seconds"],
                "agent_cost_ratio": left_account["fit_seconds"]
                / max(right_account["fit_seconds"], 1e-12),
                "normalization_profile_sha256": profile.profile_sha256,
                "test_loaded": False,
            }
        )
    deltas = np.asarray(
        [row["normalized_oriented_delta"] for row in task_rows], dtype=float
    )
    agent_max = int(arm_limits["agent"]["max_actions_per_task"])
    automl_max = int(arm_limits["automl"]["max_actions_per_task"])
    agent_compliant = all(
        row["agent_primary_configurations"] <= agent_max for row in task_rows
    )
    control_compliant = all(
        row["automl_primary_configurations"] <= automl_max for row in task_rows
    )
    source = _source_audit(agent_dir, automl_dir)
    agent_seconds = sum(row["agent_fit_seconds"] for row in task_rows)
    automl_seconds = sum(row["automl_fit_seconds"] for row in task_rows)
    interval = _bootstrap(deltas)
    worst = min(task_rows, key=lambda row: row["normalized_oriented_delta"])
    return {
        "version": LEGACY_POLICY_AUDIT_VERSION,
        "protocol_sha256": protocol["sha256"],
        "protocol_version": protocol["protocol_version"],
        "task_count": len(task_rows),
        "task_rows": task_rows,
        "source_audit": source,
        "budget_audit": {
            "declared_agent_max_actions": agent_max,
            "declared_automl_max_actions": automl_max,
            "agent_protocol_compliant": agent_compliant,
            "automl_protocol_compliant": control_compliant,
            "actual_action_parity_tasks": sum(
                row["agent_primary_configurations"]
                == row["automl_primary_configurations"]
                for row in task_rows
            ),
            "agent_same_or_lower_action_tasks": sum(
                row["agent_primary_configurations"]
                <= row["automl_primary_configurations"]
                for row in task_rows
            ),
            "agent_total_fits": sum(row["agent_total_fits"] for row in task_rows),
            "automl_total_fits": sum(row["automl_total_fits"] for row in task_rows),
        },
        "predictive_comparison": {
            "mean_normalized_oriented_delta": float(np.mean(deltas)),
            "median_normalized_oriented_delta": float(np.median(deltas)),
            "task_bootstrap_mean_95_ci": list(interval),
            "wins": int(np.sum(deltas > 1e-12)),
            "ties": int(np.sum(np.abs(deltas) <= 1e-12)),
            "losses": int(np.sum(deltas < -1e-12)),
            "tasks_within_minus_0_01": int(np.sum(deltas >= -0.01)),
            "worst_task": worst["task"],
            "worst_normalized_delta": worst["normalized_oriented_delta"],
            "exploratory_noninferiority_minus_0_01": interval[0] >= -0.01,
        },
        "cost_comparison": {
            "agent_fit_seconds": agent_seconds,
            "automl_fit_seconds": automl_seconds,
            "aggregate_agent_cost_ratio": agent_seconds / automl_seconds,
            "agent_compute_saving_supported": agent_seconds <= automl_seconds,
        },
        "formal_claim_eligible": bool(
            agent_compliant and control_compliant and source["same_source_snapshot"]
        ),
        "test_loaded": False,
        "selection_feedback_to_discovery": False,
        "claim_boundary": (
            "Post-hoc exploratory negative evidence only: both arms exceeded the "
            "preregistered action limit and source snapshots differ. It cannot certify "
            "a formal same-budget Agent-vs-AutoML claim."
        ),
    }


__all__ = [
    "EXECUTION_KERNEL_FILES",
    "LEGACY_POLICY_AUDIT_VERSION",
    "audit_legacy_agent_vs_automl",
]
