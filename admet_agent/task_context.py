"""Versioned, label-blind task context for contextual evidence transport."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

from .endpoint_contracts import endpoint_contract
from .tasks import TASKS

CONTEXT_VERSION = "admet-task-context-v1"
HARDNESS_DIMENSIONS = (
    "scarcity",
    "imbalance",
    "domain_shift",
    "local_roughness",
    "data_quality",
    "split_leakage_risk",
)
CONTEXT_DISTANCE_WEIGHTS = {
    "hardness": 0.55,
    "task_type": 0.20,
    "metric": 0.10,
    "category": 0.15,
}


def task_context_signature(
    hardness: dict[str, Any], category: str | None = None
) -> dict[str, Any]:
    dimensions = hardness.get("dimensions", {})
    missing = set(HARDNESS_DIMENSIONS) - set(dimensions)
    if missing:
        raise ValueError(f"Hardness context is missing dimensions: {sorted(missing)}")
    task = str(hardness["task"])
    resolved_category = category or endpoint_contract(task).category
    severities = {
        name: float(dimensions[name]["severity"]) for name in HARDNESS_DIMENSIONS
    }
    if any(not math.isfinite(value) or not 0 <= value <= 1 for value in severities.values()):
        raise ValueError("Hardness severities must be finite values within [0, 1]")
    return {
        "context_version": CONTEXT_VERSION,
        "task": task,
        "task_type": str(hardness["task_type"]),
        "metric": str(hardness["metric"]),
        "category": resolved_category,
        "hardness": severities,
        "hardness_evidence_sha256": str(hardness["evidence_sha256"]),
        "discovery_only": bool(hardness.get("discovery_only") is True),
    }


def context_distance(left: dict[str, Any], right: dict[str, Any]) -> float:
    if left.get("context_version") != CONTEXT_VERSION:
        raise ValueError("Left task context version is unsupported")
    if right.get("context_version") != CONTEXT_VERSION:
        raise ValueError("Right task context version is unsupported")
    hardness_distance = sum(
        abs(float(left["hardness"][name]) - float(right["hardness"][name]))
        for name in HARDNESS_DIMENSIONS
    ) / len(HARDNESS_DIMENSIONS)
    distance = (
        CONTEXT_DISTANCE_WEIGHTS["hardness"] * hardness_distance
        + CONTEXT_DISTANCE_WEIGHTS["task_type"]
        * float(left["task_type"] != right["task_type"])
        + CONTEXT_DISTANCE_WEIGHTS["metric"]
        * float(left["metric"] != right["metric"])
        + CONTEXT_DISTANCE_WEIGHTS["category"]
        * float(left["category"] != right["category"])
    )
    return float(min(1.0, max(0.0, distance)))


def context_similarity(left: dict[str, Any], right: dict[str, Any]) -> float:
    """RBF similarity fixed before the EC-VOI predictive comparison."""

    return float(math.exp(-3.0 * context_distance(left, right)))


def build_context_registry(diagnosis_run: Path) -> dict[str, Any]:
    paths = sorted(diagnosis_run.glob("tasks/*/diagnosis_seed_1.json"))
    packets = [json.loads(path.read_text(encoding="utf-8")) for path in paths]
    by_task = {packet["task"]: packet for packet in packets}
    expected = {item.name for item in TASKS.values()}
    if set(by_task) != expected:
        raise ValueError("Context registry requires all 22 diagnosis packets")
    source_files = {
        str(path.relative_to(diagnosis_run)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in paths
    }
    contexts = {
        task: task_context_signature(packet["hardness"])
        for task, packet in sorted(by_task.items())
    }
    body = {
        "version": "admet-task-context-registry-v1",
        "context_version": CONTEXT_VERSION,
        "diagnosis_run": str(diagnosis_run),
        "source_files": source_files,
        "contexts": contexts,
        "distance_weights": CONTEXT_DISTANCE_WEIGHTS,
        "similarity_kernel": "exp(-3 * weighted_context_distance)",
        "test_loaded": False,
    }
    digest = hashlib.sha256(
        json.dumps(body, sort_keys=True).encode("utf-8")
    ).hexdigest()
    return {**body, "sha256": digest}


def write_context_registry(diagnosis_run: Path, output: Path) -> dict[str, Any]:
    payload = build_context_registry(diagnosis_run)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        existing = json.loads(output.read_text(encoding="utf-8"))
        if existing.get("sha256") != payload["sha256"]:
            raise RuntimeError("Refusing to overwrite a different context registry")
    else:
        output.write_text(
            json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
        )
    return payload


def load_context_registry(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    claimed = payload.pop("sha256", None)
    actual = hashlib.sha256(
        json.dumps(payload, sort_keys=True).encode("utf-8")
    ).hexdigest()
    if claimed != actual:
        raise ValueError("Task-context registry SHA-256 is invalid")
    if payload.get("test_loaded") is not False:
        raise ValueError("Task-context registry must be test-blind")
    return {**payload, "sha256": claimed}


__all__ = [
    "build_context_registry",
    "context_distance",
    "context_similarity",
    "load_context_registry",
    "task_context_signature",
    "write_context_registry",
]
