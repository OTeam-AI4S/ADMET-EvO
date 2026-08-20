"""Pre-test audit and authorization for a globally frozen TDC-22 finalist set."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

from .research_loop import EvidenceLedger
from .tasks import TASKS


def _digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def _load_protocol(path: Path) -> tuple[dict[str, Any], bool]:
    protocol = json.loads(path.read_text(encoding="utf-8"))
    claimed = protocol.get("sha256")
    payload = {key: value for key, value in protocol.items() if key != "sha256"}
    return protocol, claimed == _digest(payload)


def _verify_source_snapshot(run_dir: Path) -> tuple[dict[str, Any], bool]:
    snapshot_dir = run_dir / "source_snapshot"
    path = snapshot_dir / "manifest.json"
    if not path.exists():
        return {}, False
    manifest = json.loads(path.read_text(encoding="utf-8"))
    files = manifest.get("files", {})
    observed = {}
    for relative in files:
        source = snapshot_dir / relative
        if not source.exists():
            return manifest, False
        observed[relative] = hashlib.sha256(source.read_bytes()).hexdigest()
    combined = hashlib.sha256(
        json.dumps(files, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    valid = observed == files and combined == manifest.get("source_sha256")
    return manifest, valid


def audit_global_freeze(
    run_dir: Path,
    protocol_path: Path,
    output_audit: Path,
    output_finalists: Path,
) -> dict[str, Any]:
    protocol, protocol_hash_valid = _load_protocol(protocol_path)
    expected = {item.name for item in TASKS.values()}
    summaries: dict[str, dict[str, Any]] = {}
    duplicate_tasks = []
    for path in sorted(run_dir.glob("tasks/*/search_seed_*.json")):
        summary = json.loads(path.read_text(encoding="utf-8"))
        task = summary.get("profile", {}).get("task")
        if task in summaries:
            duplicate_tasks.append(task)
        summaries[task] = summary

    errors: list[str] = []
    warnings: list[str] = []
    if not protocol_hash_valid:
        errors.append("benchmark protocol SHA-256 is invalid")
    if protocol.get("test_feedback_to_discovery") is not False:
        errors.append("protocol permits test feedback to discovery")
    if set(protocol.get("tasks", [])) != expected:
        errors.append("protocol does not contain the canonical 22 tasks")
    if duplicate_tasks:
        errors.append(f"multiple search summaries found for tasks: {sorted(duplicate_tasks)}")
    missing = sorted(expected - set(summaries))
    extra = sorted(set(summaries) - expected)
    if missing:
        errors.append(f"missing task summaries: {missing}")
    if extra:
        errors.append(f"non-canonical task summaries: {extra}")
    if (run_dir / "final" / "results.json").exists():
        errors.append("test results already exist in the discovery run directory")
    source_manifest, source_valid = _verify_source_snapshot(run_dir)
    if not source_valid:
        errors.append("production source snapshot is missing or has an invalid hash")
    source_hashes = {
        summary.get("source_sha256") for summary in summaries.values()
    }
    if source_hashes != {source_manifest.get("source_sha256")}:
        errors.append("task summaries are not bound to one valid source snapshot")

    finalists: dict[str, Any] = {}
    task_audits = []
    required_seeds = int(protocol.get("required_validation_seeds", 5))
    stability_threshold = float(protocol.get("stability_review_threshold", 0.05))
    for task_name in sorted(expected & set(summaries)):
        summary = summaries[task_name]
        seeds = summary.get("validation_seeds", [])
        finalist_id = summary.get("finalist_run_id")
        finalist = next(
            (
                item
                for item in summary.get("robust_results", [])
                + summary.get("ensemble_results", [])
                if item.get("run_id") == finalist_id
            ),
            None,
        )
        spec = summary.get("finalist_spec")
        task_errors = []
        if len(set(seeds)) < required_seeds:
            task_errors.append(
                f"requires {required_seeds} validation seeds; found {len(set(seeds))}"
            )
        if finalist is None or spec is None:
            task_errors.append("missing finalist result or frozen specification")
        failed = [item for item in summary.get("results", []) if item.get("status") != "completed"]
        if failed:
            task_errors.append(f"contains {len(failed)} failed experiment(s)")
        if summary.get("planner_errors"):
            task_errors.append("contains unresolved planner errors")
        evidence = summary.get("action_evidence")
        selected_axis = next(
            (
                item.get("axis")
                for item in summary.get("research_plan", {}).get("actions", [])
                if item.get("action_id")
                == summary.get("research_plan", {}).get("selected_action_id")
            ),
            None,
        )
        if selected_axis in {"feature", "model", "data", "replicate"} and not (
            evidence and evidence.get("single_axis_verified")
        ):
            task_errors.append("selected research action lacks matched/replicated evidence")

        std = finalist.get("validation_std") if finalist else None
        score = finalist.get("validation_score") if finalist else None
        standard_error = (
            float(std) / math.sqrt(len(set(seeds)))
            if std is not None and seeds
            else None
        )
        metric = summary.get("profile", {}).get("metric")
        if standard_error is None:
            task_errors.append("finalist lacks multi-seed uncertainty")
            stability_value = None
        elif metric == "mae":
            stability_value = standard_error / max(abs(float(score)), 0.1)
            if stability_value > 0.10:
                task_errors.append(
                    f"relative validation standard error {stability_value:.4f} exceeds 0.10"
                )
        else:
            stability_value = standard_error
            if standard_error > stability_threshold:
                task_errors.append(
                    f"validation standard error {standard_error:.4f} exceeds {stability_threshold:.4f}"
                )
        if evidence and evidence.get("status") in {"refuted", "inconclusive"}:
            warnings.append(
                f"{task_name}: selected action evidence is {evidence['status']}"
            )
        if spec is not None:
            finalists[task_name] = spec
        errors.extend(f"{task_name}: {message}" for message in task_errors)
        task_audits.append(
            {
                "task": task_name,
                "validation_seeds": seeds,
                "finalist_run_id": finalist_id,
                "validation_score": score,
                "validation_std": std,
                "stability_value": stability_value,
                "action_status": evidence.get("status") if evidence else None,
                "passed": not task_errors,
                "errors": task_errors,
            }
        )

    ledger = EvidenceLedger(run_dir / "evidence_ledger.jsonl").verify()
    if not ledger.get("valid"):
        errors.append("discovery evidence ledger hash chain is invalid")
    finalists_hash = _digest(finalists)
    audit_base = {
        "protocol": "tdc22-global-freeze-audit-v1",
        "benchmark_protocol_sha256": protocol.get("sha256"),
        "discovery_run_dir": str(run_dir),
        "tasks": task_audits,
        "finalists_sha256": finalists_hash,
        "evidence_ledger": ledger,
        "source_snapshot": {
            "source_sha256": source_manifest.get("source_sha256"),
            "valid": source_valid,
        },
        "warnings": warnings,
        "errors": errors,
        "test_loaded": False,
        "passed": not errors and set(finalists) == expected,
    }
    audit = {**audit_base, "authorization_sha256": _digest(audit_base)}
    output_audit.parent.mkdir(parents=True, exist_ok=True)
    output_finalists.parent.mkdir(parents=True, exist_ok=True)
    output_audit.write_text(json.dumps(audit, indent=2, sort_keys=True), encoding="utf-8")
    output_finalists.write_text(
        json.dumps(finalists, indent=2, sort_keys=True), encoding="utf-8"
    )
    return audit


def verify_freeze_authorization(
    audit_path: Path,
    finalists: dict[str, Any],
) -> dict[str, Any]:
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    claimed = audit.get("authorization_sha256")
    base = {key: value for key, value in audit.items() if key != "authorization_sha256"}
    if claimed != _digest(base):
        raise RuntimeError("Freeze authorization SHA-256 is invalid")
    if not audit.get("passed") or audit.get("test_loaded") is not False:
        raise RuntimeError("Freeze authorization did not pass the discovery-only audit")
    if audit.get("finalists_sha256") != _digest(finalists):
        raise RuntimeError("Finalists do not match the audited global freeze")
    return audit
