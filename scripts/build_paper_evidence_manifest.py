"""Hash and verify the evidence bundle supporting the v2.4 manuscript."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from scripts.build_v24_goal_scorecard import verify_upgrade_result

VALIDATION_REPORTS = {
    "router_replay": "runs/v24-router-replay/report.json",
    "router_loso": "runs/v24-router-loso/report.json",
    "router_context": "runs/v24-router-context/report.json",
    "router_error_analysis": "runs/v24-router-error-analysis/report.json",
    "rescue_ledger": "runs/v24-rescue-ledger/report.json",
    "data_isolation_audit": "runs/v23-ablation-study/data_isolation_audit.json",
}

EVIDENCE_FILES = {
    "baseline_scorecard": "reports/goal_scorecard.json",
    "leaderboard_snapshot": "reports/leaderboard_snapshot_2026-08-13.json",
    "full_freeze": "runs/v24-freeze/full-22-task.json",
    "research_plan": "docs/v24_research_plan.md",
    "paper_readiness": "reports/paper_readiness_v24.md",
    "working_manuscript": "MANUSCRIPT_V24.md",
    **VALIDATION_REPORTS,
}

SOURCE_FILES = (
    "admet_agent/evidence_router.py",
    "admet_agent/frozen_executor.py",
    "admet_agent/frozen_recipe.py",
    "scripts/build_v24_full_freeze.py",
    "scripts/build_v24_goal_scorecard.py",
    "scripts/evaluate_router_loso.py",
    "scripts/evaluate_router_context_ablation.py",
    "scripts/analyze_router_failures.py",
    "scripts/summarize_v24_rescue_experiments.py",
)


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def hash_file(path: Path) -> dict[str, Any]:
    if not path.is_file() or path.stat().st_size == 0:
        raise ValueError(f"Evidence file is missing or empty: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return {"sha256": digest.hexdigest(), "bytes": path.stat().st_size}


def validate_firewall(name: str, payload: dict[str, Any]) -> dict[str, Any]:
    declarations = {}
    for field in ("official_test_loaded", "test_labels_used", "leaderboard_used"):
        if field in payload:
            if payload[field] is not False:
                raise ValueError(f"{name} does not declare {field}=false")
            declarations[field] = False
    if not any(field in declarations for field in ("official_test_loaded", "test_labels_used")):
        raise ValueError(f"{name} lacks an explicit validation/test firewall declaration")
    return declarations


def manifest_digest(payload: dict[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def build_manifest(repo_root: Path) -> dict[str, Any]:
    root = repo_root.resolve()
    freeze = _load(root / "runs/v24-freeze/full-22-task.json")
    if freeze.get("ready_for_upgrade_finalization") is not True:
        raise ValueError("The full v2.4 freeze is not ready for finalization")

    evidence = {}
    firewalls = {}
    for name, relative in EVIDENCE_FILES.items():
        path = root / relative
        evidence[name] = {"path": relative, **hash_file(path)}
        if name in VALIDATION_REPORTS:
            firewalls[name] = validate_firewall(name, _load(path))

    sources = {
        relative: hash_file(root / relative) for relative in SOURCE_FILES
    }
    official = {}
    pending = []
    for task in sorted(freeze["upgrade_tasks"]):
        result_path = root / "runs/v24-official" / task / "final/results.json"
        if not result_path.is_file():
            pending.append(task)
            continue
        result, provenance = verify_upgrade_result(root, task, freeze["finalists"])
        official[task] = {
            **provenance,
            "result_file": hash_file(result_path),
            "result": next(iter(result["results"].values())),
            "seeds": result["seeds"],
        }

    status = "interim_pending_official_results" if pending else "official_results_complete"
    final_scorecard = root / "reports/goal_scorecard_v24.json"
    if pending and final_scorecard.exists():
        raise ValueError("A final v2.4 scorecard exists while official upgrades are pending")
    if not pending:
        evidence["final_scorecard"] = {
            "path": "reports/goal_scorecard_v24.json",
            **hash_file(final_scorecard),
        }

    payload = {
        "version": "v2.4",
        "status": status,
        "full_freeze_sha256": freeze["sha256"],
        "upgrade_tasks": sorted(freeze["upgrade_tasks"]),
        "verified_official_tasks": sorted(official),
        "pending_official_tasks": pending,
        "validation_firewalls": firewalls,
        "evidence": evidence,
        "official_results": official,
        "source_files": sources,
    }
    return {**payload, "manifest_sha256": manifest_digest(payload)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("reports/paper_evidence_manifest_v24.json"),
    )
    args = parser.parse_args()
    manifest = build_manifest(args.repo_root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "status": manifest["status"],
                "manifest_sha256": manifest["manifest_sha256"],
                "verified_official_tasks": manifest["verified_official_tasks"],
                "pending_official_tasks": manifest["pending_official_tasks"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
