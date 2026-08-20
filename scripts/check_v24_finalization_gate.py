"""Fail closed unless the 22-task v2.4 freeze is ready for one-shot upgrades."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def check_gate(report_path: Path, mapping_path: Path, task: str) -> dict:
    report = json.loads(report_path.read_text(encoding="utf-8"))
    mapping = json.loads(mapping_path.read_text(encoding="utf-8"))
    payload = {
        "protocol": report["protocol"],
        "tasks": sorted(mapping),
        "finalists": mapping,
    }
    digest = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
    checks = {
        "coverage_complete": report.get("coverage", {}).get("complete") is True,
        "all_22_tasks_present": len(mapping) == 22,
        "freeze_hash_matches": digest == report.get("sha256"),
        "upgrade_smokes_valid": report.get("upgrade_smokes_valid") is True,
        "ready_for_upgrade_finalization": (
            report.get("ready_for_upgrade_finalization") is True
        ),
        "task_is_preregistered_upgrade": task in report.get("upgrade_tasks", []),
        "no_upgrade_test_run_recorded": (
            report.get("provenance", {})
            .get(task, {})
            .get("official_test_runs_after_upgrade")
            == 0
        ),
    }
    if not all(checks.values()):
        failed = [name for name, passed in checks.items() if not passed]
        raise RuntimeError(f"v2.4 finalization gate refused {task}: {failed}")
    return {"task": task, "freeze_sha256": digest, "checks": checks}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", required=True)
    parser.add_argument(
        "--report", type=Path, default=Path("runs/v24-freeze/full-22-task.json")
    )
    parser.add_argument(
        "--mapping",
        type=Path,
        default=Path("runs/v24-freeze/full-22-task.mapping.json"),
    )
    args = parser.parse_args()
    print(
        json.dumps(
            check_gate(args.report, args.mapping, args.task),
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
