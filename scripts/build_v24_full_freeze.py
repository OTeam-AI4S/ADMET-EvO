"""Merge verified v2.1 finalists with preregistered v2.4 upgrades."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from admet_agent.tasks import TASKS

UPGRADES = {
    "DILI",
    "hERG",
    "Lipophilicity_AstraZeneca",
    "Half_Life_Obach",
}


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _digest(payload: dict[str, Any]) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def validate_legacy_task(repo_root: Path, task: str) -> tuple[dict, dict]:
    final_dir = repo_root / "runs" / "leaderboard-v21" / "tasks" / task / "final"
    freeze_path = final_dir / "freeze.json"
    result_path = final_dir / "results.json"
    freeze = _load(freeze_path)
    result = _load(result_path)
    payload = {
        "protocol": freeze["protocol"],
        "tasks": freeze["tasks"],
        "finalists": freeze["finalists"],
    }
    if freeze["sha256"] != _digest(payload):
        raise ValueError(f"Legacy freeze hash mismatch for {task}")
    if freeze["tasks"] != [task] or set(freeze["finalists"]) != {task}:
        raise ValueError(f"Legacy freeze task mismatch for {task}")
    if result.get("freeze_sha256") != freeze["sha256"]:
        raise ValueError(f"Legacy result/freeze mismatch for {task}")
    if result.get("seeds") != [1, 2, 3, 4, 5]:
        raise ValueError(f"Legacy result seeds are incomplete for {task}")
    audit = result.get("audit", [])
    if len(audit) != 5 or {int(row["seed"]) for row in audit} != set(range(1, 6)):
        raise ValueError(f"Legacy audit seeds are incomplete for {task}")
    if any(row.get("freeze_sha256") != freeze["sha256"] for row in audit):
        raise ValueError(f"Legacy audit hash mismatch for {task}")
    return freeze["finalists"][task], {
        "status": "verified",
        "freeze_sha256": freeze["sha256"],
        "freeze_path": freeze_path.relative_to(repo_root).as_posix(),
        "result_path": result_path.relative_to(repo_root).as_posix(),
        "seeds": result["seeds"],
        "official_result": result["results"],
    }


def build_full_freeze(repo_root: Path) -> dict[str, Any]:
    repo_root = repo_root.resolve()
    upgrade_manifest_path = repo_root / "runs" / "v24-freeze" / "finalists.json"
    upgrade_manifest = _load(upgrade_manifest_path)
    upgrade_mapping = upgrade_manifest["finalists"]
    if set(upgrade_mapping) != UPGRADES:
        raise ValueError("v2.4 upgrade mapping must contain exactly four tasks")
    smoke_by_task = {
        item["task"]: item.get("smoke_test", {"status": "missing"})
        for item in upgrade_manifest["candidates"]
    }
    all_tasks = {spec.name for spec in TASKS.values()}
    finalists = {}
    provenance = {}
    for task in sorted(all_tasks):
        if task in UPGRADES:
            finalists[task] = upgrade_mapping[task]
            provenance[task] = {
                "source": "v2.4_upgrade",
                "validation_only_smoke": smoke_by_task[task],
                "official_test_runs_after_upgrade": 0,
            }
        else:
            finalist, audit = validate_legacy_task(repo_root, task)
            finalists[task] = finalist
            provenance[task] = {"source": "verified_v2.1", **audit}
    if set(finalists) != all_tasks or len(finalists) != 22:
        raise ValueError("Full freeze does not cover all 22 tasks")
    payload = {
        "protocol": "tdc-admet-test-v2.4",
        "tasks": sorted(finalists),
        "finalists": finalists,
    }
    upgrade_smokes_valid = all(
        provenance[task]["validation_only_smoke"].get("status") == "valid"
        for task in UPGRADES
    )
    return {
        "sha256": _digest(payload),
        **payload,
        "coverage": {"expected": 22, "actual": len(finalists), "complete": True},
        "legacy_tasks_verified": 18,
        "upgrade_tasks": sorted(UPGRADES),
        "upgrade_smokes_valid": upgrade_smokes_valid,
        "ready_for_upgrade_finalization": upgrade_smokes_valid,
        "official_test_loaded_during_merge": False,
        "provenance": provenance,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    parser.add_argument(
        "--output", type=Path, default=Path("runs/v24-freeze/full-22-task")
    )
    args = parser.parse_args()
    report = build_full_freeze(args.repo_root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.with_suffix(".json").write_text(
        json.dumps(report, indent=2, sort_keys=True), encoding="utf-8"
    )
    args.output.with_suffix(".mapping.json").write_text(
        json.dumps(report["finalists"], indent=2, sort_keys=True), encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "tasks": report["coverage"]["actual"],
                "legacy_tasks_verified": report["legacy_tasks_verified"],
                "upgrade_smokes_valid": report["upgrade_smokes_valid"],
                "ready_for_upgrade_finalization": report[
                    "ready_for_upgrade_finalization"
                ],
                "sha256": report["sha256"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
