"""Merge verified v2.4 upgrades into the dated 22-task goal scorecard."""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
from pathlib import Path
from typing import Any

from admet_agent.runner import ADMETAgent
from admet_agent.tasks import resolve_task
from scripts.build_goal_scorecard import build_scorecard, markdown
from scripts.compare_leaderboards import expected_rank, finalist_label


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _freeze_digest(freeze: dict[str, Any]) -> str:
    payload = {
        "protocol": freeze["protocol"],
        "tasks": freeze["tasks"],
        "finalists": freeze["finalists"],
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def _saved_training_seeds(path: Path) -> dict[str, int]:
    values: dict[str, int] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        key, separator, raw = line.partition(":")
        if separator and key.strip() in {"seed", "split_seed"}:
            values[key.strip()] = int(raw.strip())
    return values


def verify_upgrade_result(
    repo_root: Path,
    task: str,
    frozen_mapping: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    final_dir = repo_root / "runs" / "v24-official" / task / "final"
    result_path = final_dir / "results.json"
    freeze_path = final_dir / "freeze.json"
    result = _load(result_path)
    freeze = _load(freeze_path)
    if freeze.get("sha256") != _freeze_digest(freeze):
        raise ValueError(f"Official freeze hash mismatch for {task}")
    if result.get("freeze_sha256") != freeze["sha256"]:
        raise ValueError(f"Official result/freeze mismatch for {task}")
    if result.get("seeds") != [1, 2, 3, 4, 5]:
        raise ValueError(f"Official seeds are incomplete for {task}")
    audit = result.get("audit", [])
    if len(audit) != 5 or {int(row["seed"]) for row in audit} != set(range(1, 6)):
        raise ValueError(f"Official audit seeds are incomplete for {task}")
    if any(row.get("freeze_sha256") != freeze["sha256"] for row in audit):
        raise ValueError(f"Official audit hash mismatch for {task}")
    expected = ADMETAgent._normalize_frozen(resolve_task(task), frozen_mapping[task])
    if result.get("finalists", {}).get(task) != expected:
        raise ValueError(f"Official finalist differs from full freeze for {task}")
    seed_records = {}
    for seed in range(1, 6):
        config = (
            final_dir
            / "artifacts"
            / task
            / f"seed_{seed}"
            / "unimol2_finetune"
            / "config.yaml"
        )
        observed = _saved_training_seeds(config)
        expected_seeds = {"seed": seed, "split_seed": seed}
        if observed != expected_seeds:
            raise ValueError(
                f"Saved Uni-Mol seeds mismatch for {task} seed {seed}: {observed}"
            )
        seed_records[str(seed)] = observed
    return result, {
        "source": result_path.relative_to(repo_root).as_posix(),
        "freeze": freeze_path.relative_to(repo_root).as_posix(),
        "freeze_sha256": freeze["sha256"],
        "training_seeds": seed_records,
    }


def build_v24_scorecard(repo_root: Path, snapshot_path: Path) -> dict[str, Any]:
    repo_root = repo_root.resolve()
    snapshot = _load(snapshot_path)
    full_freeze_path = repo_root / "runs" / "v24-freeze" / "full-22-task.json"
    full_freeze = _load(full_freeze_path)
    if full_freeze.get("ready_for_upgrade_finalization") is not True:
        raise ValueError("Full 22-task freeze is not ready")
    finalists = full_freeze["finalists"]
    upgrade_tasks = set(full_freeze["upgrade_tasks"])
    comparison_rows = []
    provenance = {}
    for task, board in snapshot["pages"].items():
        if task in upgrade_tasks:
            result, record = verify_upgrade_result(repo_root, task, finalists)
            provenance[task] = {"source_version": "v2.4", **record}
        else:
            path = (
                repo_root
                / "runs"
                / "leaderboard-v21"
                / "tasks"
                / task
                / "final"
                / "results.json"
            )
            result = _load(path)
            provenance[task] = {
                "source_version": "v2.1",
                "source": path.relative_to(repo_root).as_posix(),
                "freeze_sha256": result["freeze_sha256"],
            }
        values = next(iter(result["results"].values()))
        mean, std = float(values[0]), float(values[1])
        rank_best, rank_worst = expected_rank(
            mean, board["entries"], board["direction"]
        )
        finalist = result["finalists"][task]
        label = (
            f"recipe({finalist['recipe']})"
            if finalist.get("kind") == "recipe"
            else finalist_label(finalist)
        )
        comparison_rows.append(
            {
                "task": task,
                "metric": board["metric"],
                "mean": mean,
                "std": std,
                "estimated_rank": rank_best,
                "tie_worst_rank": rank_worst,
                "finalist": label,
            }
        )
    scorecard = build_scorecard({"tasks": comparison_rows}, snapshot)
    scorecard["freeze_sha256"] = full_freeze["sha256"]
    scorecard["upgrade_tasks"] = sorted(upgrade_tasks)
    scorecard["provenance"] = provenance
    scorecard["official_results_complete"] = len(provenance) == 22
    scorecard["rank_statistics"] = {
        "mean_rank": statistics.mean(
            row["estimated_rank"] for row in scorecard["tasks"]
        ),
        "median_rank": statistics.median(
            row["estimated_rank"] for row in scorecard["tasks"]
        ),
    }
    return scorecard


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    parser.add_argument(
        "--snapshot",
        type=Path,
        default=Path("reports/leaderboard_snapshot_2026-08-13.json"),
    )
    parser.add_argument(
        "--output", type=Path, default=Path("reports/goal_scorecard_v24")
    )
    args = parser.parse_args()
    scorecard = build_v24_scorecard(args.repo_root, args.snapshot)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.with_suffix(".json").write_text(
        json.dumps(scorecard, indent=2, sort_keys=True), encoding="utf-8"
    )
    args.output.with_suffix(".md").write_text(markdown(scorecard), encoding="utf-8")
    print(json.dumps(scorecard["status"], sort_keys=True))


if __name__ == "__main__":
    main()
