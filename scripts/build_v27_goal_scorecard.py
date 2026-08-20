"""Verify v2.7 official artifacts and rebuild the dated 22-task scorecard."""

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
from scripts.compare_leaderboards import expected_rank


TASKS = {
    "Bioavailability_Ma": {"fit_rows": 512, "predict_rows": 128},
    "CYP2C9_Substrate_CarbonMangels": {"fit_rows": 534, "predict_rows": 135},
}
CHECKPOINT_SHA256 = "6d0f8febad73e437772ebffc2ac32253d79f86ee138cfc233590ae50fb1cfeb9"
UPSTREAM_SHA256 = "74fbd1c98d9afa7fa4bda1add21efd429e20dee0a4b0fb8fa7e9b3825c21fe13"


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _digest(payload: dict[str, Any]) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_official_result(
    root: Path, task: str, mapping: dict[str, Any]
) -> tuple[dict[str, Any], dict[str, Any]]:
    final_dir = root / "runs/v27-official" / task / "final"
    result_path = final_dir / "results.json"
    freeze_path = final_dir / "freeze.json"
    result = _load(result_path)
    freeze = _load(freeze_path)
    freeze_payload = {
        "protocol": freeze["protocol"],
        "tasks": freeze["tasks"],
        "finalists": freeze["finalists"],
    }
    if freeze.get("sha256") != _digest(freeze_payload):
        raise ValueError(f"{task}: execution-lock hash mismatch")
    if result.get("freeze_sha256") != freeze["sha256"]:
        raise ValueError(f"{task}: result/execution-lock mismatch")
    if result.get("seeds") != [1, 2, 3, 4, 5]:
        raise ValueError(f"{task}: official seeds are incomplete")
    expected = ADMETAgent._normalize_frozen(resolve_task(task), mapping[task])
    if result.get("finalists", {}).get(task) != expected:
        raise ValueError(f"{task}: executed finalist differs from v2.7 freeze")
    audit = result.get("audit", [])
    if len(audit) != 5 or {int(row["seed"]) for row in audit} != set(range(1, 6)):
        raise ValueError(f"{task}: result audit is incomplete")
    if any(row.get("freeze_sha256") != freeze["sha256"] for row in audit):
        raise ValueError(f"{task}: result audit hash mismatch")

    expected_rows = TASKS[task]
    seed_records = {}
    prediction_hashes = {}
    for seed in range(1, 6):
        artifact = final_dir / "artifacts" / task / f"seed_{seed}"
        execution_path = artifact / "execution_audit.json"
        expert_dir = artifact / "maplight_gnn"
        input_path = expert_dir / "input.json"
        runtime_path = expert_dir / "runtime_report.json"
        expert_predictions_path = expert_dir / "predictions.npy"
        execution = _load(execution_path)
        input_payload = _load(input_path)
        runtime = _load(runtime_path)
        if execution.get("task") != task or execution.get("seed") != seed:
            raise ValueError(f"{task} seed {seed}: executor audit mismatch")
        if execution.get("test_labels_passed_to_executor") is not False:
            raise ValueError(f"{task} seed {seed}: executor received test labels")
        if execution.get("maplight_gnn", {}).get("predict_labels_passed") is not False:
            raise ValueError(f"{task} seed {seed}: expert received prediction labels")
        if execution.get("fit_rows") != expected_rows["fit_rows"]:
            raise ValueError(f"{task} seed {seed}: fit row-count mismatch")
        if execution.get("predict_rows") != expected_rows["predict_rows"]:
            raise ValueError(f"{task} seed {seed}: prediction row-count mismatch")
        if input_payload.get("predict_labels_present") is not False:
            raise ValueError(f"{task} seed {seed}: input reports prediction labels")
        if "predict_labels" in input_payload:
            raise ValueError(f"{task} seed {seed}: input contains prediction labels")
        if len(input_payload.get("fit_labels", [])) != expected_rows["fit_rows"]:
            raise ValueError(f"{task} seed {seed}: fit label-count mismatch")
        if len(input_payload.get("predict_smiles", [])) != expected_rows["predict_rows"]:
            raise ValueError(f"{task} seed {seed}: prediction SMILES-count mismatch")
        if runtime.get("input_sha256") != _sha256(input_path):
            raise ValueError(f"{task} seed {seed}: runtime/input hash mismatch")
        if runtime.get("predictions_sha256") != _sha256(expert_predictions_path):
            raise ValueError(f"{task} seed {seed}: expert prediction hash mismatch")
        if runtime.get("predict_labels_present") is not False:
            raise ValueError(f"{task} seed {seed}: runtime label firewall failed")
        if runtime.get("checkpoint_sha256") != CHECKPOINT_SHA256:
            raise ValueError(f"{task} seed {seed}: checkpoint hash mismatch")
        if runtime.get("upstream_sha256") != UPSTREAM_SHA256:
            raise ValueError(f"{task} seed {seed}: upstream source hash mismatch")
        if runtime.get("feature_shape") != [
            expected_rows["fit_rows"] + expected_rows["predict_rows"],
            2863,
        ]:
            raise ValueError(f"{task} seed {seed}: feature shape mismatch")
        feature_manifest_path = root / runtime["feature_manifest"]
        if not feature_manifest_path.exists():
            raise ValueError(f"{task} seed {seed}: feature manifest is missing")
        if runtime.get("feature_manifest_sha256") != _sha256(feature_manifest_path):
            raise ValueError(f"{task} seed {seed}: feature manifest hash mismatch")
        seed_records[str(seed)] = {
            "execution_audit": execution_path.relative_to(root).as_posix(),
            "runtime_report": runtime_path.relative_to(root).as_posix(),
            "input_sha256": runtime["input_sha256"],
            "expert_predictions_sha256": runtime["predictions_sha256"],
            "feature_manifest_sha256": runtime["feature_manifest_sha256"],
            "feature_cache_hit": runtime["feature_cache_hit"],
        }
        prediction_path = final_dir / f"predictions_seed_{seed}.npz"
        prediction_hashes[str(seed)] = _sha256(prediction_path)
    return result, {
        "source": result_path.relative_to(root).as_posix(),
        "freeze": freeze_path.relative_to(root).as_posix(),
        "freeze_sha256": freeze["sha256"],
        "result_file_sha256": _sha256(result_path),
        "seed_records": seed_records,
        "prediction_hashes": prediction_hashes,
        "checkpoint_sha256": CHECKPOINT_SHA256,
        "upstream_sha256": UPSTREAM_SHA256,
        "test_labels_passed_to_executor": False,
    }


def build_v27_scorecard(repo_root: Path, snapshot_path: Path) -> dict[str, Any]:
    root = repo_root.resolve()
    snapshot = _load(snapshot_path)
    prior = _load(root / "reports/goal_scorecard_v26.json")
    full_freeze_path = root / "runs/v27-maplight-gnn-freeze/full-22-task.json"
    mapping_path = root / "runs/v27-maplight-gnn-freeze/full-22-task.mapping.json"
    full_freeze = _load(full_freeze_path)
    mapping = _load(mapping_path)
    full_payload = {
        "protocol": full_freeze["protocol"],
        "tasks": sorted(mapping),
        "finalists": mapping,
    }
    if full_freeze.get("sha256") != _digest(full_payload):
        raise ValueError("v2.7 full-freeze hash mismatch")
    if full_freeze.get("finalists") != mapping:
        raise ValueError("v2.7 report/mapping mismatch")
    if full_freeze.get("ready_for_upgrade_finalization") is not True:
        raise ValueError("v2.7 full freeze was not ready")

    official_results = {}
    official_records = {}
    retained = {}
    for task in TASKS:
        official, official_record = verify_official_result(root, task, mapping)
        candidate_mean, candidate_std = map(float, next(iter(official["results"].values())))
        prior_row = next(row for row in prior["tasks"] if row["task"] == task)
        previous_mean = float(prior_row["score"])
        accepted = candidate_mean > previous_mean
        retained[task] = {
            "mean": candidate_mean if accepted else previous_mean,
            "std": candidate_std if accepted else float(prior_row["std"]),
            "accepted": accepted,
            "previous": previous_mean,
            "candidate": candidate_mean,
            "candidate_std": candidate_std,
        }
        official_results[task] = official
        official_records[task] = official_record

    comparison = []
    for row in prior["tasks"]:
        task = row["task"]
        board = snapshot["pages"][task]
        upgrade = retained.get(task)
        mean = upgrade["mean"] if upgrade else float(row["score"])
        std = upgrade["std"] if upgrade else float(row["std"])
        rank_best, rank_worst = expected_rank(mean, board["entries"], board["direction"])
        comparison.append(
            {
                "task": task,
                "metric": row["metric"],
                "mean": mean,
                "std": std,
                "estimated_rank": rank_best,
                "tie_worst_rank": rank_worst,
                "finalist": (
                    "recipe(maplight_gnn)"
                    if upgrade and upgrade["accepted"]
                    else row["finalist"]
                ),
            }
        )
    scorecard = build_scorecard({"tasks": comparison}, snapshot)
    scorecard["freeze_sha256"] = full_freeze["sha256"]
    scorecard["upgrade_tasks"] = list(TASKS)
    scorecard["provenance"] = dict(prior["provenance"])
    for task in TASKS:
        scorecard["provenance"][task] = {
            "source_version": "v2.7",
            **official_records[task],
        }
    scorecard["official_results_complete"] = True
    scorecard["rank_statistics"] = {
        "mean_rank": statistics.mean(row["estimated_rank"] for row in scorecard["tasks"]),
        "median_rank": statistics.median(row["estimated_rank"] for row in scorecard["tasks"]),
    }
    outcomes = {}
    for task in TASKS:
        task_row = next(row for row in scorecard["tasks"] if row["task"] == task)
        upgrade = retained[task]
        validation = full_freeze["provenance"][task]["fresh_confirmation"]
        outcomes[task] = {
            "decision": (
                "accept_top10_rescue"
                if upgrade["accepted"] and task_row["estimated_rank"] <= 10
                else "accept_improvement_below_top10"
                if upgrade["accepted"]
                else "reject_keep_previous"
            ),
            "previous_official": upgrade["previous"],
            "candidate_official": upgrade["candidate"],
            "candidate_official_std": upgrade["candidate_std"],
            "official_delta": upgrade["candidate"] - upgrade["previous"],
            "retained_score": upgrade["mean"],
            "fresh_validation_mean": validation["maplight_gnn_mean"],
            "validation_to_official_delta": upgrade["candidate"]
            - float(validation["maplight_gnn_mean"]),
            "estimated_rank": task_row["estimated_rank"],
            "top10_threshold": task_row["top10_threshold"],
            "remaining_top10_gap": task_row["improvement_to_top10"],
            "selection_after_official_result": False,
        }
    scorecard["rescue_outcomes"] = outcomes
    return scorecard


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    parser.add_argument(
        "--snapshot",
        type=Path,
        default=Path("reports/leaderboard_snapshot_2026-08-17.json"),
    )
    parser.add_argument("--output", type=Path, default=Path("reports/goal_scorecard_v27"))
    args = parser.parse_args()
    scorecard = build_v27_scorecard(args.repo_root, args.snapshot)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.with_suffix(".json").write_text(
        json.dumps(scorecard, indent=2, sort_keys=True), encoding="utf-8"
    )
    rendered = markdown(scorecard)
    rendered += "\n## v2.7 MapLight+GNN outcomes\n\n"
    for task, outcome in scorecard["rescue_outcomes"].items():
        rendered += (
            f"- {task}: {outcome['previous_official']:.3f} -> "
            f"{outcome['candidate_official']:.3f} +/- "
            f"{outcome['candidate_official_std']:.3f}; estimated rank "
            f"{outcome['estimated_rank']}; remaining Top-10 gap "
            f"{outcome['remaining_top10_gap']:.3f}.\n"
        )
    rendered += "\nNo post-test tuning or reweighting was performed.\n"
    rendered += (
        "\nBioavailability has a separately verified material leaderboard "
        "discrepancy (`reports/bio_leaderboard_discrepancy_v27.json`). The "
        "displayed threshold is retained until an authoritative correction.\n"
    )
    args.output.with_suffix(".md").write_text(rendered, encoding="utf-8")
    print(
        json.dumps(
            {**scorecard["status"], "rescue_outcomes": scorecard["rescue_outcomes"]},
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
