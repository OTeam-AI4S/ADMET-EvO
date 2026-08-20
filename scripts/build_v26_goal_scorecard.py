"""Verify the v2.6 CYP2C9 result and rebuild the dated 22-task scorecard."""

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

TASK = "CYP2C9_Substrate_CarbonMangels"


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _digest(payload: dict[str, Any]) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_v26_result(root: Path, mapping: dict[str, Any]) -> tuple[dict, dict]:
    final_dir = root / "runs/v26-official" / TASK / "final"
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
        raise ValueError("v2.6 execution-lock hash mismatch")
    if result.get("freeze_sha256") != freeze["sha256"]:
        raise ValueError("v2.6 result/execution-lock mismatch")
    if result.get("seeds") != [1, 2, 3, 4, 5]:
        raise ValueError("v2.6 official seeds are incomplete")
    expected = ADMETAgent._normalize_frozen(resolve_task(TASK), mapping[TASK])
    if result.get("finalists", {}).get(TASK) != expected:
        raise ValueError("v2.6 executed finalist differs from the full freeze")
    audit = result.get("audit", [])
    if len(audit) != 5 or {int(row["seed"]) for row in audit} != set(range(1, 6)):
        raise ValueError("v2.6 result audit is incomplete")
    if any(row.get("freeze_sha256") != freeze["sha256"] for row in audit):
        raise ValueError("v2.6 result audit hash mismatch")

    seed_records = {}
    prediction_hashes = {}
    for seed in range(1, 6):
        artifact = final_dir / "artifacts" / TASK / f"seed_{seed}"
        execution_path = artifact / "execution_audit.json"
        focal = artifact / "focal_minimol_16"
        manifest_path = focal / "input_manifest.json"
        runtime_path = focal / "runtime_report.json"
        input_path = focal / "input.npz"
        focal_predictions_path = focal / "predictions.npy"
        execution = _load(execution_path)
        manifest = _load(manifest_path)
        runtime = _load(runtime_path)
        if execution.get("task") != TASK or execution.get("seed") != seed:
            raise ValueError(f"v2.6 executor audit mismatch for seed {seed}")
        if execution.get("test_labels_passed_to_executor") is not False:
            raise ValueError(f"v2.6 executor received test labels for seed {seed}")
        if execution.get("fit_rows") != 534 or execution.get("predict_rows") != 135:
            raise ValueError(f"v2.6 row-count mismatch for seed {seed}")
        if execution.get("focal_minimol_16", {}).get("predict_labels_passed") is not False:
            raise ValueError(f"v2.6 focal expert received prediction labels for seed {seed}")
        if manifest.get("predict_labels_present") is not False:
            raise ValueError(f"v2.6 focal input contains prediction labels for seed {seed}")
        if manifest.get("input_sha256") != _sha256(input_path):
            raise ValueError(f"v2.6 focal input hash mismatch for seed {seed}")
        if runtime.get("input_sha256") != manifest["input_sha256"]:
            raise ValueError(f"v2.6 focal runtime/input mismatch for seed {seed}")
        if runtime.get("predictions_sha256") != _sha256(focal_predictions_path):
            raise ValueError(f"v2.6 focal prediction hash mismatch for seed {seed}")
        if runtime.get("members") != 16 or runtime.get("epochs") != 100:
            raise ValueError(f"v2.6 focal runtime changed for seed {seed}")
        if runtime.get("predict_labels_present") is not False:
            raise ValueError(f"v2.6 focal runtime reports prediction labels for seed {seed}")
        seed_records[str(seed)] = {
            "execution_audit": execution_path.relative_to(root).as_posix(),
            "runtime_report": runtime_path.relative_to(root).as_posix(),
            "input_sha256": manifest["input_sha256"],
            "focal_predictions_sha256": runtime["predictions_sha256"],
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
        "test_labels_passed_to_executor": False,
    }


def build_v26_scorecard(repo_root: Path, snapshot_path: Path) -> dict[str, Any]:
    root = repo_root.resolve()
    snapshot = _load(snapshot_path)
    prior = _load(root / "reports/goal_scorecard_v24.json")
    full_freeze_path = root / "runs/v26-cyp2c9-freeze/full-22-task.json"
    mapping_path = root / "runs/v26-cyp2c9-freeze/full-22-task.mapping.json"
    full_freeze = _load(full_freeze_path)
    mapping = _load(mapping_path)
    full_payload = {
        "protocol": full_freeze["protocol"],
        "tasks": sorted(mapping),
        "finalists": mapping,
    }
    if full_freeze.get("sha256") != _digest(full_payload):
        raise ValueError("v2.6 full-freeze hash mismatch")
    if full_freeze.get("finalists") != mapping:
        raise ValueError("v2.6 report/mapping mismatch")
    if full_freeze.get("ready_for_upgrade_finalization") is not True:
        raise ValueError("v2.6 full freeze was not ready")

    official, official_record = verify_v26_result(root, mapping)
    candidate_mean, candidate_std = map(float, next(iter(official["results"].values())))
    prior_row = next(row for row in prior["tasks"] if row["task"] == TASK)
    previous_mean = float(prior_row["score"])
    accepted = candidate_mean > previous_mean
    retained_mean = candidate_mean if accepted else previous_mean
    retained_std = candidate_std if accepted else float(prior_row["std"])

    comparison = []
    for row in prior["tasks"]:
        task = row["task"]
        board = snapshot["pages"][task]
        mean = retained_mean if task == TASK else float(row["score"])
        std = retained_std if task == TASK else float(row["std"])
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
                    "recipe(rank(2d_lightgbm@0.80+focal_minimol_16@0.20))"
                    if task == TASK and accepted
                    else row["finalist"]
                ),
            }
        )
    scorecard = build_scorecard({"tasks": comparison}, snapshot)
    scorecard["freeze_sha256"] = full_freeze["sha256"]
    scorecard["upgrade_tasks"] = [TASK]
    scorecard["provenance"] = dict(prior["provenance"])
    scorecard["provenance"][TASK] = {
        "source_version": "v2.6",
        **official_record,
    }
    scorecard["official_results_complete"] = True
    scorecard["rank_statistics"] = {
        "mean_rank": statistics.mean(row["estimated_rank"] for row in scorecard["tasks"]),
        "median_rank": statistics.median(row["estimated_rank"] for row in scorecard["tasks"]),
    }
    confirmation = full_freeze["provenance"][TASK]["fresh_confirmation"]
    cyp_row = next(row for row in scorecard["tasks"] if row["task"] == TASK)
    scorecard["rescue_outcome"] = {
        "task": TASK,
        "decision": "accept_improvement_below_top10" if accepted else "reject_keep_previous",
        "previous_official": previous_mean,
        "candidate_official": candidate_mean,
        "candidate_official_std": candidate_std,
        "official_delta": candidate_mean - previous_mean,
        "retained_score": retained_mean,
        "fresh_validation_mean": confirmation["fixed_rank_fusion"]["mean"],
        "validation_to_official_delta": candidate_mean
        - float(confirmation["fixed_rank_fusion"]["mean"]),
        "estimated_rank": cyp_row["estimated_rank"],
        "top10_threshold": cyp_row["top10_threshold"],
        "remaining_top10_gap": cyp_row["improvement_to_top10"],
        "selection_after_official_result": False,
    }
    scorecard["rejected_official_rescues"] = [
        {
            "task": "Bioavailability_Ma",
            "source": "reports/v25_bio_rescue_outcome.json",
            "candidate_official": 0.675,
            "retained_official": 0.697,
        }
    ]
    return scorecard


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    parser.add_argument(
        "--snapshot",
        type=Path,
        default=Path("reports/leaderboard_snapshot_2026-08-13.json"),
    )
    parser.add_argument("--output", type=Path, default=Path("reports/goal_scorecard_v26"))
    args = parser.parse_args()
    scorecard = build_v26_scorecard(args.repo_root, args.snapshot)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.with_suffix(".json").write_text(
        json.dumps(scorecard, indent=2, sort_keys=True), encoding="utf-8"
    )
    rendered = markdown(scorecard)
    outcome = scorecard["rescue_outcome"]
    rendered += (
        "\n## v2.6 CYP2C9 rescue outcome\n\n"
        f"The preregistered official result improved from {outcome['previous_official']:.3f} "
        f"to {outcome['candidate_official']:.3f} +/- {outcome['candidate_official_std']:.3f}, "
        f"but remains {outcome['remaining_top10_gap']:.3f} below the dated Top-10 threshold. "
        "No post-test reweighting was performed.\n"
    )
    args.output.with_suffix(".md").write_text(rendered, encoding="utf-8")
    print(json.dumps({**scorecard["status"], "rescue_outcome": outcome}, sort_keys=True))


if __name__ == "__main__":
    main()
