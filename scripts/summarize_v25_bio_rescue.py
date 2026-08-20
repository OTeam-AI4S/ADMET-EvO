"""Verify and summarize the one-shot v2.5 Bioavailability rescue result.

The v2.5 full freeze is a preregistration artifact.  The per-task finalization
lock is an execution artifact.  This verifier links both hashes, checks all
five recipe audits, and keeps the previous official result when the rescue is
not better.  It never loads TDC data or test labels.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from admet_agent.runner import ADMETAgent
from admet_agent.tasks import resolve_task

TASK = "Bioavailability_Ma"


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _digest(payload: dict[str, Any]) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def _file_record(root: Path, path: Path) -> dict[str, Any]:
    return {
        "path": path.relative_to(root).as_posix(),
        "bytes": path.stat().st_size,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


def _scorecard_row(scorecard: dict[str, Any], task: str) -> dict[str, Any]:
    return next(row for row in scorecard["tasks"] if row["task"] == task)


def build_report(repo_root: Path) -> dict[str, Any]:
    root = repo_root.resolve()
    full_report_path = root / "runs/v25-rescue-freeze/full-22-task.json"
    mapping_path = root / "runs/v25-rescue-freeze/full-22-task.mapping.json"
    final_dir = root / "runs/v25-official" / TASK / "final"
    execution_freeze_path = final_dir / "freeze.json"
    result_path = final_dir / "results.json"
    scorecard_path = root / "reports/goal_scorecard_v24.json"

    full_report = _load(full_report_path)
    mapping = _load(mapping_path)
    execution_freeze = _load(execution_freeze_path)
    result = _load(result_path)
    scorecard = _load(scorecard_path)

    full_payload = {
        "protocol": full_report["protocol"],
        "tasks": sorted(mapping),
        "finalists": mapping,
    }
    if full_report.get("sha256") != _digest(full_payload):
        raise ValueError("v2.5 full-freeze hash mismatch")
    if full_report.get("finalists") != mapping:
        raise ValueError("v2.5 report and mapping differ")
    if full_report.get("ready_for_upgrade_finalization") is not True:
        raise ValueError("v2.5 full freeze was not ready for finalization")
    if full_report.get("upgrade_tasks") != [TASK]:
        raise ValueError("v2.5 upgrade scope changed")

    execution_payload = {
        "protocol": execution_freeze["protocol"],
        "tasks": execution_freeze["tasks"],
        "finalists": execution_freeze["finalists"],
    }
    if execution_freeze.get("sha256") != _digest(execution_payload):
        raise ValueError("Bioavailability execution-lock hash mismatch")
    if result.get("freeze_sha256") != execution_freeze["sha256"]:
        raise ValueError("Bioavailability result/execution-lock mismatch")
    if result.get("seeds") != [1, 2, 3, 4, 5]:
        raise ValueError("Bioavailability official result lacks five fixed seeds")

    expected = ADMETAgent._normalize_frozen(resolve_task(TASK), mapping[TASK])
    if result.get("finalists", {}).get(TASK) != expected:
        raise ValueError("Executed Bioavailability recipe differs from v2.5 mapping")
    if execution_freeze.get("finalists", {}).get(TASK) != expected:
        raise ValueError("Execution lock differs from v2.5 mapping")

    result_audit = result.get("audit", [])
    if len(result_audit) != 5 or {
        int(row["seed"]) for row in result_audit
    } != set(range(1, 6)):
        raise ValueError("Bioavailability result audit is incomplete")
    if any(row.get("freeze_sha256") != execution_freeze["sha256"] for row in result_audit):
        raise ValueError("Bioavailability result audit hash mismatch")

    recipe_audits: dict[str, dict[str, Any]] = {}
    artifact_paths = [full_report_path, mapping_path, execution_freeze_path, result_path]
    prediction_records = {}
    for seed in range(1, 6):
        audit_path = (
            final_dir / "artifacts" / TASK / f"seed_{seed}" / "execution_audit.json"
        )
        audit = _load(audit_path)
        expected_fields = {
            "task": TASK,
            "seed": seed,
            "recipe": "minimol_morgan_kernel",
            "test_labels_passed_to_executor": False,
        }
        if any(audit.get(key) != value for key, value in expected_fields.items()):
            raise ValueError(f"Invalid recipe execution audit for seed {seed}")
        if audit.get("fit_rows") != 512 or audit.get("predict_rows") != 128:
            raise ValueError(f"Unexpected Bioavailability row counts for seed {seed}")
        recipe_audits[str(seed)] = {
            "path": audit_path.relative_to(root).as_posix(),
            "test_labels_passed_to_executor": False,
        }
        prediction_path = final_dir / f"predictions_seed_{seed}.npz"
        prediction_records[str(seed)] = _file_record(root, prediction_path)
        artifact_paths.extend([audit_path, prediction_path])

    official_values = next(iter(result["results"].values()))
    candidate_mean, candidate_std = map(float, official_values)
    baseline_row = _scorecard_row(scorecard, TASK)
    baseline_mean = float(baseline_row["score"])
    accepted = candidate_mean > baseline_mean
    retained_mean = candidate_mean if accepted else baseline_mean
    confirmation = full_report["provenance"][TASK]

    return {
        "version": "v2.5-bioavailability-rescue-outcome",
        "task": TASK,
        "decision": "accept" if accepted else "reject_keep_previous_official",
        "reason": (
            "one-shot official candidate improved the previous official score"
            if accepted
            else "one-shot official candidate underperformed the previous official score"
        ),
        "scores": {
            "previous_official": baseline_mean,
            "candidate_official": candidate_mean,
            "candidate_official_std": candidate_std,
            "official_delta": candidate_mean - baseline_mean,
            "retained_scorecard_score": retained_mean,
            "fresh_scaffold_confirmation_mean": float(
                confirmation["fresh_confirmation_mean"]
            ),
            "validation_to_official_delta": candidate_mean
            - float(confirmation["fresh_confirmation_mean"]),
        },
        "official_test_runs_after_upgrade": 1,
        "selection_after_official_result": False,
        "test_labels_passed_to_executor": False,
        "preregistration": {
            "full_22_task_freeze": full_report_path.relative_to(root).as_posix(),
            "full_22_task_freeze_sha256": full_report["sha256"],
            "upgrade_tasks": [TASK],
        },
        "execution": {
            "freeze": execution_freeze_path.relative_to(root).as_posix(),
            "freeze_sha256": execution_freeze["sha256"],
            "seeds": [1, 2, 3, 4, 5],
            "recipe_audits": recipe_audits,
            "predictions": prediction_records,
        },
        "artifacts": [_file_record(root, path) for path in artifact_paths],
    }


def markdown(report: dict[str, Any]) -> str:
    scores = report["scores"]
    return "\n".join(
        [
            "# v2.5 Bioavailability rescue outcome",
            "",
            f"Decision: **{report['decision']}**.",
            "",
            f"- Previous official AUROC: {scores['previous_official']:.4f}",
            f"- Frozen candidate official AUROC: {scores['candidate_official']:.4f}",
            f"- Official delta: {scores['official_delta']:+.4f}",
            f"- Retained scorecard AUROC: {scores['retained_scorecard_score']:.4f}",
            f"- Fresh scaffold-confirmation AUROC: {scores['fresh_scaffold_confirmation_mean']:.4f}",
            f"- Validation-to-official delta: {scores['validation_to_official_delta']:+.4f}",
            "",
            "The five fixed seeds used the preregistered recipe. All recipe audits "
            "declare `test_labels_passed_to_executor=false`; the official result was "
            "not used for further selection or reweighting.",
            "",
            f"Full 22-task preregistration hash: `{report['preregistration']['full_22_task_freeze_sha256']}`.",
            f"Per-task execution-lock hash: `{report['execution']['freeze_sha256']}`.",
            "",
        ]
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    parser.add_argument(
        "--output", type=Path, default=Path("reports/v25_bio_rescue_outcome")
    )
    args = parser.parse_args()
    report = build_report(args.repo_root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.with_suffix(".json").write_text(
        json.dumps(report, indent=2, sort_keys=True), encoding="utf-8"
    )
    args.output.with_suffix(".md").write_text(markdown(report), encoding="utf-8")
    print(json.dumps({"decision": report["decision"], **report["scores"]}, sort_keys=True))


if __name__ == "__main__":
    main()
