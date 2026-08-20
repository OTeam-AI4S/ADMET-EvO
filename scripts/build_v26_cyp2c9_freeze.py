"""Build a new 22-task freeze for the confirmed CYP2C9 focal-rank fusion."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

from admet_agent.frozen_recipe import build_execution_plan
from admet_agent.tasks import resolve_task

TASK = "CYP2C9_Substrate_CarbonMangels"
RECIPE = "rank(2d_lightgbm@0.80+focal_minimol_16@0.20)"


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


def _artifact(root: Path, path: Path) -> dict[str, Any]:
    return {
        "path": path.relative_to(root).as_posix(),
        "bytes": path.stat().st_size,
        "sha256": _sha256(path),
    }


def build(repo_root: Path) -> dict[str, Any]:
    root = repo_root.resolve()
    # Start from v2.4, not v2.5: the v2.5 Bioavailability candidate failed its
    # official one-shot evaluation and must not leak into a later mapping.
    base_report_path = root / "runs/v24-freeze/full-22-task.json"
    base_mapping_path = root / "runs/v24-freeze/full-22-task.mapping.json"
    base_report = _load(base_report_path)
    base_mapping = _load(base_mapping_path)
    base_payload = {
        "protocol": base_report["protocol"],
        "tasks": sorted(base_mapping),
        "finalists": base_mapping,
    }
    if _digest(base_payload) != base_report["sha256"]:
        raise ValueError("v2.4 base freeze hash mismatch")
    if len(base_mapping) != 22:
        raise ValueError("v2.4 base mapping is incomplete")

    confirmation_root = root / "runs/v25-cyp2c9-focal-confirmation"
    prereg_path = confirmation_root / "preregistration.json"
    input_manifest_path = confirmation_root / "input_manifest.json"
    input_path = confirmation_root / "confirmation_inputs.npz"
    result_path = confirmation_root / "result.json"
    predictions_path = confirmation_root / "predictions.npz"
    prereg = _load(prereg_path)
    manifest = _load(input_manifest_path)
    result = _load(result_path)
    prereg_payload = {
        key: value for key, value in prereg.items() if key != "preregistration_sha256"
    }
    if prereg["preregistration_sha256"] != _digest(prereg_payload):
        raise ValueError("CYP2C9 preregistration hash mismatch")
    if manifest.get("input_sha256") != _sha256(input_path):
        raise ValueError("CYP2C9 confirmation input hash mismatch")
    if result.get("preregistration_sha256") != prereg["preregistration_sha256"]:
        raise ValueError("CYP2C9 confirmation/preregistration mismatch")
    if result.get("input_sha256") != manifest["input_sha256"]:
        raise ValueError("CYP2C9 result/input mismatch")
    if result.get("decision") != "retain_for_new_freeze":
        raise ValueError("CYP2C9 fresh confirmation did not pass its promotion gate")
    if result.get("official_test_loaded") is not False or result.get("test_labels_used") is not False:
        raise ValueError("CYP2C9 confirmation is not validation-only")
    if result.get("confirmation_seeds_used_for_candidate_selection") is not False:
        raise ValueError("Fresh CYP2C9 seeds were used to alter the candidate")
    if result.get("paired_seed_improvements", 0) < 3:
        raise ValueError("CYP2C9 paired-improvement gate failed")
    candidate = result["summaries"]["fixed_rank_fusion"]
    baseline = result["summaries"]["lightgbm"]
    if float(candidate["mean"]) <= float(baseline["mean"]):
        raise ValueError("CYP2C9 mean-improvement gate failed")

    finalist = {
        "kind": "recipe",
        "recipe": RECIPE,
        "protocol": {
            "model_size": "84m",
            "learning_rate": 1e-4,
            "epochs": 20,
            "batch_size": 16,
            "inner_folds": 5,
        },
    }
    mapping = dict(base_mapping)
    mapping[TASK] = finalist
    payload = {
        "protocol": "tdc-admet-test-v2.6-cyp2c9-focal-fusion",
        "tasks": sorted(mapping),
        "finalists": mapping,
    }

    mapping_path = root / "runs/v26-cyp2c9-freeze/full-22-task.mapping.json"
    smoke_path = root / "runs/v26-cyp2c9-smoke" / TASK / "seed_1/result.json"
    smoke = _load(smoke_path) if smoke_path.exists() else None
    smoke_checks = {
        "present": smoke is not None,
        "passed": smoke is not None and smoke.get("status") == "passed",
        "task_matches": smoke is not None and smoke.get("task") == TASK,
        "recipe_matches": (
            smoke is not None
            and smoke.get("frozen_config", {}).get("recipe") == RECIPE
        ),
        "mapping_hash_matches": (
            smoke is not None
            and mapping_path.exists()
            and smoke.get("mapping_sha256") == _sha256(mapping_path)
        ),
        "official_test_not_loaded": (
            smoke is not None and smoke.get("official_test_loaded") is False
        ),
        "test_labels_not_passed": (
            smoke is not None
            and smoke.get("test_labels_passed_to_executor") is False
        ),
        "seed1_score_reproduced": (
            smoke is not None
            and math.isclose(
                float(smoke.get("score", float("nan"))),
                0.6979616375730547,
                abs_tol=0.02,
            )
        ),
        "minimol_version_recorded": (
            smoke is not None
            and smoke.get("environment", {}).get("minimol") == "1.3.5"
        ),
    }
    smoke_valid = all(smoke_checks.values())
    artifacts = [
        base_report_path,
        base_mapping_path,
        prereg_path,
        input_manifest_path,
        input_path,
        result_path,
        predictions_path,
        root / "admet_agent/features.py",
        root / "admet_agent/frozen_executor.py",
        root / "admet_agent/frozen_recipe.py",
        root / "scripts/predict_focal_minimol.py",
        root / "scripts/prepare_cyp2c9_focal_confirmation.py",
        root / "scripts/train_cyp2c9_focal_confirmation.py",
    ]
    if smoke_path.exists():
        artifacts.append(smoke_path)
    return {
        "sha256": _digest(payload),
        **payload,
        "coverage": {"expected": 22, "actual": len(mapping), "complete": len(mapping) == 22},
        "derived_from_freeze_sha256": base_report["sha256"],
        "excluded_failed_rescue": {
            "task": "Bioavailability_Ma",
            "outcome": "reports/v25_bio_rescue_outcome.json",
        },
        "upgrade_tasks": [TASK],
        "upgrade_smokes_valid": smoke_valid,
        "ready_for_upgrade_finalization": smoke_valid,
        "official_test_loaded_during_build": False,
        "provenance": {
            TASK: {
                "source": "v2.6_fresh_seed_confirmed_focal_rank_fusion",
                "official_test_runs_after_upgrade": 0,
                "execution_plan": build_execution_plan(resolve_task(TASK), RECIPE),
                "discovery_fusion_mean": 0.489476096522265,
                "fresh_confirmation": result["summaries"],
                "paired_seed_improvements": result["paired_seed_improvements"],
                "preregistration_sha256": prereg["preregistration_sha256"],
                "smoke": {
                    "status": "valid" if smoke_valid else "missing_or_invalid",
                    "checks": smoke_checks,
                },
            }
        },
        "artifacts": [_artifact(root, path) for path in artifacts],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    parser.add_argument(
        "--output", type=Path, default=Path("runs/v26-cyp2c9-freeze/full-22-task")
    )
    args = parser.parse_args()
    report = build(args.repo_root)
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
                "sha256": report["sha256"],
                "tasks": report["coverage"]["actual"],
                "smoke_valid": report["upgrade_smokes_valid"],
                "ready_for_upgrade_finalization": report["ready_for_upgrade_finalization"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
