"""Build a new 22-task freeze that replaces only Bioavailability_Ma.

The prior v2.4 freeze is immutable because four official evaluations already
reference its hash.  This builder derives a new content-addressed freeze from
that mapping, verifies the validation-only MiniMol/Morgan kernel evidence, and
requires a separate validation smoke test before finalization is allowed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

from admet_agent.frozen_recipe import build_execution_plan
from admet_agent.tasks import resolve_task

TASK = "Bioavailability_Ma"
RECIPE = "minimol_morgan_kernel"


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


def _artifact(repo_root: Path, path: Path) -> dict[str, Any]:
    return {
        "path": path.relative_to(repo_root).as_posix(),
        "bytes": path.stat().st_size,
        "sha256": _sha256(path),
    }


def build(repo_root: Path) -> dict[str, Any]:
    repo_root = repo_root.resolve()
    previous_report_path = repo_root / "runs/v24-freeze/full-22-task.json"
    previous_mapping_path = repo_root / "runs/v24-freeze/full-22-task.mapping.json"
    previous_report = _load(previous_report_path)
    previous_mapping = _load(previous_mapping_path)
    previous_payload = {
        "protocol": previous_report["protocol"],
        "tasks": sorted(previous_mapping),
        "finalists": previous_mapping,
    }
    if _digest(previous_payload) != previous_report["sha256"]:
        raise ValueError("Previous v2.4 full-freeze hash does not match its mapping")
    if len(previous_mapping) != 22:
        raise ValueError("Previous v2.4 mapping does not contain 22 tasks")

    confirmation_path = (
        repo_root
        / "runs/v24-minimol-kernel-confirmation"
        / TASK
        / "locked_kernel_confirmation.json"
    )
    confirmation_predictions_path = confirmation_path.with_name(
        "locked_kernel_confirmation_predictions.npz"
    )
    discovery_path = (
        repo_root
        / "runs/v24-minimol-validation"
        / TASK
        / "scaffold_kernels.json"
    )
    embedding_manifest_path = (
        repo_root
        / "runs/v24-minimol-validation"
        / TASK
        / "embeddings.manifest.json"
    )
    confirmation = _load(confirmation_path)
    discovery = _load(discovery_path)
    if confirmation.get("task") != TASK or confirmation.get("metric") != "auroc":
        raise ValueError("Bioavailability confirmation task or metric mismatch")
    if confirmation.get("locked_recipe") != {"kernel": "fusion_chem_0.50", "c": 10.0}:
        raise ValueError("Bioavailability locked kernel recipe changed")
    protocol = confirmation.get("protocol", {})
    if protocol.get("official_test_loaded") is not False:
        raise ValueError("Confirmation is not explicitly validation-only")
    if protocol.get("test_labels_used") is not False:
        raise ValueError("Confirmation reports test-label access")
    if protocol.get("confirmation_seeds_used_for_selection") is not False:
        raise ValueError("Fresh confirmation seeds were used for recipe selection")
    if protocol.get("seeds") != [6, 7, 8, 9, 10]:
        raise ValueError("Confirmation does not contain fresh seeds 6-10")
    scores = [float(value) for value in confirmation["scores"]]
    if len(scores) != 5 or not math.isclose(
        sum(scores) / len(scores), float(confirmation["mean"]), abs_tol=1e-12
    ):
        raise ValueError("Confirmation summary is inconsistent")
    discovery_key = "fusion_chem_0.50_c_10"
    discovery_scores = discovery["fixed_candidates"][discovery_key]["scores"]

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
    mapping = dict(previous_mapping)
    mapping[TASK] = finalist
    payload = {
        "protocol": "tdc-admet-test-v2.5-bioavailability-rescue",
        "tasks": sorted(mapping),
        "finalists": mapping,
    }

    smoke_path = (
        repo_root / "runs/v25-rescue-smoke" / TASK / "seed_1/result.json"
    )
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
            and smoke.get("mapping_sha256")
            == _sha256(
                repo_root / "runs/v25-rescue-freeze/full-22-task.mapping.json"
            )
        ) if (repo_root / "runs/v25-rescue-freeze/full-22-task.mapping.json").exists() else False,
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
                float(discovery_scores[0]),
                # The locked discovery embeddings were generated on CUDA while
                # the smoke test deliberately runs the same frozen MiniMol
                # weights on CPU.  SVC ranking is stable but tiny floating-point
                # differences can move AUROC by one or two pairs.
                abs_tol=0.01,
            )
        ),
        "minimol_version_recorded": (
            smoke is not None
            and smoke.get("environment", {}).get("minimol") == "1.3.5"
        ),
    }
    smoke_valid = all(smoke_checks.values())
    artifacts = [
        previous_report_path,
        previous_mapping_path,
        confirmation_path,
        confirmation_predictions_path,
        discovery_path,
        embedding_manifest_path,
        repo_root / "admet_agent/features.py",
        repo_root / "admet_agent/frozen_executor.py",
        repo_root / "admet_agent/frozen_recipe.py",
        repo_root / "admet_agent/runner.py",
    ]
    if smoke_path.exists():
        artifacts.append(smoke_path)
    return {
        "sha256": _digest(payload),
        **payload,
        "coverage": {"expected": 22, "actual": len(mapping), "complete": len(mapping) == 22},
        "derived_from_freeze_sha256": previous_report["sha256"],
        "upgrade_tasks": [TASK],
        "upgrade_smokes_valid": smoke_valid,
        "ready_for_upgrade_finalization": smoke_valid,
        "official_test_loaded_during_build": False,
        "provenance": {
            TASK: {
                "source": "v2.5_validation_only_rescue",
                "official_test_runs_after_upgrade": 0,
                "execution_plan": build_execution_plan(resolve_task(TASK), RECIPE),
                "discovery_scores": discovery_scores,
                "fresh_confirmation_scores": scores,
                "fresh_confirmation_mean": confirmation["mean"],
                "fresh_confirmation_std": confirmation["std"],
                "smoke": {
                    "status": "valid" if smoke_valid else "missing_or_invalid",
                    "checks": smoke_checks,
                },
            }
        },
        "artifacts": [_artifact(repo_root, path) for path in artifacts],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    parser.add_argument(
        "--output", type=Path, default=Path("runs/v25-rescue-freeze/full-22-task")
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
                "ready_for_upgrade_finalization": report[
                    "ready_for_upgrade_finalization"
                ],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
