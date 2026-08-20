"""Build the 22-task v2.7 freeze after fresh MapLight+GNN confirmation."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from admet_agent.frozen_recipe import build_execution_plan
from admet_agent.tasks import resolve_task


TASKS = ["Bioavailability_Ma", "CYP2C9_Substrate_CarbonMangels"]
RECIPE = "maplight_gnn"
CHECKPOINT_SHA256 = "6d0f8febad73e437772ebffc2ac32253d79f86ee138cfc233590ae50fb1cfeb9"


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _freeze_digest(payload: dict[str, Any]) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def _prereg_digest(payload: dict[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


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
    base_report_path = root / "runs/v26-cyp2c9-freeze/full-22-task.json"
    base_mapping_path = root / "runs/v26-cyp2c9-freeze/full-22-task.mapping.json"
    base_report = _load(base_report_path)
    base_mapping = _load(base_mapping_path)
    base_payload = {
        "protocol": base_report["protocol"],
        "tasks": sorted(base_mapping),
        "finalists": base_mapping,
    }
    if _freeze_digest(base_payload) != base_report["sha256"]:
        raise ValueError("v2.6 base freeze hash mismatch")
    if len(base_mapping) != 22:
        raise ValueError("v2.6 base mapping is incomplete")

    confirmation_root = root / "runs/v27-maplight-gnn-confirmation"
    prereg_path = confirmation_root / "preregistration.json"
    result_path = confirmation_root / "result.json"
    predictions_path = confirmation_root / "fresh_predictions.npz"
    feature_path = confirmation_root / "all_development_features.npy"
    feature_manifest_path = confirmation_root / "all_development_features.manifest.json"
    prereg = _load(prereg_path)
    result = _load(result_path)
    feature_manifest = _load(feature_manifest_path)
    prereg_payload = {
        key: value for key, value in prereg.items() if key != "preregistration_sha256"
    }
    if prereg["preregistration_sha256"] != _prereg_digest(prereg_payload):
        raise ValueError("MapLight+GNN preregistration hash mismatch")
    if result.get("preregistration_sha256") != prereg["preregistration_sha256"]:
        raise ValueError("MapLight+GNN result/preregistration mismatch")
    if result.get("promotion_gate_passed") is not True:
        raise ValueError("MapLight+GNN fresh promotion gate failed")
    protocol = result.get("protocol", {})
    if protocol.get("official_test_loaded") is not False:
        raise ValueError("MapLight+GNN confirmation loaded official test data")
    if protocol.get("test_labels_used") is not False:
        raise ValueError("MapLight+GNN confirmation used test labels")
    if protocol.get("fresh_labels_used_for_candidate_selection") is not False:
        raise ValueError("Fresh confirmation labels altered the candidate")
    if result["provenance"].get("predictions_sha256") != _sha256(predictions_path):
        raise ValueError("MapLight+GNN prediction artifact hash mismatch")
    if feature_manifest.get("feature_sha256") != _sha256(feature_path):
        raise ValueError("MapLight+GNN feature artifact hash mismatch")
    checkpoint_records = {
        item["path"]: item for item in feature_manifest["checkpoint_files"]
    }
    checkpoint_path = str(Path.home() / ".cache/molfeat/gin_supervised_masking/model.save")
    if checkpoint_records.get(checkpoint_path, {}).get("sha256") != CHECKPOINT_SHA256:
        raise ValueError("MapLight+GNN checkpoint was not content-addressed")

    for task in TASKS:
        task_result = result["tasks"][task]["summary"]
        if task_result.get("gate_passed") is not True:
            raise ValueError(f"{task} did not pass the fresh confirmation gate")
        if task_result.get("paired_seed_improvements", 0) < 3:
            raise ValueError(f"{task} paired seed gate failed")
        if task_result["maplight_gnn_mean"] <= task_result["maplight_mean"]:
            raise ValueError(f"{task} mean improvement gate failed")
        identity_records = result["identity_audit"][task]
        if any(
            record.get("cross_split_identity_overlap") != 0
            for record in identity_records.values()
        ):
            raise ValueError(f"{task} failed the molecule identity firewall")

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
    for task in TASKS:
        mapping[task] = finalist
    payload = {
        "protocol": "tdc-admet-test-v2.7-maplight-gnn-independent-audit",
        "tasks": sorted(mapping),
        "finalists": mapping,
    }
    mapping_path = root / "runs/v27-maplight-gnn-freeze/full-22-task.mapping.json"
    smoke_records = {}
    all_smokes_valid = True
    for task in TASKS:
        smoke_path = root / "runs/v27-maplight-gnn-smoke" / task / "seed_1/result.json"
        smoke = _load(smoke_path) if smoke_path.exists() else None
        runtime_path = (
            root
            / "runs/v27-maplight-gnn-smoke"
            / task
            / "seed_1/artifacts/maplight_gnn/runtime_report.json"
        )
        runtime = _load(runtime_path) if runtime_path.exists() else None
        checks = {
            "present": smoke is not None,
            "passed": smoke is not None and smoke.get("status") == "passed",
            "task_matches": smoke is not None and smoke.get("task") == task,
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
            "runtime_present": runtime is not None,
            "predict_labels_not_passed": (
                runtime is not None and runtime.get("predict_labels_present") is False
            ),
            "checkpoint_matches": (
                runtime is not None
                and runtime.get("checkpoint_sha256") == CHECKPOINT_SHA256
            ),
            "feature_width_matches": (
                runtime is not None and runtime.get("feature_shape", [None, None])[1] == 2863
            ),
        }
        smoke_records[task] = {
            "status": "valid" if all(checks.values()) else "missing_or_invalid",
            "checks": checks,
        }
        all_smokes_valid = all_smokes_valid and all(checks.values())

    artifacts = [
        base_report_path,
        base_mapping_path,
        prereg_path,
        result_path,
        predictions_path,
        feature_path,
        feature_manifest_path,
        root / "admet_agent/frozen_recipe.py",
        root / "admet_agent/frozen_executor.py",
        root / "scripts/evaluate_maplight_gnn_confirmation.py",
        root / "scripts/predict_maplight_gnn.py",
    ]
    for task in TASKS:
        smoke_path = root / "runs/v27-maplight-gnn-smoke" / task / "seed_1/result.json"
        if smoke_path.exists():
            artifacts.append(smoke_path)
    return {
        "sha256": _freeze_digest(payload),
        **payload,
        "coverage": {"expected": 22, "actual": len(mapping), "complete": len(mapping) == 22},
        "derived_from_freeze_sha256": base_report["sha256"],
        "upgrade_tasks": TASKS,
        "upgrade_smokes_valid": all_smokes_valid,
        "ready_for_upgrade_finalization": all_smokes_valid,
        "official_test_loaded_during_build": False,
        "provenance": {
            task: {
                "source": "independent_2026_audit_plus_fresh_scaffold_confirmation",
                "execution_plan": build_execution_plan(resolve_task(task), RECIPE),
                "fresh_confirmation": result["tasks"][task]["summary"],
                "identity_audit": result["identity_audit"][task],
                "preregistration_sha256": prereg["preregistration_sha256"],
                "checkpoint_sha256": CHECKPOINT_SHA256,
                "smoke": smoke_records[task],
            }
            for task in TASKS
        },
        "artifacts": [_artifact(root, path) for path in artifacts],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    parser.add_argument(
        "--output", type=Path, default=Path("runs/v27-maplight-gnn-freeze/full-22-task")
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
