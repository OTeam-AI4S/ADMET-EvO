#!/usr/bin/env python3
"""Sealed all-22 Evidence Packet data-only evaluator and closed-loop CLI."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from preflight_evidence_packet_v4 import preflight

from admet_agent.evidence_packet_closed_loop import (
    EvidencePacketClosedLoop,
    finalize_evaluation_artifact,
    load_verified_evaluation_artifact_against_data,
)
from admet_agent.evidence_packet_loop import CANONICAL_TASKS
from admet_agent.evidence_packet_splits import (
    REFERENCE_SPLIT_SEED,
    scaffold_train_validation_split,
)
from admet_agent.evidence_packets import (
    assemble_evidence_training_data,
    prepare_evidence_packet,
    sha256_file,
    validate_evidence_packet,
)
from admet_agent.features import FeatureCache
from admet_agent.schemas import ExperimentSpec
from admet_agent.selection import build_model, predict, score
from admet_agent.tasks import resolve_task

EVALUATOR_VERSION = "admet-evidence-packet-sealed-evaluator-v4.5"
FROZEN_MODEL = {
    "model": "lightgbm",
    "hyperparameters": {},
    "imbalance": "none",
}
FROZEN_FEATURES = ("morgan",)
FROZEN_IMPLEMENTATION = (
    ROOT / "admet_agent/evidence_packet_closed_loop.py",
    ROOT / "admet_agent/evidence_packet_splits.py",
    ROOT / "admet_agent/evidence_packets.py",
    ROOT / "admet_agent/external_structure_firewall.py",
    ROOT / "admet_agent/features.py",
    ROOT / "admet_agent/selection.py",
    ROOT / "admet_agent/schemas.py",
    ROOT / "admet_agent/tasks.py",
    Path(__file__).resolve(),
)


def _implementation_hashes() -> dict[str, str]:
    return {
        str(path.relative_to(ROOT)): sha256_file(path) for path in FROZEN_IMPLEMENTATION
    }


def _array_sha256(values: np.ndarray) -> str:
    array = np.ascontiguousarray(values)
    return hashlib.sha256(array.tobytes()).hexdigest()


def _load_acquisition_plan(path: Path) -> tuple[str, list[str]]:
    resolved = path.resolve()
    payload = json.loads(resolved.read_text(encoding="utf-8"))
    if payload.get("protocol") != "admet-evidence-packet-acquisition-plan-v4.1":
        raise ValueError("acquisition plan protocol mismatch")
    targets = payload.get("target_tasks")
    if (
        not isinstance(targets, list)
        or len(targets) not in {2, 3}
        or len(set(targets)) != len(targets)
        or not set(targets).issubset(CANONICAL_TASKS)
        or payload.get("target_count") != len(targets)
    ):
        raise ValueError("acquisition plan must bind exactly 2 or 3 canonical tasks")
    if payload.get("test_labels_loaded") is not False:
        raise ValueError("acquisition plan must be test-label blind")
    if payload.get("legacy_bulk_pool_used") is not False:
        raise ValueError("acquisition plan may not use the legacy bulk pool")
    ticket_tasks = [ticket.get("task") for ticket in payload.get("tickets", [])]
    if ticket_tasks != targets:
        raise ValueError("acquisition plan tickets differ from target tasks")
    return sha256_file(resolved), list(targets)


def _fit_with_sample_weight(
    model: object,
    x: np.ndarray,
    y: np.ndarray,
    sample_weight: np.ndarray,
) -> object:
    if np.allclose(sample_weight, 1.0):
        return model.fit(x, y)
    try:
        return model.fit(x, y, sample_weight=sample_weight)
    except TypeError as error:
        raise TypeError(
            "frozen model does not consume Evidence Packet sample_weight"
        ) from error


def _read_registry(registry_path: Path) -> tuple[dict[str, Any], dict[str, list[Path]]]:
    report = preflight(registry_path)
    if report["status"] == "failed_closed":
        raise RuntimeError("Evidence Packet registry failed closed")
    by_task: dict[str, list[Path]] = {task: [] for task in CANONICAL_TASKS}
    for packet in report["packets"]:
        by_task[packet["task"]].append(Path(packet["manifest_path"]))
    return report, by_task


def _load_train_validation(
    data_dir: Path, task_name: str, seed: int
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    benchmark = task_name.lower()
    source = data_dir / "admet_group" / benchmark / "train_val.csv"
    raw = pd.read_csv(source, usecols=["Drug", "Y"])
    train, validation, audit = scaffold_train_validation_split(raw, seed=seed)
    audit = {
        **audit,
        "train_val_path": str(source.resolve()),
        "train_val_sha256": sha256_file(source),
        "train_rows": len(train),
        "validation_rows": len(validation),
    }
    return train, validation, audit


def _evaluate_task(
    *,
    task_name: str,
    data_dir: Path,
    seed: int,
    weight_mode: str,
    manifest_paths: list[Path],
    output_dir: Path,
    feature_cache_root: Path,
    baseline_task: dict[str, Any] | None,
    n_jobs: int,
) -> dict[str, Any]:
    started = time.perf_counter()
    task = resolve_task(task_name)
    train, validation, split_audit = _load_train_validation(data_dir, task_name, seed)
    test_path = data_dir / "admet_group" / task_name.lower() / "test.csv"
    prepared = []
    for manifest_path in manifest_paths:
        packet = validate_evidence_packet(manifest_path)
        prepared.append(
            prepare_evidence_packet(
                packet,
                train_smiles=train["Drug"].astype(str).tolist(),
                validation_smiles=validation["Drug"].astype(str).tolist(),
                test_path=test_path,
            )
        )
    fit_frame, merge_audit = assemble_evidence_training_data(
        train, prepared, weight_mode=weight_mode
    )
    if manifest_paths and merge_audit["external_rows"] == 0:
        raise RuntimeError(
            f"{task_name}: registered packets produced zero admitted external rows"
        )

    all_smiles = (
        fit_frame["Drug"].astype(str).tolist() + validation["Drug"].astype(str).tolist()
    )
    features = FeatureCache(
        feature_cache_root / task_name, device="cpu"
    ).get_or_compute(all_smiles, FROZEN_FEATURES)
    x_fit = features[: len(fit_frame)]
    x_validation = features[len(fit_frame) :]
    y_fit = fit_frame["Y"].to_numpy(
        dtype=int if task.task_type == "classification" else float
    )
    y_validation = validation["Y"].to_numpy(
        dtype=int if task.task_type == "classification" else float
    )
    sample_weight = fit_frame["sample_weight"].to_numpy(dtype=float)

    target_mean = 0.0
    target_scale = 1.0
    model_y = y_fit
    if task.task_type == "regression":
        target_mean = float(np.mean(y_fit))
        target_scale = float(np.std(y_fit))
        if not math.isfinite(target_scale) or target_scale <= 0:
            target_scale = 1.0
        model_y = (y_fit - target_mean) / target_scale
    experiment = ExperimentSpec.from_dict(
        {
            "task": task_name,
            "seed": seed,
            "representations": list(FROZEN_FEATURES),
            **FROZEN_MODEL,
            "data_policy": "official",
            "device": "cpu",
            "n_jobs": n_jobs,
            "hypothesis": "frozen Evidence Packet data-only comparison",
        }
    )
    model = build_model(experiment, task, np.asarray(model_y))
    fitted = _fit_with_sample_weight(model, x_fit, np.asarray(model_y), sample_weight)
    predictions = np.asarray(predict(fitted, x_validation, task), dtype=float)
    if task.task_type == "regression":
        predictions = predictions * target_scale + target_mean
    raw_score = score(y_validation, predictions, task.metric)

    prediction_dir = output_dir / "predictions"
    prediction_dir.mkdir(parents=True, exist_ok=True)
    prediction_path = prediction_dir / f"{task_name}.npy"
    np.save(prediction_path, predictions, allow_pickle=False)
    if baseline_task is None:
        baseline_score = raw_score
        normalized = 0.0
    else:
        if baseline_task["metric"] != task.metric:
            raise ValueError(f"{task_name}: frozen baseline metric mismatch")
        baseline_score = float(baseline_task["score"])
        denominator = max(abs(baseline_score), 1e-12)
        normalized = (
            (baseline_score - raw_score) / denominator
            if task.metric == "mae"
            else (raw_score - baseline_score) / denominator
        )
    return {
        "status": "completed",
        "task": task_name,
        "task_type": task.task_type,
        "metric": task.metric,
        "score": float(raw_score),
        "baseline_score": float(baseline_score),
        "normalized_improvement": float(normalized),
        "native_train_rows": int(merge_audit["native_rows"]),
        "external_train_rows": int(merge_audit["external_rows"]),
        "weight_mode": weight_mode,
        "sample_weight_consumed": bool(not np.allclose(sample_weight, 1.0)),
        "merge_audit": merge_audit,
        "packet_audits": [item.audit for item in prepared],
        "split_sha256": split_audit["split_sha256"],
        "split": split_audit,
        "validation_rows": len(validation),
        "validation_target_sha256": _array_sha256(y_validation),
        "prediction_path": str(prediction_path.relative_to(output_dir)),
        "prediction_sha256": sha256_file(prediction_path),
        "fit_wall_seconds": time.perf_counter() - started,
        "test_labels_loaded": False,
    }


def run_all22_evaluation(
    *,
    registry_path: Path,
    data_dir: Path,
    output_dir: Path,
    weight_mode: str,
    baseline_artifact: Path | None,
    acquisition_plan_sha256: str,
    acquisition_target_tasks: list[str],
    seed: int = REFERENCE_SPLIT_SEED,
    n_jobs: int = 8,
    feature_cache_root: Path | None = None,
) -> Path:
    if seed != REFERENCE_SPLIT_SEED:
        raise ValueError(
            "formal closed-loop discovery is frozen to split/model seed 42"
        )
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    report, manifests_by_task = _read_registry(registry_path.resolve())
    baseline = (
        load_verified_evaluation_artifact_against_data(
            baseline_artifact,
            data_dir=data_dir,
        )
        if baseline_artifact is not None
        else None
    )
    if baseline is None and report["active_packet_count"] != 0:
        raise ValueError("baseline evaluation requires an empty packet registry")
    if baseline is not None and baseline["kind"] != "baseline":
        raise ValueError("baseline_artifact must be a baseline evaluation")

    frozen_before = _implementation_hashes()
    task_results: dict[str, dict[str, Any]] = {}
    cache_root = feature_cache_root or (
        ROOT / "checkpoints/evidence_packet_v4/features"
    )
    for task_name in CANONICAL_TASKS:
        task_results[task_name] = _evaluate_task(
            task_name=task_name,
            data_dir=data_dir,
            seed=seed,
            weight_mode=weight_mode,
            manifest_paths=manifests_by_task[task_name],
            output_dir=output_dir,
            feature_cache_root=cache_root,
            baseline_task=None if baseline is None else baseline["tasks"][task_name],
            n_jobs=n_jobs,
        )
    frozen_after = _implementation_hashes()
    if frozen_before != frozen_after:
        raise RuntimeError("frozen model/feature/evaluator implementation changed")

    normalized_scores = {
        task: float(task_results[task]["normalized_improvement"])
        for task in CANONICAL_TASKS
    }
    packet_ids = sorted(packet["packet_id"] for packet in report["packets"])
    payload = {
        "evaluator": EVALUATOR_VERSION,
        "kind": "baseline" if baseline is None else "challenger",
        "status": "completed",
        "evaluated_at": datetime.now(timezone.utc).isoformat(),
        "seed": seed,
        "split": "Auto-Research-Molprop random-scaffold greedy-fill",
        "model_config": FROZEN_MODEL,
        "feature_config": list(FROZEN_FEATURES),
        "weight_mode": weight_mode,
        "acquisition_plan_sha256": acquisition_plan_sha256,
        "acquisition_target_tasks": acquisition_target_tasks,
        "registry_path": str(registry_path.resolve()),
        "registry_sha256": sha256_file(registry_path.resolve()),
        "packet_ids": packet_ids,
        "baseline_artifact_sha256": (
            None if baseline is None else baseline["artifact_sha256"]
        ),
        "tasks": task_results,
        "normalized_scores": normalized_scores,
        "mean_normalized_improvement": (
            sum(normalized_scores.values()) / len(normalized_scores)
        ),
        "frozen_implementation_before": frozen_before,
        "frozen_implementation_after": frozen_after,
        "all_22_tasks_completed": True,
        "validation_labels_held_by_sealed_evaluator": True,
        "test_labels_loaded": False,
        "legacy_bulk_pool_used": False,
    }
    artifact_path = output_dir / "evaluation.json"
    finalize_evaluation_artifact(payload, artifact_path)
    load_verified_evaluation_artifact_against_data(
        artifact_path,
        data_dir=data_dir,
        expected_baseline_sha256=(
            None if baseline is None else baseline["artifact_sha256"]
        ),
    )
    return artifact_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--action", choices=("initialize", "trial"), required=True)
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--state", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--acquisition-plan", type=Path, required=True)
    parser.add_argument(
        "--weight-mode",
        choices=("paper_equal", "native_priority"),
        default="paper_equal",
    )
    parser.add_argument("--n-jobs", type=int, default=8)
    parser.add_argument("--feature-cache-root", type=Path)
    args = parser.parse_args()
    acquisition_plan_sha256, acquisition_target_tasks = _load_acquisition_plan(
        args.acquisition_plan
    )

    loop = EvidencePacketClosedLoop(args.state, data_dir=args.data_dir)
    if args.action == "initialize":
        baseline_path = None
    else:
        state = loop.load()
        if acquisition_plan_sha256 != state["acquisition_plan_sha256"]:
            raise ValueError("trial acquisition plan differs from initialized loop")
        if acquisition_target_tasks != state["acquisition_target_tasks"]:
            raise ValueError("trial acquisition targets differ from initialized loop")
        if args.weight_mode != state["weight_mode"]:
            raise ValueError("trial weight mode differs from initialized loop")
        baseline_snapshot = Path(state["baseline"]["snapshot_path"])
        candidates = list((baseline_snapshot / "evaluation").rglob("evaluation.json"))
        loop.validate_candidate_registry(args.registry)
        baseline_candidates = [
            path
            for path in candidates
            if json.loads(path.read_text(encoding="utf-8")).get("kind") == "baseline"
        ]
        if len(baseline_candidates) != 1:
            raise RuntimeError("could not resolve the unique frozen baseline artifact")
        baseline_path = baseline_candidates[0]

    artifact = run_all22_evaluation(
        registry_path=args.registry,
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        weight_mode=args.weight_mode,
        baseline_artifact=baseline_path,
        acquisition_plan_sha256=acquisition_plan_sha256,
        acquisition_target_tasks=acquisition_target_tasks,
        n_jobs=args.n_jobs,
        feature_cache_root=args.feature_cache_root,
    )
    if args.action == "initialize":
        state = loop.initialize(
            evaluation_artifact=artifact,
            registry_path=args.registry,
        )
        result: dict[str, Any] = {
            "action": "initialized",
            "state": str(loop.state_path),
            "snapshot_sha256": state["incumbent"]["snapshot_sha256"],
        }
    else:
        decision = loop.submit(
            evaluation_artifact=artifact,
            registry_path=args.registry,
        )
        result = {"action": "trial", "decision": decision.to_dict()}
    result["artifact"] = str(artifact)
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
