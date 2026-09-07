import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from admet_agent.evidence_packet_closed_loop import (
    REQUIRED_FROZEN_IMPLEMENTATION,
    EvidencePacketClosedLoop,
    finalize_evaluation_artifact,
    load_verified_evaluation_artifact,
    load_verified_evaluation_artifact_against_data,
)
from admet_agent.evidence_packet_loop import CANONICAL_TASKS
from admet_agent.evidence_packet_splits import scaffold_train_validation_split
from admet_agent.evidence_packets import SCHEMA_VERSION, sha256_file
from admet_agent.selection import score
from admet_agent.tasks import resolve_task


def _registry(path: Path, entries: list[dict]) -> Path:
    payload = {
        "schema_version": "admet-evidence-packet-registry-v4.1",
        "policy": {
            "active_packet_root": "data/evidence_packets_v4/packets",
            "legacy_bulk_pool_allowed": False,
        },
        "active_packets": entries,
        "quarantined_legacy_sources": [],
        "test_labels_loaded": False,
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _write_ames_packet(root: Path) -> Path:
    root.mkdir()
    records = root / "records.csv"
    pd.DataFrame(
        [{"Drug": "CCOC", "Y": 1, "source_record_id": "table-1-row-1"}]
    ).to_csv(records, index=False)
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "packet_id": "paper-ames-assay-001",
        "task": "AMES",
        "task_type": "classification",
        "source": {
            "kind": "independent_peer_reviewed_assay",
            "name": "Independent AMES study",
            "title": "Direct bacterial mutagenicity assay",
            "citation_url": "https://doi.org/10.1000/example-ames",
            "citation_type": "doi",
            "stable_identifier": "doi:10.1000/example-ames",
            "license_or_access_terms": "factual extraction with citation",
            "independence_status": "verified_independent",
            "independence_evidence": "separate study and laboratory",
        },
        "assay": {
            "endpoint_definition": "Ames bacterial mutagenicity",
            "assay_type": "direct bacterial reverse mutation assay",
            "organism": "Salmonella typhimurium",
            "biological_system": "bacterial culture",
            "source_record_locator": "Table 1 row 1",
        },
        "label": {
            "source_value_definition": "reported mutagenicity call",
            "mapping_rule": "Y=1 for mutagenic and Y=0 for non-mutagenic",
            "direction": "positive means mutagenic",
            "transform": "none",
        },
        "review": {
            "status": "modeling_ready",
            "reviewer": "data-steward",
            "reviewed_at": "2026-08-30T00:00:00Z",
            "endpoint_match": "direct",
            "notes": "independence and endpoint checked",
            "native_priority_external_weight": 0.75,
        },
        "records": {
            "path": "records.csv",
            "sha256": sha256_file(records),
            "row_count": 1,
        },
    }
    path = root / "manifest.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    return path


def _write_validation_data(root: Path) -> Path:
    smiles = [
        "c1ccccc1O",
        "c1ccccc1N",
        "C1CCCCC1O",
        "C1CCCCC1N",
        "c1ccncc1O",
        "c1ccncc1N",
        "C1CCCC1O",
        "C1CCCC1N",
        "C1CCC1O",
        "C1CCC1N",
        "c1ccc2ccccc2c1O",
        "c1ccc2ccccc2c1N",
    ]
    for task_name in CANONICAL_TASKS:
        directory = root / "admet_group" / task_name.lower()
        directory.mkdir(parents=True)
        pd.DataFrame({"Drug": smiles, "Y": [0, 1] * 6}).to_csv(
            directory / "train_val.csv", index=False
        )
    return root


def _artifact(
    root: Path,
    registry: Path,
    data_dir: Path,
    *,
    kind: str,
    packet_ids: list[str],
    baseline_sha256: str | None = None,
    baseline_tasks: dict | None = None,
    weight_mode: str = "paper_equal",
    improve_ames: bool = False,
) -> Path:
    root.mkdir()
    prediction_root = root / "predictions"
    prediction_root.mkdir()
    tasks = {}
    normalized_scores = {}
    for task_name in CANONICAL_TASKS:
        task = resolve_task(task_name)
        raw = pd.read_csv(
            data_dir / "admet_group" / task_name.lower() / "train_val.csv"
        )
        _, validation, split_audit = scaffold_train_validation_split(raw, seed=42)
        y_validation = validation["Y"].to_numpy(
            dtype=int if task.task_type == "classification" else float
        )
        predictions = (
            np.full(len(validation), 0.5, dtype=float)
            if task.task_type == "classification"
            else np.zeros(len(validation), dtype=float)
        )
        if improve_ames and task_name == "AMES":
            predictions = y_validation.astype(float)
        raw_score = score(y_validation, predictions, task.metric)
        baseline_score = (
            raw_score
            if baseline_tasks is None
            else float(baseline_tasks[task_name]["score"])
        )
        denominator = max(abs(baseline_score), 1e-12)
        normalized = (
            0.0
            if kind == "baseline"
            else (
                (baseline_score - raw_score) / denominator
                if task.metric == "mae"
                else (raw_score - baseline_score) / denominator
            )
        )
        prediction_path = prediction_root / f"{task_name}.npy"
        np.save(prediction_path, predictions, allow_pickle=False)
        tasks[task_name] = {
            "status": "completed",
            "metric": task.metric,
            "score": raw_score,
            "baseline_score": baseline_score,
            "normalized_improvement": normalized,
            "prediction_path": str(prediction_path.relative_to(root)),
            "prediction_sha256": sha256_file(prediction_path),
            "validation_rows": len(validation),
            "validation_target_sha256": hashlib.sha256(
                np.ascontiguousarray(y_validation).tobytes()
            ).hexdigest(),
            "split_sha256": split_audit["split_sha256"],
        }
        normalized_scores[task_name] = normalized
    payload = {
        "kind": kind,
        "status": "completed",
        "seed": 42,
        "model_config": {
            "model": "lightgbm",
            "hyperparameters": {},
            "imbalance": "none",
        },
        "feature_config": ["morgan"],
        "weight_mode": weight_mode,
        "acquisition_plan_sha256": "a" * 64,
        "acquisition_target_tasks": ["AMES", "CYP3A4_Veith", "CYP2C9_Veith"],
        "baseline_artifact_sha256": baseline_sha256,
        "registry_sha256": sha256_file(registry),
        "packet_ids": packet_ids,
        "tasks": tasks,
        "normalized_scores": normalized_scores,
        "mean_normalized_improvement": (
            sum(normalized_scores.values()) / len(normalized_scores)
        ),
        "frozen_implementation_before": {
            relative: sha256_file(Path(__file__).resolve().parents[1] / relative)
            for relative in REQUIRED_FROZEN_IMPLEMENTATION
        },
        "frozen_implementation_after": {
            relative: sha256_file(Path(__file__).resolve().parents[1] / relative)
            for relative in REQUIRED_FROZEN_IMPLEMENTATION
        },
        "test_labels_loaded": False,
        "legacy_bulk_pool_used": False,
    }
    artifact = root / "evaluation.json"
    finalize_evaluation_artifact(payload, artifact)
    return artifact


def test_artifact_only_loop_snapshots_data_and_rejects_without_mutation(tmp_path: Path):
    data_dir = _write_validation_data(tmp_path / "data")
    empty_registry = _registry(tmp_path / "baseline-registry.json", [])
    baseline_artifact = _artifact(
        tmp_path / "baseline-eval",
        empty_registry,
        data_dir,
        kind="baseline",
        packet_ids=[],
    )
    baseline = load_verified_evaluation_artifact(baseline_artifact)
    loop = EvidencePacketClosedLoop(tmp_path / "loop/state.json", data_dir=data_dir)
    initialized = loop.initialize(
        evaluation_artifact=baseline_artifact,
        registry_path=empty_registry,
    )
    baseline_snapshot = initialized["incumbent"]["snapshot_sha256"]
    assert (Path(initialized["incumbent"]["snapshot_path"]) / "registry.json").is_file()

    manifest = _write_ames_packet(tmp_path / "packet")
    entry = {
        "packet_id": "paper-ames-assay-001",
        "task": "AMES",
        "manifest_path": str(manifest),
        "manifest_sha256": sha256_file(manifest),
    }
    candidate_registry = _registry(tmp_path / "candidate-registry.json", [entry])
    candidate_check = loop.validate_candidate_registry(candidate_registry)
    assert candidate_check["mutation"]["action"] == "add"
    assert candidate_check["targeted_weak_task"] is True
    mismatched_artifact = _artifact(
        tmp_path / "mismatched-weight-eval",
        candidate_registry,
        data_dir,
        kind="challenger",
        packet_ids=["paper-ames-assay-001"],
        baseline_sha256=baseline["artifact_sha256"],
        baseline_tasks=baseline["tasks"],
        weight_mode="native_priority",
    )
    with pytest.raises(ValueError, match="implementation contract"):
        loop.submit(
            evaluation_artifact=mismatched_artifact,
            registry_path=candidate_registry,
        )
    assert json.loads(candidate_registry.read_text())["active_packets"] == []
    candidate_registry = _registry(candidate_registry, [entry])
    tied_artifact = _artifact(
        tmp_path / "tied-eval",
        candidate_registry,
        data_dir,
        kind="challenger",
        packet_ids=["paper-ames-assay-001"],
        baseline_sha256=baseline["artifact_sha256"],
        baseline_tasks=baseline["tasks"],
    )
    rejected = loop.submit(
        evaluation_artifact=tied_artifact,
        registry_path=candidate_registry,
    )
    assert rejected.keep is False
    assert loop.load()["incumbent"]["snapshot_sha256"] == baseline_snapshot
    assert len(list(loop.snapshot_root.iterdir())) == 1
    assert json.loads(candidate_registry.read_text())["active_packets"] == []

    candidate_registry = _registry(candidate_registry, [entry])

    improved_artifact = _artifact(
        tmp_path / "improved-eval",
        candidate_registry,
        data_dir,
        kind="challenger",
        packet_ids=["paper-ames-assay-001"],
        baseline_sha256=baseline["artifact_sha256"],
        baseline_tasks=baseline["tasks"],
        improve_ames=True,
    )
    kept = loop.submit(
        evaluation_artifact=improved_artifact,
        registry_path=candidate_registry,
    )
    state = loop.load()
    assert kept.keep is True
    assert state["incumbent"]["packet_ids"] == ["paper-ames-assay-001"]
    snapshot = Path(state["incumbent"]["snapshot_path"])
    assert (snapshot / "packets/paper-ames-assay-001/manifest.json").is_file()
    assert (snapshot / "packets/paper-ames-assay-001/records.csv").is_file()
    assert (snapshot / "evaluation/evaluation.json").is_file()
    assert state["baseline"]["snapshot_sha256"] == baseline_snapshot


def test_prediction_tampering_invalidates_evaluation_artifact(tmp_path: Path):
    data_dir = _write_validation_data(tmp_path / "data")
    registry = _registry(tmp_path / "registry.json", [])
    artifact = _artifact(
        tmp_path / "evaluation",
        registry,
        data_dir,
        kind="baseline",
        packet_ids=[],
    )
    np.save(
        artifact.parent / "predictions/AMES.npy",
        np.asarray([0.99]),
        allow_pickle=False,
    )
    with pytest.raises(ValueError, match="prediction hash mismatch"):
        load_verified_evaluation_artifact(artifact)


def test_metric_fabrication_is_recomputed_against_private_validation(tmp_path: Path):
    data_dir = _write_validation_data(tmp_path / "data")
    registry = _registry(tmp_path / "registry.json", [])
    artifact = _artifact(
        tmp_path / "evaluation",
        registry,
        data_dir,
        kind="baseline",
        packet_ids=[],
    )
    payload = json.loads(artifact.read_text())
    payload.pop("artifact_sha256")
    payload["tasks"]["AMES"]["score"] = 0.75
    finalize_evaluation_artifact(payload, artifact)
    with pytest.raises(ValueError, match="metric differs from bound predictions"):
        load_verified_evaluation_artifact_against_data(
            artifact,
            data_dir=data_dir,
        )
