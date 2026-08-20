from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from admet_agent.artifact_retention import (
    RetentionTier,
    artifact_type,
    build_retention_manifest,
    write_retention_manifest,
)
from scripts.plan_artifact_retention import main


def _write(path: Path, payload: bytes = b"artifact") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return path


def _records(manifest: dict) -> dict[str, dict]:
    return {record["path"]: record for record in manifest["artifacts"]}


def test_dry_run_classifies_evidence_base_final_and_ephemeral_artifacts(tmp_path: Path):
    runs = tmp_path / "runs"
    result = _write(runs / "pilot" / "DILI" / "results.json", b"{}")
    prediction = _write(runs / "pilot" / "DILI" / "seed_1" / "validation_predictions.npz")
    config = _write(runs / "pilot" / "DILI" / "config.json", b"{}")
    base = _write(runs / "unimol2-weights" / "modelzoo" / "84M" / "checkpoint.pt")
    official = _write(runs / "v24-official" / "DILI" / "final" / "model_0.pth")
    smoke = _write(runs / "unimol2-scale-smoke" / "DILI" / "model_0.pth")
    ablation = _write(runs / "unimol2-scale-ablation" / "DILI" / "model_0.pth")
    cache = _write(runs / "agent" / "feature_cache" / "features.npy")
    ignored = _write(runs / "agent" / "notes.txt")

    before = {path: path.read_bytes() for path in (result, prediction, config, base, official, smoke, ablation, cache, ignored)}
    manifest = build_retention_manifest([runs], project_root=tmp_path, hash_files=False)
    records = _records(manifest)

    assert records["runs/pilot/DILI/results.json"]["retention_tier"] == RetentionTier.EVIDENCE.value
    assert records["runs/pilot/DILI/seed_1/validation_predictions.npz"]["retention_tier"] == RetentionTier.EVIDENCE.value
    assert records["runs/pilot/DILI/config.json"]["retention_tier"] == RetentionTier.EVIDENCE.value
    assert records["runs/unimol2-weights/modelzoo/84M/checkpoint.pt"]["retention_tier"] == RetentionTier.DEPLOYMENT.value
    assert records["runs/v24-official/DILI/final/model_0.pth"]["retention_tier"] == RetentionTier.DEPLOYMENT.value
    assert records["runs/unimol2-scale-smoke/DILI/model_0.pth"]["retention_tier"] == RetentionTier.RECLAIM_CANDIDATE.value
    assert records["runs/unimol2-scale-ablation/DILI/model_0.pth"]["retention_tier"] == RetentionTier.COLD_STORE.value
    assert records["runs/agent/feature_cache/features.npy"]["retention_tier"] == RetentionTier.DEBUG_TTL.value
    assert "runs/agent/notes.txt" not in records
    assert manifest["summary"]["estimated_reclaimable_bytes"] == smoke.stat().st_size
    assert manifest["summary"]["estimated_local_release_after_approved_archival_bytes"] == (
        smoke.stat().st_size + ablation.stat().st_size
    )
    assert manifest["policy"]["mutates_files"] is False
    assert {path: path.read_bytes() for path in before} == before


def test_directly_referenced_smoke_checkpoint_is_protected_as_deployment(tmp_path: Path):
    runs = tmp_path / "runs"
    checkpoint = _write(runs / "smoke" / "DILI" / "model_0.pth")
    freeze = runs / "freeze" / "finalists.json"
    freeze.parent.mkdir(parents=True)
    freeze.write_text(
        json.dumps({"finalists": {"DILI": {"checkpoint": "runs/smoke/DILI/model_0.pth"}}}),
        encoding="utf-8",
    )

    manifest = build_retention_manifest([runs], project_root=tmp_path, reference_roots=[runs])
    record = _records(manifest)["runs/smoke/DILI/model_0.pth"]
    assert record["retention_tier"] == RetentionTier.DEPLOYMENT.value
    assert record["referenced"] is True
    assert record["reference_status"] == "direct"
    assert record["referenced_by"] == ["runs/freeze/finalists.json"]
    assert record["reclaimable"] is False
    assert checkpoint.exists()


def test_default_hashes_small_evidence_but_skips_large_checkpoint(tmp_path: Path):
    small = _write(tmp_path / "runs" / "task" / "results.json", b'{"ok": true}')
    large = _write(tmp_path / "runs" / "smoke" / "model_0.pth", b"x" * 32)
    manifest = build_retention_manifest(
        [tmp_path / "runs"], project_root=tmp_path, hash_max_bytes=16
    )
    records = _records(manifest)
    assert records["runs/task/results.json"]["sha256"] == hashlib.sha256(small.read_bytes()).hexdigest()
    assert records["runs/task/results.json"]["sha256_status"] == "computed"
    assert records["runs/smoke/model_0.pth"]["sha256"] is None
    assert records["runs/smoke/model_0.pth"]["sha256_status"] == "skipped_size"
    assert large.read_bytes() == b"x" * 32


def test_only_requested_roots_are_inventoried_and_missing_root_warns(tmp_path: Path):
    _write(tmp_path / "runs" / "selected" / "model.joblib")
    _write(tmp_path / "runs" / "excluded" / "model.joblib")
    manifest = build_retention_manifest(
        [tmp_path / "runs" / "selected", tmp_path / "runs" / "missing"],
        project_root=tmp_path,
        hash_files=False,
    )
    assert [record["path"] for record in manifest["artifacts"]] == [
        "runs/selected/model.joblib"
    ]
    assert manifest["warnings"] == ["run root does not exist: runs/missing"]


def test_writer_and_cli_emit_json_plan_without_artifact_mutation(tmp_path: Path, capsys):
    checkpoint = _write(tmp_path / "runs" / "screen" / "model_0.pth", b"checkpoint")
    output = tmp_path / "reports" / "plan.json"
    assert main(
        [
            "--project-root",
            str(tmp_path),
            "--run-root",
            str(tmp_path / "runs"),
            "--output",
            str(output),
            "--no-hash",
        ]
    ) == 0
    printed = json.loads(capsys.readouterr().out)
    saved = json.loads(output.read_text(encoding="utf-8"))
    assert printed["mode"] == "dry_run_read_only"
    assert saved["summary"]["estimated_reclaimable_bytes"] == len(b"checkpoint")
    assert checkpoint.read_bytes() == b"checkpoint"
    assert saved["artifacts"][0]["sha256_status"] == "disabled"

    second = write_retention_manifest(saved, tmp_path / "reports" / "second.json")
    assert second.exists()


def test_invalid_empty_roots_and_hash_limit_are_rejected(tmp_path: Path):
    with pytest.raises(ValueError, match="at least one run root"):
        build_retention_manifest([], project_root=tmp_path)
    with pytest.raises(ValueError, match="hash_max_bytes"):
        build_retention_manifest([tmp_path], project_root=tmp_path, hash_max_bytes=-1)


def test_artifact_type_ignores_python_pth_and_recognizes_requested_families():
    assert artifact_type(Path("site-packages/distutils-precedence.pth")) is None
    assert artifact_type(Path("run/unimol2/model_3.pth")) == "unimol2_checkpoint"
    assert artifact_type(Path("run/artifacts/model.joblib")) == "trained_model"
    assert artifact_type(Path("run/feature_cache/key.npy")) == "feature_cache"
    assert artifact_type(Path("run/predictions.npz")) == "predictions"
    assert artifact_type(Path("run/results.json")) == "results"
    assert artifact_type(Path("run/input_manifest.json")) == "manifest"
    assert artifact_type(Path("run/finalists.mapping.json")) == "manifest"
