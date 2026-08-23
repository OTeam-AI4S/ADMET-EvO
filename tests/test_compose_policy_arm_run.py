from __future__ import annotations

import hashlib
import json
import sqlite3
from pathlib import Path

import pytest

from admet_agent.benchmark_protocol import write_default_protocol
from scripts.compose_policy_arm_run import compose_policy_arm_run


def _write_protocol(path: Path, tasks: list[str]) -> dict:
    protocol = write_default_protocol(path)
    protocol["tasks"] = tasks
    protocol.pop("sha256", None)
    protocol["sha256"] = hashlib.sha256(
        json.dumps(protocol, sort_keys=True).encode()
    ).hexdigest()
    path.write_text(json.dumps(protocol), encoding="utf-8")
    return json.loads(path.read_text(encoding="utf-8"))


def _make_shard(root: Path, protocol: dict, task: str, arm: str = "automl") -> None:
    run_id = f"{task}-fit"
    task_dir = root / "tasks" / task
    artifact = root / "artifacts" / run_id
    task_dir.mkdir(parents=True)
    artifact.mkdir(parents=True)
    (artifact / "experiment.json").write_text(
        json.dumps({"task": task, "seed": 1, "model": "lightgbm"}),
        encoding="utf-8",
    )
    summary = {
        "profile": {"task": task},
        "benchmark_arm": arm,
        "benchmark_protocol_sha256": protocol["sha256"],
        "source_sha256": protocol.get("implementation_source_sha256"),
        "validation_seeds": protocol["discovery_seeds"],
        "finalist_run_id": run_id,
        "finalist_spec": {"model": "lightgbm"},
        "results": [{"run_id": run_id, "seed": 1, "status": "completed"}],
        "robust_results": [{"run_id": run_id}],
    }
    (task_dir / "search_seed_1.json").write_text(
        json.dumps(summary), encoding="utf-8"
    )


def test_composes_exact_task_cover_and_records_aborted_overhead(tmp_path: Path) -> None:
    protocol_path = tmp_path / "protocol.json"
    protocol = _write_protocol(protocol_path, ["HIA_Hou", "BBB_Martins"])
    source_a, source_b = tmp_path / "a", tmp_path / "b"
    _make_shard(source_a, protocol, "HIA_Hou")
    _make_shard(source_b, protocol, "BBB_Martins")
    database = source_a / "experiments.sqlite"
    connection = sqlite3.connect(database)
    connection.execute(
        "CREATE TABLE runs (task TEXT, status TEXT, elapsed_seconds REAL)"
    )
    connection.execute("INSERT INTO runs VALUES ('HIA_Hou','completed',12.5)")
    connection.commit()
    connection.close()

    output = tmp_path / "composed"
    result = compose_policy_arm_run(
        protocol_path=protocol_path,
        arm="automl",
        task_sources={"HIA_Hou": source_a, "BBB_Martins": source_b},
        output_dir=output,
        aborted_task_sources={"HIA_Hou": source_a},
    )

    assert (output / "tasks/HIA_Hou/search_seed_1.json").is_file()
    assert (output / "artifacts/HIA_Hou-fit/experiment.json").is_file()
    assert result["canonical_candidates_from_aborted_runs"] is False
    assert result["aborted_runs"][0]["completed_fits"] == 1
    assert result["aborted_runs"][0]["excluded_task"] == "HIA_Hou"
    manifest = json.loads(
        (output / "composition_manifest.json").read_text(encoding="utf-8")
    )
    assert len(manifest["task_shards"]) == 2


def test_rejects_incomplete_task_cover(tmp_path: Path) -> None:
    protocol_path = tmp_path / "protocol.json"
    protocol = _write_protocol(protocol_path, ["HIA_Hou", "BBB_Martins"])
    source = tmp_path / "a"
    _make_shard(source, protocol, "HIA_Hou")
    with pytest.raises(ValueError, match="exactly cover"):
        compose_policy_arm_run(
            protocol_path=protocol_path,
            arm="automl",
            task_sources={"HIA_Hou": source},
            output_dir=tmp_path / "out",
        )


def test_allows_explicit_protocol_task_subset(tmp_path: Path) -> None:
    protocol_path = tmp_path / "protocol.json"
    protocol = _write_protocol(protocol_path, ["HIA_Hou", "BBB_Martins"])
    source = tmp_path / "a"
    _make_shard(source, protocol, "HIA_Hou")
    result = compose_policy_arm_run(
        protocol_path=protocol_path,
        arm="automl",
        task_sources={"HIA_Hou": source},
        expected_tasks=["HIA_Hou"],
        output_dir=tmp_path / "out",
    )
    assert result["protocol_task_count"] == 2
    assert result["composed_tasks"] == ["HIA_Hou"]


def test_rejects_existing_output(tmp_path: Path) -> None:
    protocol_path = tmp_path / "protocol.json"
    protocol = _write_protocol(protocol_path, ["HIA_Hou"])
    source = tmp_path / "a"
    _make_shard(source, protocol, "HIA_Hou")
    output = tmp_path / "out"
    output.mkdir()
    with pytest.raises(FileExistsError):
        compose_policy_arm_run(
            protocol_path=protocol_path,
            arm="automl",
            task_sources={"HIA_Hou": source},
            output_dir=output,
        )
