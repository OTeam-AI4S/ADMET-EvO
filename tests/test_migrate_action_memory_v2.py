import hashlib
import json

import pytest

from scripts.migrate_action_memory_v2 import migrate


def _row(**updates):
    row = {
        "task": "AMES",
        "action_id": "feature-domain-shift",
        "axis": "feature",
        "evidence_protocol": "discovery_only_matched_action_v1",
        "transport_eligible": True,
        "single_axis_verified": True,
        "paired_seeds": 5,
        "mean_oriented_delta": 0.01,
        "delta_std": 0.0,
        "record_sha256": "a" * 64,
        "transport_protocol_sha256": "b" * 64,
        "transport_scope": "global_admet",
    }
    return {**row, **updates}


def _write(path, rows):
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def test_migration_is_hash_bound_and_uses_preregistered_se_floor(tmp_path):
    source = tmp_path / "memory.jsonl"
    _write(source, [_row()])
    rows, manifest = migrate(source)
    assert len(rows) == 1
    assert rows[0]["native_standard_error"] == pytest.approx(0.0005)
    assert rows[0]["effect_standardization_scale"] == pytest.approx(0.005)
    assert rows[0]["pair_id"] == "legacy-grid:" + "a" * 64
    assert rows[0]["source_record_sha256"] == "a" * 64
    assert manifest["source_sha256"] == hashlib.sha256(source.read_bytes()).hexdigest()
    assert manifest["test_loaded"] is False


def test_migration_excludes_ineligible_or_unreconstructable_evidence(tmp_path):
    source = tmp_path / "memory.jsonl"
    _write(
        source,
        [
            _row(task="A", transport_eligible=False),
            _row(task="B", paired_seeds=2),
            _row(task="C", mean_oriented_delta=None),
        ],
    )
    rows, manifest = migrate(source)
    assert rows == []
    assert manifest["excluded_records"] == 3
    assert manifest["exclusions_by_reason"] == {
        "insufficient_paired_seeds": 1,
        "matched_effect_statistics_missing": 1,
        "not_transport_eligible": 1,
    }


def test_duplicate_endpoint_action_is_rejected(tmp_path):
    source = tmp_path / "memory.jsonl"
    _write(source, [_row(), _row(record_sha256="c" * 64)])
    with pytest.raises(ValueError, match="duplicate endpoint/action"):
        migrate(source)


def test_migration_refuses_post_hoc_scale_or_se_floor_changes(tmp_path):
    source = tmp_path / "memory.jsonl"
    _write(source, [_row()])
    with pytest.raises(ValueError, match="scale is frozen"):
        migrate(source, effect_standardization_scale=0.01)
    with pytest.raises(ValueError, match="floor fraction is frozen"):
        migrate(source, standard_error_floor_fraction=0.2)
