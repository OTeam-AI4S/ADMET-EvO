import hashlib
import json

import pytest

from scripts.check_v24_finalization_gate import check_gate


def _fixture(tmp_path, ready=True):
    mapping = {f"Task_{index}": {"model": "x"} for index in range(21)}
    mapping["DILI"] = {"kind": "recipe", "recipe": "finetuned_unimol2"}
    payload = {
        "protocol": "tdc-admet-test-v2.4",
        "tasks": sorted(mapping),
        "finalists": mapping,
    }
    digest = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
    report = {
        "protocol": payload["protocol"],
        "sha256": digest,
        "coverage": {"complete": True},
        "upgrade_smokes_valid": ready,
        "ready_for_upgrade_finalization": ready,
        "upgrade_tasks": ["DILI"],
        "provenance": {"DILI": {"official_test_runs_after_upgrade": 0}},
    }
    report_path = tmp_path / "report.json"
    mapping_path = tmp_path / "mapping.json"
    report_path.write_text(json.dumps(report), encoding="utf-8")
    mapping_path.write_text(json.dumps(mapping), encoding="utf-8")
    return report_path, mapping_path


def test_gate_accepts_hashed_ready_one_shot_upgrade(tmp_path):
    report, mapping = _fixture(tmp_path)
    result = check_gate(report, mapping, "DILI")
    assert all(result["checks"].values())


def test_gate_fails_closed_before_all_smokes_pass(tmp_path):
    report, mapping = _fixture(tmp_path, ready=False)
    with pytest.raises(RuntimeError, match="smokes"):
        check_gate(report, mapping, "DILI")
