import hashlib
import json

import pytest

from scripts.build_v24_full_freeze import _digest, validate_legacy_task


def test_digest_is_canonical_json_sha256():
    payload = {"b": 2, "a": 1}
    expected = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
    assert _digest(payload) == expected


def test_legacy_validator_rejects_tampered_freeze(tmp_path):
    final = tmp_path / "runs" / "leaderboard-v21" / "tasks" / "AMES" / "final"
    final.mkdir(parents=True)
    freeze = {
        "protocol": "tdc-admet-test-v1",
        "tasks": ["AMES"],
        "finalists": {"AMES": {"kind": "single"}},
        "sha256": "tampered",
    }
    (final / "freeze.json").write_text(json.dumps(freeze), encoding="utf-8")
    (final / "results.json").write_text("{}", encoding="utf-8")

    with pytest.raises(ValueError, match="hash mismatch"):
        validate_legacy_task(tmp_path, "AMES")
