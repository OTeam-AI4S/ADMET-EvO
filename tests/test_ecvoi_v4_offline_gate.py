import json
from pathlib import Path

import pytest

from scripts.evaluate_ecvoi_v4_offline_gate import evaluate


ROOT = Path(__file__).resolve().parents[1]
MEMORY = ROOT / "reports/ecvoi_v2_matched_effect_memory_v1.jsonl"
CONTEXTS = ROOT / "reports/admet22_action_context_registry_v1.json"


def test_v4_offline_gate_is_target_held_out_and_preserves_negative_control():
    result = evaluate(memory_path=MEMORY, context_path=CONTEXTS)

    assert result["test_loaded"] is False
    assert result["leaderboard_loaded"] is False
    assert result["passed"] is False
    assert result["gates"]["target_endpoint_rows_admitted_max_zero"] is True
    assert (
        result["gates"]["shuffled_diagnosis_does_not_equal_or_outperform"]
        is False
    )
    assert all(item["target_endpoint_rows_admitted"] == 0 for item in result["results"])
    assert result["aggregates"]["full"] == result["aggregates"]["shuffled_diagnosis"]


def test_v4_offline_gate_rejects_tampered_memory(tmp_path: Path):
    rows = [json.loads(line) for line in MEMORY.read_text().splitlines() if line]
    rows[0]["native_oriented_delta"] += 0.01
    tampered = tmp_path / "tampered.jsonl"
    tampered.write_text("\n".join(json.dumps(row) for row in rows) + "\n")

    with pytest.raises(RuntimeError, match="observation hash mismatch"):
        evaluate(memory_path=tampered, context_path=CONTEXTS)
