from pathlib import Path

from scripts.build_self_correction_schedule import build

ROOT = Path(__file__).parents[1]


def test_historical_schedule_is_minimal_and_does_not_claim_completion():
    payload = build(ROOT / "runs/v3-agent-tdc22-v5")
    assert payload["task_count"] == 22
    assert payload["trigger_status_counts"] == {
        "inconclusive": 2,
        "refuted": 2,
        "supported": 18,
    }
    assert payload["decision_counts"] == {
        "repair_evidence": 1,
        "stop": 20,
        "switch_axis": 1,
    }
    assert payload["scheduled_tasks"] == [
        "CYP2C9_Substrate_CarbonMangels",
        "hERG",
    ]
    assert payload["estimated_additional_fits"] == 12
    assert payload["counts_as_completed_self_correction"] is False
    assert payload["test_loaded"] is False
    assert payload["training_started"] is False
    selected = [
        row for row in payload["decisions"] if row["execution_status"] == "scheduled"
    ]
    assert all("unimol2" not in str(row["next_plan"]).lower() for row in selected)
