import json
from pathlib import Path

import pytest

from admet_agent.feedback_policy_controls import compare_feedback_controls

ROOT = Path(__file__).parents[1]


def _inputs():
    schedule = json.loads(
        (ROOT / "reports/self_correction_schedule_v1.json").read_text()
    )
    smoke = json.loads((ROOT / "reports/self_correction_smoke_v1.json").read_text())
    return schedule, smoke


def test_feedback_controls_separate_automl_process_gap_from_rules_tie():
    report = compare_feedback_controls(*_inputs())
    arms = report["arms"]
    assert report["case_count"] == 2
    assert arms["agent_feedback_loop"]["macro_process_score"] == 1.0
    assert arms["complete_rules_feedback_loop"] == arms["agent_feedback_loop"]
    assert arms["expanded_automl_scalar_search"]["macro_process_score"] < 1.0
    assert arms["conventional_automl_scalar_search"]["macro_process_score"] < 1.0
    assert report["test_loaded"] is False
    assert report["claim_scope"].startswith("locked_two_case")


def test_feedback_control_rejects_non_validation_input():
    schedule, smoke = _inputs()
    with pytest.raises(ValueError, match="validation-only"):
        compare_feedback_controls({**schedule, "test_loaded": True}, smoke)
