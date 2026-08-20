from pathlib import Path

from scripts.build_self_correction_smoke_report import build

ROOT = Path(__file__).parents[1]


def test_completed_smoke_reports_behavior_without_inventing_predictive_support():
    payload = build(
        ROOT / "runs/v5-self-correction-smoke",
        ROOT / "runs/v3-agent-tdc22-v5",
    )
    assert payload["scheduled_actions"] == 2
    assert payload["completed_actions"] == 2
    assert payload["completion_rate"] == 1.0
    assert payload["completed_evidence_repairs"] == 1
    assert payload["supported_child_hypotheses"] == 0
    assert payload["resolved_child_hypotheses"] == 0
    assert payload["all_post_followup_stopped"] is True
    assert payload["total_fits"] == 12
    assert payload["test_loaded"] is False
    correction = payload["process_metrics"]["self_correction_rate"]
    assert correction["numerator"] == 1
    assert correction["denominator"] == 1
    assert correction["rate"] == 1.0
    assert "does not demonstrate" in payload["claim_boundary"]
