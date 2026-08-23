from pathlib import Path

from scripts.build_chembl_herg_external_ticket_report import build


ROOT = Path(__file__).resolve().parents[1]


def test_real_external_ticket_retains_failures_and_forbids_gain_claim():
    payload = build(ROOT)
    assert payload["frozen_protocol_attempts"] == 2
    assert payload["raw_artifacts_accepted"] == 0
    assert payload["model_fits"] == 0
    assert payload["unsafe_merges"] == 0
    assert payload["external_gain_demonstrated"] is False
    assert payload["self_correction"]["observed"] is True
    assert payload["self_correction"]["automatic_third_retry_allowed"] is False
    assert payload["firewall"]["test_feedback_used"] is False
    assert payload["firewall"]["strict_no_test_access_claim_allowed"] is False
