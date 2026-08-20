import json
from pathlib import Path

ROOT = Path(__file__).parents[1]


def test_legacy_audit_is_retained_as_negative_not_formal_evidence():
    report = json.loads(
        (ROOT / "reports/legacy_agent_automl_audit_v1.json").read_text()
    )
    assert report["task_count"] == 22
    assert report["formal_claim_eligible"] is False
    assert report["budget_audit"]["agent_protocol_compliant"] is False
    assert report["budget_audit"]["automl_protocol_compliant"] is False
    assert report["budget_audit"]["actual_action_parity_tasks"] == 21
    assert report["budget_audit"]["agent_same_or_lower_action_tasks"] == 22
    assert report["source_audit"]["execution_kernel_identical"] is True
    assert report["source_audit"]["same_source_snapshot"] is False
    predictive = report["predictive_comparison"]
    assert predictive["mean_normalized_oriented_delta"] < 0
    assert predictive["task_bootstrap_mean_95_ci"][1] < 0
    assert predictive["exploratory_noninferiority_minus_0_01"] is False
    assert report["cost_comparison"]["aggregate_agent_cost_ratio"] > 1
    assert report["cost_comparison"]["agent_compute_saving_supported"] is False
    assert report["test_loaded"] is False
