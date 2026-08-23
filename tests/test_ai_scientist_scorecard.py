from pathlib import Path

from admet_agent.ai_scientist_scorecard import _assert_no_test_loaded, build_evidence_scorecard


def test_scorecard_preserves_claim_boundaries_and_raw_gates():
    payload = build_evidence_scorecard(Path("reports"))
    gates = payload["claim_gates"]
    assert gates["application_performance_qualified"] is True
    assert gates["local_integrated_process_advantage_over_rules"] is True
    assert gates["local_data_advantage_over_expanded_automl"] is True
    assert gates["key_process_ablations_executed"] is True
    assert gates["formal_22_task_same_budget_predictive_control_complete"] is True
    assert gates["external_data_discovery_endpoint_and_firewall_mechanism_ready"] is True
    assert gates["independently_authored_external_process_validation_complete"] is False
    assert payload["overall_status"] != "complete"
    assert payload["test_loaded"] is False
    assert payload["test_accessed_in_current_goal_session"] is True
    assert payload["test_feedback_used_after_incident"] is False
    predictive = payload["evidence_vector"]["same_budget_predictive_control"]
    assert predictive["noninferiority_passed"] is False
    assert predictive["logical_charged_fits"] == 154
    assert predictive["physical_unique_fits"] == 113
    external = payload["evidence_vector"]["real_external_data_ticket"]
    assert external["frozen_protocol_attempts"] == 2
    assert external["fits"] == 0
    assert external["unsafe_merges"] == 0
    admission = payload["evidence_vector"]["autonomous_data_admission_stack"]
    assert admission["candidate_sources_discovered"] == 3
    assert admission["direct_merges_allowed"] == 0
    assert admission["test_labels_loaded"] is False
    graph = payload["evidence_vector"]["bounded_graph_fusion"]
    assert graph["gin_ready"] is True
    assert graph["distinct_mpnn_ready"] is False
    assert graph["maximum_ticket_fits"] == 20


def test_ablation_vector_retains_negative_memory_and_safe_no_data_result():
    vector = build_evidence_scorecard(Path("reports"))["evidence_vector"]["key_process_ablations"]
    assert vector["no_memory"]["behavioral_advantage_demonstrated"] is False
    assert vector["no_memory"]["selection_changed_tasks"] == 0
    assert vector["no_hardness"]["ablation_micro_recall"] == 0.0
    assert vector["no_data"]["ablation_unsafe_merges"] == 0
    assert vector["no_data"]["ablation_decision_accuracy"] < vector["no_data"]["full_agent_decision_accuracy"]


def test_scorecard_distinguishes_measured_and_estimated_compute():
    vector = build_evidence_scorecard(Path("reports"))["evidence_vector"]
    cost = vector["cost_aware_stopping"]
    assert cost["minimal_pilot_actual_fits"] == 14
    assert cost["minimal_pilot_fits_avoided"] == 6
    assert cost["minimal_pilot_saving_fraction"] == 0.3
    assert "executed" in cost["measurement"]
    assert "estimates" in cost["measurement"]


def test_test_firewall_rejects_nested_true_value():
    try:
        _assert_no_test_loaded({"nested": [{"test_loaded": True}]})
    except ValueError as error:
        assert "Test firewall violation" in str(error)
    else:
        raise AssertionError("nested test access was not rejected")
