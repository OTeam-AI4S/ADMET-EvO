import pytest

from admet_agent.committee_activation import ConditionalCommitteeActivationPolicy


def _plan(action_id="feature-domain-shift", axis="feature", compute_cost=0.2):
    action = {
        "action_id": action_id,
        "axis": axis,
        "compute_cost": compute_cost,
    }
    return {
        "selected_action_id": action_id,
        "actions": [action],
        "test_loaded": False,
    }


def test_routine_feature_action_stays_single_agent():
    decision = ConditionalCommitteeActivationPolicy().decide(_plan())
    assert decision.mode == "single_agent"
    assert decision.roles == ()
    assert decision.activate is False


def test_selected_data_action_activates_data_steward_and_skeptic():
    decision = ConditionalCommitteeActivationPolicy().decide(
        _plan("data-leakage-safe", "data")
    )
    assert decision.mode == "targeted_review"
    assert decision.roles == ("data_steward", "skeptic")


def test_scheduled_model_correction_activates_targeted_review():
    correction = {
        "decision_kind": "switch_axis",
        "selected_axis": "model",
        "execution_status": "scheduled",
        "test_loaded": False,
    }
    decision = ConditionalCommitteeActivationPolicy().decide(
        _plan(), correction=correction
    )
    assert decision.stage == "post_validation"
    assert decision.roles == ("model_researcher", "skeptic")


def test_stopped_correction_does_not_trigger_review():
    correction = {
        "decision_kind": "stop",
        "selected_axis": None,
        "execution_status": "stopped",
    }
    assert (
        ConditionalCommitteeActivationPolicy()
        .decide(_plan(), correction=correction)
        .mode
        == "single_agent"
    )


def test_cross_axis_disagreement_activates_full_committee():
    decision = ConditionalCommitteeActivationPolicy().decide(
        _plan(), risk_context={"cross_axis_disagreement": True}
    )
    assert decision.mode == "full_committee"
    assert decision.roles == ("data_steward", "model_researcher", "skeptic")


def test_high_cost_action_is_reviewed_without_changing_fit_budget():
    decision = ConditionalCommitteeActivationPolicy().decide(
        _plan("model-unimol2", "model", 0.9)
    )
    assert decision.roles == ("model_researcher", "skeptic")


def test_test_derived_fields_are_rejected_recursively():
    with pytest.raises(ValueError, match="Test-like fields"):
        ConditionalCommitteeActivationPolicy().decide(
            _plan(), risk_context={"nested_test_score": True}
        )


def test_decision_hash_is_deterministic():
    policy = ConditionalCommitteeActivationPolicy()
    first = policy.decide(_plan("data-quality-filter", "data"))
    second = policy.decide(_plan("data-quality-filter", "data"))
    assert first.decision_sha256 == second.decision_sha256
