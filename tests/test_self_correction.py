import pytest

from admet_agent.self_correction import ValidationSelfCorrectionPlanner


def _action(action_id, axis, *, payload=None):
    return {
        "action_id": action_id,
        "axis": axis,
        "hypothesis": f"hypothesis for {action_id}",
        "falsification_test": "paired validation comparison",
        "payload": payload or {},
        "compute_cost": 0.25,
        "utility": 0.5,
    }


def _plan():
    return {
        "task": "AMES",
        "hardness_evidence_sha256": "hardness",
        "selected_action_id": "feature-first",
        "effect_priority": 0.8,
        "actions": [
            _action("feature-first", "feature"),
            _action(
                "data-external-quarantine",
                "data",
                payload={"merge_status": "quarantined"},
            ),
            _action("model-second", "model"),
            _action("stop-no-defensible-action", "stop"),
        ],
    }


def _evidence(status, **overrides):
    return {
        "action_id": "feature-first",
        "axis": "feature",
        "status": status,
        "validation_seeds": [1, 2, 3],
        "single_axis_verified": True,
        **overrides,
    }


def test_refutation_switches_axis_and_never_auto_admits_quarantined_data():
    decision = ValidationSelfCorrectionPlanner().decide(_plan(), _evidence("refuted"))
    assert decision.kind == "switch_axis"
    assert decision.selected_axis == "model"
    assert decision.estimated_additional_fits == 6
    assert decision.execution_status == "scheduled"
    child = decision.next_plan["actions"][0]
    assert child["source_action_id"] == "model-second"
    assert child["corrects_action_id"] == "feature-first"
    assert child["parent_action_id"] == "feature-first"
    assert decision.test_loaded is False
    assert decision == ValidationSelfCorrectionPlanner().decide(
        _plan(), _evidence("refuted")
    )


def test_incomplete_evidence_schedules_pair_repair_without_claiming_completion():
    decision = ValidationSelfCorrectionPlanner().decide(
        _plan(),
        _evidence(
            "inconclusive",
            validation_seeds=[1],
            single_axis_verified=False,
        ),
    )
    assert decision.kind == "repair_evidence"
    assert decision.selected_axis == "feature"
    assert decision.estimated_additional_fits == 4
    assert decision.requires_execution is True
    assert decision.next_plan["execution_status"] == "scheduled"


def test_resolved_small_effect_switches_axis_instead_of_repeating_search():
    decision = ValidationSelfCorrectionPlanner().decide(
        _plan(), _evidence("inconclusive")
    )
    assert decision.kind == "switch_axis"
    assert decision.selected_axis == "model"
    assert "effect_below_resolution_gate" in decision.reason_codes


def test_supported_action_stops_when_application_is_already_qualified():
    decision = ValidationSelfCorrectionPlanner().decide(_plan(), _evidence("supported"))
    assert decision.kind == "stop"
    assert decision.requires_execution is False
    assert decision.estimated_additional_fits == 0
    assert "avoid_redundant_search" in decision.reason_codes


def test_budget_zero_stops_and_test_fields_are_rejected():
    planner = ValidationSelfCorrectionPlanner()
    stopped = planner.decide(_plan(), _evidence("refuted"), remaining_action_budget=0)
    assert stopped.kind == "stop"
    assert "action_budget_exhausted" in stopped.reason_codes
    with pytest.raises(ValueError, match="Test-like"):
        planner.decide(_plan(), {**_evidence("refuted"), "test_score": 0.9})


def test_expensive_unimol_style_followup_is_not_scheduled_after_qualification():
    plan = _plan()
    plan["actions"] = [
        _action("model-first", "model"),
        {
            **_action("feature-expensive", "feature"),
            "compute_cost": 0.65,
            "payload": {"representations": ["unimol2", "morgan"]},
        },
        _action("stop-no-defensible-action", "stop"),
    ]
    plan["selected_action_id"] = "model-first"
    evidence = {
        **_evidence("refuted"),
        "action_id": "model-first",
        "axis": "model",
    }
    decision = ValidationSelfCorrectionPlanner().decide(plan, evidence)
    assert decision.kind == "stop"
    assert decision.selected_action_id is None
    assert "no_unattempted_executable_cross_axis_action" in decision.reason_codes
