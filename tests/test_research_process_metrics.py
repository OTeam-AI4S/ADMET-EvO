from __future__ import annotations

import pytest

from admet_agent.research_process_metrics import extract_research_process_metrics


def plan(
    task, action_id, *, falsification=True, family=None, attempt_id=None, order=None
):
    action = {
        "action_id": action_id,
        "axis": "feature",
        "falsification_test": "paired comparison" if falsification else "",
    }
    if family:
        action["action_family"] = family
    if attempt_id:
        action["attempt_id"] = attempt_id
    if order is not None:
        action["sequence_index"] = order
    return {"task": task, "selected_action_id": action_id, "actions": [action]}


def evidence(
    task,
    action_id,
    status,
    *,
    verified=True,
    treatment="treatment",
    control="control",
    **extra,
):
    return {
        "task": task,
        "action_id": action_id,
        "axis": "feature",
        "status": status,
        "single_axis_verified": verified,
        "treatment_run_id": treatment,
        "control_run_id": control,
        **extra,
    }


def fit(run_id, task, seed, seconds, status="completed"):
    return {
        "run_id": run_id,
        "task": task,
        "seed": seed,
        "status": status,
        "elapsed_seconds": seconds,
    }


def summary(task, action_id, status, *, racing=None):
    value = {
        "profile": {"task": task},
        "validation_seeds": [1, 2, 3, 4, 5],
        "research_plan": plan(task, action_id),
        "action_evidence": evidence(task, action_id, status),
        "results": [
            fit(f"{task}-1", task, 1, 2.0),
            fit(f"{task}-2", task, 2, 3.0),
            fit(f"{task}-3", task, 3, 4.0, "failed"),
        ],
    }
    if racing is not None:
        value["robust_racing"] = racing
    return value


def ledger_outcome(task, action_id, status):
    return {
        "event_type": "research_action_evaluated",
        "payload": {
            "task": task,
            "evidence": evidence(task, action_id, status),
        },
    }


def test_complete_metrics_cover_causal_discipline_and_resource_ledger():
    summaries = [
        summary("AMES", "feature-a", "supported", racing={"planned_fits_avoided": 3}),
        summary("DILI", "feature-b", "refuted", racing={"planned_fits_avoided": 6}),
        summary("hERG", "feature-c", "inconclusive"),
    ]
    ledger = [
        ledger_outcome("AMES", "feature-a", "supported"),
        ledger_outcome("DILI", "feature-b", "refuted"),
        {
            "event_type": "experiment_completed",
            "payload": {"result": fit("AMES-1", "AMES", 1, 2.0)},
        },
    ]
    card = extract_research_process_metrics(summaries, ledger)
    assert card.tasks == ("AMES", "DILI", "hERG")
    assert card.unique_task_actions == 3
    assert card.single_axis_contrast_validity.rate == pytest.approx(1.0)
    assert card.treatment_control_completeness.rate == pytest.approx(1.0)
    assert card.falsification_resolution.numerator == 2
    assert card.falsification_resolution.denominator == 3
    assert card.falsification_resolution.rate == pytest.approx(2 / 3)
    assert card.outcome_counts.to_dict() == {
        "supported": 1,
        "refuted": 1,
        "inconclusive": 1,
        "missing_or_unknown": 0,
        "valid_single_axis": 3,
        "unique_outcomes": 3,
    }
    assert card.negative_result_retention.rate == pytest.approx(1.0)
    assert card.preregistered_action_discipline.rate == pytest.approx(1.0)
    assert card.resources.total_fits == 9
    assert card.resources.primary_fits == 3
    assert card.resources.replicate_fits == 6
    assert card.resources.completed_fits == 6
    assert card.resources.failed_fits == 3
    assert card.resources.fit_seconds == pytest.approx(27.0)
    assert card.resources.failed_fit_seconds == pytest.approx(12.0)
    assert card.resources.planned_fits_avoided == 9
    assert card.to_dict()["version"] == "research-process-metrics-v1"


def test_missing_fields_reduce_conservative_rate_and_report_coverage():
    rows = [
        evidence("AMES", "a", "supported"),
        {"task": "DILI", "action_id": "b", "axis": "feature", "status": "supported"},
    ]
    card = extract_research_process_metrics(action_outcomes=rows)
    metric = card.single_axis_contrast_validity
    assert metric.numerator == 1
    assert metric.denominator == 2
    assert metric.evaluable == 1
    assert metric.missing == 1
    assert metric.rate == pytest.approx(0.5)
    assert metric.observed_rate == pytest.approx(1.0)
    assert metric.coverage == pytest.approx(0.5)
    assert metric.missing_ids == ("DILI/b",)
    pair = card.treatment_control_completeness
    assert pair.rate == pytest.approx(0.5)
    assert card.preregistered_action_discipline.rate == 0.0
    assert card.preregistered_action_discipline.observed_rate is None
    assert card.preregistered_action_discipline.missing == 2


def test_no_ledger_does_not_claim_negative_result_retention():
    card = extract_research_process_metrics(
        action_outcomes=[evidence("DILI", "a", "refuted")]
    )
    retention = card.negative_result_retention
    assert retention.denominator == 1
    assert retention.evaluable == 0
    assert retention.rate == 0.0
    assert retention.observed_rate is None
    assert retention.coverage == 0.0
    assert retention.missing_ids == ("DILI/a",)


def test_task_action_dedup_prevents_summary_ledger_pseudoreplication():
    row = summary("AMES", "feature-a", "supported")
    ledger = [ledger_outcome("AMES", "feature-a", "supported")]
    card = extract_research_process_metrics([row, row], ledger)
    assert card.outcome_counts.unique_outcomes == 1
    assert card.outcome_counts.supported == 1
    assert card.unique_task_actions == 1
    assert card.input_coverage["duplicate_task_summaries_ignored"] == 1
    assert card.resources.total_fits == 3


def test_self_correction_requires_explicit_or_linked_evidence():
    refuted = evidence("AMES", "old", "refuted", self_corrected=False)
    corrected = evidence("DILI", "old", "refuted", self_corrected=True)
    missing = evidence("hERG", "old", "refuted")
    child = evidence(
        "AMES",
        "new",
        "supported",
        parent_action_id="old",
        action_family="feature-family",
    )
    card = extract_research_process_metrics(
        action_outcomes=[refuted, corrected, missing, child]
    )
    metric = card.self_correction_rate
    assert metric.numerator == 2
    assert metric.denominator == 3
    assert metric.evaluable == 2
    assert metric.rate == pytest.approx(2 / 3)
    assert metric.coverage == pytest.approx(2 / 3)
    assert metric.missing_ids == ("hERG/old",)


def test_scheduled_correction_is_not_counted_as_completed_self_correction():
    refuted = evidence("AMES", "old", "refuted")
    scheduled = {
        "event_type": "corrective_action_scheduled",
        "payload": {
            "task": "AMES",
            "decision": {
                "trigger_action_id": "old",
                "selected_action_id": "new--correct-123",
                "execution_status": "scheduled",
                "test_loaded": False,
            },
            "test_loaded": False,
        },
    }
    metric = extract_research_process_metrics(
        ledger_records=[scheduled], action_outcomes=[refuted]
    ).self_correction_rate
    assert metric.numerator == 0
    assert metric.denominator == 1
    assert metric.evaluable == 0
    assert metric.coverage == 0.0
    assert metric.missing_ids == ("AMES/old",)


def test_duplicate_and_refuted_family_repeat_waste_needs_explicit_attempts():
    rows = [
        evidence(
            "AMES",
            "a",
            "refuted",
            attempt_id="1",
            sequence_index=1,
            action_family="feature-family",
        ),
        evidence(
            "AMES",
            "a",
            "refuted",
            attempt_id="2",
            sequence_index=2,
            action_family="feature-family",
        ),
        evidence(
            "AMES",
            "b",
            "supported",
            attempt_id="3",
            sequence_index=3,
            action_family="feature-family",
        ),
        evidence(
            "AMES",
            "c",
            "supported",
            attempt_id="4",
            sequence_index=4,
            action_family="feature-family",
            parent_action_id="a",
        ),
    ]
    waste = extract_research_process_metrics(action_outcomes=rows).repeat_waste
    assert waste.logical_attempts == 4
    assert waste.duplicate_task_action_attempts == 1
    assert waste.refuted_family_repeat_attempts == 2
    assert waste.overlapping_waste_attempts == 1
    assert waste.total_repeat_waste == 2
    assert waste.waste_rate == pytest.approx(0.5)
    assert waste.ordering_coverage == pytest.approx(1.0)


def test_numeric_attempt_order_is_not_sorted_lexically():
    rows = [
        evidence(
            "AMES",
            "later",
            "supported",
            attempt_id="later",
            sequence_index=10,
            action_family="family",
        ),
        evidence(
            "AMES",
            "first",
            "refuted",
            attempt_id="first",
            sequence_index=2,
            action_family="family",
        ),
    ]
    waste = extract_research_process_metrics(action_outcomes=rows).repeat_waste
    assert waste.refuted_family_repeat_attempts == 1


@pytest.mark.parametrize(
    "payload",
    [
        {"test_score": 0.9},
        {"nested": [{"heldout_predictions": [0.1]}]},
        {"payload": {"test_loaded": True}},
        {"nested": {"agentTestScore": 0.8}},
    ],
)
def test_test_like_keys_are_rejected_recursively(payload):
    with pytest.raises(ValueError, match="Test-like keys are forbidden"):
        extract_research_process_metrics(search_summaries=payload)


def test_falsification_test_is_safe_metadata_not_a_test_split_key():
    card = extract_research_process_metrics(
        search_summaries={
            "profile": {"task": "AMES"},
            "research_plan": plan("AMES", "a"),
            "action_evidence": evidence("AMES", "a", "supported"),
            "results": [],
        }
    )
    assert card.preregistered_action_discipline.rate == pytest.approx(1.0)


def test_false_test_loaded_audit_sentinel_is_allowed_but_carries_no_test_content():
    card = extract_research_process_metrics(
        ledger_records={
            "event_type": "research_action_evaluated",
            "payload": {
                "task": "AMES",
                "test_loaded": False,
                "evidence": evidence("AMES", "a", "supported"),
            },
        }
    )
    assert card.tasks == ("AMES",)


def test_ledger_plan_must_precede_first_outcome_to_be_preregistered():
    action = evidence("AMES", "a", "supported")
    plan_event = {
        "event_type": "research_plan_created",
        "payload": {"task": "AMES", "research_plan": plan("AMES", "a")},
    }
    outcome_event = {
        "event_type": "research_action_evaluated",
        "payload": {"task": "AMES", "evidence": action},
    }
    prior = extract_research_process_metrics(ledger_records=[plan_event, outcome_event])
    late = extract_research_process_metrics(ledger_records=[outcome_event, plan_event])
    assert prior.preregistered_action_discipline.rate == pytest.approx(1.0)
    assert late.preregistered_action_discipline.rate == pytest.approx(0.0)


def test_robust_racing_missing_avoided_count_is_explicit():
    row = summary("AMES", "a", "supported", racing={"decisions": []})
    card = extract_research_process_metrics(row)
    assert card.resources.racing_summaries == 1
    assert card.resources.planned_fits_avoided == 0
    assert card.resources.racing_missing_avoided_count_tasks == ("AMES",)
