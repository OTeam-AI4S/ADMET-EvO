from __future__ import annotations

from dataclasses import replace

import pytest

from admet_agent.metric_normalization import (
    build_metric_normalization_profile,
    paired_normalized_deltas,
)
from admet_agent.research_capability import (
    PROCESS_DIMENSIONS,
    CapabilityThresholds,
    EndpointValidationEvidence,
    FrozenCertificationEvidence,
    ProcessAdvantageEvidence,
    ProcessEvidence,
    SafetyEvidence,
    evaluate_research_capability,
)
from admet_agent.tasks import TASKS

SHA = "a" * 64


def _endpoints(
    *,
    application_margin: float = 0.01,
    automl_delta: float | None = 0.005,
    automl_lcb: float | None = -0.005,
) -> list[EndpointValidationEvidence]:
    return [
        EndpointValidationEvidence(
            task=task.name,
            normalized_application_margin=application_margin,
            normalized_effect_vs_strong_fixed=0.02,
            validation_seed_count=5,
            normalization_profile_sha256=SHA,
            evidence_sha256=SHA,
            expanded_automl_normalized_delta=automl_delta,
            expanded_automl_one_sided_lcb=automl_lcb,
            expanded_automl_same_budget=(None if automl_delta is None else True),
            expanded_automl_protocol_match=(None if automl_delta is None else True),
        )
        for task in TASKS.values()
    ]


def _process(**changes) -> ProcessEvidence:
    base = ProcessEvidence(
        faults_present=10,
        faults_detected=9,
        false_fault_alarms=1,
        remediation_opportunities=10,
        successful_remediations=9,
        causal_contrasts_attempted=10,
        valid_causal_contrasts=9,
        falsification_opportunities=10,
        completed_falsifications=9,
        negative_results_observed=10,
        negative_results_retained=10,
        routing_brier_score=0.1,
        routing_expected_calibration_error=0.1,
        self_correction_opportunities=10,
        successful_self_corrections=9,
        attempted_actions=10,
        useful_actions=9,
        baseline_compute_seconds=100,
        agent_compute_seconds=100,
        stopping_decisions=10,
        appropriate_stops=9,
        external_sources_used=2,
        complete_source_manifests=2,
        data_leakage_incidents=0,
        data_admission_opportunities=10,
        correct_data_admission_decisions=9,
        unsafe_data_admissions=0,
        reproducibility_requests=10,
        reproducible_runs=9,
    )
    return replace(base, **changes)


def _frozen(**changes) -> FrozenCertificationEvidence:
    base = FrozenCertificationEvidence(
        certified_tasks=tuple(task.name for task in TASKS.values()),
        freeze_manifest_sha256=SHA,
        certification_report_sha256=SHA,
        freeze_audit_passed=True,
        finalists_locked_before_test=True,
        one_shot_certification=True,
        official_test_loaded_during_discovery=False,
        test_feedback_to_discovery=False,
    )
    return replace(base, **changes)


def _safety(frozen=True, **changes) -> SafetyEvidence:
    base = SafetyEvidence(
        critical_faults_present=20,
        critical_faults_detected=20,
        test_feedback_events=0,
        protocol_violations=0,
        required_hash_items=100,
        verified_hash_items=100,
        frozen_certification=_frozen() if frozen else None,
    )
    return replace(base, **changes)


def _advantage(value: float = 0.01, **changes) -> ProcessAdvantageEvidence:
    base = ProcessAdvantageEvidence(
        controls_evaluated=(
            "random_action_policy",
            "deterministic_rule_policy",
            "conventional_automl",
            "expanded_space_automl",
        ),
        same_budget=True,
        protocol_hash_match=True,
        dimension_lower_bounds={name: value for name in PROCESS_DIMENSIONS},
    )
    return replace(base, **changes)


def test_scorecard_consumes_dimensionless_metric_effects():
    profile = build_metric_normalization_profile("LD50_Zhu", [0, 10, 20, 30, 40])
    effect = paired_normalized_deltas(
        profile,
        [10 - 0.02 * profile.gain_scale],
        [10],
    )[0]
    rows = _endpoints()
    index = next(i for i, row in enumerate(rows) if row.task == "LD50_Zhu")
    rows[index] = replace(rows[index], normalized_effect_vs_strong_fixed=effect)
    card = evaluate_research_capability(
        rows, _process(), _safety(), process_advantage=_advantage()
    )
    assert (
        card.predictive_sufficiency.mean_normalized_effect_vs_strong_fixed
        == pytest.approx(0.02)
    )


def test_predictive_gate_tracks_floor_rate_worst_regret_and_coverage():
    rows = _endpoints()
    rows[0] = replace(rows[0], normalized_application_margin=-0.03)
    rows[1] = replace(rows[1], validation_seed_count=4)
    card = evaluate_research_capability(rows, _process(), _safety())
    gate = card.predictive_sufficiency
    assert gate.task_coverage == pytest.approx(21 / 22)
    assert gate.worst_normalized_application_regret == pytest.approx(0.03)
    assert gate.passed is False
    assert card.claim_level == "insufficient_or_unsafe_evidence"


def test_expanded_automl_gate_is_independent_and_stronger_than_predictive_gate():
    rows = _endpoints(automl_delta=None, automl_lcb=None)
    card = evaluate_research_capability(
        rows, _process(), _safety(), process_advantage=_advantage()
    )
    assert card.predictive_sufficiency.passed is True
    assert card.process_advantage.passed is True
    assert card.expanded_automl_noninferiority.passed is False
    assert card.claim_level == "frozen_application_qualified_process_advantage"


def test_safety_gate_requires_critical_recall_hashes_and_zero_violations():
    unsafe = _safety(
        critical_faults_detected=18,
        verified_hash_items=99,
        protocol_violations=1,
    )
    card = evaluate_research_capability(_endpoints(), _process(), unsafe)
    assert card.safety.validation_passed is False
    assert "critical_fault_recall_below_threshold" in card.safety.blockers
    assert "hash_coverage_below_threshold" in card.safety.blockers
    assert "protocol_violation_detected" in card.safety.blockers


def test_missing_safety_evidence_is_not_interpreted_as_full_credit():
    missing = SafetyEvidence(0, 0, 0, 0, 0, 0)
    card = evaluate_research_capability(_endpoints(), _process(), missing)
    assert card.safety.critical_fault_recall is None
    assert card.safety.hash_coverage is None
    assert card.safety.validation_passed is False


def test_process_components_do_not_compensate_for_one_failed_dimension():
    weak = _process(successful_self_corrections=1)
    card = evaluate_research_capability(
        _endpoints(), weak, _safety(), process_advantage=_advantage()
    )
    assert card.process_dimensions["self_correction"].score == pytest.approx(0.1)
    assert card.process_dimensions["self_correction"].passed is False
    assert all(
        dimension.score >= 0.6
        for name, dimension in card.process_dimensions.items()
        if name != "self_correction" and dimension.score is not None
    )
    assert card.process_advantage.passed is False
    assert card.claim_level == "frozen_application_performance_qualified"


def test_missing_process_denominator_reduces_coverage_instead_of_scoring_one():
    card = evaluate_research_capability(
        _endpoints(),
        _process(remediation_opportunities=0, successful_remediations=0),
        _safety(),
    )
    remediation = card.process_dimensions["remediation"]
    assert remediation.score is None
    assert remediation.coverage == 0
    assert remediation.passed is False
    assert card.process_dimension_coverage == pytest.approx(8 / 9)


def test_data_stewardship_requires_admission_evidence_and_never_compensates_unsafe_merge():
    missing = evaluate_research_capability(
        _endpoints(),
        _process(
            external_sources_used=0,
            complete_source_manifests=0,
            data_admission_opportunities=0,
            correct_data_admission_decisions=0,
        ),
        _safety(),
    )
    dimension = missing.process_dimensions["data_stewardship"]
    assert dimension.score is None
    assert dimension.coverage == 0.0
    assert dimension.passed is False

    unsafe = evaluate_research_capability(
        _endpoints(),
        _process(unsafe_data_admissions=1),
        _safety(),
    )
    dimension = unsafe.process_dimensions["data_stewardship"]
    assert dimension.components["unsafe_merge_avoidance"] == 0.0
    assert dimension.score == 0.0
    assert dimension.passed is False


def test_process_advantage_requires_every_dimension_and_same_budget_controls():
    bounds = {name: 0.01 for name in PROCESS_DIMENSIONS}
    bounds["routing_calibration"] = 0.0
    advantage = _advantage(
        dimension_lower_bounds=bounds,
        same_budget=False,
    )
    card = evaluate_research_capability(
        _endpoints(), _process(), _safety(), process_advantage=advantage
    )
    assert card.process_advantage.passed is False
    assert card.process_advantage.nonpositive_dimensions == ("routing_calibration",)
    assert "process_controls_not_same_budget" in card.process_advantage.blockers


@pytest.mark.parametrize(
    "process_advantage,automl,frozen,expected",
    [
        (None, True, True, "frozen_application_performance_qualified"),
        (_advantage(), False, True, "frozen_application_qualified_process_advantage"),
        (
            _advantage(),
            True,
            False,
            "validation_process_advantage_plus_automl_noninferiority",
        ),
        (
            _advantage(),
            True,
            True,
            "frozen_process_advantage_plus_automl_noninferiority",
        ),
    ],
)
def test_claim_ladder_is_strict(process_advantage, automl, frozen, expected):
    endpoints = (
        _endpoints() if automl else _endpoints(automl_delta=None, automl_lcb=None)
    )
    card = evaluate_research_capability(
        endpoints,
        _process(),
        _safety(frozen=frozen),
        process_advantage=process_advantage,
    )
    assert card.claim_level == expected


def test_frozen_claim_requires_complete_one_shot_firewall_metadata():
    frozen = _frozen(
        certified_tasks=tuple(task.name for task in list(TASKS.values())[:-1]),
        one_shot_certification=False,
        test_feedback_to_discovery=True,
    )
    card = evaluate_research_capability(
        _endpoints(),
        _process(),
        _safety(frozen_certification=frozen),
        process_advantage=_advantage(),
    )
    assert card.safety.validation_passed is True
    assert card.safety.frozen_certification_passed is False
    assert card.claim_level == "validation_process_advantage_plus_automl_noninferiority"
    assert (
        "frozen_certification_task_coverage_incomplete" in card.safety.frozen_blockers
    )
    assert "certification_not_one_shot" in card.safety.frozen_blockers
    assert "frozen_test_feedback_to_discovery" in card.safety.frozen_blockers


def test_test_feedback_blocks_even_validation_claim():
    card = evaluate_research_capability(
        _endpoints(),
        _process(),
        _safety(test_feedback_events=1),
        process_advantage=_advantage(),
    )
    assert card.safety.validation_passed is False
    assert card.claim_level == "insufficient_or_unsafe_evidence"


def test_duplicate_endpoint_and_invalid_hash_are_rejected():
    rows = _endpoints()
    with pytest.raises(ValueError, match="Only one endpoint"):
        evaluate_research_capability([*rows, rows[0]], _process(), _safety())
    with pytest.raises(ValueError, match="SHA-256"):
        replace(rows[0], evidence_sha256="invalid")


def test_to_dict_keeps_dimension_vector_and_contains_no_test_scores():
    card = evaluate_research_capability(
        _endpoints(), _process(), _safety(), process_advantage=_advantage()
    )
    payload = card.to_dict()
    assert set(payload["process_dimensions"]) == set(PROCESS_DIMENSIONS)
    assert "overall_score" not in payload
    assert "test_score" not in str(payload).lower()
    assert payload["version"] == "admet-research-capability-v3"


def test_claim_flags_keep_application_process_and_automl_claims_independent():
    card = evaluate_research_capability(
        _endpoints(automl_delta=None, automl_lcb=None),
        _process(),
        _safety(),
        process_advantage=_advantage(),
    )
    flags = card.claim_flags
    assert flags["validation_application_performance_qualified"] is True
    assert flags["frozen_application_performance_certified"] is True
    assert flags["validation_process_advantage"] is True
    assert flags["expanded_automl_predictive_noninferiority"] is False
    assert flags["frozen_application_qualified_process_advantage"] is True
    assert flags["frozen_process_advantage_plus_automl_noninferiority"] is False


def test_threshold_boundaries_are_inclusive():
    thresholds = CapabilityThresholds(
        application_floor_pass_rate=1.0,
        maximum_application_regret=0.0,
    )
    card = evaluate_research_capability(
        _endpoints(application_margin=0.0),
        _process(),
        _safety(),
        process_advantage=_advantage(),
        thresholds=thresholds,
    )
    assert card.predictive_sufficiency.passed is True
