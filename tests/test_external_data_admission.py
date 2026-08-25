from copy import deepcopy

import pytest

from admet_agent.external_data_admission import (
    evaluate_external_data_admission,
    require_task_admission,
)


def _audit() -> dict:
    return {
        "run_mode": "data_audit_only",
        "task": "Example",
        "task_type": "classification",
        "curation": {
            "output_identities": 500,
            "selected_before_standardization": 510,
            "valid_structure_rows": 505,
            "invalid_structure_rows": 5,
            "conflicting_identity_rows": 5,
        },
        "target_alignment": {
            "exact_overlap_identities": 20,
            "exact_overlap_label_disagreement_fraction": 0.05,
            "absolute_positive_rate_shift": 0.10,
        },
        "chemical_space_alignment": {"fraction_nearest_at_least_0_4": 0.20},
        "safety": {"tdc_test_loaded": False},
    }


def test_admission_requires_automated_manual_and_lineage_approval():
    result = evaluate_external_data_admission(
        _audit(),
        endpoint_review_status="approved_direct",
        lineage_status="cleared_independent",
    )
    assert result["automated_quality_passed"] is True
    assert result["permitted_for_direct_augmentation"] is True
    assert result["model_scores_consulted"] is False


def test_low_quality_fails_closed_even_if_manually_approved():
    audit = deepcopy(_audit())
    audit["target_alignment"]["exact_overlap_label_disagreement_fraction"] = 0.30
    result = evaluate_external_data_admission(
        audit,
        endpoint_review_status="approved_direct",
        lineage_status="cleared_independent",
    )
    assert result["permitted_for_direct_augmentation"] is False
    assert "exact_overlap_target_consistency" in result["failed_checks"]


def test_bridge_and_pending_lineage_never_enter_direct_augmentation():
    bridge = evaluate_external_data_admission(
        _audit(), endpoint_review_status="bridge_only", lineage_status="cleared_independent"
    )
    pending = evaluate_external_data_admission(
        _audit(), endpoint_review_status="approved_direct", lineage_status="pending"
    )
    assert bridge["permitted_for_direct_augmentation"] is False
    assert bridge["permitted_for_controlled_bridge_augmentation"] is True
    assert pending["decision"] == "fail_closed"


def test_clean_low_similarity_bridge_can_enter_controlled_tier_only():
    audit = _audit()
    audit["chemical_space_alignment"]["fraction_nearest_at_least_0_4"] = 0.033
    audit["target_alignment"]["exact_overlap_identities"] = 1
    audit["target_alignment"]["absolute_positive_rate_shift"] = 0.20
    result = evaluate_external_data_admission(
        audit, endpoint_review_status="bridge_only", lineage_status="pending"
    )
    assert result["permitted_for_direct_augmentation"] is False
    assert result["permitted_for_controlled_bridge_augmentation"] is True
    assert result["decision"] == "admit_controlled_bridge"


def test_rejects_non_audit_or_test_accessed_artifacts():
    not_audit = deepcopy(_audit())
    not_audit["run_mode"] = "model_confirmation"
    with pytest.raises(ValueError, match="data_audit_only"):
        evaluate_external_data_admission(
            not_audit,
            endpoint_review_status="approved_direct",
            lineage_status="cleared_independent",
        )
    test_accessed = deepcopy(_audit())
    test_accessed["safety"]["tdc_test_loaded"] = True
    with pytest.raises(ValueError, match="test data"):
        evaluate_external_data_admission(
            test_accessed,
            endpoint_review_status="approved_direct",
            lineage_status="cleared_independent",
        )


def test_runner_gate_fails_closed_and_accepts_only_explicit_admission(tmp_path):
    report = {
        "model_scores_consulted": False,
        "test_labels_loaded": False,
        "decisions": {
            "blocked": {
                "permitted_for_direct_augmentation": False,
                "failed_checks": ["chemical_space_coverage"],
                "endpoint_review_status": "approved_direct",
                "lineage_status": "cleared_independent",
            },
            "admitted": {"permitted_for_direct_augmentation": True},
            "bridge": {"permitted_for_controlled_bridge_augmentation": True},
        },
    }
    path = tmp_path / "admission.json"
    path.write_text(__import__("json").dumps(report))
    with pytest.raises(RuntimeError, match="failed closed"):
        require_task_admission(path, "blocked")
    assert require_task_admission(path, "admitted")[
        "permitted_for_direct_augmentation"
    ] is True
    assert require_task_admission(
        path, "bridge", augmentation_mode="controlled_bridge"
    )["permitted_for_controlled_bridge_augmentation"] is True
