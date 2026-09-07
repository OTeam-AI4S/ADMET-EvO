from copy import deepcopy
import hashlib
import json

import pytest

from admet_agent.external_data_admission import (
    evaluate_external_data_admission,
    require_bound_task_admission,
    require_task_admission,
    weight_policy_for_admission,
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


def test_admission_tier_assigns_conservative_weights():
    direct = {
        "permitted_for_direct_augmentation": True,
        "endpoint_review_status": "approved_direct",
        "lineage_status": "cleared_independent",
        "resolved_augmentation_mode": "direct",
    }
    assert weight_policy_for_admission(direct).external_mass_ratio == 0.20
    assert weight_policy_for_admission(direct).external_row_weight_cap == 0.50
    assert (
        weight_policy_for_admission(direct).external_to_min_native_weight_ratio_cap
        == 0.50
    )
    bridge = {
        "permitted_for_controlled_bridge_augmentation": True,
        "endpoint_review_status": "bridge_only",
        "lineage_status": "pending",
        "resolved_augmentation_mode": "controlled_bridge",
    }
    policy = weight_policy_for_admission(bridge)
    assert policy.external_mass_ratio == 0.02
    assert policy.external_row_weight_cap == 0.10


def test_bound_gate_rejects_source_substitution(tmp_path):
    source = tmp_path / "source.tab"
    source.write_text("X\tY\nCC\t1\n")
    source_sha = hashlib.sha256(source.read_bytes()).hexdigest()
    audit = {
        "task": "Example",
        "run_mode": "data_audit_only",
        "safety": {"tdc_test_loaded": False},
        "source": {
            "name": "independent_assay",
            "path": str(source.resolve()),
            "sha256": source_sha,
        },
    }
    audit_path = tmp_path / "audit.json"
    audit_path.write_text(json.dumps(audit))
    report = {
        "model_scores_consulted": False,
        "test_labels_loaded": False,
        "decisions": {
            "Example": {
                "audit_path": str(audit_path),
                "audit_sha256": hashlib.sha256(audit_path.read_bytes()).hexdigest(),
                "permitted_for_direct_augmentation": True,
                "endpoint_review_status": "approved_direct",
                "lineage_status": "cleared_independent",
            }
        },
    }
    report_path = tmp_path / "report.json"
    report_path.write_text(json.dumps(report))
    admitted = require_bound_task_admission(
        report_path,
        "Example",
        source_path=source,
        source_name="independent_assay",
    )
    assert admitted["audit_artifact_verified"] is True
    source.write_text("X\tY\nCCC\t0\n")
    with pytest.raises(RuntimeError, match="SHA-256 mismatch"):
        require_bound_task_admission(
            report_path,
            "Example",
            source_path=source,
            source_name="independent_assay",
        )
