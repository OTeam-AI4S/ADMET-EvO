import importlib.util
from pathlib import Path


SCRIPT = Path(__file__).parents[1] / "scripts/build_paper_evidence_manifest_v27.py"


def _load_script():
    spec = importlib.util.spec_from_file_location("paper_manifest_v27", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_v27_paper_manifest_verifies_scorecard_firewall_and_bio_policy():
    module = _load_script()
    root = Path(__file__).parents[1]
    manifest = module.build_manifest(root)
    assert manifest["status"] == (
        "official_results_complete_goal_pending_authoritative_bio_resolution"
    )
    assert manifest["scorecard_status"] == {
        "top5": 18,
        "top10": 21,
        "top5_deficit": 0,
        "top10_deficit": 1,
        "goal_met": False,
    }
    assert manifest["verified_official_tasks"] == [
        "Bioavailability_Ma",
        "CYP2C9_Substrate_CarbonMangels",
    ]
    assert manifest["validation_firewall"]["official_test_loaded"] is False
    assert manifest["validation_firewall"]["cross_split_identity_overlap"] == 0
    assert manifest["bio_discrepancy_policy"]["threshold_override_allowed"] is False
    assert manifest["router_controls"]["task_count"] == 22
    assert manifest["router_controls"]["validation_only_firewall"] is True
    assert manifest["router_controls"]["memory_claim_gate_passed"] is False
    assert manifest["router_controls"]["selection_claim_gate_passed"] is False
    assert manifest["router_controls"]["maximum_relative_cost_error"] <= 1e-9
    assert manifest["primary_model_effects"] == {
        "contrast_count": 3,
        "post_hoc_descriptive": True,
        "selection_or_freeze_changed": False,
    }
    assert manifest["validation_calibration_residuals"] == {
        "task_count": 22,
        "classification_task_count": 13,
        "regression_task_count": 9,
        "identity_overlap_count": 0,
        "scaffold_overlap_count": 0,
    }
    assert manifest["compute_memory_appendix"]["router_task_count"] == 22
    assert manifest["compute_memory_appendix"]["selected_to_score_total_cost_ratio"] < 1
    assert manifest["compute_memory_appendix"]["peak_memory_coverage"] == {
        "records_total": 31,
        "records_measured": 0,
        "records_missing": 31,
    }
    assert manifest["bibliography"] == {
        "entry_count": 15,
        "citation_audit_passed": True,
    }
    assert manifest["third_party_release_audit"] == {
        "essential_dependency_count": 2,
        "final_mapping_minimol_used": False,
        "registration_free_checkpoint_access_verified": True,
        "source_code_licenses_verified": True,
        "checkpoint_specific_licenses_verified": False,
        "third_party_checkpoints_in_bundle": False,
        "gin_official_parameter_equivalence_verified": True,
        "ready_for_external_deposit": True,
    }
    assert manifest["router_metric_risk_guard"]["task_count"] == 22
    assert manifest["router_metric_risk_guard"]["fresh_validation_seeds"] == [4, 5]
    assert manifest["router_metric_risk_guard"]["identity_overlap_total"] == 0
    assert manifest["router_metric_risk_guard"]["claim_gate_passed"] is False
    assert manifest["router_metric_risk_guard"]["significance_gate_passed"] is False
    assert manifest["router_metric_risk_guard"]["selection_or_official_freeze_changed"] is False
    assert manifest["paper_figures"] == {
        "figure_count": 4,
        "formats": ["svg", "png"],
        "output_count": 9,
        "official_test_loaded": False,
        "selection_or_freeze_changed": False,
        "leaderboard_rank_summary": {
            "task_count": 22,
            "top5": 18,
            "top10": 21,
            "conservative_deficit_task": "Bioavailability_Ma",
        },
        "router_claim_gate_passed": False,
        "chemical_space_worse_counts": {
            "classification": [11, 13],
            "regression": [7, 9],
        },
    }
    assert manifest["jcheminf_submission"] == {
        "article_type": "Research Article",
        "abstract_word_count": 255,
        "contribution_heading_present": True,
        "reference_count": 15,
        "supplement_task_count": 22,
        "graphical_abstract": {
            "width_px": 920,
            "height_px": 300,
            "max_bytes": 150000,
        },
        "output_count": 5,
        "unresolved_blocker_count": 6,
        "official_test_loaded_during_generation": False,
        "selection_or_freeze_changed": False,
    }
    assert manifest["reviewer_bundle_builder"] == {
        "deterministic_archive_builder_included": True,
        "gpu_environment_syntax_guard_passed": True,
        "official_test_labels_permitted": False,
        "credential_files_permitted": False,
        "external_deposit_complete": False,
        "third_party_checkpoint_access_verified": True,
        "checkpoint_specific_licenses_verified": False,
    }
    assert len(manifest["manifest_sha256"]) == 64
