"""Hash and verify the evidence bundle supporting the v2.7 manuscript."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from scripts.build_paper_evidence_manifest import hash_file, manifest_digest
from scripts.build_v27_goal_scorecard import TASKS, build_v27_scorecard, verify_official_result


EVIDENCE_FILES = {
    "leaderboard_snapshot": "reports/leaderboard_snapshot_2026-08-17.json",
    "goal_scorecard": "reports/goal_scorecard_v27.json",
    "bio_discrepancy_audit": "reports/bio_leaderboard_discrepancy_v27.json",
    "bio_correction_packet": "reports/tdc_bio_correction_packet_v27.md",
    "paper_readiness": "reports/paper_readiness_v27.md",
    "goal_status": "reports/goal_status_v27_2026-08-17.md",
    "working_manuscript": "MANUSCRIPT_V24.md",
    "full_freeze": "runs/v27-maplight-gnn-freeze/full-22-task.json",
    "full_freeze_mapping": "runs/v27-maplight-gnn-freeze/full-22-task.mapping.json",
    "fresh_confirmation": "runs/v27-maplight-gnn-confirmation/result.json",
    "preregistration": "runs/v27-maplight-gnn-confirmation/preregistration.json",
    "router_loso": "runs/v24-router-loso/report.json",
    "router_context": "runs/v24-router-context/report.json",
    "router_error_analysis": "runs/v24-router-error-analysis/report.json",
    "router_controls_preregistration": "runs/v27-router-controls/preregistration.json",
    "router_controls": "runs/v27-router-controls/report.json",
    "primary_model_effects": "reports/primary_model_effects_v27.json",
    "validation_calibration_residuals": "reports/validation_calibration_residuals_v27.json",
    "compute_memory_appendix": "reports/compute_memory_appendix_v27.json",
    "bibliography": "paper/references.bib",
    "bibliography_audit": "reports/bibliography_audit_v27.json",
    "third_party_release_audit": "reports/third_party_release_audit_v27.json",
    "third_party_release_audit_markdown": "reports/third_party_release_audit_v27.md",
    "gin_checkpoint_equivalence": "reports/gin_checkpoint_equivalence_v27.json",
    "venue_strategy": "reports/venue_strategy_v27.md",
    "router_risk_guard_preregistration": "runs/v27-router-risk-guard/preregistration.json",
    "router_risk_guard": "runs/v27-router-risk-guard/report.json",
    "paper_figures": "paper/figures/manifest.json",
    "paper_figure_captions": "paper/figures/captions.md",
    "jcheminf_submission": "paper/submission/jcheminformatics/manifest.json",
    "jcheminf_submission_checklist": "paper/submission/jcheminformatics/submission_checklist.md",
    "rescue_ledger": "runs/v24-rescue-ledger/report.json",
}

SOURCE_FILES = (
    "admet_agent/evidence_router.py",
    "admet_agent/frozen_executor.py",
    "admet_agent/frozen_recipe.py",
    "scripts/predict_maplight_gnn.py",
    "scripts/evaluate_maplight_gnn_confirmation.py",
    "scripts/build_v27_maplight_gnn_freeze.py",
    "scripts/build_v27_goal_scorecard.py",
    "scripts/audit_bio_leaderboard_v27.py",
    "scripts/ablate_router_memory_and_random.py",
    "scripts/summarize_primary_model_effects_v27.py",
    "scripts/analyze_validation_calibration_residuals_v27.py",
    "scripts/build_compute_memory_appendix_v27.py",
    "scripts/audit_bibliography_v27.py",
    "scripts/audit_third_party_release_v27.py",
    "scripts/verify_gin_checkpoint_equivalence_v27.py",
    "scripts/evaluate_router_metric_risk_guard_v27.py",
    "scripts/build_paper_figures_v27.py",
    "scripts/build_jcheminf_submission_v27.py",
    "scripts/build_reviewer_bundle_v27.py",
    "environment-cpu.yml",
    "environment-gpu.yml",
    "requirements-repro.txt",
)


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def build_manifest(repo_root: Path) -> dict[str, Any]:
    root = repo_root.resolve()
    snapshot_path = root / EVIDENCE_FILES["leaderboard_snapshot"]
    rebuilt_scorecard = build_v27_scorecard(root, snapshot_path)
    persisted_scorecard = _load(root / EVIDENCE_FILES["goal_scorecard"])
    if rebuilt_scorecard != persisted_scorecard:
        raise ValueError("Persisted v2.7 scorecard differs from verified rebuild")
    if persisted_scorecard["official_results_complete"] is not True:
        raise ValueError("v2.7 official results are incomplete")

    mapping = _load(root / EVIDENCE_FILES["full_freeze_mapping"])
    official = {}
    for task in sorted(TASKS):
        result, provenance = verify_official_result(root, task, mapping)
        official[task] = {
            **provenance,
            "result": next(iter(result["results"].values())),
            "seeds": result["seeds"],
        }

    confirmation = _load(root / EVIDENCE_FILES["fresh_confirmation"])
    protocol = confirmation["protocol"]
    if protocol["official_test_loaded"] is not False:
        raise ValueError("Fresh confirmation loaded the official test set")
    if protocol["test_labels_used"] is not False:
        raise ValueError("Fresh confirmation used test labels")
    if protocol["fresh_labels_used_for_candidate_selection"] is not False:
        raise ValueError("Fresh confirmation labels altered the candidate")
    if confirmation["promotion_gate_passed"] is not True:
        raise ValueError("Fresh confirmation gate did not pass")
    if any(
        int(split["cross_split_identity_overlap"]) != 0
        for task in confirmation["identity_audit"].values()
        for split in task.values()
    ):
        raise ValueError("Fresh confirmation has cross-split identity overlap")

    discrepancy = _load(root / EVIDENCE_FILES["bio_discrepancy_audit"])
    if discrepancy["assessment"]["status"] != (
        "confirmed_material_leaderboard_discrepancy"
    ):
        raise ValueError("Bio discrepancy status is not confirmed")
    if discrepancy["goal_effect"]["threshold_override_allowed"] is not False:
        raise ValueError("Bio audit improperly overrides the displayed threshold")
    if not all(discrepancy["checks"].values()):
        raise ValueError("Bio discrepancy evidence contains a failed check")

    router_controls = _load(root / EVIDENCE_FILES["router_controls"])
    if router_controls["task_count"] != 22:
        raise ValueError("Router controls do not cover all 22 tasks")
    if any(
        router_controls[field] is not False
        for field in ("official_test_loaded", "official_results_used", "leaderboard_used")
    ):
        raise ValueError("Router controls violate the validation-only firewall")
    if router_controls["matched_cost_random"]["maximum_relative_cost_error"] > 1e-9:
        raise ValueError("Matched-random control does not exactly match expected cost")
    if router_controls["memory_ablation"]["claim_gate"]["passed"] is not False:
        raise ValueError("Unexpected full-memory improvement claim")
    if router_controls["matched_cost_random"]["claim_gate"]["passed"] is not False:
        raise ValueError("Unexpected selection-intelligence claim")

    primary_effects = _load(root / EVIDENCE_FILES["primary_model_effects"])
    if primary_effects["selection_or_freeze_changed"] is not False:
        raise ValueError("Descriptive effect analysis improperly changed selection")
    if primary_effects["official_test_loaded"] is not False:
        raise ValueError("Primary effect analysis loaded official test data")
    if len(primary_effects["contrasts"]) != 3:
        raise ValueError("Primary effect analysis is incomplete")

    validation_audit = _load(root / EVIDENCE_FILES["validation_calibration_residuals"])
    if validation_audit["task_count"] != 22:
        raise ValueError("Validation calibration/residual audit is incomplete")
    if validation_audit["classification_task_count"] != 13 or validation_audit["regression_task_count"] != 9:
        raise ValueError("Validation audit task-type coverage is inconsistent")
    if any(
        validation_audit[field] is not False
        for field in ("official_test_loaded", "official_results_used", "leaderboard_used_for_analysis")
    ):
        raise ValueError("Validation calibration/residual audit violates the firewall")
    if any(
        row["identity_overlap_count"] != 0
        or row["train_valid_scaffold_overlap_count"] != 0
        for row in validation_audit["tasks"]
    ):
        raise ValueError("Validation audit contains identity or scaffold leakage")

    compute_appendix = _load(root / EVIDENCE_FILES["compute_memory_appendix"])
    if compute_appendix["router_replay"]["task_count"] != 22:
        raise ValueError("Compute appendix router coverage is incomplete")
    if compute_appendix["official_test_loaded"] is not False:
        raise ValueError("Compute appendix loaded official test data")
    if compute_appendix["peak_memory_coverage"]["records_measured"] != 0:
        raise ValueError("Unexpected historical peak-memory measurements")
    if compute_appendix["peak_memory_coverage"]["records_missing"] <= 0:
        raise ValueError("Compute appendix does not disclose peak-memory missingness")

    bibliography_audit = _load(root / EVIDENCE_FILES["bibliography_audit"])
    if bibliography_audit["passed"] is not True:
        raise ValueError("Bibliography or manuscript citation audit failed")
    if bibliography_audit["entry_count"] != 15:
        raise ValueError("Bibliography coverage is incomplete")
    if any(
        bibliography_audit[field]
        for field in (
            "missing_required_keys",
            "missing_required_fields",
            "missing_persistent_identifiers",
            "duplicate_identifiers",
            "unresolved_manuscript_citations",
            "uncited_bibliography_entries",
        )
    ):
        raise ValueError("Bibliography audit contains unresolved metadata")

    third_party = _load(root / EVIDENCE_FILES["third_party_release_audit"])
    if third_party["scope"] != {
        "frozen_task_count": 22,
        "essential_dependency_count": 2,
        "final_mapping_minimol_used": False,
        "rejected_experiments_are_not_release_dependencies": True,
    }:
        raise ValueError("Third-party release scope does not match the frozen mapping")
    if third_party["gates"]["registration_free_checkpoint_access_verified"] is not True:
        raise ValueError("Third-party checkpoint access is not verified")
    if third_party["gates"]["source_code_licenses_verified"] is not True:
        raise ValueError("Third-party source-code licenses are not verified")
    if third_party["gates"]["checkpoint_specific_licenses_verified"] is not False:
        raise ValueError("Expected fail-closed GIN artifact-license boundary changed")
    if third_party["gates"]["ready_for_external_deposit"] is not True:
        raise ValueError("Link-only reviewer bundle is not ready for external deposit")
    if any(
        third_party["policy"][field] is not False
        for field in ("official_test_loaded", "selection_or_freeze_changed")
    ):
        raise ValueError("Third-party release audit violates the evidence firewall")

    risk_guard = _load(root / EVIDENCE_FILES["router_risk_guard"])
    if risk_guard["task_count"] != 22 or risk_guard["fresh_validation_seeds"] != [4, 5]:
        raise ValueError("Router risk guard does not cover the preregistered fresh protocol")
    if any(
        risk_guard[field] is not False
        for field in ("official_test_loaded", "official_results_used", "leaderboard_used")
    ):
        raise ValueError("Router risk guard violates the validation-only firewall")
    if risk_guard["selection_or_official_freeze_changed"] is not False:
        raise ValueError("Router risk guard improperly changed the official freeze")
    if risk_guard["identity_overlap_total"] != 0:
        raise ValueError("Router risk guard fresh splits contain identity leakage")

    figures = _load(root / EVIDENCE_FILES["paper_figures"])
    if figures["figure_count"] != 4 or figures["formats"] != ["svg", "png"]:
        raise ValueError("Main-text figure set is incomplete")
    if any(
        figures[field] is not False
        for field in (
            "official_test_loaded",
            "leaderboard_used_for_selection",
            "selection_or_freeze_changed",
        )
    ):
        raise ValueError("Main-text figures violate the discovery firewall")
    if len(figures["outputs"]) != 9:
        raise ValueError("Main-text figure output inventory is incomplete")
    for relative, expected in figures["outputs"].items():
        if hash_file(root / relative) != expected:
            raise ValueError(f"Main-text figure output hash mismatch: {relative}")
    rank_figure = figures["figure_data"]["figure2_leaderboard_ranks"]
    if (rank_figure["task_count"], rank_figure["top5"], rank_figure["top10"]) != (22, 18, 21):
        raise ValueError("Leaderboard figure does not match the frozen scorecard")
    router_figure = figures["figure_data"]["figure3_router_tradeoff"]
    if (
        router_figure["task_count"] != 22
        or router_figure["fresh_validation_seeds"] != [4, 5]
        or router_figure["claim_gate_passed"] is not False
        or router_figure["aggregate_cost_ratio"] >= 0.35
    ):
        raise ValueError("Router trade-off figure does not match fresh evidence")
    space_figure = figures["figure_data"]["figure4_chemical_space_risk"]
    if (
        space_figure["classification_worse_low_similarity"] != 11
        or space_figure["classification_tasks"] != 13
        or space_figure["regression_worse_low_similarity"] != 7
        or space_figure["regression_tasks"] != 9
        or space_figure["identity_overlap_total"] != 0
        or space_figure["scaffold_overlap_total"] != 0
    ):
        raise ValueError("Chemical-space figure does not match the residual audit")

    submission = _load(root / EVIDENCE_FILES["jcheminf_submission"])
    if (
        submission["target_journal"] != "Journal of Cheminformatics"
        or submission["article_type"] != "Research Article"
        or submission["full_freeze_sha256"] != persisted_scorecard["freeze_sha256"]
    ):
        raise ValueError("Submission package target or freeze is inconsistent")
    if submission["manuscript"]["abstract_word_count"] > 350:
        raise ValueError("Submission abstract exceeds the journal limit")
    if submission["manuscript"]["contribution_heading_present"] is not True:
        raise ValueError("Submission abstract lacks the required Contribution heading")
    if submission["manuscript"]["citation_count"] != 15 or submission["manuscript"]["reference_count"] != 15:
        raise ValueError("Submission reference coverage is incomplete")
    if any(
        submission[field] is not False
        for field in ("official_test_loaded_during_generation", "selection_or_freeze_changed")
    ):
        raise ValueError("Submission generation violates the evidence firewall")
    submission_supplement = submission["supplement"]
    if any(
        submission_supplement[field] != 22
        for field in ("task_count", "router_control_task_count", "fresh_guard_task_count", "chemical_space_task_count")
    ):
        raise ValueError("Submission supplement does not cover all 22 tasks")
    if submission_supplement["identity_overlap_total"] != 0 or submission_supplement["scaffold_overlap_total"] != 0:
        raise ValueError("Submission supplement reports split leakage")
    if not submission["unresolved_submission_blockers"]:
        raise ValueError("Submission package incorrectly claims no remaining author actions")
    for relative, expected in submission["outputs"].items():
        if hash_file(root / relative) != expected:
            raise ValueError(f"Submission output hash mismatch: {relative}")

    gpu_environment = (root / "environment-gpu.yml").read_text(encoding="utf-8")
    if "\n+#" in gpu_environment:
        raise ValueError("GPU environment contains an accidental patch marker")

    evidence = {
        name: {"path": relative, **hash_file(root / relative)}
        for name, relative in EVIDENCE_FILES.items()
    }
    sources = {
        relative: hash_file(root / relative) for relative in SOURCE_FILES
    }
    payload = {
        "version": "v2.7",
        "status": "official_results_complete_goal_pending_authoritative_bio_resolution",
        "full_freeze_sha256": persisted_scorecard["freeze_sha256"],
        "scorecard_status": persisted_scorecard["status"],
        "verified_official_tasks": sorted(official),
        "validation_firewall": {
            "official_test_loaded": False,
            "test_labels_used": False,
            "fresh_labels_used_for_candidate_selection": False,
            "cross_split_identity_overlap": 0,
        },
        "bio_discrepancy_policy": {
            "status": discrepancy["assessment"]["status"],
            "threshold_override_allowed": False,
            "authoritative_correction_obtained": False,
        },
        "router_controls": {
            "task_count": router_controls["task_count"],
            "validation_only_firewall": True,
            "memory_claim_gate_passed": False,
            "selection_claim_gate_passed": False,
            "maximum_relative_cost_error": router_controls["matched_cost_random"][
                "maximum_relative_cost_error"
            ],
        },
        "primary_model_effects": {
            "contrast_count": len(primary_effects["contrasts"]),
            "post_hoc_descriptive": True,
            "selection_or_freeze_changed": False,
        },
        "validation_calibration_residuals": {
            "task_count": validation_audit["task_count"],
            "classification_task_count": validation_audit["classification_task_count"],
            "regression_task_count": validation_audit["regression_task_count"],
            "identity_overlap_count": sum(
                row["identity_overlap_count"] for row in validation_audit["tasks"]
            ),
            "scaffold_overlap_count": sum(
                row["train_valid_scaffold_overlap_count"] for row in validation_audit["tasks"]
            ),
        },
        "compute_memory_appendix": {
            "router_task_count": compute_appendix["router_replay"]["task_count"],
            "selected_to_score_total_cost_ratio": compute_appendix["router_replay"][
                "selected_to_score_total_cost_ratio"
            ],
            "peak_memory_coverage": compute_appendix["peak_memory_coverage"],
        },
        "bibliography": {
            "entry_count": bibliography_audit["entry_count"],
            "citation_audit_passed": bibliography_audit["passed"],
        },
        "third_party_release_audit": {
            "essential_dependency_count": third_party["scope"]["essential_dependency_count"],
            "final_mapping_minimol_used": third_party["scope"]["final_mapping_minimol_used"],
            "registration_free_checkpoint_access_verified": third_party["gates"][
                "registration_free_checkpoint_access_verified"
            ],
            "source_code_licenses_verified": third_party["gates"][
                "source_code_licenses_verified"
            ],
            "checkpoint_specific_licenses_verified": third_party["gates"][
                "checkpoint_specific_licenses_verified"
            ],
            "third_party_checkpoints_in_bundle": False,
            "gin_official_parameter_equivalence_verified": third_party["dependencies"][
                "gin_supervised_masking"
            ]["checkpoint"]["official_parameter_equivalence"]["verified"],
            "ready_for_external_deposit": True,
        },
        "router_metric_risk_guard": {
            "task_count": risk_guard["task_count"],
            "fresh_validation_seeds": risk_guard["fresh_validation_seeds"],
            "identity_overlap_total": risk_guard["identity_overlap_total"],
            "claim_gate_passed": risk_guard["claim_gate"]["passed"],
            "significance_gate_passed": risk_guard["significance_gate_passed"],
            "selection_or_official_freeze_changed": False,
        },
        "paper_figures": {
            "figure_count": figures["figure_count"],
            "formats": figures["formats"],
            "output_count": len(figures["outputs"]),
            "official_test_loaded": False,
            "selection_or_freeze_changed": False,
            "leaderboard_rank_summary": {
                "task_count": rank_figure["task_count"],
                "top5": rank_figure["top5"],
                "top10": rank_figure["top10"],
                "conservative_deficit_task": rank_figure["conservative_deficit_task"],
            },
            "router_claim_gate_passed": router_figure["claim_gate_passed"],
            "chemical_space_worse_counts": {
                "classification": [
                    space_figure["classification_worse_low_similarity"],
                    space_figure["classification_tasks"],
                ],
                "regression": [
                    space_figure["regression_worse_low_similarity"],
                    space_figure["regression_tasks"],
                ],
            },
        },
        "jcheminf_submission": {
            "article_type": submission["article_type"],
            "abstract_word_count": submission["manuscript"]["abstract_word_count"],
            "contribution_heading_present": submission["manuscript"]["contribution_heading_present"],
            "reference_count": submission["manuscript"]["reference_count"],
            "supplement_task_count": submission_supplement["task_count"],
            "graphical_abstract": submission["graphical_abstract"],
            "output_count": len(submission["outputs"]),
            "unresolved_blocker_count": len(submission["unresolved_submission_blockers"]),
            "official_test_loaded_during_generation": False,
            "selection_or_freeze_changed": False,
        },
        "reviewer_bundle_builder": {
            "deterministic_archive_builder_included": True,
            "gpu_environment_syntax_guard_passed": True,
            "official_test_labels_permitted": False,
            "credential_files_permitted": False,
            "external_deposit_complete": False,
            "third_party_checkpoint_access_verified": True,
            "checkpoint_specific_licenses_verified": False,
        },
        "evidence": evidence,
        "official_results": official,
        "source_files": sources,
    }
    return {**payload, "manifest_sha256": manifest_digest(payload)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("reports/paper_evidence_manifest_v27.json"),
    )
    args = parser.parse_args()
    manifest = build_manifest(args.repo_root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "status": manifest["status"],
                "manifest_sha256": manifest["manifest_sha256"],
                "scorecard_status": manifest["scorecard_status"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
