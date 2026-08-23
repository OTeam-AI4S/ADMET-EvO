"""Freeze the matched-budget public-cohort action screen before outcomes exist."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    cohort_path = Path("reports/ecvoi_v5_public_cohort_v4.csv")
    cohort_manifest_path = Path("reports/ecvoi_v5_public_cohort_manifest_v4.json")
    mechanism_path = Path("reports/ecvoi_v5_mechanism_protocol_v1.json")
    cohort_manifest = json.loads(cohort_manifest_path.read_text())
    mechanism = json.loads(mechanism_path.read_text())
    all_endpoints = sorted(cohort_manifest["endpoint_summary"])
    primary = sorted(
        endpoint
        for endpoint, summary in cohort_manifest["endpoint_summary"].items()
        if summary["rows"] >= 100
    )
    stress = sorted(set(all_endpoints) - set(primary))
    implementation = [
        Path("admet_agent/public_action_matrix.py"),
        Path("admet_agent/imbalance_action.py"),
        Path("scripts/run_ecvoi_v5_public_action_screen.py"),
        Path("tests/test_public_action_matrix.py"),
    ]
    payload = {
        "version": "ecvoi-v5-public-action-screen-protocol-v1",
        "frozen_at": "2026-08-23",
        "scope": "development_action_effect_generation_only_before_policy_analysis",
        "cohort": {
            "path": str(cohort_path.resolve()),
            "sha256": _sha256(cohort_path),
            "manifest_sha256": _sha256(cohort_manifest_path),
            "governance_promotion_passed": cohort_manifest["promotion_passed"],
        },
        "mechanism_protocol_file_sha256": _sha256(mechanism_path),
        "mechanism_protocol_self_sha256": mechanism["sha256"],
        "endpoint_scope": all_endpoints,
        "primary_endpoint_scope": primary,
        "small_sample_stress_scope": stress,
        "screen_seed": 17,
        "confirmation_seeds_locked_not_executed": [29, 43],
        "outer_folds": 5,
        "inner_folds": 5,
        "estimator": {
            "family": "ExtraTrees",
            "trees": 200,
            "min_samples_leaf": 2,
            "max_features": "sqrt",
            "workers": 4,
        },
        "actions": {
            "baseline": {"representation": ["morgan_r2_2048"], "class_weight": None, "threshold": 0.5, "outer_fit_cost": 1},
            "feature_rich": {"representation": ["morgan_r2_2048", "avalon_1024", "erg", "rdkit2d"], "class_weight": None, "threshold": 0.5, "outer_fit_cost": 1},
            "nested_imbalance": {"classification_only": True, "representation": ["morgan_r2_2048"], "class_weight": "balanced", "threshold": "five-fold group-disjoint inner OOF MCC", "outer_fit_cost": 6},
            "stop": {"oriented_effect": 0.0, "fit_cost": 0, "always_available": True},
            "replicate": {"availability": "only_after_seed17_policy_and_uncertainty_protocol_are_frozen", "fresh_seeds": [29, 43]},
            "data": {"available_in_this_matrix": False, "reason": "no second contract-compatible supervised source was preregistered for the same public-cohort endpoint", "separate_real_loop": "frozen OpenADMET external propensity experiment"},
        },
        "state_encoder": {
            "features": ["task_type", "log_train_rows", "train_minority_fraction", "train_group_density", "median_sampled_validation_to_train_tanimoto", "source_id_for_heldout_partition_only"],
            "similarity_sample_size": 256,
            "validation_labels_used": False,
            "outer_validation_baseline_score_used_as_state": False,
        },
        "metrics": {"classification": "MCC_maximize", "regression": "negative_MAE_maximize"},
        "matched_control_contract": {
            "same_action_effect_registry": True,
            "same_outer_folds_and_seed": True,
            "same_data_permissions": True,
            "fit_cost_and_wall_time_recorded": True,
            "controls_cannot_request_extra_fits": True,
        },
        "screen_integrity_gate": {
            "expected_result_rows": 245,
            "failures": 0,
            "test_blind_leaderboard_access": 0,
            "promotion_analysis_in_runner": False,
        },
        "promotion_gate_inherited_unchanged": mechanism["promotion_gate"],
        "implementation_sha256": {str(path): _sha256(path) for path in implementation},
        "test_labels_loaded": False,
        "blind_or_leaderboard_feedback_used": False,
    }
    encoded = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    payload["sha256"] = hashlib.sha256(encoded.encode()).hexdigest()
    output = Path("reports/ecvoi_v5_public_action_screen_protocol_v1.json")
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(payload["sha256"])


if __name__ == "__main__":
    main()
