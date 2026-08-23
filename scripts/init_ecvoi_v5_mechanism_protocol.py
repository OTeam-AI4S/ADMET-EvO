"""Freeze the EC-VOI v5 mechanism before binding a new evidence cohort."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    implementation = [
        Path("admet_agent/ecvoi_v5_policy.py"),
        Path("admet_agent/imbalance_action.py"),
        Path("tests/test_ecvoi_v5_policy.py"),
    ]
    payload = {
        "version": "ecvoi-v5-mechanism-protocol-v1",
        "frozen_at": "2026-08-23",
        "scope": "mechanism_only_before_new_endpoint_source_cohort_binding",
        "predecessor_evidence": {
            "v4_budget_protocol_sha256": "d6396a249adf7b93884cae9bbef7bff64a1d83773266e25a05f6c3b95c4d82b5",
            "v4_budget_gate_passed": False,
            "openadmet_applicability_protocol_sha256": "9f5be7a4df914ca70beeff3fbec377bb7155d1f2eccc0f5fd33bfd2c91a904cb",
            "openadmet_applicability_gate_passed": False,
        },
        "implementation_sha256": {
            str(path): _sha256(path) for path in implementation
        },
        "eligibility": {
            "practical_gain": 0.005,
            "harm_margin": 0.005,
            "harm_probability_limit": 0.20,
            "cost_penalty": 0.00025,
            "information_gain_weight_initial": 0.0,
            "stop": "explicit_zero_gain_zero_cost_zero_harm_always_eligible",
            "budget_fill_required": False,
        },
        "imbalance_action": {
            "name": "nested_group_oof_mcc_threshold_plus_class_balanced_boosting",
            "threshold_grid": "0.05_to_0.95_step_0.01",
            "threshold_data": "outer_train_inner_group_disjoint_oof_only",
            "inner_fits_count_toward_budget": True,
            "outer_evaluation_labels_used": False,
        },
        "required_actions": [
            "feature_rich",
            "nested_imbalance",
            "data_if_endpoint_contract_admits",
            "replicate",
            "stop",
        ],
        "new_cohort_requirements": {
            "minimum_endpoint_source_units": 8,
            "task_types": ["classification", "regression"],
            "required_state_axes": ["imbalance", "structural_shift", "source_shift"],
            "leave_out": "endpoint_and_source",
            "same_outer_split_seed_data_permissions": True,
            "official_blind_may_be_separate_certification": True,
        },
        "promotion_gate": {
            "full_regret_reduction_vs_no_diagnosis": ">=0.20",
            "full_regret_reduction_vs_shuffled": ">=0.20",
            "full_total_gain_strictly_above_both": True,
            "harmful_interventions": 0,
            "at_least_one_strict_anchor_win": True,
            "matched_budget_and_permissions": True,
            "endpoint_source_cluster_bootstrap_direction_consistent": True,
            "test_blind_leaderboard_access": 0,
        },
        "unlock_policy": {
            "tdc_19_remains_sealed": True,
            "unlock_only_if_all_new_cohort_gates_pass": True,
            "known_v4_confirmation_cannot_validate_v5": True,
            "openadmet_training_oof_cannot_validate_v5": True,
        },
        "test_labels_loaded": False,
        "blind_test_opened": False,
        "leaderboard_feedback_used": False,
    }
    encoded = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    payload["sha256"] = hashlib.sha256(encoded.encode()).hexdigest()
    output = Path("reports/ecvoi_v5_mechanism_protocol_v1.json")
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(payload["sha256"])


if __name__ == "__main__":
    main()
