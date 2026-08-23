"""Freeze applicability protocol v2 after a record-key-only v1 abort."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    predecessor = Path("reports/openadmet_cyp_applicability_protocol_v1.json")
    result_path = Path("runs/openadmet-cyp-stage-b/b1_cv_result.json").resolve()
    result = json.loads(result_path.read_text())
    prediction_path = Path(result["prediction_path"]).resolve()
    implementation = [
        Path("admet_agent/applicability.py"),
        Path("scripts/analyze_openadmet_applicability_v1.py"),
        Path("scripts/analyze_openadmet_applicability_v2.py"),
    ]
    payload = {
        "version": "openadmet-cyp-applicability-protocol-v2",
        "frozen_at": "2026-08-23",
        "scope": "training_only_oof_post_selection_diagnostic",
        "aborted_predecessor_protocol_sha256": _sha256(predecessor),
        "compatibility_adapter_scope": "add_source_table_and_source_row_to_seed_collapse_key_only",
        "input_result_path": str(result_path),
        "input_result_sha256": _sha256(result_path),
        "input_prediction_path": str(prediction_path),
        "input_prediction_sha256": _sha256(prediction_path),
        "implementation_sha256": {
            str(path): _sha256(path) for path in implementation
        },
        "seeds": [1, 2, 3],
        "uncertainty_contract": {
            "distance": "1_minus_nearest_outer_train_morgan2_2048_tanimoto",
            "model_regression": "seed_prediction_disagreement_percentile_rank",
            "model_classification": "mean_seed_disagreement_and_one_minus_margin_percentile_ranks",
            "combined": "equal_mean_of_available_endpoint_local_percentile_ranks",
            "labels_used_to_construct_uncertainty": False,
            "primary_loss_classification": "brier_row_loss",
            "primary_loss_regression": "official_interval_aware_ST_RAE_numerator_row_loss",
            "high_error": "strictly_above_endpoint_loss_q75",
            "risk_coverage_grid": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
            "conformal": "cross_outer_fold_absolute_residual_with_constant_or_0.25_plus_combined_rank_scale",
        },
        "gate": {
            "combined_beats_both_single_signals": ">=4_of_6_endpoints",
            "combined_detects_high_error": ">=4_of_6_endpoints_AUROC_gt_0.5",
            "low_similarity_has_higher_loss": ">=4_of_6_endpoints",
            "forbidden_blind_access": 0,
        },
        "claim_boundary": {
            "allowed": "OpenADMET training OOF post-selection applicability diagnostic",
            "forbidden": [
                "prospective_blind_UQ_validation",
                "official_test_superiority",
                "universal_calibration_claim",
            ],
        },
        "blind_test_opened": False,
        "test_labels_loaded": False,
        "leaderboard_feedback_used": False,
    }
    encoded = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    payload["sha256"] = hashlib.sha256(encoded.encode()).hexdigest()
    output = Path("reports/openadmet_cyp_applicability_protocol_v2.json")
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(payload["sha256"])


if __name__ == "__main__":
    main()
