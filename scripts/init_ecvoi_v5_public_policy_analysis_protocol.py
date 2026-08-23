"""Freeze cross-source policy analysis and statistics before seed-17 outcomes."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    action_protocol_path = Path("reports/ecvoi_v5_public_action_screen_protocol_v2.json")
    mechanism_path = Path("reports/ecvoi_v5_mechanism_protocol_v1.json")
    action_protocol = json.loads(action_protocol_path.read_text())
    mechanism = json.loads(mechanism_path.read_text())
    forbidden_outputs = [
        Path("reports/ecvoi_v5_public_action_screen_seed17_v2.csv"),
        Path("reports/ecvoi_v5_public_action_screen_seed17_manifest_v2.json"),
    ]
    if any(path.exists() for path in forbidden_outputs):
        raise RuntimeError("Refusing to freeze analysis after action outcomes exist")
    implementation = [
        Path("admet_agent/public_policy_analysis.py"),
        Path("admet_agent/ecvoi_v5_policy.py"),
        Path("scripts/analyze_ecvoi_v5_public_policy.py"),
        Path("tests/test_public_policy_analysis.py"),
    ]
    payload = {
        "version": "ecvoi-v5-public-policy-analysis-protocol-v1",
        "frozen_at": "2026-08-23",
        "outcomes_available_at_freeze": False,
        "action_screen_protocol_file_sha256": _sha256(action_protocol_path),
        "action_screen_protocol_self_sha256": action_protocol["sha256"],
        "primary_endpoint_scope": action_protocol["primary_endpoint_scope"],
        "small_sample_stress_excluded_from_primary_gate": action_protocol["small_sample_stress_scope"],
        "cross_fit": {
            "unit": "endpoint",
            "heldout_rule": "remove held endpoint and every endpoint from its source",
            "outer_fold_action_effects": "mean only after complete pairing",
            "heldout_endpoint_outcomes_used_to_fit_policy": False,
            "heldout_source_outcomes_used_to_fit_policy": False,
        },
        "state_encoder": action_protocol["state_encoder"],
        "effect_model": {
            "family": "standardized ridge",
            "alpha": 4.0,
            "source_cluster_bootstrap_draws": 256,
            "predictive_residual_floor": 0.01,
            "full": "state-conditioned",
            "no_diagnosis": "intercept-only",
            "shuffled_diagnosis": "training state rows permuted independently of effects",
        },
        "posterior_bootstraps": 256,
        "cluster_bootstraps": 2000,
        "cluster_direction_probability_threshold": 0.90,
        "seed": 20260823,
        "eligibility": {
            "practical_gain": mechanism["eligibility"]["practical_gain"],
            "harm_margin": mechanism["eligibility"]["harm_margin"],
            "harm_probability_limit": mechanism["eligibility"]["harm_probability_limit"],
            "cost_penalty": mechanism["eligibility"]["cost_penalty"],
            "information_gain_weight": mechanism["eligibility"]["information_gain_weight_initial"],
        },
        "candidate_research_actions": ["feature_rich", "nested_imbalance", "stop"],
        "controls": ["no_diagnosis", "shuffled_diagnosis", "rules", "boosting_anchor", "automl_expanded", "full_no_harm"],
        "rules_control": {"nested_if_minority_fraction_lte": 0.10, "feature_if_structural_support_lte": 0.35, "otherwise": "stop"},
        "matched_max_fits_per_endpoint": 21,
        "costs": {"stop": 0, "feature_rich": 1, "nested_imbalance": 6, "boosting_anchor": 1, "automl_expanded": 21},
        "primary_estimands": ["total_oriented_gain", "mean_oracle_regret", "harmful_interventions", "gain_per_fit"],
        "promotion_gate": mechanism["promotion_gate"],
        "implementation_sha256": {str(path): _sha256(path) for path in implementation},
        "test_labels_loaded": False,
        "blind_or_leaderboard_feedback_used": False,
        "tdc_19_locked_until_all_checks_pass": True,
    }
    encoded = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    payload["sha256"] = hashlib.sha256(encoded.encode()).hexdigest()
    output = Path("reports/ecvoi_v5_public_policy_analysis_protocol_v1.json")
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(payload["sha256"])


if __name__ == "__main__":
    main()
