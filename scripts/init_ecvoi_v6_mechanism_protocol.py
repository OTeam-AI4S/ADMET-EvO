"""Freeze v6 after v5 failure analysis and before any fresh-cohort outcomes."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    implementation = [
        Path("admet_agent/public_policy_v6.py"),
        Path("admet_agent/ecvoi_v5_policy.py"),
        Path("admet_agent/public_policy_analysis.py"),
        Path("tests/test_public_policy_v6.py"),
    ]
    payload = {
        "version": "ecvoi-v6-identifiability-shrinkage-mechanism-v2",
        "supersedes": {
            "protocol": "ecvoi-v6-identifiability-shrinkage-mechanism-v1",
            "reason": "pre-outcome unit test exposed a read-only sample-weight array defect",
            "fresh_outcomes_seen_before_supersession": False,
        },
        "frozen_at": "2026-08-23",
        "development_evidence": {
            "cohort": "ecvoi-v5-public-cohort-v4",
            "v5_outcomes_available": True,
            "use": "mechanism development and negative-control replay only",
            "eligible_for_v6_promotion": False,
            "observed_failure": "state-conditioned v5 did not beat source-balanced intercept controls",
        },
        "fresh_outcomes_available_at_freeze": False,
        "mechanism": {
            "heldout_rule": "remove every endpoint from held endpoint source",
            "prior": "equal-source-weight action-effect mean",
            "diagnostic_model": "standardized ridge alpha=4 with inverse source-size weights",
            "identifiability_test": "inner leave-one-source-out squared-loss comparison against prior",
            "minimum_training_sources": 5,
            "minimum_source_win_fraction": 0.60,
            "minimum_median_loss_gain": 0.0,
            "full_weight_sources": 9,
            "fallback": "diagnostic shrinkage weight exactly zero",
            "shrinkage": "positive normalized median loss gain times source-count reliability",
            "uncertainty": "Dirichlet source-weight bootstrap plus source-balanced residual noise",
        },
        "eligibility": {
            "practical_gain": 0.005,
            "harm_margin": 0.005,
            "harm_probability_limit": 0.20,
            "cost_penalty": 0.00025,
            "information_gain_weight": 0.0,
        },
        "actions": ["feature_rich", "nested_imbalance", "stop"],
        "action_costs": {"feature_rich": 1, "nested_imbalance": 6, "stop": 0},
        "fresh_confirmation_requirements": {
            "minimum_scientific_sources": 9,
            "minimum_eligible_endpoints": 18,
            "source_identity_audited": True,
            "endpoint_contract_and_license_audited": True,
            "identity_and_near-structure_firewall": True,
            "frozen_before_action_outcomes": True,
            "controls": [
                "source-balanced no-diagnosis", "unshrunk state model",
                "shuffled diagnostics", "fixed rules", "boosting anchor", "expanded AutoML",
            ],
            "matched_max_fits_per_endpoint": 21,
            "primary_gate": {
                "mean_oracle_regret_reduction_vs_no_diagnosis": ">=0.15",
                "total_gain_above_no_diagnosis": ">0",
                "total_gain_above_unshrunk_state": ">0",
                "harmful_interventions": 0,
                "source_cluster_positive_direction_probability": ">=0.90",
                "at_least_one_strict_boosting_anchor_win": True,
            },
        },
        "implementation_sha256": {str(path): _sha256(path) for path in implementation},
        "test_or_leaderboard_feedback_used": False,
        "tdc_19_locked": True,
    }
    encoded = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    payload["sha256"] = hashlib.sha256(encoded.encode()).hexdigest()
    output = Path("reports/ecvoi_v6_mechanism_protocol_v2.json")
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(payload["sha256"])


if __name__ == "__main__":
    main()
