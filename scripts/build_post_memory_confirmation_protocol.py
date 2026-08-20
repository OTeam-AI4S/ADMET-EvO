"""Freeze an independent fresh-seed test of the negative-memory pruning gate."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from admet_agent.source_snapshot import production_source_manifest


def _digest(payload: dict) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def main() -> None:
    root = Path(__file__).parents[1]
    runner = root / "scripts/run_post_memory_confirmation.py"
    body = {
        "protocol_version": "post-memory-negative-gate-confirmation-v1",
        "objective": (
            "Test whether pruning model-estimator-family after prospective refutation "
            "saves treatment fits without material missed oriented effect."
        ),
        "endpoints": [
            "Solubility_AqSolDB",
            "PPBR_AZ",
            "CYP3A4_Veith",
            "Clearance_Microsome_AZ",
            "DILI",
        ],
        "selection_rule": (
            "Before confirmation outcomes, choose one non-transport-training endpoint "
            "per ADMET category with the largest no-memory model-estimator-family "
            "utility among endpoints exposing that action on frozen seed-1 hardness."
        ),
        "excluded_transport_training_endpoints": [
            "Caco2_Wang",
            "BBB_Martins",
            "CYP2C9_Veith",
            "Clearance_Hepatocyte_AZ",
            "LD50_Zhu",
        ],
        "validation_seeds": [6, 7, 8, 9, 10],
        "action_id": "model-estimator-family",
        "memory_decision": "prune",
        "no_memory_decision": "execute",
        "negative_gate": {
            "minimum_transport_endpoints": 5,
            "minimum_refuted_rate": 0.80,
            "maximum_posterior_support": 0.35,
            "frozen_observed_refuted_rate": 0.80,
            "frozen_posterior_support": 0.2777777777777778,
        },
        "representations": ["morgan", "rdkit2d"],
        "hyperparameters": {
            "extra_trees": {
                "n_estimators": 300,
                "min_samples_leaf": 2,
                "max_features": "sqrt",
            },
            "random_forest": {
                "n_estimators": 300,
                "min_samples_leaf": 2,
                "max_features": "sqrt",
            },
        },
        "minimum_effect": 0.005,
        "minimum_direction_agreement": 0.80,
        "pruning_success_gate": {
            "maximum_supported_endpoints": 0,
            "maximum_mean_positive_regret": 0.005,
            "minimum_treatment_fits_avoided": 25,
        },
        "maximum_total_evaluation_fits": 50,
        "max_compute_minutes_per_endpoint": 30,
        "source_sha256": production_source_manifest()["source_sha256"],
        "runner_sha256": hashlib.sha256(runner.read_bytes()).hexdigest(),
        "test_loaded": False,
        "test_feedback_to_discovery": False,
    }
    output = root / "runs/v5-post-memory-confirmation/protocol.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps({**body, "sha256": _digest(body)}, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
