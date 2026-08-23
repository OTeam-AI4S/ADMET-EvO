"""Freeze the smallest experiment needed to activate the category-size policy."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from admet_agent.source_snapshot import production_source_manifest


def _digest(payload: object) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def main() -> None:
    root = Path(__file__).parents[1]
    runner = root / "scripts/run_minimal_exact_pair_pilot.py"
    rich = ["morgan", "avalon", "erg", "rdkit2d"]
    body = {
        "protocol_version": "minimal-category-size-exact-pair-pilot-v1",
        "objective": "Resolve only the two exact matched pairs missing from the five-category policy.",
        "tasks": {
            "CYP2C9_Veith": {
                "category": "metabolism",
                "axis": "imbalance",
                "representations": rich,
                "control": {
                    "model": "extra_trees", "imbalance": "none",
                    "hypothesis": "matched imbalance control: unweighted ExtraTrees",
                },
                "treatment": {
                    "model": "extra_trees", "imbalance": "class_weight",
                    "hypothesis": "imbalance treatment: class-weighted ExtraTrees",
                },
            },
            "LD50_Zhu": {
                "category": "toxicity",
                "axis": "model",
                "representations": rich,
                "control": {
                    "model": "lightgbm", "imbalance": "none",
                    "hypothesis": "matched model control: rich-2D LightGBM",
                },
                "treatment": {
                    "model": "extra_trees", "imbalance": "none",
                    "hypothesis": "model treatment: rich-2D ExtraTrees",
                },
            },
        },
        "screen_seeds": [1, 2],
        "confirmation_seeds": [3, 4, 5],
        "promotion_gate": {
            "all_screen_directions_must_be_positive": True,
            "minimum_mean_oriented_gain": 0.005,
        },
        "resolution_gate_after_five_seeds": {
            "minimum_absolute_mean_oriented_gain": 0.005,
            "minimum_direction_agreement": 0.80,
        },
        "maximum_initial_fits": 8,
        "maximum_total_fits": 20,
        "default_unimol2": False,
        "default_committee": False,
        "default_external_data_search": False,
        "max_compute_minutes_per_fit": 30,
        "source_sha256": production_source_manifest()["source_sha256"],
        "runner_sha256": hashlib.sha256(runner.read_bytes()).hexdigest(),
        "test_loaded": False,
        "test_feedback_to_discovery": False,
    }
    output = root / "runs/minimal-exact-pair-pilot-v1/protocol.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps({**body, "sha256": _digest(body)}, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
