#!/usr/bin/env python3
"""Freeze the source-admitted 22-endpoint three-axis ablation v3 protocol."""

from __future__ import annotations

import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from run_classic_ml_external_data_confirmation import TASKS  # noqa: E402


OUTPUT = ROOT / "reports/all22_three_axis_ablation_v3/protocol.json"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    implementation = [
        ROOT / "scripts/init_all22_three_axis_ablation_protocol_v2.py",
        ROOT / "scripts/run_all22_three_axis_ablation.py",
        ROOT / "scripts/run_all22_three_axis_ablation_v2.py",
        ROOT / "scripts/preflight_all22_three_axis_data_v3.py",
        ROOT / "scripts/build_all22_three_axis_ablation_summary.py",
        ROOT / "scripts/build_all22_three_axis_ablation_summary_v2.py",
        ROOT / "scripts/slurm_all22_three_axis_ablation_v3.sh",
        ROOT / "scripts/slurm_all22_three_axis_finalize_v3.sh",
        ROOT / "admet_agent/data_axis_v2.py",
        ROOT / "admet_agent/features.py",
        ROOT / "admet_agent/selection.py",
        ROOT / "admet_agent/adaptive_external_data.py",
        ROOT / "admet_agent/external_data_admission.py",
        ROOT / "admet_agent/external_structure_firewall.py",
    ]
    data_path = ROOT / "data/ADMET_v3_clean_external_candidate_pool.csv"
    audit_path = data_path.with_suffix(data_path.suffix + ".audit.json")
    admission_path = (
        ROOT
        / "reports/classic_ml_external_data_all22_audit_v2/admission_report_v2.json"
    )
    payload = {
        "protocol": "all22-three-axis-agent-ablation-v3",
        "frozen_at": datetime.now(timezone.utc).isoformat(),
        "objective": "reviewer-facing causal ablation of model, feature, and data-centric agent optimization",
        "tasks": list(TASKS),
        "baseline": {
            "model": "lightgbm",
            "representations": ["morgan"],
            "hyperparameters": {},
            "imbalance": "none",
            "data": "official_only_leakage_safe",
        },
        "seeds": {
            "discovery": [21, 22, 23, 24],
            "confirmation": [31, 32, 33, 34],
            "certification_model_seeds": [41, 42, 43, 44],
        },
        "selection": {
            "unit": "paired mean normalized oriented validation improvement across seeds",
            "risk_adjustment": "mean minus 0.5 standard error",
            "practical_threshold": 0.005,
            "confirmation_practical_threshold": 0.005,
            "positive_seed_fraction": 0.75,
            "no_regret_inheritance": True,
            "tie_break": "lower_complexity_then_lexicographic",
            "test_seed_selection_forbidden": True,
        },
        "model_axis": {
            "M0": "fixed LightGBM",
            "M1": "variance-aware default model-family selection",
            "M2": "task-level parameter and imbalance optimization with M1 inheritance",
            "M3": "two- or three-family probability/raw prediction fusion with M2 inheritance",
            "families": [
                "lightgbm", "xgboost", "extra_trees", "random_forest",
                "catboost", "balanced_extra_trees_for_classification",
                "svm_rbf_if_n_train_le_5000",
            ],
            "tune_top_default_families": 4,
            "parameter_variants_per_family": 3,
            "two_member_fusion_weights": [0.20, 0.35, 0.50, 0.65, 0.80],
            "three_member_fusion_weights": [
                [1 / 3, 1 / 3, 1 / 3], [0.5, 0.25, 0.25],
                [0.25, 0.5, 0.25], [0.25, 0.25, 0.5],
            ],
            "locked": ["morgan", "official_data", "split", "metric"],
        },
        "feature_axis": {
            "F0": "Morgan radius-2 bit fingerprint",
            "F1": "variance-aware single-view representation selection",
            "F2": "fixed and top-single pairwise composed-view search with F1 inheritance",
            "F3": "complexity-aware union of complementary composed views with F2 inheritance",
            "single_views": [
                ["morgan"], ["morgan_r3"], ["morgan_count"], ["avalon"],
                ["erg"], ["rdkit2d"], ["maccs"], ["rdkit_fp"],
                ["atom_pair"], ["torsion"],
            ],
            "composed_views": [
                ["morgan", "rdkit2d"],
                ["morgan", "avalon", "erg", "rdkit2d"],
                ["morgan_count", "atom_pair", "maccs", "rdkit2d"],
                ["morgan", "rdkit_fp", "torsion", "rdkit2d"],
                ["morgan_r3", "rdkit2d"],
                ["avalon", "erg", "rdkit2d"],
            ],
            "locked": ["lightgbm_default", "official_data", "split", "metric"],
        },
        "data_axis": {
            "D0": "official training partition, mandatory holdout firewall",
            "D1": "train-only identity curation and conflict-aware replicate handling",
            "D2": "train-only scaffold/target sample valuation and weighting with D1 inheritance",
            "D3": "selective external acquisition with source admission, coreset, mass, ratio, and abstention",
            "native_policies": ["official", "curated"],
            "weight_policies": ["none", "scaffold_balance", "target_balance", "combined"],
            "external_row_budgets": [10, 20, 30],
            "external_ratios": [0.05, 0.10, 0.20],
            "external_selection_methods": ["relevance_diverse_coreset", "source_matched_random"],
            "external_weight_policy": {
                "direct_cleared": {
                    "total_mass_ratio_cap": 0.20,
                    "per_row_weight_cap": 0.50,
                    "max_fraction_of_min_native_weight": 0.50,
                },
                "controlled_bridge_cleared": {
                    "total_mass_ratio_cap": 0.05,
                    "per_row_weight_cap": 0.25,
                    "max_fraction_of_min_native_weight": 0.50,
                },
                "controlled_bridge_pending": {
                    "total_mass_ratio_cap": 0.02,
                    "per_row_weight_cap": 0.10,
                    "max_fraction_of_min_native_weight": 0.50,
                },
                "shared_or_unresolved": {
                    "total_mass_ratio_cap": 0.0,
                    "per_row_weight_cap": 0.0,
                    "max_fraction_of_min_native_weight": 0.0,
                },
            },
            "minimum_relevance": 0.20,
            "validation_near_neighbor_threshold": 0.80,
            "hidden_test_near_analog_threshold": 0.90,
            "same_source_test_overlap_rejection_threshold": 0.05,
            "same_source_rule": "reject when raw exact test overlap fraction is greater than 0.05",
            "safety_filter_always_on": True,
            "heldout_labels_for_curation_or_weighting": False,
            "locked": ["lightgbm_default", "morgan", "split", "metric"],
        },
        "factorial": {
            "cells": ["000", "100", "010", "001", "110", "101", "011", "111"],
            "axis_order": ["model", "feature", "data"],
            "on_level": "frozen M3/F3/D3 proposed in discovery without hidden-test labels",
        },
        "routing": {
            "candidates": ["000", "100", "010", "001", "110", "101", "011", "111"],
            "proposal_data": "discovery validation only",
            "final_selection_data": "independent confirmation validation only",
            "hidden_test_selection_forbidden": True,
            "threshold": 0.005,
        },
        "certification": {
            "fit_data": "entire official train_val plus frozen curated/weighted/admitted external rows",
            "evaluation": "official held-out test",
            "test_structures_available_to_firewall": True,
            "test_labels_available_before_certification": False,
            "report": "mean and sample SD over preregistered certification model seeds",
        },
        "external_data": {
            "path": str(data_path.resolve()),
            "sha256": _sha256(data_path),
            "audit_path": str(audit_path.resolve()),
            "audit_sha256": _sha256(audit_path),
            "admission_report_path": str(admission_path.resolve()),
            "admission_report_sha256": _sha256(admission_path),
            "source_priority": [
                "endpoint-matched FDA/regulatory assay",
                "endpoint-matched independent literature assay",
                "controlled bridge only when explicitly admitted",
                "unresolved aggregate database remains quarantined",
            ],
        },
        "implementation_sha256": {
            str(path.relative_to(ROOT)): _sha256(path) for path in implementation
        },
        "test_labels_loaded": False,
    }
    encoded = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    payload["sha256"] = hashlib.sha256(encoded.encode()).hexdigest()
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"output": str(OUTPUT), "sha256": payload["sha256"]}))


if __name__ == "__main__":
    main()
