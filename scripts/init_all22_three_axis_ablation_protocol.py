#!/usr/bin/env python3
"""Freeze the 22-endpoint model/feature/data ablation before outcomes exist."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

from run_classic_ml_external_data_confirmation import TASKS


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "reports/all22_three_axis_ablation_v1/protocol.json"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    implementation = [
        ROOT / "scripts/run_all22_three_axis_ablation.py",
        ROOT / "scripts/build_all22_three_axis_ablation_summary.py",
        ROOT / "scripts/slurm_all22_three_axis_ablation.sh",
        ROOT / "scripts/slurm_all22_three_axis_finalize.sh",
        ROOT / "admet_agent/features.py",
        ROOT / "admet_agent/selection.py",
        ROOT / "admet_agent/adaptive_external_data.py",
        ROOT / "admet_agent/heldout_structure_firewall.py",
    ]
    data_path = ROOT / "data/ADMET_v3_clean_external_candidate_pool.csv"
    audit_path = data_path.with_suffix(data_path.suffix + ".audit.json")
    payload = {
        "protocol": "all22-three-axis-agent-ablation-v1",
        "frozen_at": datetime.now(timezone.utc).isoformat(),
        "tasks": list(TASKS),
        "baseline": {
            "model": "lightgbm",
            "representations": ["morgan"],
            "hyperparameters": {},
            "imbalance": "none",
            "data": "official_only",
            "description": "historical fixed Morgan/LightGBM controller",
        },
        "seeds": {
            "discovery": [2, 3, 4, 5],
            "confirmation": [8, 9, 10, 11],
            "certification_model_seeds": [8, 9, 10, 11],
        },
        "selection": {
            "unit": "mean normalized oriented validation improvement across seeds",
            "practical_threshold": 0.005,
            "positive_seed_fraction": 0.75,
            "tie_break": "simpler_then_lexicographic",
            "test_seed_selection_forbidden": True,
        },
        "model_axis": {
            "M0": "fixed LightGBM",
            "M1": "default-parameter model-family selection",
            "M2": "model-family plus task-level parameter/imbalance selection",
            "M3": "validation-selected blend of complementary tuned models",
            "families": [
                "lightgbm",
                "xgboost",
                "extra_trees",
                "random_forest",
                "catboost",
                "svm_rbf_if_n_train_le_5000",
            ],
            "tune_top_default_families": 3,
            "parameter_variants_per_family": 3,
            "fusion_weights": [0.25, 0.5, 0.75],
            "locked": ["representations", "official_data", "split", "metric"],
        },
        "feature_axis": {
            "F0": "Morgan radius-2 bit fingerprint",
            "F1": "single-view representation selection",
            "F2": "task-adaptive composed-view selection",
            "F3": "feature-level union of complementary selected views",
            "single_views": [
                ["morgan"],
                ["morgan_r3"],
                ["morgan_count"],
                ["avalon"],
                ["erg"],
                ["rdkit2d"],
            ],
            "composed_views": [
                ["morgan", "rdkit2d"],
                ["morgan", "avalon", "erg", "rdkit2d"],
                ["morgan_count", "atom_pair", "maccs", "rdkit2d"],
                ["morgan", "rdkit_fp", "torsion", "rdkit2d"],
            ],
            "locked": ["lightgbm_default", "official_data", "split", "metric"],
        },
        "data_axis": {
            "D0": "official data only",
            "D1": "fixed-ratio random sample from admitted safe external evidence",
            "D2": "relevance-diverse coreset at fixed ratio 1x",
            "D3": "adaptive relevance-diverse coreset ratio",
            "ratios": [0.5, 1.0, 2.0, 4.0, 8.0],
            "external_mass_ratio": 1.0,
            "minimum_relevance": 0.20,
            "validation_near_neighbor_threshold": 0.80,
            "hidden_test_near_analog_threshold": 0.90,
            "same_source_test_overlap_rejection_threshold": 0.05,
            "same_source_rule": "reject when raw exact test overlap fraction is greater than 0.05",
            "safety_filter_always_on": True,
            "locked": ["lightgbm_default", "morgan", "split", "metric"],
        },
        "factorial": {
            "cells": ["000", "100", "010", "001", "110", "101", "011", "111"],
            "axis_order": ["model", "feature", "data"],
            "on_level": "frozen M3/F3/D3 selected without test labels",
        },
        "routing": {
            "candidates": ["baseline", "M3", "F3", "D3", "MF", "MD", "FD", "MFD"],
            "threshold": 0.005,
            "selection_data": "discovery validation only",
        },
        "certification": {
            "fit_data": "entire official train_val plus frozen admitted external rows",
            "evaluation": "official hidden test",
            "test_structures_available_to_firewall": True,
            "test_labels_available_before_certification": False,
            "report": "mean and sample SD over preregistered model seeds",
        },
        "external_data": {
            "path": str(data_path.resolve()),
            "sha256": _sha256(data_path),
            "audit_path": str(audit_path.resolve()),
            "audit_sha256": _sha256(audit_path),
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
