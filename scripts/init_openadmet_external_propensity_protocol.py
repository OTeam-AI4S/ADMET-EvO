"""Freeze the matched OpenADMET/Veith auxiliary propensity protocol."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build(args: argparse.Namespace) -> dict[str, Any]:
    implementation = args.implementation.resolve()
    longform = args.longform.resolve()
    longform_manifest = longform.with_suffix(".manifest.json")
    bridge = args.external_bridge.resolve()
    bridge_manifest_path = bridge.with_suffix(".manifest.json")
    baseline = args.baseline_reference.resolve()
    bridge_manifest = json.loads(bridge_manifest_path.read_text())
    baseline_payload = json.loads(baseline.read_text())
    if bridge_manifest.get("training_only") is not True:
        raise RuntimeError("External bridge is not training-only")
    if bridge_manifest.get("blind_test_opened") is not False:
        raise RuntimeError("External bridge indicates blind-test access")
    if bridge_manifest.get("exact_primary_endpoint_merge_forbidden") is not True:
        raise RuntimeError("External bridge route is not auxiliary-only")
    if baseline_payload.get("training_only") is not True:
        raise RuntimeError("Baseline reference is not training-only")
    if baseline_payload.get("blind_test_opened") is not False:
        raise RuntimeError("Baseline reference indicates blind-test access")
    payload = {
        "version": "openadmet-cyp-external-propensity-protocol-v1",
        "scope": "training_only_matched_external_data_trial",
        "implementation_path": str(implementation),
        "implementation_source_sha256": _sha256(implementation),
        "inputs": {
            "longform_path": str(longform),
            "longform_sha256": _sha256(longform),
            "longform_manifest_sha256": _sha256(longform_manifest),
            "external_bridge_path": str(bridge),
            "external_bridge_sha256": _sha256(bridge),
            "external_bridge_manifest_sha256": _sha256(bridge_manifest_path),
            "baseline_reference_path": str(baseline),
            "baseline_reference_sha256": _sha256(baseline),
        },
        "arms": {
            "control": "frozen_B1_rich_2D_LightGBM",
            "treatment": "same_B1_plus_three_fold_local_Veith_propensities",
        },
        "route": "source_conditioned_auxiliary_propensity_features",
        "exact_primary_endpoint_merge": False,
        "outer_folds": [0, 1, 2, 3, 4],
        "seeds": [1, 2, 3],
        "threads": 8,
        "baseline_reproduction_tolerance": 1e-12,
        "main_representations": ["morgan", "avalon", "erg", "rdkit2d"],
        "source_representations": ["morgan"],
        "source_holdout": "sha256_chemistry_group_mod_5",
        "openadmet_validation_firewall": (
            "exclude_every_bridge_record_whose_identity_or_chemistry_group_maps_to_the_current_outer_fold"
        ),
        "primary_statistical_unit": "outer_chemistry_fold_after_seed_average",
        "promotion_gate": {
            "mean_st_rae_gain_min": -0.05,
            "mean_mcc_gain_min": -0.03,
            "at_least_one_mean_primary_gain_strictly_positive": True,
            "worst_fold_st_rae_gain_min": -0.10,
            "worst_fold_mcc_gain_min": -0.10,
            "mean_brier_gain_min": -0.02,
            "test_or_leaderboard_access_max": 0,
        },
        "blind_test_can_override": False,
        "sealed_certification_can_override": False,
        "test_labels_loaded": False,
        "leaderboard_feedback_used": False,
    }
    payload["sha256"] = hashlib.sha256(
        json.dumps(payload, sort_keys=True).encode()
    ).hexdigest()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--implementation",
        type=Path,
        default=Path("scripts/run_openadmet_external_propensity_bridge.py"),
    )
    parser.add_argument(
        "--longform",
        type=Path,
        default=Path("reports/openadmet_cyp_training_longform_v1.csv"),
    )
    parser.add_argument(
        "--external-bridge",
        type=Path,
        default=Path("reports/openadmet_tdc_veith_external_bridge_v1.csv"),
    )
    parser.add_argument(
        "--baseline-reference",
        type=Path,
        default=Path("runs/openadmet-cyp-stage-b/b1_cv_result.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("reports/openadmet_external_propensity_protocol_v1.json"),
    )
    return parser.parse_args()


if __name__ == "__main__":
    print(json.dumps(build(parse_args()), indent=2, sort_keys=True))
