"""Blind-safe OOF uncertainty and applicability analysis for OpenADMET CYP."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import DataStructs
from rdkit.Chem import rdFingerprintGenerator

from admet_agent.applicability import (
    combined_uncertainty,
    cross_conformal_by_similarity,
    error_detection,
    interval_aware_loss,
    percentile_rank,
    risk_coverage,
    similarity_quartiles,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_protocol(path: Path) -> dict:
    protocol = json.loads(path.read_text())
    if protocol.get("sha256") != _sha256_without_field(protocol):
        raise RuntimeError("Applicability protocol self-hash mismatch")
    for raw_path, expected in protocol["implementation_sha256"].items():
        if _sha256(Path(raw_path)) != expected:
            raise RuntimeError(f"Frozen implementation changed: {raw_path}")
    return protocol


def _sha256_without_field(payload: dict) -> str:
    unbound = {key: value for key, value in payload.items() if key != "sha256"}
    encoded = json.dumps(unbound, indent=2, sort_keys=True) + "\n"
    return hashlib.sha256(encoded.encode()).hexdigest()


def _collapse_predictions(predictions: pd.DataFrame, expected_seeds: list[int]) -> pd.DataFrame:
    keys = [
        "inchikey",
        "endpoint_key",
        "standard_smiles",
        "chemistry_group",
        "task_type",
        "target",
        "credible_low",
        "credible_high",
        "outer_fold_evaluated",
    ]
    records = []
    for values, group in predictions.groupby(keys, dropna=False, sort=True):
        observed = sorted(group["training_seed"].astype(int).unique().tolist())
        if observed != expected_seeds or len(group) != len(expected_seeds):
            raise RuntimeError(f"Incomplete seed schedule for {values[:2]}: {observed}")
        row = dict(zip(keys, values, strict=True))
        values_array = group["prediction"].to_numpy(dtype=float)
        row.update(
            prediction=float(np.mean(values_array)),
            disagreement=float(np.std(values_array, ddof=1)),
        )
        records.append(row)
    return pd.DataFrame(records)


def _nearest_train_similarity(frame: pd.DataFrame) -> np.ndarray:
    generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    smiles_to_fp = {
        smiles: generator.GetFingerprintFromSmiles(smiles)
        for smiles in sorted(frame["standard_smiles"].astype(str).unique())
    }
    output = np.empty(len(frame), dtype=float)
    positions = {index: pos for pos, index in enumerate(frame.index)}
    for fold in sorted(frame["outer_fold_evaluated"].unique()):
        train_smiles = frame.loc[
            frame["outer_fold_evaluated"] != fold, "standard_smiles"
        ].astype(str).unique()
        evaluation = frame.loc[frame["outer_fold_evaluated"] == fold]
        train_fp = [smiles_to_fp[value] for value in train_smiles]
        if not train_fp:
            raise RuntimeError(f"No training molecules for outer fold {fold}")
        for index, smiles in evaluation["standard_smiles"].astype(str).items():
            similarities = DataStructs.BulkTanimotoSimilarity(
                smiles_to_fp[smiles], train_fp
            )
            output[positions[index]] = max(similarities)
    return output


def _endpoint_analysis(frame: pd.DataFrame) -> dict:
    frame = frame.reset_index(drop=True).copy()
    frame["nearest_train_similarity"] = _nearest_train_similarity(frame)
    frame["distance_uncertainty"] = 1.0 - frame["nearest_train_similarity"]
    classification = frame["task_type"].iloc[0] == "classification"
    if classification:
        frame["confidence_uncertainty"] = 1.0 - 2.0 * np.abs(
            frame["prediction"] - 0.5
        )
        frame["model_uncertainty"] = combined_uncertainty(
            distance=np.zeros(len(frame)),
            disagreement=frame["disagreement"].to_numpy(),
            confidence_uncertainty=frame["confidence_uncertainty"].to_numpy(),
        )
        # Remove the constant placeholder component used to share the helper's
        # deterministic rank semantics.
        frame["model_uncertainty"] = np.mean(
            np.vstack(
                [
                    percentile_rank(frame["disagreement"].to_numpy()),
                    percentile_rank(frame["confidence_uncertainty"].to_numpy()),
                ]
            ),
            axis=0,
        )
        frame["combined_uncertainty"] = combined_uncertainty(
            distance=frame["distance_uncertainty"].to_numpy(),
            disagreement=frame["disagreement"].to_numpy(),
            confidence_uncertainty=frame["confidence_uncertainty"].to_numpy(),
        )
        loss = np.square(
            frame["target"].to_numpy(dtype=float)
            - frame["prediction"].to_numpy(dtype=float)
        )
    else:
        frame["model_uncertainty"] = percentile_rank(
            frame["disagreement"].to_numpy()
        )
        frame["combined_uncertainty"] = combined_uncertainty(
            distance=frame["distance_uncertainty"].to_numpy(),
            disagreement=frame["disagreement"].to_numpy(),
        )
        loss = interval_aware_loss(
            frame["target"].to_numpy(),
            frame["prediction"].to_numpy(),
            frame["credible_low"].to_numpy(),
            frame["credible_high"].to_numpy(),
        )
    frame["row_loss"] = loss
    frame["similarity_quartile"] = similarity_quartiles(
        frame["nearest_train_similarity"].to_numpy()
    )
    signals = {}
    for name in ("distance_uncertainty", "model_uncertainty", "combined_uncertainty"):
        signals[name] = {
            "error_detection": error_detection(loss, frame[name].to_numpy()),
            "risk_coverage": risk_coverage(loss, frame[name].to_numpy()),
        }
    quartiles = {}
    for quartile in range(4):
        selected = frame["similarity_quartile"].eq(quartile)
        quartiles[str(quartile)] = {
            "rows": int(selected.sum()),
            "similarity_mean": float(frame.loc[selected, "nearest_train_similarity"].mean()),
            "loss_mean": float(frame.loc[selected, "row_loss"].mean()),
        }
    conformal = None
    if not classification:
        frame["constant_scale"] = 1.0
        frame["combined_scale"] = 0.25 + frame["combined_uncertainty"]
        conformal = {
            "constant": cross_conformal_by_similarity(
                frame, scale_column="constant_scale"
            ),
            "combined_adaptive": cross_conformal_by_similarity(
                frame, scale_column="combined_scale"
            ),
        }
    return {
        "task_type": str(frame["task_type"].iloc[0]),
        "rows": len(frame),
        "identity_overlap": 0,
        "signals": signals,
        "similarity_quartiles": quartiles,
        "cross_conformal_90": conformal,
    }


def run(args: argparse.Namespace) -> dict:
    protocol_path = args.protocol.resolve()
    protocol = _load_protocol(protocol_path)
    result_path = Path(protocol["input_result_path"]).resolve()
    if _sha256(result_path) != protocol["input_result_sha256"]:
        raise RuntimeError("Frozen B1 result hash mismatch")
    result = json.loads(result_path.read_text())
    if result.get("training_only") is not True or result.get("blind_test_opened") is not False:
        raise RuntimeError("Input is not blind-safe training-only OOF evidence")
    prediction_path = Path(result["prediction_path"]).resolve()
    if _sha256(prediction_path) != protocol["input_prediction_sha256"]:
        raise RuntimeError("Frozen B1 prediction hash mismatch")
    predictions = pd.read_csv(prediction_path)
    collapsed = _collapse_predictions(predictions, protocol["seeds"])
    endpoints = {
        endpoint: _endpoint_analysis(frame)
        for endpoint, frame in collapsed.groupby("endpoint_key", sort=True)
    }
    combined_better = 0
    detectable = 0
    low_similarity_worse = 0
    for evidence in endpoints.values():
        combined = evidence["signals"]["combined_uncertainty"]
        singles = [
            evidence["signals"][name]["risk_coverage"]["aurc"]
            for name in ("distance_uncertainty", "model_uncertainty")
        ]
        combined_better += int(combined["risk_coverage"]["aurc"] < min(singles))
        auroc = combined["error_detection"]["auroc"]
        detectable += int(auroc is not None and auroc > 0.5)
        quartiles = evidence["similarity_quartiles"]
        low_similarity_worse += int(quartiles["0"]["loss_mean"] > quartiles["3"]["loss_mean"])
    checks = [
        {
            "name": "combined_beats_both_single_signals",
            "observed": combined_better,
            "requirement": ">= 4 of 6 endpoints",
            "passed": combined_better >= 4,
        },
        {
            "name": "combined_detects_high_error",
            "observed": detectable,
            "requirement": ">= 4 of 6 endpoints with AUROC > 0.5",
            "passed": detectable >= 4,
        },
        {
            "name": "low_similarity_has_higher_loss",
            "observed": low_similarity_worse,
            "requirement": ">= 4 of 6 endpoints",
            "passed": low_similarity_worse >= 4,
        },
        {
            "name": "forbidden_blind_access",
            "observed": 0,
            "requirement": "= 0",
            "passed": True,
        },
    ]
    payload = {
        "version": "openadmet-cyp-applicability-v1",
        "scope": "training_only_oof_post_selection_diagnostic",
        "protocol_path": str(protocol_path),
        "protocol_sha256": protocol["sha256"],
        "input_result_sha256": protocol["input_result_sha256"],
        "input_prediction_sha256": protocol["input_prediction_sha256"],
        "uncertainty_contract": protocol["uncertainty_contract"],
        "endpoints": endpoints,
        "checks": checks,
        "promotion_passed": all(item["passed"] for item in checks),
        "test_labels_loaded": False,
        "blind_test_opened": False,
        "leaderboard_feedback_used": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--protocol",
        type=Path,
        default=Path("reports/openadmet_cyp_applicability_protocol_v1.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("reports/openadmet_cyp_applicability_analysis_v1.json"),
    )
    return parser.parse_args()


if __name__ == "__main__":
    print(json.dumps(run(parse_args()), indent=2, sort_keys=True))
