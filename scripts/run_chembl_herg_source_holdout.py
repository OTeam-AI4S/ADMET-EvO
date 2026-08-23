"""Run the frozen data-only hERG matched comparison on ChEMBL source holdout."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import average_precision_score, balanced_accuracy_score, roc_auc_score

from admet_agent.data_governance import MolecularDataCurator, canonical_smiles, frame_sha256
from admet_agent.external_herg import admission_decision, curate_chembl_herg
from admet_agent.features import MolecularFeaturizer
from admet_agent.guard import validate_tdc_discovery_path


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_jsonl_gzip(path: Path) -> list[dict]:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _metrics(y: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    return {
        "auroc": float(roc_auc_score(y, prediction)),
        "auprc": float(average_precision_score(y, prediction)),
        "balanced_accuracy": float(balanced_accuracy_score(y, prediction >= 0.5)),
    }


def paired_stratified_bootstrap_delta(
    y: np.ndarray,
    control: np.ndarray,
    treatment: np.ndarray,
    *,
    seed: int = 20260820,
    samples: int = 5000,
) -> dict[str, float]:
    positive = np.flatnonzero(y == 1)
    negative = np.flatnonzero(y == 0)
    if not len(positive) or not len(negative):
        raise ValueError("Both source-heldout classes are required")
    rng = np.random.default_rng(seed)
    deltas = np.empty(samples, dtype=float)
    for index in range(samples):
        selected = np.concatenate(
            [
                rng.choice(positive, size=len(positive), replace=True),
                rng.choice(negative, size=len(negative), replace=True),
            ]
        )
        deltas[index] = roc_auc_score(y[selected], treatment[selected]) - roc_auc_score(
            y[selected], control[selected]
        )
    return {
        "mean": float(deltas.mean()),
        "lower_95": float(np.quantile(deltas, 0.025)),
        "upper_95": float(np.quantile(deltas, 0.975)),
        "bootstrap_samples": samples,
    }


def run(
    root: Path,
    protocol_path: Path,
    raw_path: Path,
    acquisition_path: Path,
    output_dir: Path,
) -> dict:
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    if protocol["status"] != "frozen_before_acquisition":
        raise ValueError("External protocol is not frozen")
    if protocol["firewall"]["frozen_recipe_mutation_allowed"] is not False:
        raise ValueError("External ticket must not mutate the TDC recipe")
    for relative, expected in protocol["source_hashes"].items():
        if _sha(root / relative) != expected:
            raise ValueError(f"Protocol-bound source changed: {relative}")
    acquisition = json.loads(acquisition_path.read_text(encoding="utf-8"))
    if _sha(raw_path) != acquisition["raw_gzip_sha256"]:
        raise ValueError("Raw ChEMBL checksum differs from acquisition manifest")

    tdc_path = validate_tdc_discovery_path(root / "data/admet_group/herg/train_val.csv")
    tdc_raw = pd.read_csv(tdc_path)
    tdc_curated, tdc_audit = MolecularDataCurator().curate(
        tdc_raw, task_type="classification", policy="quality_filtered"
    )
    tdc_smiles = {value for value in tdc_raw["Drug"].map(canonical_smiles) if value is not None}
    external_train, source_heldout, external_audit = curate_chembl_herg(
        _load_jsonl_gzip(raw_path), tdc_train_smiles=tdc_smiles
    )
    admission = admission_decision(external_audit)

    output_dir.mkdir(parents=True, exist_ok=True)
    external_train_path = output_dir / "chembl_source_train.csv"
    heldout_path = output_dir / "chembl_source_heldout.csv"
    external_train.to_csv(external_train_path, index=False)
    source_heldout.to_csv(heldout_path, index=False)
    (output_dir / "LICENSE_EXTERNAL_DATA.md").write_text(
        "ChEMBL-derived tables in this directory are distributed under CC BY-SA 3.0.\n\n"
        "Source: ChEMBL current data API, acquired 2026-08-20. "
        "https://chembl.gitbook.io/chembl-interface-documentation/about\n",
        encoding="utf-8",
    )
    base = {
        "protocol": protocol["protocol"],
        "protocol_sha256": _sha(protocol_path),
        "acquisition": acquisition,
        "tdc_train": {
            "input_rows": len(tdc_raw),
            "curated_rows": len(tdc_curated),
            "audit": tdc_audit.to_dict(),
            "curated_sha256": frame_sha256(tdc_curated),
        },
        "external_audit": external_audit,
        "admission": admission,
        "curated_artifacts": {
            "source_train": str(external_train_path),
            "source_train_sha256": _sha(external_train_path),
            "source_heldout": str(heldout_path),
            "source_heldout_sha256": _sha(heldout_path),
        },
        "firewall": {
            "known_incident": protocol["firewall"]["known_incident"],
            "test_accessed_in_session": True,
            "test_feedback_used": False,
            "tdc_test_read_by_runner": False,
            "frozen_recipe_mutated": False,
        },
    }
    if not admission["admitted_for_matched_external_evaluation"]:
        return {**base, "status": "quarantined_by_preregistered_admission_gate", "fits": 0}

    all_smiles = (
        tdc_curated["Drug"].tolist()
        + external_train["Drug"].tolist()
        + source_heldout["Drug"].tolist()
    )
    features = MolecularFeaturizer(("morgan", "rdkit2d")).transform(all_smiles)
    n_tdc = len(tdc_curated)
    n_external = len(external_train)
    x_tdc = features[:n_tdc]
    x_external = features[n_tdc : n_tdc + n_external]
    x_heldout = features[n_tdc + n_external :]
    y_tdc = tdc_curated["Y"].to_numpy(dtype=int)
    y_external = external_train["Y"].to_numpy(dtype=int)
    y_heldout = source_heldout["Y"].to_numpy(dtype=int)
    seeds = tuple(protocol["matched_evaluation"]["seeds"])
    per_seed = []
    control_predictions = []
    treatment_predictions = []
    for seed in seeds:
        parameters = {
            "n_estimators": 500,
            "min_samples_leaf": 2,
            "max_features": "sqrt",
            "class_weight": "balanced",
            "n_jobs": 8,
            "random_state": int(seed),
        }
        control = ExtraTreesClassifier(**parameters).fit(x_tdc, y_tdc)
        treatment = ExtraTreesClassifier(**parameters).fit(
            np.vstack([x_tdc, x_external]), np.concatenate([y_tdc, y_external])
        )
        control_prediction = control.predict_proba(x_heldout)[:, 1]
        treatment_prediction = treatment.predict_proba(x_heldout)[:, 1]
        control_predictions.append(control_prediction)
        treatment_predictions.append(treatment_prediction)
        control_metrics = _metrics(y_heldout, control_prediction)
        treatment_metrics = _metrics(y_heldout, treatment_prediction)
        per_seed.append(
            {
                "seed": seed,
                "control": control_metrics,
                "treatment": treatment_metrics,
                "delta": {key: treatment_metrics[key] - control_metrics[key] for key in control_metrics},
            }
        )
    control_mean_prediction = np.mean(control_predictions, axis=0)
    treatment_mean_prediction = np.mean(treatment_predictions, axis=0)
    bootstrap = paired_stratified_bootstrap_delta(
        y_heldout, control_mean_prediction, treatment_mean_prediction
    )
    mean_delta = float(np.mean([row["delta"]["auroc"] for row in per_seed]))
    gain_supported = bootstrap["lower_95"] > 0 and mean_delta >= 0.01
    predictions = source_heldout[["Drug", "Y", "document_chembl_ids"]].copy()
    predictions["control_prediction"] = control_mean_prediction
    predictions["treatment_prediction"] = treatment_mean_prediction
    predictions_path = output_dir / "source_heldout_predictions.csv"
    predictions.to_csv(predictions_path, index=False)
    return {
        **base,
        "status": "matched_external_evaluation_complete",
        "fits": 2 * len(seeds),
        "only_changed_axis": "external_training_data",
        "per_seed": per_seed,
        "mean_auroc_delta": mean_delta,
        "paired_stratified_bootstrap": bootstrap,
        "gain_gate": {
            "mean_delta_at_least_0.01": mean_delta >= 0.01,
            "bootstrap_lower_95_above_zero": bootstrap["lower_95"] > 0,
            "external_data_gain_supported": gain_supported,
        },
        "predictions_path": str(predictions_path),
        "predictions_sha256": _sha(predictions_path),
        "claim_boundary": "External source-heldout auxiliary hERG evidence only; does not change or certify TDC test performance.",
    }


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, default=root / "runs/chembl-herg-source-heldout-v2/protocol.json")
    parser.add_argument("--raw", type=Path, default=root / "data/external/chembl/herg_ic50.raw.jsonl.gz")
    parser.add_argument("--acquisition", type=Path, default=root / "data/external/chembl/herg_ic50.acquisition.json")
    parser.add_argument("--output-dir", type=Path, default=root / "runs/chembl-herg-source-heldout-v2")
    args = parser.parse_args()
    result = run(root, args.protocol, args.raw, args.acquisition, args.output_dir)
    (args.output_dir / "result.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
