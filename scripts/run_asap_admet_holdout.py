"""Evaluate frozen recipe families on the official ASAP 2025 temporal holdout."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

from admet_agent.external_structure_firewall import _standard_molecule
from admet_agent.features import MolecularFeaturizer
from admet_agent.public_action_matrix import evaluate_action
from scripts.build_openadmet_cyp_split_manifest import _chemistry_group
from scripts.run_new_admet_endpoint import _evaluate_gpu_catboost


ENDPOINTS = ("HLM", "MLM", "KSOL", "LogD", "MDR1-MDCKII")
SEEDS = (17, 29, 43)
ACTIONS = ("baseline", "feature_rich", "boosting_anchor", "automl_expanded", "gpu_catboost")
FLAGGED_GLOBAL_INDICES = {
    "HLM": {519, 524, 547, 553},
    "MLM": {515, 518, 521, 524, 525},
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _official_target(values: pd.Series, endpoint: str) -> np.ndarray:
    numeric = pd.to_numeric(values, errors="coerce").to_numpy(dtype=float)
    if endpoint == "LogD":
        return numeric
    return np.log10(np.clip(numeric, a_min=0.0, a_max=None) + 1.0)


def _prepare(frame: pd.DataFrame, endpoint: str) -> pd.DataFrame:
    rows = []
    transformed = _official_target(frame[endpoint], endpoint)
    for position, (source_index, raw_smiles) in enumerate(
        frame[["source_index", "CXSMILES"]].itertuples(index=False, name=None)
    ):
        target = transformed[position]
        if not np.isfinite(target):
            continue
        molecule = _standard_molecule(str(raw_smiles))
        if molecule is None:
            continue
        rows.append({
            "source_index": int(source_index),
            "standard_smiles": molecule.smiles,
            "inchikey": molecule.inchikey,
            "chemistry_group": _chemistry_group(molecule.smiles),
            "target": float(target),
        })
    output = pd.DataFrame(rows)
    if output["inchikey"].duplicated().any():
        output = (
            output.groupby("inchikey", as_index=False)
            .agg(
                source_index=("source_index", "first"),
                standard_smiles=("standard_smiles", "first"),
                chemistry_group=("chemistry_group", "first"),
                target=("target", "mean"),
            )
        )
    return output


def run(args: argparse.Namespace) -> dict:
    index = args.array_index
    if index is None:
        raw = os.environ.get("SLURM_ARRAY_TASK_ID")
        if raw is None:
            raise ValueError("--array-index or SLURM_ARRAY_TASK_ID is required")
        index = int(raw)
    if not 0 <= index < len(ENDPOINTS) * len(SEEDS):
        raise ValueError(f"Array index {index} is out of range")
    endpoint = ENDPOINTS[index // len(SEEDS)]
    seed = SEEDS[index % len(SEEDS)]

    data_path = args.data.resolve()
    raw = pd.read_csv(data_path).reset_index(names="source_index")
    train_raw = raw.loc[raw["Set"].astype(str).eq("Train")].copy()
    test_raw = raw.loc[raw["Set"].astype(str).eq("Test")].copy()
    flagged = FLAGGED_GLOBAL_INDICES.get(endpoint, set())
    flagged_present = sorted(set(test_raw["source_index"].astype(int)) & flagged)
    test_raw = test_raw.loc[~test_raw["source_index"].isin(flagged)].copy()

    train = _prepare(train_raw, endpoint)
    test = _prepare(test_raw, endpoint)
    exact_overlap = set(train["inchikey"]) & set(test["inchikey"])
    overlap_rows_removed = int(train["inchikey"].isin(exact_overlap).sum())
    train = train.loc[~train["inchikey"].isin(exact_overlap)].copy()
    if set(train["inchikey"]) & set(test["inchikey"]):
        raise RuntimeError("Identity firewall failed to remove temporal-split overlap")
    group_overlap = set(train["chemistry_group"]) & set(test["chemistry_group"])

    combined = pd.concat([
        train[["inchikey", "standard_smiles"]],
        test[["inchikey", "standard_smiles"]],
    ], ignore_index=True).drop_duplicates("inchikey").reset_index(drop=True)
    identity_index = {value: idx for idx, value in enumerate(combined["inchikey"])}
    smiles = combined["standard_smiles"].astype(str).tolist()
    features = {
        "baseline": MolecularFeaturizer(("morgan",)).transform(smiles),
        "feature_rich": MolecularFeaturizer(
            ("morgan", "avalon", "erg", "rdkit2d")
        ).transform(smiles),
    }
    train_index = np.asarray([identity_index[value] for value in train["inchikey"]])
    test_index = np.asarray([identity_index[value] for value in test["inchikey"]])

    rows = []
    failures = []
    for action in ACTIONS:
        representation = "baseline" if action == "baseline" else "feature_rich"
        try:
            if action == "gpu_catboost":
                result = _evaluate_gpu_catboost(
                    features[representation][train_index],
                    train["target"].to_numpy(),
                    features[representation][test_index],
                    test["target"].to_numpy(),
                    task_type="regression",
                    seed=seed,
                    workers=args.workers,
                )
            else:
                result = evaluate_action(
                    features[representation][train_index],
                    train["target"].to_numpy(),
                    train["chemistry_group"].astype(str).to_numpy(),
                    features[representation][test_index],
                    test["target"].to_numpy(),
                    action=action,
                    task_type="regression",
                    seed=seed,
                    trees=args.trees,
                    workers=args.workers,
                    inner_splits=args.inner_folds,
                )
            rows.append({
                "endpoint_id": f"asap2025::{endpoint}",
                "source_id": "asap_polaris_openadmet_2025",
                "seed": seed,
                "train_rows": len(train),
                "test_rows": len(test),
                "test_groups_seen_in_train": len(group_overlap),
                **result,
            })
        except Exception as error:
            failures.append({
                "endpoint_id": endpoint,
                "seed": seed,
                "action": action,
                "error_type": type(error).__name__,
                "error": str(error),
            })

    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = args.output_dir / f"{endpoint}__seed{seed}.csv"
    pd.DataFrame(rows).to_csv(output_path, index=False)
    manifest = {
        "version": "asap-admet-official-temporal-holdout-v1",
        "endpoint_id": endpoint,
        "seed": seed,
        "data_path": str(data_path),
        "data_sha256": _sha256(data_path),
        "train_rows": len(train),
        "test_rows": len(test),
        "official_transform": "identity" if endpoint == "LogD" else "log10(max(y, 0) + 1)",
        "flagged_global_indices_removed": flagged_present,
        "exact_identity_overlap_detected": len(exact_overlap),
        "train_overlap_rows_removed": overlap_rows_removed,
        "exact_identity_overlap_after_filter": 0,
        "test_groups_seen_in_train": len(group_overlap),
        "actions": list(ACTIONS),
        "primary_recipe": "boosting_anchor",
        "primary_recipe_provenance": (
            "fixed from the independent v5 new-ADMET confirmation before ASAP scores"
        ),
        "automl_selection": "group-disjoint inner CV on temporal training rows only",
        "output_path": str(output_path.resolve()),
        "output_sha256": _sha256(output_path),
        "failures": failures,
        "integrity_passed": len(rows) == len(ACTIONS) and not failures,
        "test_labels_used_for_training_or_model_selection": False,
        "leaderboard_feedback_used": False,
    }
    output_path.with_suffix(".manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(manifest, sort_keys=True))
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path("external/asap_admet_2025/ADMET.csv"))
    parser.add_argument("--array-index", type=int)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--trees", type=int, default=200)
    parser.add_argument("--inner-folds", type=int, default=5)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
