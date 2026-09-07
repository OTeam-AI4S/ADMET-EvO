"""Evaluate frozen recipe families on the six official Fang 2023 ADME holdouts."""

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


TASKS = (
    ("fang::HLM_CLint", "ADME_HLM"),
    ("fang::RLM_CLint", "ADME_RLM"),
    ("fang::MDR1_MDCK_ER", "ADME_MDR1_ER"),
    ("fang::Solubility", "ADME_Sol"),
    ("fang::human_PPB", "ADME_hPPB"),
    ("fang::rat_PPB", "ADME_rPPB"),
)
SEEDS = (17, 29, 43)
ACTIONS = ("baseline", "feature_rich", "boosting_anchor", "automl_expanded", "gpu_catboost")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _prepare(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for raw_smiles, target in frame[["smiles", "activity"]].itertuples(index=False, name=None):
        molecule = _standard_molecule(str(raw_smiles))
        if molecule is None or not np.isfinite(float(target)):
            continue
        rows.append({
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
    if not 0 <= index < len(TASKS) * len(SEEDS):
        raise ValueError(f"Array index {index} is out of range")
    endpoint_id, stem = TASKS[index // len(SEEDS)]
    seed = SEEDS[index % len(SEEDS)]
    train_path = args.data_root / f"{stem}_train.csv"
    test_path = args.data_root / f"{stem}_test.csv"
    train = _prepare(pd.read_csv(train_path))
    test = _prepare(pd.read_csv(test_path))
    exact_overlap = set(train["inchikey"]) & set(test["inchikey"])
    overlap_rows_removed = int(train["inchikey"].isin(exact_overlap).sum())
    train = train.loc[~train["inchikey"].isin(exact_overlap)].copy()
    group_overlap = set(train["chemistry_group"]) & set(test["chemistry_group"])
    if set(train["inchikey"]) & set(test["inchikey"]):
        raise RuntimeError("Identity firewall failed to remove official split overlap")

    combined = pd.concat([
        train[["inchikey", "standard_smiles"]],
        test[["inchikey", "standard_smiles"]],
    ], ignore_index=True)
    combined = combined.drop_duplicates("inchikey").reset_index(drop=True)
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
        representation = (
            "feature_rich"
            if action in {"feature_rich", "boosting_anchor", "automl_expanded", "gpu_catboost"}
            else "baseline"
        )
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
                "endpoint_id": endpoint_id,
                "source_id": "fang_jcim_2023",
                "seed": seed,
                "train_rows": len(train),
                "test_rows": len(test),
                "train_groups": int(train["chemistry_group"].nunique()),
                "test_groups": int(test["chemistry_group"].nunique()),
                "test_groups_seen_in_train": len(group_overlap),
                **result,
            })
        except Exception as error:
            failures.append({
                "endpoint_id": endpoint_id,
                "seed": seed,
                "action": action,
                "error_type": type(error).__name__,
                "error": str(error),
            })
    output_path = args.output_dir / f"{stem}__seed{seed}.csv"
    args.output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output_path, index=False)
    manifest = {
        "version": "biogen-fang-fixed-holdout-v1",
        "endpoint_id": endpoint_id,
        "seed": seed,
        "train_path": str(train_path.resolve()),
        "train_sha256": _sha256(train_path),
        "test_path": str(test_path.resolve()),
        "test_sha256": _sha256(test_path),
        "train_rows": len(train),
        "test_rows": len(test),
        "exact_identity_overlap_detected": len(exact_overlap),
        "train_overlap_rows_removed": overlap_rows_removed,
        "exact_identity_overlap_after_filter": 0,
        "test_groups_seen_in_train": len(group_overlap),
        "actions": list(ACTIONS),
        "primary_recipe": "boosting_anchor",
        "primary_recipe_provenance": (
            "fixed from the independent v5 new-ADMET confirmation before Fang scores"
        ),
        "automl_selection": "group-disjoint inner CV on official training rows only",
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
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("external/computational_adme/MPNN"),
    )
    parser.add_argument("--array-index", type=int)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--trees", type=int, default=200)
    parser.add_argument("--inner-folds", type=int, default=5)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
