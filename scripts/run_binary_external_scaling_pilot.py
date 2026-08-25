"""Run a reusable discovery-only external-data scaling pilot.

The runner is intentionally limited to binary TDC tasks.  It never opens a
TDC test file, removes exact discovery overlap, and for every validation fold
keeps external structures only when max Morgan Tanimoto is strictly < 0.80.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from rdkit import Chem
from scipy import sparse
from sklearn.metrics import roc_auc_score
from tdc.benchmark_group import admet_group

from admet_agent.herg_data_scaling import (
    curate_binary_structures,
    fingerprints_to_csr,
    morgan_fingerprints,
    near_neighbor_keep_mask,
    standardize_smiles,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _model(seed: int, n_jobs: int) -> lgb.LGBMClassifier:
    return lgb.LGBMClassifier(
        objective="binary",
        n_estimators=350,
        learning_rate=0.05,
        num_leaves=31,
        min_child_samples=20,
        subsample=0.9,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        class_weight="balanced",
        random_state=seed,
        n_jobs=n_jobs,
        verbosity=-1,
    )


def _standardized(frame: pd.DataFrame) -> pd.DataFrame:
    structures = standardize_smiles(frame["Drug"].astype(str).tolist())
    if len(structures) != len(frame):
        raise RuntimeError("TDC discovery split contains invalid structures")
    structures["Y"] = frame["Y"].astype(int).to_numpy()
    return structures


def _score(train_x, train_y, valid_x, valid_y, seed, n_jobs) -> float:
    prediction = _model(seed, n_jobs).fit(train_x, train_y).predict_proba(valid_x)[:, 1]
    return float(roc_auc_score(valid_y, prediction))


def run(args: argparse.Namespace) -> dict:
    source = args.source.resolve()
    source_hash = _sha256(source)
    if args.expected_source_sha256 and source_hash != args.expected_source_sha256:
        raise RuntimeError("Frozen external source SHA-256 mismatch")
    raw = pd.read_csv(source, sep=args.separator)
    external, curation = curate_binary_structures(
        raw,
        smiles_column=args.smiles_column,
        label_column=args.label_column,
        id_column=args.id_column,
        majority_cap=args.majority_cap,
    )

    discovery_path = args.data_dir / "admet_group" / args.benchmark.lower() / "train_val.csv"
    discovery = pd.read_csv(discovery_path, usecols=["Drug", "Y"])
    discovery_ids = set(
        standardize_smiles(discovery["Drug"].astype(str).tolist())["standard_inchikey"]
    )
    exact_mask = external["standard_inchikey"].isin(discovery_ids)
    exact_exclusions = int(exact_mask.sum())
    external = external.loc[~exact_mask].reset_index(drop=True)
    fingerprintable = external["standard_smiles"].map(
        lambda value: Chem.MolFromSmiles(str(value)) is not None
    )
    fingerprint_parse_exclusions = int((~fingerprintable).sum())
    external = external.loc[fingerprintable].reset_index(drop=True)
    if external.empty:
        raise RuntimeError("No external identities survived exact-overlap firewall")
    external_fps = morgan_fingerprints(external["standard_smiles"].tolist())
    external_x = fingerprints_to_csr(external_fps)
    external_y = external["Y"].to_numpy(dtype=int)

    group = admet_group(path=str(args.data_dir))
    per_seed = []
    for seed in args.seed:
        train_raw, valid_raw = group.get_train_valid_split(
            benchmark=args.benchmark, split_type="default", seed=seed
        )
        train, valid = _standardized(train_raw), _standardized(valid_raw)
        train_x = fingerprints_to_csr(morgan_fingerprints(train["standard_smiles"].tolist()))
        valid_fps = morgan_fingerprints(valid["standard_smiles"].tolist())
        valid_x = fingerprints_to_csr(valid_fps)
        train_y = train["Y"].to_numpy(dtype=int)
        valid_y = valid["Y"].to_numpy(dtype=int)
        keep = near_neighbor_keep_mask(
            external_fps, valid_fps, threshold=args.near_threshold
        )
        baseline = _score(train_x, train_y, valid_x, valid_y, seed, args.n_jobs)
        augmented = _score(
            sparse.vstack([train_x, external_x[keep]]),
            np.concatenate([train_y, external_y[keep]]),
            valid_x,
            valid_y,
            seed,
            args.n_jobs,
        )
        per_seed.append(
            {
                "seed": seed,
                "tdc_train_rows": len(train),
                "tdc_validation_rows": len(valid),
                "external_firewalled_rows": int(keep.sum()),
                "near_neighbor_exclusions": int((~keep).sum()),
                "baseline_auroc": baseline,
                "augmented_auroc": augmented,
                "delta": augmented - baseline,
            }
        )

    def summarize(key: str) -> dict:
        values = np.asarray([row[key] for row in per_seed], dtype=float)
        return {
            "mean": float(values.mean()),
            "std": float(values.std()),
            "positive_seed_fraction": float((values > 0).mean()),
        }

    delta = summarize("delta")
    status = (
        "supported"
        if delta["mean"] >= 0.005 and delta["positive_seed_fraction"] >= 2 / 3
        else "refuted" if delta["mean"] <= -0.005 else "inconclusive"
    )
    result = {
        "protocol": "tdc-binary-task-conditioned-external-scaling-v1",
        "task": args.task,
        "benchmark": args.benchmark,
        "status": status,
        "source": {
            "name": args.source_name,
            "path": str(source),
            "sha256": source_hash,
            "raw_rows": len(raw),
            "lineage_status": args.lineage_status,
        },
        "curation": {
            **curation.to_dict(),
            "exact_tdc_discovery_overlap_exclusions": exact_exclusions,
            "post_standardization_fingerprint_parse_exclusions": fingerprint_parse_exclusions,
            "post_exact_firewall_identities": len(external),
            "near_neighbor_rule": f"max Morgan Tanimoto < {args.near_threshold:.2f}",
        },
        "per_seed": per_seed,
        "summary": {
            "baseline_auroc": summarize("baseline_auroc"),
            "augmented_auroc": summarize("augmented_auroc"),
            "delta": delta,
        },
        "safety": {
            "tdc_test_loaded": False,
            "leaderboard_feedback_used": False,
            "test_labels_accessible": False,
        },
        "claim_boundary": "Discovery validation only. A paper-level independent-data claim additionally requires source-lineage clearance and evaluator-owned test confirmation.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", required=True)
    parser.add_argument("--benchmark", required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--source-name", required=True)
    parser.add_argument("--smiles-column", default="Drug")
    parser.add_argument("--label-column", default="Y")
    parser.add_argument("--id-column", default="Drug_ID")
    parser.add_argument("--separator", default="\t")
    parser.add_argument("--expected-source-sha256")
    parser.add_argument("--lineage-status", default="pending_manual_clearance")
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--majority-cap", type=int, default=30_000)
    parser.add_argument("--near-threshold", type=float, default=0.80)
    parser.add_argument("--seed", type=int, action="append", default=None)
    parser.add_argument("--n-jobs", type=int, default=8)
    args = parser.parse_args()
    args.seed = tuple(dict.fromkeys(args.seed or (1, 2, 3)))
    if args.near_threshold != 0.80:
        raise ValueError("Paper protocol freezes the strict Tanimoto threshold at 0.80")
    return args


if __name__ == "__main__":
    payload = run(parse_args())
    print(json.dumps({"status": payload["status"], **payload["summary"]}, indent=2))
