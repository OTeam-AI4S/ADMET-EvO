"""Run a discovery-only external-data scaling pilot for regression tasks.

External labels must already be converted to the exact TDC endpoint scale by
an endpoint-specific curator.  This runner standardizes structures, collapses
replicates by median, removes exact overlap, and keeps only candidates whose
maximum Morgan Tanimoto to each validation fold is strictly below 0.80.
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
from scipy.stats import spearmanr
from sklearn.metrics import mean_absolute_error
from tdc.benchmark_group import admet_group

from admet_agent.external_structure_firewall import _standard_molecule
from admet_agent.herg_data_scaling import (
    fingerprints_to_csr,
    morgan_fingerprints,
    near_neighbor_keep_mask,
    standardize_smiles,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _model(seed: int, n_jobs: int) -> lgb.LGBMRegressor:
    return lgb.LGBMRegressor(
        objective="regression_l1",
        n_estimators=350,
        learning_rate=0.05,
        num_leaves=31,
        min_child_samples=20,
        subsample=0.9,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        random_state=seed,
        n_jobs=n_jobs,
        verbosity=-1,
    )


def _curate(frame: pd.DataFrame, args: argparse.Namespace) -> tuple[pd.DataFrame, dict]:
    values = pd.to_numeric(frame[args.label_column], errors="coerce")
    work = frame.loc[values.notna(), [args.id_column, args.smiles_column]].copy()
    work["Y"] = values.loc[work.index].astype(float)
    rows, invalid = [], 0
    for record_id, smiles, target in work.itertuples(index=False, name=None):
        molecule = _standard_molecule(str(smiles))
        if molecule is None:
            invalid += 1
            continue
        rows.append(
            {
                "source_record_id": str(record_id),
                "standard_smiles": molecule.smiles,
                "standard_inchikey": molecule.inchikey,
                "Y": float(target),
            }
        )
    standardized = pd.DataFrame(rows)
    if standardized.empty:
        raise ValueError("No valid external structures")
    spread = standardized.groupby("standard_inchikey")["Y"].agg(lambda x: float(x.max()-x.min()))
    collapsed = (
        standardized.groupby("standard_inchikey", as_index=False)
        .agg(standard_smiles=("standard_smiles", "first"), Y=("Y", "median"), replicate_count=("Y", "size"))
        .reset_index(drop=True)
    )
    return collapsed, {
        "input_rows": len(frame),
        "numeric_label_rows": len(work),
        "invalid_structure_rows": invalid,
        "valid_structure_rows": len(standardized),
        "output_identities": len(collapsed),
        "replicate_rows": int((collapsed["replicate_count"] > 1).sum()),
        "median_within_identity_range": float(spread.median()),
    }


def _standardized_tdc(frame: pd.DataFrame) -> pd.DataFrame:
    structures = standardize_smiles(frame["Drug"].astype(str).tolist())
    if len(structures) != len(frame):
        raise RuntimeError("TDC split contains invalid structures")
    structures["Y"] = pd.to_numeric(frame["Y"], errors="raise").to_numpy(dtype=float)
    return structures


def _score(y_true: np.ndarray, prediction: np.ndarray, metric: str) -> float:
    if metric == "mae":
        return float(mean_absolute_error(y_true, prediction))
    if metric == "spearman":
        return float(spearmanr(y_true, prediction).statistic)
    raise ValueError(f"Unsupported regression metric {metric!r}")


def run(args: argparse.Namespace) -> dict:
    source = args.source.resolve()
    source_hash = _sha256(source)
    if args.expected_source_sha256 and source_hash != args.expected_source_sha256:
        raise RuntimeError("Frozen external source SHA-256 mismatch")
    external, audit = _curate(pd.read_csv(source, sep=args.separator), args)
    discovery_path = args.data_dir / "admet_group" / args.benchmark.lower() / "train_val.csv"
    discovery = pd.read_csv(discovery_path, usecols=["Drug", "Y"])
    discovery_ids = set(standardize_smiles(discovery["Drug"].astype(str).tolist())["standard_inchikey"])
    exact = external["standard_inchikey"].isin(discovery_ids)
    audit["exact_tdc_discovery_overlap_exclusions"] = int(exact.sum())
    external = external.loc[~exact].reset_index(drop=True)
    fingerprintable = external["standard_smiles"].map(lambda x: Chem.MolFromSmiles(str(x)) is not None)
    audit["post_standardization_fingerprint_parse_exclusions"] = int((~fingerprintable).sum())
    external = external.loc[fingerprintable].reset_index(drop=True)
    audit["post_exact_firewall_identities"] = len(external)
    if len(external) < args.minimum_external_rows:
        raise RuntimeError(f"Only {len(external)} external identities survived; minimum is {args.minimum_external_rows}")
    external_fps = morgan_fingerprints(external["standard_smiles"].tolist())
    external_x = fingerprints_to_csr(external_fps)
    external_y = external["Y"].to_numpy(dtype=float)

    group = admet_group(path=str(args.data_dir))
    per_seed = []
    for seed in args.seed:
        train_raw, valid_raw = group.get_train_valid_split(benchmark=args.benchmark, split_type="default", seed=seed)
        train, valid = _standardized_tdc(train_raw), _standardized_tdc(valid_raw)
        train_x = fingerprints_to_csr(morgan_fingerprints(train["standard_smiles"].tolist()))
        valid_fps = morgan_fingerprints(valid["standard_smiles"].tolist())
        valid_x = fingerprints_to_csr(valid_fps)
        train_y, valid_y = train["Y"].to_numpy(), valid["Y"].to_numpy()
        keep = near_neighbor_keep_mask(external_fps, valid_fps, threshold=args.near_threshold)
        baseline_prediction = _model(seed, args.n_jobs).fit(train_x, train_y).predict(valid_x)
        augmented_prediction = _model(seed, args.n_jobs).fit(
            sparse.vstack([train_x, external_x[keep]]), np.concatenate([train_y, external_y[keep]])
        ).predict(valid_x)
        baseline = _score(valid_y, baseline_prediction, args.metric)
        augmented = _score(valid_y, augmented_prediction, args.metric)
        oriented_delta = baseline - augmented if args.metric == "mae" else augmented - baseline
        per_seed.append(
            {
                "seed": seed,
                "baseline_score": baseline,
                "augmented_score": augmented,
                "oriented_delta": oriented_delta,
                "external_firewalled_rows": int(keep.sum()),
                "near_neighbor_exclusions": int((~keep).sum()),
            }
        )

    deltas = np.asarray([row["oriented_delta"] for row in per_seed])
    status = "supported" if deltas.mean() >= 0.005 and (deltas > 0).mean() >= 2/3 else "refuted" if deltas.mean() <= -0.005 else "inconclusive"
    result = {
        "protocol": "tdc-regression-task-conditioned-external-scaling-v1",
        "task": args.task,
        "benchmark": args.benchmark,
        "metric": args.metric,
        "status": status,
        "source": {"name": args.source_name, "path": str(source), "sha256": source_hash, "lineage_status": args.lineage_status},
        "curation": {**audit, "near_neighbor_rule": f"max Morgan Tanimoto < {args.near_threshold:.2f}"},
        "per_seed": per_seed,
        "summary": {
            "baseline_mean": float(np.mean([row["baseline_score"] for row in per_seed])),
            "augmented_mean": float(np.mean([row["augmented_score"] for row in per_seed])),
            "oriented_delta_mean": float(deltas.mean()),
            "oriented_delta_std": float(deltas.std()),
            "positive_seed_fraction": float((deltas > 0).mean()),
        },
        "safety": {"tdc_test_loaded": False, "test_labels_accessible": False},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", required=True)
    parser.add_argument("--benchmark", required=True)
    parser.add_argument("--metric", choices=("mae", "spearman"), required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--source-name", required=True)
    parser.add_argument("--smiles-column", default="X")
    parser.add_argument("--label-column", default="Y")
    parser.add_argument("--id-column", default="ID")
    parser.add_argument("--separator", default="\t")
    parser.add_argument("--expected-source-sha256")
    parser.add_argument("--lineage-status", default="pending_manual_clearance")
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--minimum-external-rows", type=int, default=30)
    parser.add_argument("--near-threshold", type=float, default=0.80)
    parser.add_argument("--seed", type=int, action="append", default=None)
    parser.add_argument("--n-jobs", type=int, default=8)
    args = parser.parse_args()
    args.seed = tuple(dict.fromkeys(args.seed or (1, 2, 3)))
    if args.near_threshold != 0.80:
        raise ValueError("Paper protocol freezes the strict Tanimoto threshold at 0.80")
    return args


if __name__ == "__main__":
    print(json.dumps(run(parse_args()), indent=2))
