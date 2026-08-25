"""Run the minimal external-data scaling experiment for TDC hERG.

This is a discovery-only matched comparison.  It uses the frozen hERGCentral
cache as a bridge source, removes exact overlap with the complete TDC
train/validation discovery table, and optionally removes near neighbours of
each validation fold.  The TDC test table is never opened.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from tdc.benchmark_group import admet_group

from admet_agent.herg_data_scaling import (
    curate_binary_structures,
    fingerprints_to_csr,
    morgan_fingerprints,
    near_neighbor_keep_mask,
    standardize_smiles,
)

EXPECTED_SOURCE_SHA256 = (
    "5775a6900c3be6a2db65d357411fd3f768189de611dc2da3456dab4acbcb3ee8"
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


def _standardized_split(frame: pd.DataFrame) -> pd.DataFrame:
    structures = standardize_smiles(frame["Drug"].astype(str).tolist())
    if len(structures) != len(frame):
        raise RuntimeError("TDC discovery split contains invalid structures")
    output = structures.copy()
    output["Y"] = frame["Y"].astype(int).to_numpy()
    return output


def _fit_score(
    train_x,
    train_y: np.ndarray,
    valid_x,
    valid_y: np.ndarray,
    *,
    seed: int,
    n_jobs: int,
) -> float:
    prediction = _model(seed, n_jobs).fit(train_x, train_y).predict_proba(valid_x)[:, 1]
    return float(roc_auc_score(valid_y, prediction))


def run(args: argparse.Namespace) -> dict:
    source = args.source.resolve()
    if _sha256(source) != args.expected_source_sha256:
        raise RuntimeError("Frozen hERGCentral source SHA-256 mismatch")

    raw = pd.read_csv(source, sep="\t", usecols=["ID", "X", "hERG_inhib"])
    external, curation = curate_binary_structures(
        raw,
        smiles_column="X",
        label_column="hERG_inhib",
        id_column="ID",
        majority_cap=args.majority_cap,
    )
    discovery_table = pd.read_csv(
        args.data_dir / "admet_group/herg/train_val.csv", usecols=["Drug", "Y"]
    )
    discovery_structures = standardize_smiles(
        discovery_table["Drug"].astype(str).tolist()
    )
    discovery_ids = set(discovery_structures["standard_inchikey"])
    exact_overlap = external["standard_inchikey"].isin(discovery_ids)
    exact_overlap_rows = int(exact_overlap.sum())
    external = external.loc[~exact_overlap].reset_index(drop=True)
    external_fingerprints = morgan_fingerprints(external["standard_smiles"].tolist())
    external_x = fingerprints_to_csr(external_fingerprints)
    external_y = external["Y"].to_numpy(dtype=int)

    group = admet_group(path=str(args.data_dir))
    per_seed = []
    for seed in args.seed:
        train_raw, valid_raw = group.get_train_valid_split(
            benchmark="herg", split_type="default", seed=seed
        )
        train = _standardized_split(train_raw)
        valid = _standardized_split(valid_raw)
        train_x = fingerprints_to_csr(
            morgan_fingerprints(train["standard_smiles"].tolist())
        )
        valid_fingerprints = morgan_fingerprints(valid["standard_smiles"].tolist())
        valid_x = fingerprints_to_csr(valid_fingerprints)
        train_y = train["Y"].to_numpy(dtype=int)
        valid_y = valid["Y"].to_numpy(dtype=int)

        keep = near_neighbor_keep_mask(
            external_fingerprints,
            valid_fingerprints,
            threshold=args.near_threshold,
        )
        baseline = _fit_score(
            train_x, train_y, valid_x, valid_y, seed=seed, n_jobs=args.n_jobs
        )
        exact_only = _fit_score(
            np.vstack([]) if False else __import__("scipy").sparse.vstack([train_x, external_x]),
            np.concatenate([train_y, external_y]),
            valid_x,
            valid_y,
            seed=seed,
            n_jobs=args.n_jobs,
        )
        firewalled = _fit_score(
            __import__("scipy").sparse.vstack([train_x, external_x[keep]]),
            np.concatenate([train_y, external_y[keep]]),
            valid_x,
            valid_y,
            seed=seed,
            n_jobs=args.n_jobs,
        )
        per_seed.append(
            {
                "seed": seed,
                "tdc_train_rows": len(train),
                "tdc_validation_rows": len(valid),
                "external_exact_only_rows": len(external),
                "external_firewalled_rows": int(keep.sum()),
                "near_neighbor_exclusions": int((~keep).sum()),
                "baseline_auroc": baseline,
                "exact_only_auroc": exact_only,
                "firewalled_auroc": firewalled,
                "exact_only_delta": exact_only - baseline,
                "firewalled_delta": firewalled - baseline,
            }
        )

    def summary(key: str) -> dict:
        values = np.asarray([row[key] for row in per_seed], dtype=float)
        return {
            "mean": float(values.mean()),
            "std": float(values.std()),
            "positive_seed_fraction": float((values > 0).mean()),
        }

    firewalled_delta = summary("firewalled_delta")
    minimum_effect = 0.005
    status = (
        "supported"
        if firewalled_delta["mean"] >= minimum_effect
        and firewalled_delta["positive_seed_fraction"] >= 2 / 3
        else "refuted"
        if firewalled_delta["mean"] <= -minimum_effect
        else "inconclusive"
    )
    result = {
        "protocol": "tdc-herg-external-data-scaling-pilot-v1",
        "task": "hERG",
        "status": status,
        "minimum_effect": minimum_effect,
        "source": {
            "name": "TDC hERGCentral hERG_inhib",
            "role": "bridge_not_endpoint_equivalent_direct_merge",
            "path": str(source),
            "sha256": _sha256(source),
            "raw_rows": len(raw),
        },
        "curation": {
            **curation.to_dict(),
            "exact_tdc_discovery_overlap_exclusions": exact_overlap_rows,
            "post_exact_firewall_identities": len(external),
            "majority_cap": args.majority_cap,
            "near_neighbor_threshold": args.near_threshold,
        },
        "model": "LightGBM 350 trees on Morgan radius-2 2048-bit fingerprints",
        "seeds": list(args.seed),
        "per_seed": per_seed,
        "summary": {
            "baseline_auroc": summary("baseline_auroc"),
            "exact_only_auroc": summary("exact_only_auroc"),
            "firewalled_auroc": summary("firewalled_auroc"),
            "exact_only_delta": summary("exact_only_delta"),
            "firewalled_delta": firewalled_delta,
        },
        "safety": {
            "tdc_test_loaded": False,
            "leaderboard_feedback_used": False,
            "external_source_selected_before_model_results": True,
        },
        "claim_boundary": (
            "Discovery validation pilot only; it tests whether a frozen, bridge-grade "
            "external source improves TDC validation. It does not establish TDC test gain."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        type=Path,
        default=Path("external/ecvoi_v6_tdc_cache/herg_central.tab"),
    )
    parser.add_argument("--expected-source-sha256", default=EXPECTED_SOURCE_SHA256)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--output", type=Path, default=Path("reports/herg_data_scaling_pilot_v1.json"))
    parser.add_argument("--majority-cap", type=int, default=30_000)
    parser.add_argument("--near-threshold", type=float, default=0.80)
    parser.add_argument("--seed", type=int, action="append", default=None)
    parser.add_argument("--n-jobs", type=int, default=8)
    args = parser.parse_args()
    args.seed = tuple(dict.fromkeys(args.seed or (1, 2, 3)))
    return args


if __name__ == "__main__":
    payload = run(parse_args())
    print(json.dumps({"status": payload["status"], **payload["summary"]}, indent=2))
