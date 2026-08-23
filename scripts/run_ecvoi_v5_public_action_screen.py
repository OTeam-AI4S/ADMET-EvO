"""Execute the protocol-frozen seed-17 matched action screen."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from admet_agent.features import MolecularFeaturizer
from admet_agent.public_action_matrix import ACTIONS, evaluate_action, sampled_structural_support


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _protocol(path: Path) -> dict:
    payload = json.loads(path.read_text())
    unbound = {key: value for key, value in payload.items() if key != "sha256"}
    expected = hashlib.sha256((json.dumps(unbound, indent=2, sort_keys=True) + "\n").encode()).hexdigest()
    if payload.get("sha256") != expected:
        raise RuntimeError("Action-screen protocol self-hash mismatch")
    for source, digest in payload["implementation_sha256"].items():
        if _sha256(Path(source)) != digest:
            raise RuntimeError(f"Frozen action-screen implementation changed: {source}")
    return payload


def run(args: argparse.Namespace) -> dict:
    protocol = _protocol(args.protocol)
    cohort_path = Path(protocol["cohort"]["path"])
    if _sha256(cohort_path) != protocol["cohort"]["sha256"]:
        raise RuntimeError("Public cohort hash mismatch")
    cohort = pd.read_csv(cohort_path)
    eligible = set(protocol["endpoint_scope"])
    cohort = cohort.loc[cohort["endpoint_id"].isin(eligible)].copy()
    identities = (
        cohort[["inchikey", "standard_smiles"]]
        .drop_duplicates("inchikey")
        .sort_values("inchikey", kind="stable")
        .reset_index(drop=True)
    )
    identity_index = {value: index for index, value in enumerate(identities["inchikey"])}
    smiles = identities["standard_smiles"].astype(str).tolist()
    features = {
        "baseline": MolecularFeaturizer(("morgan",)).transform(smiles),
        "feature_rich": MolecularFeaturizer(("morgan", "avalon", "erg", "rdkit2d")).transform(smiles),
    }
    rows = []
    failures = []
    for endpoint_id in sorted(eligible):
        endpoint = cohort.loc[cohort["endpoint_id"] == endpoint_id].copy()
        task_type = str(endpoint["task_type"].iloc[0])
        actions = ["baseline", "feature_rich", "boosting_anchor", "automl_expanded"]
        if task_type == "classification":
            actions.append("nested_imbalance")
        for outer_fold in range(protocol["outer_folds"]):
            train = endpoint.loc[endpoint["outer_fold"] != outer_fold]
            valid = endpoint.loc[endpoint["outer_fold"] == outer_fold]
            train_index = np.asarray([identity_index[value] for value in train["inchikey"]], dtype=int)
            valid_index = np.asarray([identity_index[value] for value in valid["inchikey"]], dtype=int)
            structural_support = sampled_structural_support(
                features["baseline"][train_index],
                train["inchikey"].astype(str).to_numpy(),
                features["baseline"][valid_index],
                valid["inchikey"].astype(str).to_numpy(),
                sample_size=int(protocol["state_encoder"]["similarity_sample_size"]),
                seed=int(protocol["screen_seed"]),
            )
            positive_rate = float(train["target"].mean()) if task_type == "classification" else None
            minority_fraction = min(positive_rate, 1.0 - positive_rate) if positive_rate is not None else None
            for action in actions:
                representation = "feature_rich" if action in {"feature_rich", "boosting_anchor", "automl_expanded"} else "baseline"
                try:
                    result = evaluate_action(
                        features[representation][train_index],
                        train["target"].to_numpy(),
                        train["chemistry_group"].astype(str).to_numpy(),
                        features[representation][valid_index],
                        valid["target"].to_numpy(),
                        action=action,
                        task_type=task_type,
                        seed=int(protocol["screen_seed"] + outer_fold),
                        trees=int(protocol["estimator"]["trees"]),
                        workers=int(protocol["estimator"]["workers"]),
                        inner_splits=int(protocol["inner_folds"]),
                    )
                    rows.append({"endpoint_id": endpoint_id, "source_id": str(endpoint["source_id"].iloc[0]), "outer_fold": outer_fold, "train_rows": len(train), "valid_rows": len(valid), "train_chemistry_groups": int(train["chemistry_group"].nunique()), "train_positive_rate": positive_rate, "train_minority_fraction": minority_fraction, "median_sampled_validation_to_train_tanimoto": structural_support, **result})
                except Exception as error:
                    failures.append({"endpoint_id": endpoint_id, "outer_fold": outer_fold, "action": action, "error_type": type(error).__name__, "error": str(error)})
    output = pd.DataFrame(rows).sort_values(["endpoint_id", "outer_fold", "action"], kind="stable")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(args.output, index=False)
    expected_fits = sum(
        4 + (1 if str(frame["task_type"].iloc[0]) == "classification" else 0)
        for _, frame in cohort.groupby("endpoint_id")
    ) * protocol["outer_folds"]
    manifest = {
        "version": "ecvoi-v5-public-action-screen-v2",
        "protocol_sha256": protocol["sha256"],
        "cohort_sha256": protocol["cohort"]["sha256"],
        "output_sha256": _sha256(args.output),
        "result_rows": len(output),
        "expected_result_rows": expected_fits,
        "failures": failures,
        "integrity_passed": len(output) == expected_fits and not failures,
        "test_labels_loaded": False,
        "blind_or_leaderboard_feedback_used": False,
        "analysis_or_promotion_performed": False,
    }
    args.manifest.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, default=Path("reports/ecvoi_v5_public_action_screen_protocol_v2.json"))
    parser.add_argument("--output", type=Path, default=Path("reports/ecvoi_v5_public_action_screen_seed17_v2.csv"))
    parser.add_argument("--manifest", type=Path, default=Path("reports/ecvoi_v5_public_action_screen_seed17_manifest_v2.json"))
    return parser.parse_args()


if __name__ == "__main__":
    print(json.dumps(run(parse_args()), indent=2, sort_keys=True))
