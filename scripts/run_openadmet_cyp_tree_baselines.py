"""Run fixed LightGBM baselines on OpenADMET chemistry-group outer folds."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
from importlib.metadata import version
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier, LGBMRegressor
from sklearn.metrics import matthews_corrcoef

from admet_agent.features import FeatureCache
from scripts.openadmet_cyp_metrics import official_st_rae
from scripts.openadmet_split_safety import assert_disjoint_training_evaluation

RICH_2D = ("morgan", "avalon", "erg", "rdkit2d")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _endpoint_metric(
    *,
    task_type: str,
    validation: pd.DataFrame,
) -> dict:
    if task_type == "classification":
        binary = (validation["prediction"].to_numpy() >= 0.5).astype(int)
        return {
            "mcc": float(
                matthews_corrcoef(validation["target"].astype(int), binary)
            ),
            "rows": len(validation),
        }
    return {
        "st_rae": official_st_rae(
            target=validation["target"].to_numpy(),
            prediction=validation["prediction"].to_numpy(),
            lower=validation["credible_low"].to_numpy(),
            upper=validation["credible_high"].to_numpy(),
        ),
        "mae": float(
            np.mean(np.abs(validation["prediction"] - validation["target"]))
        ),
        "rows": len(validation),
    }


def _load_minimol(
    *, embedding: Path, embedding_input: Path, expected_input_sha256: str
) -> tuple[np.ndarray, dict[str, int], dict]:
    embedding = embedding.resolve()
    embedding_input = embedding_input.resolve()
    if _sha256(embedding_input) != expected_input_sha256:
        raise RuntimeError("Long-form manifest does not bind this embedding input")
    payload = json.loads(embedding_input.read_text())
    if payload.get("contains_labels") is not False:
        raise RuntimeError("Embedding input must be label-free")
    if payload.get("official_test_loaded") is not False:
        raise RuntimeError("Embedding input indicates blind-test access")
    manifest_path = embedding.with_suffix(".manifest.json")
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("input_sha256") != _sha256(embedding_input):
        raise RuntimeError("Embedding input hash mismatch")
    if manifest.get("output_sha256") != _sha256(embedding):
        raise RuntimeError("Embedding output hash mismatch")
    values = np.load(embedding, allow_pickle=False)
    if values.shape != (len(payload["inchikeys"]), 512):
        raise RuntimeError(f"Unexpected embedding shape {values.shape}")
    identity_index = {
        value: index for index, value in enumerate(payload["inchikeys"])
    }
    return values, identity_index, manifest


def run(args: argparse.Namespace) -> dict:
    longform_path = args.longform.resolve()
    manifest_path = longform_path.with_suffix(".manifest.json")
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("training_only") is not True:
        raise RuntimeError("Long-form artifact is not training-only")
    if manifest.get("blind_test_opened") is not False:
        raise RuntimeError("Long-form artifact indicates blind-test access")
    if manifest["output_sha256"] != _sha256(longform_path):
        raise RuntimeError("Long-form table hash mismatch")
    longform = pd.read_csv(longform_path)
    main = longform.loc[
        longform["partition"].eq("development")
        & longform["fidelity"].isin(
            {"primary_direct", "primary_classification"}
        )
    ].copy()
    identity = (
        main[["inchikey", "standard_smiles"]]
        .drop_duplicates("inchikey")
        .sort_values("inchikey", kind="stable")
        .reset_index(drop=True)
    )
    if args.baseline == "B1":
        features = FeatureCache(args.feature_cache.resolve()).get_or_compute(
            identity["standard_smiles"].astype(str).tolist(), RICH_2D
        )
        identity_index = {
            value: index for index, value in enumerate(identity["inchikey"])
        }
        feature_contract = {
            "representations": list(RICH_2D),
            "feature_width": int(features.shape[1]),
            "embedding_input_sha256": None,
            "embedding_sha256": None,
        }
    else:
        if args.embedding is None or args.embedding_input is None:
            raise ValueError("B2 requires --embedding and --embedding-input")
        features, identity_index, embedding_manifest = _load_minimol(
            embedding=args.embedding,
            embedding_input=args.embedding_input,
            expected_input_sha256=manifest["minimol_input_sha256"],
        )
        feature_contract = {
            "representations": ["frozen_minimol_512"],
            "feature_width": 512,
            "embedding_input_sha256": embedding_manifest["input_sha256"],
            "embedding_sha256": embedding_manifest["output_sha256"],
        }

    folds = [int(value) for value in args.outer_folds.split(",")]
    seeds = [int(value) for value in args.seeds.split(",")]
    records = []
    prediction_frames = []
    for outer in folds:
        for seed in seeds:
            endpoint_metrics = {}
            for endpoint, endpoint_frame in main.groupby("endpoint_key"):
                train = endpoint_frame.loc[endpoint_frame["outer_fold"] != outer].copy()
                validation = endpoint_frame.loc[
                    endpoint_frame["outer_fold"] == outer
                ].copy()
                assert_disjoint_training_evaluation(
                    train,
                    validation,
                    context=f"{args.baseline}/{endpoint}/outer-{outer}",
                )
                train_index = np.asarray(
                    [identity_index[value] for value in train["inchikey"]]
                )
                validation_index = np.asarray(
                    [identity_index[value] for value in validation["inchikey"]]
                )
                if train["task_type"].iloc[0] == "classification":
                    model = LGBMClassifier(
                        n_estimators=600,
                        num_leaves=31,
                        learning_rate=0.04,
                        subsample=0.85,
                        subsample_freq=1,
                        colsample_bytree=0.7,
                        min_child_samples=15,
                        reg_lambda=2.0,
                        class_weight="balanced",
                        n_jobs=args.threads,
                        random_state=seed,
                        verbosity=-1,
                    )
                    model.fit(features[train_index], train["target"].astype(int))
                    prediction = model.predict_proba(features[validation_index])[:, 1]
                else:
                    model = LGBMRegressor(
                        n_estimators=600,
                        num_leaves=31,
                        learning_rate=0.04,
                        subsample=0.85,
                        subsample_freq=1,
                        colsample_bytree=0.7,
                        min_child_samples=15,
                        reg_lambda=2.0,
                        objective="regression_l1",
                        n_jobs=args.threads,
                        random_state=seed,
                        verbosity=-1,
                    )
                    model.fit(features[train_index], train["target"])
                    prediction = model.predict(features[validation_index])
                validation["prediction"] = prediction
                validation["predictive_std"] = np.nan
                validation["outer_fold_evaluated"] = outer
                validation["training_seed"] = seed
                endpoint_metrics[endpoint] = _endpoint_metric(
                    task_type=str(train["task_type"].iloc[0]),
                    validation=validation,
                )
                prediction_frames.append(validation)
            regression = [
                value["st_rae"]
                for value in endpoint_metrics.values()
                if "st_rae" in value and value["st_rae"] is not None
            ]
            classification = [
                value["mcc"]
                for value in endpoint_metrics.values()
                if "mcc" in value
            ]
            records.append(
                {
                    "outer_fold": outer,
                    "seed": seed,
                    "endpoint_metrics": endpoint_metrics,
                    "macro_st_rae": float(np.mean(regression)),
                    "macro_mcc": float(np.mean(classification)),
                }
            )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    predictions = pd.concat(prediction_frames, ignore_index=True)
    prediction_path = args.output_dir / f"{args.baseline.lower()}_outer_predictions.csv"
    predictions.to_csv(prediction_path, index=False)
    result = {
        "version": "openadmet-cyp-tree-baseline-cv-v1",
        "baseline": args.baseline,
        "training_only": True,
        "blind_test_opened": False,
        "sealed_certification_opened": False,
        "longform_sha256": _sha256(longform_path),
        "longform_manifest_sha256": _sha256(manifest_path),
        "feature_contract": feature_contract,
        "outer_folds": folds,
        "seeds": seeds,
        "threads": args.threads,
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "lightgbm": version("lightgbm"),
        },
        "fold_seed_results": records,
        "mean_macro_st_rae": float(
            np.mean([record["macro_st_rae"] for record in records])
        ),
        "mean_macro_mcc": float(
            np.mean([record["macro_mcc"] for record in records])
        ),
        "prediction_path": str(prediction_path.resolve()),
        "prediction_sha256": _sha256(prediction_path),
        "test_labels_loaded": False,
        "leaderboard_feedback_used": False,
    }
    result_path = args.output_dir / f"{args.baseline.lower()}_cv_result.json"
    result_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return {**result, "result_path": str(result_path.resolve())}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--longform",
        type=Path,
        default=Path("reports/openadmet_cyp_training_longform_v1.csv"),
    )
    parser.add_argument("--baseline", choices=("B1", "B2"), required=True)
    parser.add_argument("--embedding", type=Path)
    parser.add_argument("--embedding-input", type=Path)
    parser.add_argument("--outer-folds", default="0,1,2")
    parser.add_argument("--seeds", default="1")
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument(
        "--feature-cache", type=Path, default=Path("cache/openadmet-cyp-features")
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("runs/openadmet-cyp-stage-a")
    )
    return parser.parse_args()


if __name__ == "__main__":
    print(json.dumps(run(parse_args()), indent=2, sort_keys=True))
