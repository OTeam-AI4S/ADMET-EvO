"""Evaluate a fold-safe Veith auxiliary propensity bridge on frozen B1.

The binary qHTS labels are never merged with OpenADMET pIC50 or TDI targets.
Instead, fold-local source models emit three CYP propensity features.  The
matched control and treatment use the same B1 features, estimator recipe,
chemistry folds and seeds.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
from importlib.metadata import version
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier, LGBMRegressor
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

from admet_agent.features import FeatureCache
from scripts.openadmet_cyp_metrics import official_st_rae
from scripts.openadmet_split_safety import assert_disjoint_training_evaluation
from scripts.run_openadmet_cyp_tree_baselines import RICH_2D


ISOFORMS = ("CYP2C9", "CYP2D6", "CYP3A4")
SOURCE_REPRESENTATIONS = ("morgan",)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _protocol(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    claimed = value.get("sha256")
    payload = {key: item for key, item in value.items() if key != "sha256"}
    actual = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
    if claimed != actual:
        raise RuntimeError("External propensity protocol SHA-256 is invalid")
    return value


def _source_fold(group: str, folds: int = 5) -> int:
    digest = hashlib.sha256(str(group).encode()).digest()
    return int.from_bytes(digest[:8], "big") % folds


def _excluded_outer_folds(value: Any) -> set[int]:
    if pd.isna(value) or str(value) == "":
        return set()
    return {int(item) for item in str(value).split(";") if item != ""}


def _main_model(*, task_type: str, seed: int, threads: int):
    common = {
        "n_estimators": 600,
        "num_leaves": 31,
        "learning_rate": 0.04,
        "subsample": 0.85,
        "subsample_freq": 1,
        "colsample_bytree": 0.7,
        "min_child_samples": 15,
        "reg_lambda": 2.0,
        "n_jobs": threads,
        "random_state": seed,
        "verbosity": -1,
    }
    if task_type == "classification":
        return LGBMClassifier(**common, class_weight="balanced")
    return LGBMRegressor(**common, objective="regression_l1")


def _source_model(*, seed: int, threads: int) -> LGBMClassifier:
    return LGBMClassifier(
        n_estimators=400,
        num_leaves=31,
        learning_rate=0.04,
        subsample=0.85,
        subsample_freq=1,
        colsample_bytree=0.7,
        min_child_samples=15,
        reg_lambda=2.0,
        class_weight="balanced",
        n_jobs=threads,
        random_state=seed,
        verbosity=-1,
    )


def _endpoint_metrics(frame: pd.DataFrame) -> dict[str, float | int | None]:
    task_type = str(frame["task_type"].iloc[0])
    if task_type == "classification":
        target = frame["target"].astype(int).to_numpy()
        prediction = frame["prediction"].to_numpy()
        binary = (prediction >= 0.5).astype(int)
        numerator = int(np.sum((binary == 1) & (target == 1))) * int(
            np.sum((binary == 0) & (target == 0))
        ) - int(np.sum((binary == 1) & (target == 0))) * int(
            np.sum((binary == 0) & (target == 1))
        )
        denominator = np.sqrt(
            max(1, int(np.sum(binary == 1)))
            * max(1, int(np.sum(binary == 0)))
            * max(1, int(np.sum(target == 1)))
            * max(1, int(np.sum(target == 0)))
        )
        return {
            "mcc": float(numerator / denominator),
            "brier": float(brier_score_loss(target, prediction)),
            "rows": len(frame),
        }
    return {
        "st_rae": official_st_rae(
            target=frame["target"].to_numpy(),
            prediction=frame["prediction"].to_numpy(),
            lower=frame["credible_low"].to_numpy(),
            upper=frame["credible_high"].to_numpy(),
        ),
        "mae": float(np.mean(np.abs(frame["prediction"] - frame["target"]))),
        "rows": len(frame),
    }


def _aggregate_endpoint_metrics(values: dict[str, dict[str, Any]]) -> dict[str, float]:
    regression = [
        float(item["st_rae"])
        for item in values.values()
        if item.get("st_rae") is not None
    ]
    classification = [
        float(item["mcc"]) for item in values.values() if item.get("mcc") is not None
    ]
    brier = [
        float(item["brier"])
        for item in values.values()
        if item.get("brier") is not None
    ]
    return {
        "macro_st_rae": float(np.mean(regression)),
        "macro_mcc": float(np.mean(classification)),
        "macro_brier": float(np.mean(brier)),
    }


def _reference_index(reference: dict[str, Any]) -> dict[tuple[int, int], dict[str, Any]]:
    return {
        (int(item["outer_fold"]), int(item["seed"])): item
        for item in reference["fold_seed_results"]
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    protocol_path = args.protocol.resolve()
    protocol = _protocol(protocol_path)
    if protocol.get("implementation_source_sha256") != _sha256(Path(__file__)):
        raise RuntimeError("External propensity implementation source mismatch")
    longform_path = args.longform.resolve()
    longform_manifest_path = longform_path.with_suffix(".manifest.json")
    longform_manifest = json.loads(longform_manifest_path.read_text())
    if longform_manifest.get("training_only") is not True:
        raise RuntimeError("Long-form artifact is not training-only")
    if longform_manifest.get("blind_test_opened") is not False:
        raise RuntimeError("Long-form artifact indicates blind-test access")
    if longform_manifest["output_sha256"] != _sha256(longform_path):
        raise RuntimeError("Long-form table hash mismatch")
    if protocol["inputs"]["longform_sha256"] != _sha256(longform_path):
        raise RuntimeError("Protocol long-form hash mismatch")
    if protocol["inputs"]["longform_manifest_sha256"] != _sha256(
        longform_manifest_path
    ):
        raise RuntimeError("Protocol long-form manifest hash mismatch")

    bridge_path = args.external_bridge.resolve()
    bridge_manifest_path = bridge_path.with_suffix(".manifest.json")
    bridge_manifest = json.loads(bridge_manifest_path.read_text())
    if bridge_manifest.get("training_only") is not True:
        raise RuntimeError("External bridge is not training-only")
    if bridge_manifest.get("blind_test_opened") is not False:
        raise RuntimeError("External bridge indicates blind-test access")
    if bridge_manifest.get("exact_primary_endpoint_merge_forbidden") is not True:
        raise RuntimeError("External bridge did not forbid exact primary-target merge")
    if bridge_manifest["output_sha256"] != _sha256(bridge_path):
        raise RuntimeError("External bridge hash mismatch")
    if protocol["inputs"]["external_bridge_sha256"] != _sha256(bridge_path):
        raise RuntimeError("Protocol external bridge hash mismatch")
    if protocol["inputs"]["external_bridge_manifest_sha256"] != _sha256(
        bridge_manifest_path
    ):
        raise RuntimeError("Protocol external bridge manifest hash mismatch")

    reference_path = args.baseline_reference.resolve()
    reference = json.loads(reference_path.read_text())
    if reference.get("baseline") != "B1":
        raise RuntimeError("Baseline reference is not B1")
    if reference.get("training_only") is not True:
        raise RuntimeError("Baseline reference is not training-only")
    if reference.get("blind_test_opened") is not False:
        raise RuntimeError("Baseline reference indicates blind-test access")
    if protocol["inputs"]["baseline_reference_sha256"] != _sha256(reference_path):
        raise RuntimeError("Protocol baseline-reference hash mismatch")

    longform = pd.read_csv(longform_path)
    main = longform.loc[
        longform["partition"].eq("development")
        & longform["fidelity"].isin({"primary_direct", "primary_classification"})
    ].copy()
    bridge = pd.read_csv(bridge_path)
    if set(bridge["isoform"].unique()) != set(ISOFORMS):
        raise RuntimeError("External bridge isoform contract mismatch")
    if set(bridge["task_type"].unique()) != {"classification"}:
        raise RuntimeError("External bridge contains a non-classification target")
    if set(bridge["fidelity"].unique()) != {"external_bridge"}:
        raise RuntimeError("External bridge fidelity contract mismatch")

    main_identity = (
        main[["inchikey", "standard_smiles"]]
        .drop_duplicates("inchikey")
        .sort_values("inchikey", kind="stable")
        .reset_index(drop=True)
    )
    bridge_identity = bridge[["inchikey", "standard_smiles"]].drop_duplicates(
        "inchikey"
    )
    source_identity = (
        pd.concat([main_identity, bridge_identity], ignore_index=True)
        .drop_duplicates("inchikey", keep="first")
        .sort_values("inchikey", kind="stable")
        .reset_index(drop=True)
    )
    main_features = FeatureCache(args.main_feature_cache.resolve()).get_or_compute(
        main_identity["standard_smiles"].astype(str).tolist(), RICH_2D
    )
    source_features = FeatureCache(
        args.source_feature_cache.resolve()
    ).get_or_compute(
        source_identity["standard_smiles"].astype(str).tolist(),
        SOURCE_REPRESENTATIONS,
    )
    main_index = {
        identity: index for index, identity in enumerate(main_identity["inchikey"])
    }
    source_index = {
        identity: index for index, identity in enumerate(source_identity["inchikey"])
    }
    main_source_index = np.asarray(
        [source_index[value] for value in main_identity["inchikey"]], dtype=int
    )

    folds = [int(value) for value in args.outer_folds.split(",")]
    seeds = [int(value) for value in args.seeds.split(",")]
    if folds != protocol["outer_folds"] or seeds != protocol["seeds"]:
        raise RuntimeError("Fold or seed schedule differs from the frozen protocol")
    if args.threads != protocol["threads"]:
        raise RuntimeError("Thread setting differs from the frozen protocol")
    if args.reference_tolerance != protocol["baseline_reproduction_tolerance"]:
        raise RuntimeError("Baseline reproduction tolerance differs from protocol")
    reference_index = _reference_index(reference)
    records: list[dict[str, Any]] = []
    prediction_frames: list[pd.DataFrame] = []
    reference_differences: list[float] = []
    for outer in folds:
        eligible_bridge = bridge.loc[
            ~bridge["excluded_outer_folds"].map(
                lambda value, fold=outer: fold in _excluded_outer_folds(value)
            )
        ].copy()
        eligible_bridge["source_holdout_fold"] = eligible_bridge[
            "chemistry_group"
        ].map(_source_fold)
        for seed in seeds:
            propensity = np.zeros((len(main_identity), len(ISOFORMS)), dtype=np.float32)
            source_metrics: dict[str, Any] = {}
            for column, isoform in enumerate(ISOFORMS):
                source = eligible_bridge.loc[eligible_bridge["isoform"] == isoform]
                source_train = source.loc[source["source_holdout_fold"] != outer].copy()
                source_holdout = source.loc[source["source_holdout_fold"] == outer].copy()
                if set(source_train["chemistry_group"]) & set(
                    source_holdout["chemistry_group"]
                ):
                    raise RuntimeError("External source chemistry-group leakage")
                if source_train["target"].nunique() != 2 or source_holdout[
                    "target"
                ].nunique() != 2:
                    raise RuntimeError("External source fold does not contain both classes")
                train_index = np.asarray(
                    [source_index[value] for value in source_train["inchikey"]]
                )
                holdout_index = np.asarray(
                    [source_index[value] for value in source_holdout["inchikey"]]
                )
                model = _source_model(seed=seed, threads=args.threads)
                model.fit(
                    source_features[train_index], source_train["target"].astype(int)
                )
                source_prediction = model.predict_proba(
                    source_features[holdout_index]
                )[:, 1]
                target = source_holdout["target"].astype(int).to_numpy()
                source_metrics[isoform] = {
                    "train_rows": len(source_train),
                    "heldout_rows": len(source_holdout),
                    "heldout_groups": int(source_holdout["chemistry_group"].nunique()),
                    "auroc": float(roc_auc_score(target, source_prediction)),
                    "average_precision": float(
                        average_precision_score(target, source_prediction)
                    ),
                    "brier": float(brier_score_loss(target, source_prediction)),
                }
                propensity[:, column] = model.predict_proba(
                    source_features[main_source_index]
                )[:, 1]

            arm_metrics: dict[str, dict[str, Any]] = {
                "b1_control": {},
                "b1_external_propensity": {},
            }
            for endpoint, endpoint_frame in main.groupby("endpoint_key"):
                train = endpoint_frame.loc[endpoint_frame["outer_fold"] != outer].copy()
                validation = endpoint_frame.loc[
                    endpoint_frame["outer_fold"] == outer
                ].copy()
                assert_disjoint_training_evaluation(
                    train,
                    validation,
                    context=f"external-propensity/{endpoint}/outer-{outer}",
                )
                train_index = np.asarray([main_index[value] for value in train["inchikey"]])
                validation_index = np.asarray(
                    [main_index[value] for value in validation["inchikey"]]
                )
                task_type = str(train["task_type"].iloc[0])
                for arm in ("b1_control", "b1_external_propensity"):
                    if arm == "b1_control":
                        features = main_features
                    else:
                        features = np.column_stack((main_features, propensity))
                    model = _main_model(
                        task_type=task_type, seed=seed, threads=args.threads
                    )
                    model.fit(features[train_index], train["target"])
                    scored = validation.copy()
                    if task_type == "classification":
                        scored["prediction"] = model.predict_proba(
                            features[validation_index]
                        )[:, 1]
                    else:
                        scored["prediction"] = model.predict(
                            features[validation_index]
                        )
                    scored["arm"] = arm
                    scored["outer_fold_evaluated"] = outer
                    scored["training_seed"] = seed
                    arm_metrics[arm][endpoint] = _endpoint_metrics(scored)
                    prediction_frames.append(scored)
            aggregate = {
                arm: _aggregate_endpoint_metrics(metrics)
                for arm, metrics in arm_metrics.items()
            }
            reference_row = reference_index[(outer, seed)]
            reference_differences.extend(
                [
                    abs(
                        aggregate["b1_control"]["macro_st_rae"]
                        - float(reference_row["macro_st_rae"])
                    ),
                    abs(
                        aggregate["b1_control"]["macro_mcc"]
                        - float(reference_row["macro_mcc"])
                    ),
                ]
            )
            records.append(
                {
                    "outer_fold": outer,
                    "seed": seed,
                    "source_heldout_metrics": source_metrics,
                    "endpoint_metrics": arm_metrics,
                    "aggregate": aggregate,
                }
            )

    maximum_reference_difference = max(reference_differences, default=float("inf"))
    if maximum_reference_difference > args.reference_tolerance:
        raise RuntimeError(
            "Matched B1 control did not reproduce the frozen baseline reference"
        )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    predictions = pd.concat(prediction_frames, ignore_index=True)
    prediction_path = args.output_dir / "external_propensity_predictions.csv"
    predictions.to_csv(prediction_path, index=False)

    means = {
        arm: {
            metric: float(
                np.mean([row["aggregate"][arm][metric] for row in records])
            )
            for metric in ("macro_st_rae", "macro_mcc", "macro_brier")
        }
        for arm in ("b1_control", "b1_external_propensity")
    }
    fold_deltas = []
    for outer in folds:
        rows = [row for row in records if row["outer_fold"] == outer]
        fold_deltas.append(
            {
                "outer_fold": outer,
                "st_rae_gain": float(
                    np.mean(
                        [
                            row["aggregate"]["b1_control"]["macro_st_rae"]
                            - row["aggregate"]["b1_external_propensity"][
                                "macro_st_rae"
                            ]
                            for row in rows
                        ]
                    )
                ),
                "mcc_gain": float(
                    np.mean(
                        [
                            row["aggregate"]["b1_external_propensity"]["macro_mcc"]
                            - row["aggregate"]["b1_control"]["macro_mcc"]
                            for row in rows
                        ]
                    )
                ),
                "brier_gain": float(
                    np.mean(
                        [
                            row["aggregate"]["b1_control"]["macro_brier"]
                            - row["aggregate"]["b1_external_propensity"][
                                "macro_brier"
                            ]
                            for row in rows
                        ]
                    )
                ),
            }
        )
    result = {
        "version": "openadmet-cyp-external-propensity-bridge-v1",
        "protocol_path": str(protocol_path),
        "protocol_sha256": protocol["sha256"],
        "implementation_source_sha256": _sha256(Path(__file__)),
        "training_only": True,
        "blind_test_opened": False,
        "sealed_certification_opened": False,
        "route": "source_conditioned_auxiliary_propensity_features",
        "exact_primary_endpoint_merge": False,
        "longform_sha256": _sha256(longform_path),
        "longform_manifest_sha256": _sha256(longform_manifest_path),
        "external_bridge_sha256": _sha256(bridge_path),
        "external_bridge_manifest_sha256": _sha256(bridge_manifest_path),
        "baseline_reference_path": str(reference_path),
        "baseline_reference_sha256": _sha256(reference_path),
        "baseline_reference_max_abs_difference": maximum_reference_difference,
        "outer_folds": folds,
        "seeds": seeds,
        "source_holdout": "sha256_chemistry_group_mod_5",
        "main_representations": list(RICH_2D),
        "source_representations": list(SOURCE_REPRESENTATIONS),
        "external_propensity_columns": list(ISOFORMS),
        "fold_seed_results": records,
        "arm_means": means,
        "outer_fold_mean_deltas": fold_deltas,
        "prediction_path": str(prediction_path.resolve()),
        "prediction_sha256": _sha256(prediction_path),
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "lightgbm": version("lightgbm"),
        },
        "test_labels_loaded": False,
        "leaderboard_feedback_used": False,
    }
    result_path = args.output_dir / "external_propensity_result.json"
    result_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return {**result, "result_path": str(result_path.resolve())}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
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
    parser.add_argument("--outer-folds", default="0,1,2,3,4")
    parser.add_argument("--seeds", default="1,2,3")
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--reference-tolerance", type=float, default=1e-12)
    parser.add_argument(
        "--main-feature-cache",
        type=Path,
        default=Path("cache/openadmet-cyp-features"),
    )
    parser.add_argument(
        "--source-feature-cache",
        type=Path,
        default=Path("cache/openadmet-cyp-external-source-features"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("runs/openadmet-cyp-external-propensity-v1"),
    )
    return parser.parse_args()


if __name__ == "__main__":
    print(json.dumps(run(parse_args()), indent=2, sort_keys=True))
