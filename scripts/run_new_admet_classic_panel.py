"""Run fixed RF, linear SVM, and XGBoost baselines on frozen endpoint folds."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import matthews_corrcoef, mean_absolute_error
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC, LinearSVR
from xgboost import XGBClassifier, XGBRegressor

from admet_agent.features import MolecularFeaturizer


MODELS = ("random_forest", "linear_svm", "xgboost")
RICH_REPRESENTATIONS = ("morgan", "avalon", "erg", "rdkit2d")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "__", value).strip("_")


def _resolve(cohort: pd.DataFrame, args: argparse.Namespace) -> tuple[str, int]:
    endpoints = sorted(cohort["endpoint_id"].astype(str).unique())
    if args.endpoint:
        if args.endpoint not in endpoints or args.seed is None:
            raise ValueError("--endpoint must be known and accompanied by --seed")
        return args.endpoint, int(args.seed)
    index = args.array_index
    if index is None:
        index = int(os.environ["SLURM_ARRAY_TASK_ID"])
    seeds = [int(value) for value in args.seeds.split(",") if value]
    if not 0 <= index < len(endpoints) * len(seeds):
        raise ValueError("Array index out of bounds")
    return endpoints[index // len(seeds)], seeds[index % len(seeds)]


def _fit_predict(
    model_name: str,
    task_type: str,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_valid: np.ndarray,
    *,
    seed: int,
    workers: int,
) -> np.ndarray:
    if model_name == "random_forest":
        if task_type == "classification":
            model = RandomForestClassifier(
                n_estimators=400, max_features="sqrt", min_samples_leaf=1,
                class_weight="balanced", random_state=seed, n_jobs=workers,
            )
        else:
            model = RandomForestRegressor(
                n_estimators=400, max_features="sqrt", min_samples_leaf=1,
                random_state=seed, n_jobs=workers,
            )
    elif model_name == "linear_svm":
        if task_type == "classification":
            model = make_pipeline(
                StandardScaler(with_mean=False),
                LinearSVC(C=1.0, class_weight="balanced", max_iter=20000, random_state=seed),
            )
        else:
            model = make_pipeline(
                StandardScaler(with_mean=False),
                LinearSVR(C=1.0, epsilon=0.0, max_iter=20000, random_state=seed),
            )
    elif model_name == "xgboost":
        common = dict(
            n_estimators=500, max_depth=6, learning_rate=0.05, subsample=0.8,
            colsample_bytree=0.8, tree_method="hist", random_state=seed,
            n_jobs=workers, verbosity=0,
        )
        if task_type == "classification":
            positives = max(1, int(np.sum(y_train == 1)))
            negatives = max(1, int(np.sum(y_train == 0)))
            model = XGBClassifier(
                objective="binary:logistic", eval_metric="logloss",
                scale_pos_weight=negatives / positives, **common,
            )
        else:
            model = XGBRegressor(objective="reg:absoluteerror", eval_metric="mae", **common)
    else:
        raise ValueError(model_name)
    model.fit(x_train, y_train)
    return np.asarray(model.predict(x_valid))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cohort", type=Path, required=True)
    parser.add_argument("--endpoint")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--seeds", default="29,43")
    parser.add_argument("--array-index", type=int)
    parser.add_argument("--models", default=",".join(MODELS))
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--holdout-folds", default="")
    args = parser.parse_args()

    cohort_path = args.cohort.resolve()
    cohort = pd.read_csv(cohort_path)
    endpoint_id, seed = _resolve(cohort, args)
    endpoint = cohort.loc[cohort["endpoint_id"].astype(str).eq(endpoint_id)].copy()
    task_type = str(endpoint["task_type"].iloc[0])
    source_id = str(endpoint["source_id"].iloc[0])
    models = tuple(value for value in args.models.split(",") if value)
    if set(models) - set(MODELS):
        raise ValueError("Unknown classic model")
    if int((endpoint.groupby("inchikey")["outer_fold"].nunique() > 1).sum()):
        raise RuntimeError("Identity leakage across folds")
    if int((endpoint.groupby("chemistry_group")["outer_fold"].nunique() > 1).sum()):
        raise RuntimeError("Chemistry-group leakage across folds")

    identities = endpoint[["inchikey", "standard_smiles"]].drop_duplicates("inchikey")
    identities = identities.sort_values("inchikey", kind="stable").reset_index(drop=True)
    position = {key: index for index, key in enumerate(identities["inchikey"])}
    smiles = identities["standard_smiles"].astype(str).tolist()
    features = {
        "morgan": MolecularFeaturizer(("morgan",)).transform(smiles),
        "rich": MolecularFeaturizer(RICH_REPRESENTATIONS).transform(smiles),
    }
    rows, failures = [], []
    holdout_folds = tuple(
        int(value) for value in re.split(r"[:,]", args.holdout_folds) if value.strip()
    )
    if holdout_folds:
        evaluation_splits = [(
            34,
            endpoint.loc[~endpoint["outer_fold"].astype(int).isin(holdout_folds)],
            endpoint.loc[endpoint["outer_fold"].astype(int).isin(holdout_folds)],
        )]
    else:
        evaluation_splits = [
            (
                int(fold),
                endpoint.loc[endpoint["outer_fold"].astype(int).ne(fold)],
                endpoint.loc[endpoint["outer_fold"].astype(int).eq(fold)],
            )
            for fold in sorted(endpoint["outer_fold"].astype(int).unique())
        ]
    for fold, train, valid in evaluation_splits:
        train_index = np.asarray([position[key] for key in train["inchikey"]])
        valid_index = np.asarray([position[key] for key in valid["inchikey"]])
        for model_name in models:
            representation = "morgan" if model_name == "linear_svm" else "rich"
            started = time.monotonic()
            try:
                prediction = _fit_predict(
                    model_name, task_type,
                    features[representation][train_index], train["target"].to_numpy(),
                    features[representation][valid_index],
                    seed=seed + int(fold), workers=args.workers,
                )
                if task_type == "classification":
                    raw_score = float(matthews_corrcoef(valid["target"].astype(int), prediction.astype(int)))
                    metric, oriented = "mcc", raw_score
                else:
                    raw_score = float(mean_absolute_error(valid["target"], prediction))
                    metric, oriented = "mae", -raw_score
                rows.append({
                    "endpoint_id": endpoint_id, "source_id": source_id,
                    "confirmation_seed": seed, "outer_fold": int(fold),
                    "action": model_name, "task_type": task_type, "metric": metric,
                    "raw_score": raw_score, "oriented_score": oriented,
                    "representation": representation, "train_rows": len(train),
                    "valid_rows": len(valid),
                    "wall_seconds": float(time.monotonic() - started),
                    "outer_validation_labels_used_for_training_or_threshold": False,
                })
            except Exception as error:
                failures.append({
                    "endpoint_id": endpoint_id, "confirmation_seed": seed,
                    "outer_fold": int(fold), "action": model_name,
                    "error_type": type(error).__name__, "error": str(error),
                })
    output = pd.DataFrame(rows).sort_values(["outer_fold", "action"], kind="stable")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    path = args.output_dir / f"{_safe_name(endpoint_id)}__seed{seed}.csv"
    output.to_csv(path, index=False)
    manifest = {
        "version": "new-admet-classic-panel-v1", "cohort_path": str(cohort_path),
        "cohort_sha256": _sha256(cohort_path), "endpoint_id": endpoint_id,
        "confirmation_seed": seed, "models": list(models), "result_rows": len(output),
        "expected_result_rows": len(evaluation_splits) * len(models), "failures": failures,
        "output_path": str(path.resolve()), "output_sha256": _sha256(path),
        "integrity_passed": len(output) == len(evaluation_splits) * len(models) and not failures,
        "evaluation_protocol": "fixed_holdout" if holdout_folds else "cross_validation",
        "holdout_folds": list(holdout_folds),
        "outer_validation_labels_used_for_training_or_threshold": False,
    }
    path.with_suffix(".manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps(manifest, sort_keys=True))


if __name__ == "__main__":
    main()
