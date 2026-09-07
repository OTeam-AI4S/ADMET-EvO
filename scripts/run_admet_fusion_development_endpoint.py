"""Run OOF-safe fixed single and pair-fusion actions for one ADMET endpoint."""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import os
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, CatBoostRegressor
from lightgbm import LGBMClassifier, LGBMRegressor
from sklearn.ensemble import (
    ExtraTreesClassifier, ExtraTreesRegressor,
    RandomForestClassifier, RandomForestRegressor,
)
from sklearn.metrics import matthews_corrcoef, mean_absolute_error
from xgboost import XGBClassifier, XGBRegressor

from admet_agent.features import MolecularFeaturizer


SINGLE_ACTIONS = (
    "extra_trees_morgan", "extra_trees_rich", "lightgbm_rich",
    "random_forest_rich", "xgboost_rich", "catboost_rich",
)
RICH_REPRESENTATIONS = ("morgan", "avalon", "erg", "rdkit2d")
FUSION_WEIGHTS = (0.25, 0.50, 0.75)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _safe(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "__", value).strip("_")


def _score(y: np.ndarray, prediction: np.ndarray, task_type: str) -> float:
    if task_type == "classification":
        return float(matthews_corrcoef(y.astype(int), (prediction >= 0.5).astype(int)))
    return -float(mean_absolute_error(y.astype(float), prediction.astype(float)))


def _model(action: str, task_type: str, seed: int, workers: int, y: np.ndarray):
    if action.startswith("extra_trees"):
        common = dict(
            n_estimators=400, min_samples_leaf=2, max_features="sqrt",
            random_state=seed, n_jobs=workers,
        )
        return ExtraTreesClassifier(**common) if task_type == "classification" else ExtraTreesRegressor(**common)
    if action == "random_forest_rich":
        common = dict(
            n_estimators=400, max_features="sqrt", min_samples_leaf=1,
            random_state=seed, n_jobs=workers,
        )
        if task_type == "classification":
            return RandomForestClassifier(class_weight="balanced", **common)
        return RandomForestRegressor(**common)
    if action == "lightgbm_rich":
        common = dict(
            n_estimators=250, num_leaves=31, min_child_samples=10,
            learning_rate=0.05, random_state=seed, n_jobs=workers,
            verbosity=-1, deterministic=True, force_col_wise=True,
        )
        return LGBMClassifier(**common) if task_type == "classification" else LGBMRegressor(**common)
    if action == "xgboost_rich":
        common = dict(
            n_estimators=500, max_depth=6, learning_rate=0.05, subsample=0.8,
            colsample_bytree=0.8, tree_method="hist", random_state=seed,
            n_jobs=workers, verbosity=0,
        )
        if task_type == "classification":
            positives, negatives = max(1, int(np.sum(y == 1))), max(1, int(np.sum(y == 0)))
            return XGBClassifier(
                objective="binary:logistic", eval_metric="logloss",
                scale_pos_weight=negatives / positives, **common,
            )
        return XGBRegressor(objective="reg:absoluteerror", eval_metric="mae", **common)
    if action == "catboost_rich":
        common = dict(
            iterations=600, depth=8, learning_rate=0.05, l2_leaf_reg=3.0,
            random_seed=seed, task_type="GPU", devices="0", thread_count=workers,
            verbose=False, allow_writing_files=False,
        )
        if task_type == "classification":
            return CatBoostClassifier(loss_function="Logloss", auto_class_weights="Balanced", **common)
        return CatBoostRegressor(loss_function="MAE", **common)
    raise ValueError(action)


def _predict(model: object, x: np.ndarray, task_type: str) -> np.ndarray:
    if task_type == "classification":
        return np.asarray(model.predict_proba(x)[:, 1], dtype=float)
    return np.asarray(model.predict(x), dtype=float)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cohort", type=Path, required=True)
    parser.add_argument("--endpoint")
    parser.add_argument("--array-index", type=int)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    cohort_path = args.cohort.resolve()
    cohort = pd.read_csv(cohort_path)
    endpoints = sorted(cohort["endpoint_id"].astype(str).unique())
    index = args.array_index
    if args.endpoint is None:
        if index is None:
            index = int(os.environ["SLURM_ARRAY_TASK_ID"])
        if not 0 <= index < len(endpoints):
            raise ValueError("Array index out of bounds")
        endpoint_id = endpoints[index]
    else:
        endpoint_id = args.endpoint
    endpoint = cohort.loc[cohort["endpoint_id"].astype(str).eq(endpoint_id)].copy()
    if endpoint.empty:
        raise ValueError("Unknown endpoint")
    folds = [int(value) for value in sorted(endpoint["outer_fold"].astype(int).unique())]
    if folds != [0, 1, 2]:
        raise RuntimeError(f"Fusion development requires only frozen folds 0,1,2; got {folds}")
    if int((endpoint.groupby("chemistry_group")["outer_fold"].nunique() > 1).sum()):
        raise RuntimeError("Chemistry-group leakage")
    task_type = str(endpoint["task_type"].iloc[0])
    metric = "mcc" if task_type == "classification" else "mae"
    identities = endpoint[["inchikey", "standard_smiles"]].drop_duplicates("inchikey")
    identities = identities.sort_values("inchikey", kind="stable").reset_index(drop=True)
    position = {value: index for index, value in enumerate(identities["inchikey"])}
    smiles = identities["standard_smiles"].astype(str).tolist()
    features = {
        "morgan": MolecularFeaturizer(("morgan",)).transform(smiles),
        "rich": MolecularFeaturizer(RICH_REPRESENTATIONS).transform(smiles),
    }
    oof = {action: np.full(len(endpoint), np.nan, dtype=float) for action in SINGLE_ACTIONS}
    fold_scores = {action: [] for action in SINGLE_ACTIONS}
    fit_seconds = {action: 0.0 for action in SINGLE_ACTIONS}
    for fold in folds:
        train_mask = endpoint["outer_fold"].astype(int).ne(fold).to_numpy()
        valid_mask = ~train_mask
        train = endpoint.loc[train_mask]
        valid = endpoint.loc[valid_mask]
        train_index = np.asarray([position[value] for value in train["inchikey"]], dtype=int)
        valid_index = np.asarray([position[value] for value in valid["inchikey"]], dtype=int)
        y_train = train["target"].to_numpy()
        y_valid = valid["target"].to_numpy()
        for action in SINGLE_ACTIONS:
            representation = "morgan" if action == "extra_trees_morgan" else "rich"
            started = time.monotonic()
            model = _model(action, task_type, args.seed + fold, args.workers, y_train)
            model.fit(features[representation][train_index], y_train)
            prediction = _predict(model, features[representation][valid_index], task_type)
            fit_seconds[action] += time.monotonic() - started
            oof[action][valid_mask] = prediction
            fold_scores[action].append(_score(y_valid, prediction, task_type))
    if any(np.isnan(values).any() for values in oof.values()):
        raise RuntimeError("Incomplete OOF prediction matrix")

    rows = []
    for action in SINGLE_ACTIONS:
        rows.append({
            "endpoint_id": endpoint_id, "task_type": task_type, "metric": metric,
            "action": action, "kind": "single", "left": action, "right": None,
            "left_weight": 1.0, "fold0": fold_scores[action][0],
            "fold1": fold_scores[action][1], "fold2": fold_scores[action][2],
            "mean_oriented_score": float(np.mean(fold_scores[action])),
            "gain_over_stronger_member": 0.0, "positive_fold_rate": 1.0,
            "fusion_supported": False, "fit_seconds": fit_seconds[action],
        })
    for left, right in itertools.combinations(SINGLE_ACTIONS, 2):
        for left_weight in FUSION_WEIGHTS:
            values = left_weight * oof[left] + (1.0 - left_weight) * oof[right]
            scores = []
            for fold in folds:
                mask = endpoint["outer_fold"].astype(int).eq(fold).to_numpy()
                scores.append(_score(endpoint.loc[mask, "target"].to_numpy(), values[mask], task_type))
            stronger = left if np.mean(fold_scores[left]) >= np.mean(fold_scores[right]) else right
            deltas = np.asarray(scores) - np.asarray(fold_scores[stronger])
            gain = float(np.mean(deltas))
            positive_rate = float(np.mean(deltas > 0))
            supported = gain >= 0.005 and positive_rate >= (2.0 / 3.0)
            rows.append({
                "endpoint_id": endpoint_id, "task_type": task_type, "metric": metric,
                "action": f"blend({left}@{left_weight:.2f}+{right}@{1-left_weight:.2f})",
                "kind": "fusion", "left": left, "right": right,
                "left_weight": left_weight, "fold0": scores[0], "fold1": scores[1],
                "fold2": scores[2], "mean_oriented_score": float(np.mean(scores)),
                "gain_over_stronger_member": gain, "positive_fold_rate": positive_rate,
                "fusion_supported": supported, "fit_seconds": 0.0,
            })
    table = pd.DataFrame(rows)
    admissible = table.loc[table["kind"].eq("single") | table["fusion_supported"]].copy()
    admissible["_kind_priority"] = admissible["kind"].map({"single": 0, "fusion": 1})
    selected = admissible.sort_values(
        ["mean_oriented_score", "_kind_priority", "action"],
        ascending=[False, True, True], kind="stable",
    ).iloc[0]
    baseline = float(table.loc[table["action"].eq("extra_trees_morgan"), "mean_oriented_score"].iloc[0])
    table["selected"] = table["action"].eq(selected["action"])
    args.output_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{_safe(endpoint_id)}__seed{args.seed}"
    result_path = args.output_dir / f"{stem}.csv"
    prediction_path = args.output_dir / f"{stem}.predictions.npz"
    manifest_path = args.output_dir / f"{stem}.manifest.json"
    table.to_csv(result_path, index=False)
    np.savez_compressed(
        prediction_path,
        inchikey=endpoint["inchikey"].astype(str).to_numpy(),
        outer_fold=endpoint["outer_fold"].astype(int).to_numpy(),
        target=endpoint["target"].to_numpy(),
        **oof,
    )
    frozen = {
        "kind": str(selected["kind"]), "action": str(selected["action"]),
        "left": str(selected["left"]),
        "right": None if pd.isna(selected["right"]) else str(selected["right"]),
        "left_weight": float(selected["left_weight"]),
        "threshold": 0.5 if task_type == "classification" else None,
    }
    manifest = {
        "version": "admet-toxicology-oof-fusion-development-v1",
        "cohort_path": str(cohort_path), "cohort_sha256": _sha256(cohort_path),
        "endpoint_id": endpoint_id, "seed": args.seed, "folds": folds,
        "single_actions": list(SINGLE_ACTIONS), "fixed_fusion_weights": list(FUSION_WEIGHTS),
        "fusion_gate": {"minimum_mean_gain": 0.005, "minimum_positive_folds": 2},
        "selected_recipe": frozen,
        "development_score": float(selected["mean_oriented_score"]),
        "development_baseline": baseline,
        "development_gain": float(selected["mean_oriented_score"] - baseline),
        "result_path": str(result_path.resolve()), "result_sha256": _sha256(result_path),
        "prediction_path": str(prediction_path.resolve()), "prediction_sha256": _sha256(prediction_path),
        "holdout_rows_loaded": False, "holdout_labels_loaded": False,
        "integrity_passed": len(table) == len(SINGLE_ACTIONS) + 45,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "endpoint_id": endpoint_id, "selected": frozen,
        "development_score": manifest["development_score"],
        "development_gain": manifest["development_gain"],
        "integrity_passed": manifest["integrity_passed"],
    }, sort_keys=True))


if __name__ == "__main__":
    main()
