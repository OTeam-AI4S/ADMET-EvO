"""Run one leakage-controlled endpoint/seed shard from a frozen cohort table."""

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
from sklearn.metrics import matthews_corrcoef, mean_absolute_error

from admet_agent.features import MolecularFeaturizer
from admet_agent.public_action_matrix import evaluate_action, sampled_structural_support


DEFAULT_ACTIONS = (
    "baseline",
    "feature_rich",
    "boosting_anchor",
    "automl_expanded",
    "nested_imbalance",
)
OPTIONAL_ACTIONS = ("gpu_catboost",)
RICH_REPRESENTATIONS = ("morgan", "avalon", "erg", "rdkit2d")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "__", value).strip("_")


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _resolve_shard(
    cohort: pd.DataFrame,
    *,
    endpoint: str | None,
    seed: int | None,
    seeds: tuple[int, ...],
    array_index: int | None,
) -> tuple[str, int]:
    endpoints = tuple(sorted(cohort["endpoint_id"].astype(str).unique()))
    if endpoint is not None:
        if endpoint not in endpoints:
            raise ValueError(f"Unknown endpoint {endpoint!r}")
        if seed is None:
            raise ValueError("--seed is required with --endpoint")
        return endpoint, int(seed)
    if array_index is None:
        raw = os.environ.get("SLURM_ARRAY_TASK_ID")
        if raw is None:
            raise ValueError("Use --endpoint/--seed or provide --array-index")
        array_index = int(raw)
    shard_count = len(endpoints) * len(seeds)
    if not 0 <= array_index < shard_count:
        raise ValueError(f"array index {array_index} outside [0, {shard_count})")
    return endpoints[array_index // len(seeds)], seeds[array_index % len(seeds)]


def _validate_endpoint(endpoint: pd.DataFrame) -> tuple[str, str, int]:
    required = {
        "endpoint_id",
        "source_id",
        "task_type",
        "inchikey",
        "standard_smiles",
        "chemistry_group",
        "target",
        "outer_fold",
    }
    if missing := required - set(endpoint.columns):
        raise ValueError(f"Cohort is missing columns: {sorted(missing)}")
    if endpoint["endpoint_id"].nunique() != 1:
        raise ValueError("Shard must contain exactly one endpoint")
    task_types = endpoint["task_type"].astype(str).unique()
    source_ids = endpoint["source_id"].astype(str).unique()
    if len(task_types) != 1 or task_types[0] not in {"classification", "regression"}:
        raise ValueError("Endpoint must have one supported task type")
    if len(source_ids) != 1:
        raise ValueError("Endpoint must have one source_id")
    folds = sorted(endpoint["outer_fold"].astype(int).unique())
    if folds != list(range(len(folds))) or len(folds) < 2:
        raise ValueError(f"Outer folds must be contiguous from zero, got {folds}")
    identity_folds = endpoint.groupby("inchikey")["outer_fold"].nunique()
    group_folds = endpoint.groupby("chemistry_group")["outer_fold"].nunique()
    if int((identity_folds > 1).sum()) or int((group_folds > 1).sum()):
        raise RuntimeError("Identity or chemistry-group leakage across outer folds")
    return task_types[0], source_ids[0], len(folds)


def _evaluate_gpu_catboost(
    x_train: np.ndarray,
    target_train: np.ndarray,
    x_valid: np.ndarray,
    target_valid: np.ndarray,
    *,
    task_type: str,
    seed: int,
    workers: int,
) -> dict:
    """Fit the fixed A100 CatBoost anchor without outer-label adaptation."""
    from catboost import CatBoostClassifier, CatBoostRegressor

    common = {
        "iterations": 600,
        "depth": 8,
        "learning_rate": 0.05,
        "l2_leaf_reg": 3.0,
        "random_seed": seed,
        "task_type": "GPU",
        "devices": "0",
        "thread_count": workers,
        "verbose": False,
        "allow_writing_files": False,
    }
    started = time.monotonic()
    if task_type == "classification":
        model = CatBoostClassifier(
            loss_function="Logloss",
            auto_class_weights="Balanced",
            **common,
        )
        model.fit(x_train, np.asarray(target_train, dtype=int))
        probability = model.predict_proba(x_valid)[:, 1]
        raw_score = float(
            matthews_corrcoef(
                np.asarray(target_valid, dtype=int),
                (probability >= 0.5).astype(int),
            )
        )
        oriented_score, metric = raw_score, "mcc"
    elif task_type == "regression":
        model = CatBoostRegressor(loss_function="MAE", **common)
        model.fit(x_train, np.asarray(target_train, dtype=float))
        raw_score = float(mean_absolute_error(target_valid, model.predict(x_valid)))
        oriented_score, metric = -raw_score, "mae"
    else:
        raise ValueError(f"Unsupported task type: {task_type}")
    return {
        "action": "gpu_catboost",
        "task_type": task_type,
        "metric": metric,
        "raw_score": raw_score,
        "oriented_score": oriented_score,
        "threshold": 0.5 if task_type == "classification" else None,
        "selected_automl_candidate": None,
        "fits": 1,
        "wall_seconds": float(time.monotonic() - started),
        "outer_validation_labels_used_for_training_or_threshold": False,
    }


def run(args: argparse.Namespace) -> dict:
    cohort_path = args.cohort.resolve()
    cohort = pd.read_csv(cohort_path)
    seeds = tuple(int(value) for value in args.seeds.split(",") if value.strip())
    if not seeds:
        raise ValueError("At least one confirmation seed is required")
    endpoint_id, seed = _resolve_shard(
        cohort,
        endpoint=args.endpoint,
        seed=args.seed,
        seeds=seeds,
        array_index=args.array_index,
    )
    endpoint = cohort.loc[cohort["endpoint_id"].astype(str).eq(endpoint_id)].copy()
    task_type, source_id, outer_folds = _validate_endpoint(endpoint)

    identities = (
        endpoint[["inchikey", "standard_smiles"]]
        .drop_duplicates("inchikey")
        .sort_values("inchikey", kind="stable")
        .reset_index(drop=True)
    )
    identity_index = {value: index for index, value in enumerate(identities["inchikey"])}
    smiles = identities["standard_smiles"].astype(str).tolist()
    features = {
        "baseline": MolecularFeaturizer(("morgan",)).transform(smiles),
        "feature_rich": MolecularFeaturizer(RICH_REPRESENTATIONS).transform(smiles),
    }

    requested_actions = tuple(value for value in args.actions.split(",") if value)
    unknown = set(requested_actions) - set(DEFAULT_ACTIONS + OPTIONAL_ACTIONS)
    if unknown:
        raise ValueError(f"Unknown actions: {sorted(unknown)}")
    actions = tuple(
        value
        for value in requested_actions
        if not (value == "nested_imbalance" and task_type != "classification")
    )
    rows: list[dict] = []
    failures: list[dict] = []
    holdout_folds = tuple(
        int(value) for value in re.split(r"[:,]", args.holdout_folds) if value.strip()
    )
    if holdout_folds:
        if not set(holdout_folds).issubset(set(endpoint["outer_fold"].astype(int))):
            raise ValueError("Holdout folds are not present in the cohort")
        evaluation_splits = [(
            34,
            endpoint.loc[~endpoint["outer_fold"].astype(int).isin(holdout_folds)],
            endpoint.loc[endpoint["outer_fold"].astype(int).isin(holdout_folds)],
        )]
    else:
        evaluation_splits = [
            (
                outer_fold,
                endpoint.loc[endpoint["outer_fold"].astype(int).ne(outer_fold)],
                endpoint.loc[endpoint["outer_fold"].astype(int).eq(outer_fold)],
            )
            for outer_fold in range(outer_folds)
        ]
    for outer_fold, train, valid in evaluation_splits:
        train_index = np.asarray([identity_index[value] for value in train["inchikey"]], dtype=int)
        valid_index = np.asarray([identity_index[value] for value in valid["inchikey"]], dtype=int)
        if set(train["inchikey"]) & set(valid["inchikey"]):
            raise RuntimeError("Outer-fold identity leakage")
        if set(train["chemistry_group"]) & set(valid["chemistry_group"]):
            raise RuntimeError("Outer-fold chemistry-group leakage")
        support = sampled_structural_support(
            features["baseline"][train_index],
            train["inchikey"].astype(str).to_numpy(),
            features["baseline"][valid_index],
            valid["inchikey"].astype(str).to_numpy(),
            sample_size=args.similarity_sample_size,
            seed=seed,
        )
        positive_rate = float(train["target"].mean()) if task_type == "classification" else None
        minority_fraction = (
            min(positive_rate, 1.0 - positive_rate)
            if positive_rate is not None
            else None
        )
        for action in actions:
            representation = (
                "feature_rich"
                if action
                in {"feature_rich", "boosting_anchor", "automl_expanded", "gpu_catboost"}
                else "baseline"
            )
            try:
                if action == "gpu_catboost":
                    result = _evaluate_gpu_catboost(
                        features[representation][train_index],
                        train["target"].to_numpy(),
                        features[representation][valid_index],
                        valid["target"].to_numpy(),
                        task_type=task_type,
                        seed=seed + outer_fold,
                        workers=args.workers,
                    )
                else:
                    result = evaluate_action(
                        features[representation][train_index],
                        train["target"].to_numpy(),
                        train["chemistry_group"].astype(str).to_numpy(),
                        features[representation][valid_index],
                        valid["target"].to_numpy(),
                        action=action,
                        task_type=task_type,
                        seed=seed + outer_fold,
                        trees=args.trees,
                        workers=args.workers,
                        inner_splits=args.inner_folds,
                    )
                rows.append(
                    {
                        "endpoint_id": endpoint_id,
                        "source_id": source_id,
                        "confirmation_seed": seed,
                        "outer_fold": outer_fold,
                        "train_rows": len(train),
                        "valid_rows": len(valid),
                        "train_chemistry_groups": int(train["chemistry_group"].nunique()),
                        "train_positive_rate": positive_rate,
                        "train_minority_fraction": minority_fraction,
                        "median_sampled_validation_to_train_tanimoto": support,
                        **result,
                    }
                )
            except Exception as error:  # preserve a complete auditable failure denominator
                failures.append(
                    {
                        "endpoint_id": endpoint_id,
                        "confirmation_seed": seed,
                        "outer_fold": outer_fold,
                        "action": action,
                        "error_type": type(error).__name__,
                        "error": str(error),
                    }
                )

    output = pd.DataFrame(rows)
    if len(output):
        output = output.sort_values(["outer_fold", "action"], kind="stable")
    safe = _safe_name(endpoint_id)
    output_path = args.output or args.output_dir / f"{safe}__seed{seed}.csv"
    manifest_path = args.manifest or output_path.with_suffix(".manifest.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(output_path, index=False)
    expected_rows = len(evaluation_splits) * len(actions)
    payload = {
        "version": "new-admet-endpoint-confirmation-v1",
        "cohort_path": str(cohort_path),
        "cohort_sha256": _sha256(cohort_path),
        "endpoint_id": endpoint_id,
        "source_id": source_id,
        "task_type": task_type,
        "confirmation_seed": seed,
        "outer_folds": outer_folds,
        "evaluation_protocol": "fixed_holdout" if holdout_folds else "cross_validation",
        "holdout_folds": list(holdout_folds),
        "actions": list(actions),
        "rows": len(endpoint),
        "result_rows": len(output),
        "expected_result_rows": expected_rows,
        "output_path": str(output_path.resolve()),
        "output_sha256": _sha256(output_path),
        "failures": failures,
        "integrity_passed": len(output) == expected_rows and not failures,
        "selection_contract": (
            "automl_expanded selects only inside each outer-training partition; "
            "outer labels are evaluation-only"
        ),
        "official_test_labels_loaded": False,
        "leaderboard_feedback_used": False,
    }
    _write_json(manifest_path, payload)
    print(json.dumps(payload, sort_keys=True))
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cohort", type=Path, required=True)
    parser.add_argument("--endpoint")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--seeds", default="29,43")
    parser.add_argument("--array-index", type=int)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--actions", default=",".join(DEFAULT_ACTIONS))
    parser.add_argument("--trees", type=int, default=200)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--inner-folds", type=int, default=5)
    parser.add_argument("--similarity-sample-size", type=int, default=256)
    parser.add_argument("--holdout-folds", default="")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
