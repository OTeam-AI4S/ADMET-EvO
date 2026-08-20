"""Leakage-safe Uni-Mol2 multi-expert pilot on TDC validation splits.

The script intentionally never calls ``group.get`` and never reads a TDC test
frame. Each outer seed uses TDC's default scaffold train/validation split.
Within the outer training partition, five scaffold folds generate out-of-fold
predictions for stacking.
"""

from __future__ import annotations

import argparse
import gc
import json
import platform
import time
from importlib.metadata import version
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from rdkit.Chem.Scaffolds import MurckoScaffold
from scipy.stats import rankdata
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import GroupKFold
from tdc.benchmark_group import admet_group

from admet_agent.features import FeatureCache
from admet_agent.runner import _columns
from admet_agent.schemas import ExperimentSpec
from admet_agent.selection import build_model, predict, score
from admet_agent.tasks import resolve_task

PILOT_TASKS = (
    "CYP2C9_Substrate_CarbonMangels",
    "DILI",
    "Bioavailability_Ma",
    "Lipophilicity_AstraZeneca",
    "Half_Life_Obach",
    "hERG",
)
RICH_2D = ("morgan", "avalon", "erg", "rdkit2d")


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True), encoding="utf-8")


def _scaffolds(smiles: list[str]) -> np.ndarray:
    values = []
    for value in smiles:
        try:
            scaffold = MurckoScaffold.MurckoScaffoldSmiles(
                smiles=str(value), includeChirality=True
            )
        except (RuntimeError, ValueError):
            scaffold = str(value)
        values.append(scaffold or str(value))
    return np.asarray(values)


def _rank_columns(values: np.ndarray) -> np.ndarray:
    return np.column_stack(
        [rankdata(values[:, index], method="average") / len(values) for index in range(values.shape[1])]
    )


def _base_recipes(task_name: str, task_type: str) -> dict[str, tuple[str, tuple[str, ...], dict]]:
    svm_parameters = {"C": 3.0, "gamma": "scale"}
    if task_type == "regression":
        svm_parameters["epsilon"] = 0.1
    return {
        "2d_lightgbm": ("lightgbm", RICH_2D, {}),
        "2d_extra_trees": ("extra_trees", RICH_2D, {}),
        "2d_svm": ("svm_rbf", RICH_2D, svm_parameters),
        "frozen_unimol2_lightgbm": ("lightgbm", ("unimol2",), {}),
        "frozen_unimol2_extra_trees": ("extra_trees", ("unimol2",), {}),
        "hybrid_unimol2_2d_lightgbm": (
            "lightgbm",
            ("unimol2", *RICH_2D),
            {},
        ),
    }


def _fit_tabular_experts(
    task_name: str,
    seed: int,
    task,
    cache: FeatureCache,
    train: pd.DataFrame,
    valid: pd.DataFrame,
    smiles_col: str,
    y_col: str,
    inner_folds: int,
    workers: int,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], dict[str, float]]:
    smiles = train[smiles_col].astype(str).tolist() + valid[smiles_col].astype(str).tolist()
    y_train = train[y_col].to_numpy()
    groups = _scaffolds(train[smiles_col].astype(str).tolist())
    splitter = list(GroupKFold(n_splits=inner_folds).split(train, y_train, groups))
    features: dict[tuple[str, ...], np.ndarray] = {}
    oof_predictions: dict[str, np.ndarray] = {}
    valid_predictions: dict[str, np.ndarray] = {}
    elapsed: dict[str, float] = {}

    features[RICH_2D] = cache.get_or_compute(smiles, RICH_2D)
    features[("unimol2",)] = cache.get_or_compute(smiles, ("unimol2",))
    features[("unimol2", *RICH_2D)] = np.concatenate(
        [features[("unimol2",)], features[RICH_2D]], axis=1
    )

    for name, (model_name, representations, hyperparameters) in _base_recipes(
        task_name, task.task_type
    ).items():
        if representations not in features:
            features[representations] = cache.get_or_compute(smiles, representations)
        x_all = features[representations]
        x_train, x_valid = x_all[: len(train)], x_all[len(train) :]
        oof = np.zeros(len(train), dtype=float)
        started = time.monotonic()
        for fold_index, (fit_indices, holdout_indices) in enumerate(splitter):
            spec = ExperimentSpec(
                task=task_name,
                seed=seed * 100 + fold_index,
                representations=representations,
                model=model_name,
                imbalance="class_weight" if task.task_type == "classification" else "none",
                n_jobs=workers,
                hyperparameters=hyperparameters,
                hypothesis="inner-scaffold OOF expert",
            )
            model = build_model(spec, task, y_train[fit_indices])
            model.fit(x_train[fit_indices], y_train[fit_indices])
            oof[holdout_indices] = predict(model, x_train[holdout_indices], task)
        full_spec = ExperimentSpec(
            task=task_name,
            seed=seed,
            representations=representations,
            model=model_name,
            imbalance="class_weight" if task.task_type == "classification" else "none",
            n_jobs=workers,
            hyperparameters=hyperparameters,
            hypothesis="outer-validation expert",
        )
        full_model = build_model(full_spec, task, y_train)
        full_model.fit(x_train, y_train)
        oof_predictions[name] = oof
        valid_predictions[name] = predict(full_model, x_valid, task)
        elapsed[name] = time.monotonic() - started
        del full_model
        gc.collect()
    return oof_predictions, valid_predictions, elapsed


def _fit_unimol2(
    task,
    train: pd.DataFrame,
    valid: pd.DataFrame,
    smiles_col: str,
    y_col: str,
    model_dir: Path,
    epochs: int,
    inner_folds: int,
    batch_size: int,
    model_size: str = "84m",
    learning_rate: float = 1e-4,
) -> tuple[np.ndarray, np.ndarray, float]:
    from unimol_tools import MolPredict, MolTrain

    train_frame = train[[smiles_col, y_col]].rename(
        columns={smiles_col: "SMILES", y_col: "TARGET"}
    )
    predict_frame = valid[[smiles_col]].rename(columns={smiles_col: "SMILES"})
    predict_frame["TARGET"] = -1.0
    metric = {
        "auroc": "auc",
        "auprc": "auprc",
        "mae": "mae",
        "spearman": "spearmanr",
    }[task.metric]
    started = time.monotonic()
    trainer = MolTrain(
        task="classification" if task.task_type == "classification" else "regression",
        data_type="molecule",
        epochs=epochs,
        learning_rate=learning_rate,
        batch_size=batch_size,
        early_stopping=max(2, min(5, epochs // 3)),
        metrics=metric,
        split="scaffold",
        kfold=inner_folds,
        save_path=str(model_dir),
        smiles_col="SMILES",
        target_cols=["TARGET"],
        target_normalize="auto",
        use_cuda=True,
        use_amp=True,
        use_ddp=False,
        use_gpu="0",
        model_name="unimolv2",
        model_size=model_size,
        conf_cache_level=1,
    )
    trainer.fit(train_frame)
    oof = np.asarray(trainer.cv_pred, dtype=float).reshape(-1)
    predictor = MolPredict(load_model=str(model_dir))
    outer = np.asarray(predictor.predict(predict_frame), dtype=float).reshape(-1)
    return oof, outer, time.monotonic() - started


def _stack_predictions(
    task,
    y_train: np.ndarray,
    oof: dict[str, np.ndarray],
    valid: dict[str, np.ndarray],
) -> tuple[dict[str, np.ndarray], dict]:
    direction = 1 if task.maximize else -1
    oof_scores = {
        name: score(y_train, values, task.metric) for name, values in oof.items()
    }
    ordered = sorted(oof_scores, key=lambda name: direction * oof_scores[name], reverse=True)
    selected = ordered[: min(4, len(ordered))]
    x_oof = np.column_stack([oof[name] for name in selected])
    x_valid = np.column_stack([valid[name] for name in selected])
    if task.task_type == "classification":
        meta = LogisticRegression(C=0.1, class_weight="balanced", max_iter=2000)
        meta.fit(x_oof, y_train)
        stacked = meta.predict_proba(x_valid)[:, 1]
    else:
        if task.metric == "spearman":
            x_oof = _rank_columns(x_oof)
            x_valid = _rank_columns(x_valid)
        meta = Ridge(alpha=10.0)
        meta.fit(x_oof, y_train)
        stacked = meta.predict(x_valid)
    top_three = selected[: min(3, len(selected))]
    fixed_rank = np.mean(
        [rankdata(valid[name], method="average") / len(valid[name]) for name in top_three],
        axis=0,
    )
    if task.metric == "mae":
        fixed_rank = np.mean([valid[name] for name in top_three], axis=0)
    return {
        "oof_stacker": np.asarray(stacked),
        "fixed_rank_top3": np.asarray(fixed_rank),
    }, {
        "oof_scores": oof_scores,
        "selected_experts": selected,
        "meta_model": type(meta).__name__,
    }


def run_task(args: argparse.Namespace) -> dict:
    task = resolve_task(args.task)
    if task.name not in PILOT_TASKS:
        raise ValueError(f"{task.name} is not in the six-task pilot")
    output_dir = args.output_dir / task.name
    output_dir.mkdir(parents=True, exist_ok=True)
    group = admet_group(path=str(args.data_dir))
    cache = FeatureCache(
        args.output_dir / "shared_feature_cache",
        device="cuda",
        unimol_batch_size=args.batch_size,
    )
    result = {
        "task": task.name,
        "metric": task.metric,
        "protocol": {
            "outer_split": "TDC default scaffold validation",
            "outer_seeds": args.seeds,
            "inner_split": f"{args.inner_folds}-fold scaffold OOF",
            "test_labels_used": False,
            "unimol2": "84m full fine-tune",
            "epochs": args.epochs,
            "batch_size": args.batch_size,
        },
        "environment": {
            "python": platform.python_version(),
            "numpy": version("numpy"),
            "torch": version("torch"),
            "unimol_tools": version("unimol-tools"),
        },
        "seeds": {},
    }
    for seed in args.seeds:
        train, valid_frame = group.get_train_valid_split(
            benchmark=task.name,
            split_type="default",
            seed=seed,
        )
        smiles_col, y_col = _columns(train)
        y_train = train[y_col].to_numpy()
        y_valid = valid_frame[y_col].to_numpy()
        oof, predictions, elapsed = _fit_tabular_experts(
            task.name,
            seed,
            task,
            cache,
            train,
            valid_frame,
            smiles_col,
            y_col,
            args.inner_folds,
            args.workers,
        )
        fine_oof, fine_valid, fine_elapsed = _fit_unimol2(
            task,
            train,
            valid_frame,
            smiles_col,
            y_col,
            output_dir / f"seed_{seed}" / "unimol2_finetune",
            args.epochs,
            args.inner_folds,
            args.batch_size,
        )
        oof["finetuned_unimol2"] = fine_oof
        predictions["finetuned_unimol2"] = fine_valid
        elapsed["finetuned_unimol2"] = fine_elapsed
        ensembles, stack_metadata = _stack_predictions(task, y_train, oof, predictions)
        predictions.update(ensembles)
        scores = {name: score(y_valid, values, task.metric) for name, values in predictions.items()}
        seed_dir = output_dir / f"seed_{seed}"
        seed_dir.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(seed_dir / "validation_predictions.npz", **predictions)
        joblib.dump(stack_metadata, seed_dir / "stack_metadata.joblib")
        result["seeds"][str(seed)] = {
            "n_train": len(train),
            "n_valid": len(valid_frame),
            "scores": scores,
            "elapsed_seconds": elapsed,
            "stack": stack_metadata,
        }
        _write_json(output_dir / "results.partial.json", result)
        print(task.name, f"seed={seed}", json.dumps(scores, sort_keys=True), flush=True)
        gc.collect()
        try:
            import torch

            torch.cuda.empty_cache()
        except ImportError:
            pass
    expert_names = sorted(
        set.intersection(
            *(set(seed_result["scores"]) for seed_result in result["seeds"].values())
        )
    )
    result["summary"] = {
        name: {
            "mean": float(np.mean([seed_result["scores"][name] for seed_result in result["seeds"].values()])),
            "std": float(np.std([seed_result["scores"][name] for seed_result in result["seeds"].values()])),
        }
        for name in expert_names
    }
    direction = 1 if task.maximize else -1
    result["champion"] = max(
        expert_names,
        key=lambda name: direction * result["summary"][name]["mean"],
    )
    _write_json(output_dir / "results.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", required=True, choices=PILOT_TASKS)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--output-dir", type=Path, default=Path("runs/unimol2-pilot-v22"))
    parser.add_argument("--seeds", default="1,2,3,4,5")
    parser.add_argument("--inner-folds", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    args.seeds = [int(value) for value in args.seeds.split(",")]
    print(json.dumps(run_task(args), indent=2), flush=True)


if __name__ == "__main__":
    main()
