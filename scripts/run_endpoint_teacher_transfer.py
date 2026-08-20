"""Validation-only cross-endpoint teacher features for the two weakest tasks.

For every target scaffold split, auxiliary teachers use only auxiliary development
rows. Any molecule appearing anywhere in the target development pool is removed
from every teacher's fit set, preventing exact-molecule shortcuts. The script never
calls ``admet_group.get`` and never accesses an official test frame.
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem
from scipy.stats import rankdata
from tdc.benchmark_group import admet_group

from admet_agent.features import FeatureCache
from admet_agent.runner import _columns
from admet_agent.schemas import ExperimentSpec
from admet_agent.selection import build_model, predict, score
from admet_agent.tasks import resolve_task

RICH_2D = ("morgan", "avalon", "erg", "rdkit2d")
TEACHERS = {
    "Bioavailability_Ma": (
        "HIA_Hou",
        "Pgp_Broccatelli",
        "BBB_Martins",
        "Caco2_Wang",
        "Lipophilicity_AstraZeneca",
        "Solubility_AqSolDB",
    ),
    "CYP2C9_Substrate_CarbonMangels": (
        "CYP2C9_Veith",
        "CYP2D6_Veith",
        "CYP3A4_Veith",
        "CYP2D6_Substrate_CarbonMangels",
        "CYP3A4_Substrate_CarbonMangels",
    ),
}
CANDIDATES = {
    "baseline_lightgbm": ("lightgbm", "base"),
    "teacher_only_lightgbm": ("lightgbm", "teacher"),
    "teacher_aug_lightgbm": ("lightgbm", "augmented"),
    "teacher_aug_extra_trees": ("extra_trees", "augmented"),
    "teacher_aug_svm": ("svm_rbf", "augmented"),
}


def _canonical(value: str) -> str:
    molecule = Chem.MolFromSmiles(str(value))
    return Chem.MolToSmiles(molecule, canonical=True) if molecule is not None else str(value)


def exclude_target_molecules(
    auxiliary: pd.DataFrame,
    auxiliary_smiles_col: str,
    target_smiles: list[str],
) -> tuple[pd.DataFrame, int]:
    blocked = {_canonical(value) for value in target_smiles}
    keep = [
        _canonical(value) not in blocked
        for value in auxiliary[auxiliary_smiles_col].astype(str)
    ]
    filtered = auxiliary.loc[keep].reset_index(drop=True)
    return filtered, len(auxiliary) - len(filtered)


def _teacher_matrix(
    group,
    cache: FeatureCache,
    target_smiles: list[str],
    target_name: str,
    seed: int,
    workers: int,
) -> tuple[np.ndarray, list[dict]]:
    columns = []
    audit = []
    for teacher_name in TEACHERS[target_name]:
        teacher = resolve_task(teacher_name)
        aux_train, aux_valid = group.get_train_valid_split(
            benchmark=teacher.name, split_type="default", seed=1
        )
        auxiliary = pd.concat([aux_train, aux_valid], ignore_index=True)
        smiles_col, y_col = _columns(auxiliary)
        auxiliary, excluded = exclude_target_molecules(
            auxiliary, smiles_col, target_smiles
        )
        if len(auxiliary) < 20:
            raise ValueError(f"Too few rows remain for teacher {teacher.name}")
        smiles = auxiliary[smiles_col].astype(str).tolist()
        features = cache.get_or_compute([*smiles, *target_smiles], RICH_2D)
        x_aux = features[: len(smiles)]
        x_target = features[len(smiles) :]
        y_aux = auxiliary[y_col].to_numpy()
        experiment = ExperimentSpec(
            task=teacher.name,
            seed=seed,
            representations=RICH_2D,
            model="lightgbm",
            imbalance=(
                "class_weight" if teacher.task_type == "classification" else "none"
            ),
            device="cpu",
            n_jobs=workers,
            hypothesis="cross-endpoint teacher trained without target molecules",
        )
        model = build_model(experiment, teacher, y_aux)
        model.fit(x_aux, y_aux)
        values = predict(model, x_target, teacher)
        columns.extend(
            [values, rankdata(values, method="average") / len(values)]
        )
        audit.append(
            {
                "teacher": teacher.name,
                "task_type": teacher.task_type,
                "fit_rows": len(auxiliary),
                "excluded_exact_target_molecules": excluded,
                "target_labels_used": False,
                "official_test_loaded": False,
            }
        )
    return np.column_stack(columns), audit


def _target_candidates(
    cache: FeatureCache,
    task,
    seed: int,
    train: pd.DataFrame,
    valid: pd.DataFrame,
    teacher_features: np.ndarray,
    smiles_col: str,
    y_col: str,
    workers: int,
) -> tuple[dict[str, float], dict[str, np.ndarray]]:
    smiles = train[smiles_col].astype(str).tolist() + valid[smiles_col].astype(str).tolist()
    base = cache.get_or_compute(smiles, RICH_2D)
    views = {
        "base": base,
        "teacher": teacher_features,
        "augmented": np.concatenate([base, teacher_features], axis=1),
    }
    y_train = train[y_col].to_numpy()
    y_valid = valid[y_col].to_numpy()
    scores = {}
    predictions = {}
    for name, (model_name, view_name) in CANDIDATES.items():
        params = {"C": 3.0, "gamma": "scale"} if model_name == "svm_rbf" else {}
        experiment = ExperimentSpec(
            task=task.name,
            seed=seed,
            representations=RICH_2D,
            model=model_name,
            imbalance="class_weight",
            device="cpu",
            n_jobs=workers,
            hyperparameters=params,
            hypothesis="inductive cross-endpoint teacher evidence",
        )
        model = build_model(experiment, task, y_train)
        x_all = views[view_name]
        model.fit(x_all[: len(train)], y_train)
        values = predict(model, x_all[len(train) :], task)
        predictions[name] = values
        scores[name] = score(y_valid, values, task.metric)
    return scores, predictions


def summarize(seed_scores: dict[int, dict[str, float]]) -> dict:
    summaries = {
        name: {
            "scores": [seed_scores[seed][name] for seed in sorted(seed_scores)],
            "mean": statistics.mean(
                seed_scores[seed][name] for seed in sorted(seed_scores)
            ),
            "std": statistics.pstdev(
                seed_scores[seed][name] for seed in sorted(seed_scores)
            ),
        }
        for name in CANDIDATES
    }
    loso = {}
    for penalty in (0.0, 0.25, 0.5, 1.0):
        selected = {}
        heldout_scores = []
        for heldout in sorted(seed_scores):
            fit_seeds = [seed for seed in seed_scores if seed != heldout]
            utilities = {
                name: statistics.mean(seed_scores[seed][name] for seed in fit_seeds)
                - penalty
                * statistics.pstdev(seed_scores[seed][name] for seed in fit_seeds)
                for name in CANDIDATES
            }
            winner = max(utilities, key=utilities.get)
            selected[str(heldout)] = winner
            heldout_scores.append(seed_scores[heldout][winner])
        loso[f"lcb_{penalty}"] = {
            "selected": selected,
            "scores": heldout_scores,
            "mean": statistics.mean(heldout_scores),
            "std": statistics.pstdev(heldout_scores),
        }
    return {
        "summaries": summaries,
        "best_fixed_candidate": max(summaries, key=lambda name: summaries[name]["mean"]),
        "loso": loso,
    }


def run_task(args: argparse.Namespace) -> dict:
    task = resolve_task(args.task)
    if task.name not in TEACHERS:
        raise ValueError(f"No teacher set registered for {task.name}")
    group = admet_group(path=str(args.data_dir))
    output = args.output_dir / task.name
    output.mkdir(parents=True, exist_ok=True)
    cache = FeatureCache(args.output_dir / "feature_cache", device="cpu")
    seed_scores = {}
    seed_audits = {}
    for seed in args.seeds:
        started = time.monotonic()
        train, valid = group.get_train_valid_split(
            benchmark=task.name, split_type="default", seed=seed
        )
        smiles_col, y_col = _columns(train)
        target_smiles = (
            train[smiles_col].astype(str).tolist()
            + valid[smiles_col].astype(str).tolist()
        )
        teacher_features, teacher_audit = _teacher_matrix(
            group, cache, target_smiles, task.name, seed, args.workers
        )
        scores, predictions = _target_candidates(
            cache,
            task,
            seed,
            train,
            valid,
            teacher_features,
            smiles_col,
            y_col,
            args.workers,
        )
        seed_dir = output / f"seed_{seed}"
        seed_dir.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(seed_dir / "validation_predictions.npz", **predictions)
        seed_scores[seed] = scores
        seed_audits[str(seed)] = {
            "n_train": len(train),
            "n_valid": len(valid),
            "teacher_features": teacher_features.shape[1],
            "teachers": teacher_audit,
            "scores": scores,
            "elapsed_seconds": time.monotonic() - started,
        }
        partial = {
            "task": task.name,
            "metric": task.metric,
            "test_labels_used": False,
            "official_test_loaded": False,
            "seeds": seed_audits,
        }
        (output / "results.partial.json").write_text(
            json.dumps(partial, indent=2, sort_keys=True), encoding="utf-8"
        )
        print(task.name, seed, json.dumps(scores, sort_keys=True), flush=True)
    report = {
        "task": task.name,
        "metric": task.metric,
        "protocol": "five outer TDC scaffold seeds; fixed auxiliary endpoint set",
        "target_molecule_exclusion": "canonical exact matches removed from every teacher",
        "test_labels_used": False,
        "official_test_loaded": False,
        "seeds": seed_audits,
        **summarize(seed_scores),
    }
    (output / "results.json").write_text(
        json.dumps(report, indent=2, sort_keys=True), encoding="utf-8"
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", required=True, choices=sorted(TEACHERS))
    parser.add_argument("--seeds", default="1,2,3,4,5")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument(
        "--output-dir", type=Path, default=Path("runs/v24-endpoint-transfer")
    )
    args = parser.parse_args()
    args.seeds = [int(value) for value in args.seeds.split(",")]
    report = run_task(args)
    print(
        json.dumps(
            {
                "task": report["task"],
                "best_fixed_candidate": report["best_fixed_candidate"],
                "mean": report["summaries"][report["best_fixed_candidate"]]["mean"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
