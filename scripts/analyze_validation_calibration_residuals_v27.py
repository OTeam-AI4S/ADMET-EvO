"""Post-selection calibration and chemical-space residual audit for 22 tasks.

Only development train/validation scaffold splits and saved validation
predictions are used.  The analysis never loads the official test split and
cannot alter a frozen finalist.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import rdFingerprintGenerator
from rdkit.Chem.Scaffolds import MurckoScaffold
from scipy.stats import rankdata
from tdc.benchmark_group import admet_group

from admet_agent.frozen_recipe import aggregate_predictions, parse_recipe

RDLogger.DisableLog("rdApp.warning")
RDLogger.DisableLog("rdApp.error")

UPGRADED_TASKS = {
    "Bioavailability_Ma",
    "CYP2C9_Substrate_CarbonMangels",
    "DILI",
    "Half_Life_Obach",
    "Lipophilicity_AstraZeneca",
    "hERG",
}
V24_RECIPES = {
    "DILI": "raw(2d_lightgbm@0.33+2d_extra_trees@0.33+finetuned_unimol2@0.33)",
    "hERG": "raw(2d_lightgbm@0.50+finetuned_unimol2@0.50)",
    "Half_Life_Obach": (
        "rank(finetuned_unimol2@0.33+fixed_rank_top3@0.33+"
        "rank_target_xgboost@0.33)"
    ),
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def split_sha256(train: Any, valid: Any) -> str:
    digest = hashlib.sha256()
    for label, frame in (("train", train), ("valid", valid)):
        for drug, target in zip(frame["Drug"], frame["Y"]):
            digest.update(
                f"{label}\t{str(drug)}\t{float(target):.17g}\n".encode("utf-8")
            )
    return digest.hexdigest()


def canonical_identity(smiles: str) -> str:
    mol = Chem.MolFromSmiles(str(smiles))
    return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=False) if mol else ""


def scaffold(smiles: str) -> str:
    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None:
        return ""
    return MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=False)


def nearest_train_similarity(train_smiles: list[str], valid_smiles: list[str]) -> np.ndarray:
    generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    train_fps = []
    for value in train_smiles:
        mol = Chem.MolFromSmiles(str(value))
        train_fps.append(generator.GetFingerprint(mol) if mol else None)
    valid_fps = []
    for value in valid_smiles:
        mol = Chem.MolFromSmiles(str(value))
        valid_fps.append(generator.GetFingerprint(mol) if mol else None)
    usable_train = [value for value in train_fps if value is not None]
    if not usable_train:
        raise ValueError("No valid training fingerprints")
    return np.asarray(
        [
            max(DataStructs.BulkTanimotoSimilarity(value, usable_train))
            if value is not None
            else math.nan
            for value in valid_fps
        ],
        dtype=float,
    )


def adaptive_ece(y_true: np.ndarray, probability: np.ndarray, bins: int = 10) -> float:
    order = np.argsort(probability, kind="stable")
    groups = [group for group in np.array_split(order, min(bins, len(order))) if len(group)]
    return float(
        sum(
            len(group) / len(order)
            * abs(float(np.mean(probability[group])) - float(np.mean(y_true[group])))
            for group in groups
        )
    )


def classification_summary(y_true: np.ndarray, probability: np.ndarray) -> dict[str, float]:
    probability = np.clip(np.asarray(probability, dtype=float), 0.0, 1.0)
    y_true = np.asarray(y_true, dtype=float)
    brier = float(np.mean((probability - y_true) ** 2))
    prevalence = float(np.mean(y_true))
    null_brier = prevalence * (1.0 - prevalence)
    return {
        "n": int(len(y_true)),
        "prevalence": prevalence,
        "mean_probability": float(np.mean(probability)),
        "calibration_in_the_large": float(np.mean(probability) - prevalence),
        "brier": brier,
        "brier_skill_vs_prevalence": 1.0 - brier / null_brier if null_brier > 0 else math.nan,
        "adaptive_ece_10": adaptive_ece(y_true, probability),
    }


def regression_error(y_true: np.ndarray, prediction: np.ndarray, metric: str) -> tuple[np.ndarray, np.ndarray, str]:
    y_true = np.asarray(y_true, dtype=float)
    prediction = np.asarray(prediction, dtype=float)
    if metric == "spearman":
        y_true = rankdata(y_true, method="average") / len(y_true)
        prediction = rankdata(prediction, method="average") / len(prediction)
        unit = "percentile_rank"
    else:
        unit = "native_target"
    residual = prediction - y_true
    return residual, np.abs(residual), unit


def subset_mean(values: np.ndarray, mask: np.ndarray) -> float | None:
    selected = values[mask & np.isfinite(values)]
    return float(np.mean(selected)) if len(selected) else None


def _load_v21_prediction(root: Path, task: str) -> tuple[np.ndarray, Path, str]:
    task_root = root / "runs/leaderboard-v21/tasks" / task
    search_path = task_root / "tasks" / task / "search_seed_1.json"
    search = json.loads(search_path.read_text(encoding="utf-8"))
    run_id = search["finalist_run_id"]
    artifact = task_root / "artifacts" / run_id
    npz_path = artifact / "validation_predictions.npz"
    npy_path = artifact / "validation_predictions.npy"
    if npz_path.is_file():
        with np.load(npz_path) as payload:
            prediction = np.asarray(payload["seed_1"], dtype=float)
        path = npz_path
    elif npy_path.is_file():
        prediction = np.asarray(np.load(npy_path), dtype=float)
        path = npy_path
    else:
        raise FileNotFoundError(f"No validation prediction for {task} finalist {run_id}")
    return prediction, path, f"v2.1 frozen finalist {run_id}"


def _load_recipe_prediction(root: Path, task: str) -> tuple[np.ndarray, Path, str]:
    if task in {"DILI", "hERG"}:
        path = root / "runs/unimol2-pilot-v22" / task / "seed_1/validation_predictions.npz"
    else:
        path = root / "runs/half-life-fusion-ablation/seed_1/validation_predictions.npz"
    with np.load(path) as payload:
        arrays = {name: np.asarray(payload[name], dtype=float) for name in payload.files}
    recipe = V24_RECIPES[task]
    return aggregate_predictions(parse_recipe(recipe), arrays), path, f"v2.4 frozen recipe {recipe}"


def load_prediction(root: Path, task: str) -> tuple[np.ndarray, int, Path, str]:
    if task not in UPGRADED_TASKS:
        prediction, path, lineage = _load_v21_prediction(root, task)
        return prediction, 1, path, lineage
    if task in V24_RECIPES:
        prediction, path, lineage = _load_recipe_prediction(root, task)
        return prediction, 1, path, lineage
    if task == "Lipophilicity_AstraZeneca":
        path = root / (
            "runs/unimol2-scale-ablation-84m-b8/84m/"
            "Lipophilicity_AstraZeneca/lr_0.0001/seed_1/predictions.npz"
        )
        with np.load(path) as payload:
            prediction = np.asarray(payload["validation"], dtype=float)
        return prediction, 1, path, "v2.4 frozen Uni-Mol2 84M validation"
    if task in {"Bioavailability_Ma", "CYP2C9_Substrate_CarbonMangels"}:
        path = root / "runs/v27-maplight-gnn-confirmation/fresh_predictions.npz"
        with np.load(path) as payload:
            prediction = np.asarray(payload[f"{task}_seed_6_maplight_gnn"], dtype=float)
        return prediction, 6, path, "v2.7 fresh-confirmation MapLight+GNN"
    raise KeyError(task)


def analyze_task(
    root: Path,
    group: Any,
    task_row: dict[str, Any],
) -> dict[str, Any]:
    task = task_row["task"]
    metric = task_row["metric"]
    prediction, seed, prediction_path, lineage = load_prediction(root, task)
    train, valid = group.get_train_valid_split(benchmark=task, split_type="default", seed=seed)
    y_true = valid["Y"].to_numpy(dtype=float)
    if prediction.ndim != 1 or len(prediction) != len(valid) or not np.isfinite(prediction).all():
        raise ValueError(f"Prediction alignment failed for {task}")
    train_smiles = train["Drug"].astype(str).tolist()
    valid_smiles = valid["Drug"].astype(str).tolist()
    train_ids = {canonical_identity(value) for value in train_smiles}
    valid_ids = {canonical_identity(value) for value in valid_smiles}
    train_scaffolds = {scaffold(value) for value in train_smiles}
    valid_scaffolds = [scaffold(value) for value in valid_smiles]
    valid_scaffold_counts = Counter(valid_scaffolds)
    similarities = nearest_train_similarity(train_smiles, valid_smiles)
    finite = np.isfinite(similarities)
    low_cut, high_cut = np.quantile(similarities[finite], [0.25, 0.75])
    low = similarities <= low_cut
    high = similarities >= high_cut
    singleton = np.asarray([valid_scaffold_counts[value] == 1 for value in valid_scaffolds])
    repeated = ~singleton
    record: dict[str, Any] = {
        "task": task,
        "metric": metric,
        "analysis_seed": seed,
        "lineage": lineage,
        "n_train": len(train),
        "n_valid": len(valid),
        "prediction_artifact": prediction_path.relative_to(root).as_posix(),
        "prediction_sha256": sha256_file(prediction_path),
        "split_sha256": split_sha256(train, valid),
        "identity_overlap_count": len((train_ids & valid_ids) - {""}),
        "train_valid_scaffold_overlap_count": len((train_scaffolds & set(valid_scaffolds)) - {""}),
        "valid_unique_scaffolds": len(set(valid_scaffolds) - {""}),
        "valid_singleton_scaffold_fraction": float(np.mean(singleton)),
        "nearest_train_tanimoto": {
            "mean": float(np.nanmean(similarities)),
            "median": float(np.nanmedian(similarities)),
            "q25": float(low_cut),
            "q75": float(high_cut),
        },
    }
    if metric in {"auroc", "auprc"}:
        overall = classification_summary(y_true, prediction)
        low_summary = classification_summary(y_true[low], prediction[low])
        high_summary = classification_summary(y_true[high], prediction[high])
        record["analysis_type"] = "classification_calibration"
        record["calibration"] = overall
        record["chemical_space"] = {
            "low_similarity_n": int(low.sum()),
            "high_similarity_n": int(high.sum()),
            "low_similarity_brier": low_summary["brier"],
            "high_similarity_brier": high_summary["brier"],
            "low_minus_high_brier": low_summary["brier"] - high_summary["brier"],
        }
    else:
        residual, error, unit = regression_error(y_true, prediction, metric)
        record["analysis_type"] = "regression_residual"
        record["residual"] = {
            "unit": unit,
            "mean_bias": float(np.mean(residual)),
            "mean_absolute_error": float(np.mean(error)),
            "root_mean_squared_error": float(np.sqrt(np.mean(residual**2))),
            "median_absolute_error": float(np.median(error)),
            "p90_absolute_error": float(np.quantile(error, 0.9)),
        }
        low_error = subset_mean(error, low)
        high_error = subset_mean(error, high)
        record["chemical_space"] = {
            "low_similarity_n": int(low.sum()),
            "high_similarity_n": int(high.sum()),
            "low_similarity_mean_absolute_error": low_error,
            "high_similarity_mean_absolute_error": high_error,
            "low_minus_high_mean_absolute_error": (
                low_error - high_error if low_error is not None and high_error is not None else None
            ),
            "singleton_scaffold_n": int(singleton.sum()),
            "repeated_scaffold_n": int(repeated.sum()),
            "singleton_scaffold_mean_absolute_error": subset_mean(error, singleton),
            "repeated_scaffold_mean_absolute_error": subset_mean(error, repeated),
        }
    return record


def build_report(root: Path, data_dir: Path) -> dict[str, Any]:
    root = root.resolve()
    scorecard_path = root / "reports/goal_scorecard_v27.json"
    scorecard = json.loads(scorecard_path.read_text(encoding="utf-8"))
    group = admet_group(path=str(data_dir))
    tasks = [analyze_task(root, group, row) for row in scorecard["tasks"]]
    classifications = [row for row in tasks if row["analysis_type"] == "classification_calibration"]
    regressions = [row for row in tasks if row["analysis_type"] == "regression_residual"]
    class_deltas = [row["chemical_space"]["low_minus_high_brier"] for row in classifications]
    reg_deltas = [row["chemical_space"]["low_minus_high_mean_absolute_error"] for row in regressions]
    return {
        "version": "v27-validation-calibration-residuals-1",
        "analysis_role": "post_selection_descriptive_validation_audit",
        "selection_or_freeze_changed": False,
        "official_test_loaded": False,
        "official_results_used": False,
        "leaderboard_used_for_analysis": False,
        "split_protocol": (
            "one saved default scaffold validation split per task; seed 1 for v2.1/v2.4 "
            "lineages and preregistered fresh seed 6 for v2.7 MapLight+GNN"
        ),
        "limitations": (
            "Finalists were selected using development validation evidence, so this is a "
            "post-selection diagnostic, not an unbiased performance estimate. Spearman "
            "residuals are computed in percentile-rank space."
        ),
        "task_count": len(tasks),
        "classification_task_count": len(classifications),
        "regression_task_count": len(regressions),
        "aggregate": {
            "mean_classification_brier": statistics.mean(row["calibration"]["brier"] for row in classifications),
            "mean_classification_adaptive_ece": statistics.mean(row["calibration"]["adaptive_ece_10"] for row in classifications),
            "classification_low_similarity_brier_worse_tasks": sum(value > 0 for value in class_deltas),
            "classification_low_similarity_brier_better_tasks": sum(value < 0 for value in class_deltas),
            "regression_low_similarity_error_worse_tasks": sum(value > 0 for value in reg_deltas if value is not None),
            "regression_low_similarity_error_better_tasks": sum(value < 0 for value in reg_deltas if value is not None),
        },
        "scorecard": {
            "path": scorecard_path.relative_to(root).as_posix(),
            "sha256": sha256_file(scorecard_path),
        },
        "tasks": tasks,
    }


def markdown(report: dict[str, Any]) -> str:
    lines = [
        "# v2.7 validation calibration and chemical-space residual audit",
        "",
        report["split_protocol"] + ". Official test data and official results were not loaded.",
        "",
        "## Classification calibration",
        "",
        "| Task | Seed | Brier | Brier skill | adaptive ECE | mean p - prevalence | low-high similarity Brier |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in report["tasks"]:
        if row["analysis_type"] != "classification_calibration":
            continue
        cal = row["calibration"]
        lines.append(
            f"| {row['task']} | {row['analysis_seed']} | {cal['brier']:.4f} | "
            f"{cal['brier_skill_vs_prevalence']:.3f} | {cal['adaptive_ece_10']:.4f} | "
            f"{cal['calibration_in_the_large']:.4f} | "
            f"{row['chemical_space']['low_minus_high_brier']:.4f} |"
        )
    lines.extend(
        [
            "",
            "## Regression residuals",
            "",
            "| Task | Metric | Error unit | Mean absolute error | Bias | p90 error | low-high similarity error | singleton-repeat scaffold error |",
            "|---|---|---|---:|---:|---:|---:|---:|",
        ]
    )
    for row in report["tasks"]:
        if row["analysis_type"] != "regression_residual":
            continue
        residual = row["residual"]
        chemical = row["chemical_space"]
        singleton = chemical["singleton_scaffold_mean_absolute_error"]
        repeated = chemical["repeated_scaffold_mean_absolute_error"]
        scaffold_delta = singleton - repeated if singleton is not None and repeated is not None else None
        lines.append(
            f"| {row['task']} | {row['metric']} | {residual['unit']} | "
            f"{residual['mean_absolute_error']:.4f} | {residual['mean_bias']:.4f} | "
            f"{residual['p90_absolute_error']:.4f} | "
            f"{chemical['low_minus_high_mean_absolute_error']:.4f} | "
            f"{scaffold_delta:.4f} |" if scaffold_delta is not None else
            f"| {row['task']} | {row['metric']} | {residual['unit']} | "
            f"{residual['mean_absolute_error']:.4f} | {residual['mean_bias']:.4f} | "
            f"{residual['p90_absolute_error']:.4f} | "
            f"{chemical['low_minus_high_mean_absolute_error']:.4f} | NA |"
        )
    aggregate = report["aggregate"]
    lines.extend(
        [
            "",
            "## Aggregate observations",
            "",
            f"- Mean Brier across classification tasks: {aggregate['mean_classification_brier']:.4f}.",
            f"- Mean adaptive ECE across classification tasks: {aggregate['mean_classification_adaptive_ece']:.4f}.",
            (
                "- Low-similarity molecules had worse Brier in "
                f"{aggregate['classification_low_similarity_brier_worse_tasks']} classification tasks "
                f"and better Brier in {aggregate['classification_low_similarity_brier_better_tasks']}."
            ),
            (
                "- Low-similarity molecules had worse residual error in "
                f"{aggregate['regression_low_similarity_error_worse_tasks']} regression tasks "
                f"and better error in {aggregate['regression_low_similarity_error_better_tasks']}."
            ),
            "",
            report["limitations"],
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument(
        "--output", type=Path, default=Path("reports/validation_calibration_residuals_v27")
    )
    args = parser.parse_args()
    report = build_report(args.root, args.data_dir)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.with_suffix(".json").write_text(
        json.dumps(report, indent=2, sort_keys=True), encoding="utf-8"
    )
    args.output.with_suffix(".md").write_text(markdown(report), encoding="utf-8")
    print(json.dumps({"task_count": report["task_count"], **report["aggregate"]}, sort_keys=True))


if __name__ == "__main__":
    main()
