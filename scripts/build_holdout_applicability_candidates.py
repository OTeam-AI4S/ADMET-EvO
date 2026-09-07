#!/usr/bin/env python3
"""Build three candidate applicability-domain figures from frozen official test predictions.

The analysis is strictly post-freeze. It loads the fixed TDC official test set only
to align labels/structures with already-saved predictions and never changes model
selection, recipes, or freeze artifacts.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")

import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import rdFingerprintGenerator
from rdkit.Chem.Scaffolds import MurckoScaffold
from scipy.stats import rankdata, spearmanr
from sklearn.metrics import average_precision_score, mean_absolute_error, roc_auc_score
from tdc.benchmark_group import admet_group


ROOT = Path(__file__).resolve().parents[1]
SCORECARD = ROOT / "reports" / "goal_scorecard_v27.json"
OUTPUT_DIR = ROOT / "outputs" / "figures" / "holdout_applicability_candidates"
SOURCE_DATA = OUTPUT_DIR / "holdout_applicability_source.csv"
SUMMARY_JSON = OUTPUT_DIR / "holdout_applicability_summary.json"
FONT_REGULAR = ROOT / "assets" / "fonts" / "msyh.ttc"
FONT_BOLD = ROOT / "assets" / "fonts" / "msyhbd.ttc"

MM_PER_INCH = 25.4
WIDTH_MM = 183
HEIGHT_MM = 94
COMBINED_WIDTH_MM = 280
COMBINED_HEIGHT_MM = 88
CORAL = "#C95F55"
TEAL = "#178778"
PALE_YELLOW = "#FFF2C9"
PALE_GREY = "#EEF2F5"
GRID = "#D7DEE4"
TEXT = "#26333E"
MUTED = "#657483"
NEUTRAL = "#AEBCC6"
CLASSIFICATION = "Classification"
REGRESSION = "Regression"
SEEDS = (1, 2, 3, 4, 5)
COVERAGES = np.arange(0.25, 1.001, 0.05)
WHITE_TEXT_SCALE = 5.0
WHITE_TITLE_PT = 10.5 * WHITE_TEXT_SCALE
WHITE_LABEL_PT = 8.0 * WHITE_TEXT_SCALE
WHITE_TICK_PT = 7.0 * WHITE_TEXT_SCALE
WHITE_DIRECT_LABEL_PT = 7.3 * WHITE_TEXT_SCALE
WHITE_FOOTER_PT = 6.3 * WHITE_TEXT_SCALE
WHITE_CANVAS_SCALE = 3.0
WHITE_LINE_SCALE = 5.0
GAIN_TEXT_SCALE = 1.2
COVERAGE_TEXT_SCALE = 0.65


def configure_style() -> None:
    for font_path in (FONT_REGULAR, FONT_BOLD):
        if not font_path.is_file():
            raise FileNotFoundError(font_path)
        fm.fontManager.addfont(font_path)
    resolved = Path(fm.findfont(fm.FontProperties(family="Microsoft YaHei"))).resolve()
    if resolved not in {FONT_REGULAR.resolve(), FONT_BOLD.resolve()}:
        raise RuntimeError(f"Microsoft YaHei resolved unexpectedly: {resolved}")
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Microsoft YaHei", "Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 7.5,
            "axes.titlesize": 10.5,
            "axes.labelsize": 8,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 0.8,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.facecolor": "white",
        }
    )


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_identity(smiles: str) -> str:
    molecule = Chem.MolFromSmiles(str(smiles))
    return Chem.MolToSmiles(molecule, canonical=True, isomericSmiles=False) if molecule else ""


def scaffold(smiles: str) -> str:
    molecule = Chem.MolFromSmiles(str(smiles))
    if molecule is None:
        return ""
    return MurckoScaffold.MurckoScaffoldSmiles(mol=molecule, includeChirality=False)


def percentile_rank(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    if len(values) == 1:
        return np.asarray([0.5])
    return (rankdata(values, method="average") - 1.0) / (len(values) - 1.0)


def official_metric(target: np.ndarray, prediction: np.ndarray, metric: str) -> float:
    if metric == "auroc":
        return float(roc_auc_score(target, prediction))
    if metric == "auprc":
        return float(average_precision_score(target, prediction))
    if metric == "mae":
        return float(mean_absolute_error(target, prediction))
    if metric == "spearman":
        return float(spearmanr(target, prediction).statistic)
    raise KeyError(metric)


def point_loss(target: np.ndarray, prediction: np.ndarray, metric: str) -> np.ndarray:
    if metric in {"auroc", "auprc"}:
        if np.min(prediction) < -1e-8 or np.max(prediction) > 1 + 1e-8:
            raise ValueError("Classification predictions must be probabilities")
        return (np.clip(prediction, 0, 1) - target) ** 2
    if metric == "spearman":
        return np.abs(percentile_rank(prediction) - percentile_rank(target))
    if metric == "mae":
        return np.abs(prediction - target)
    raise KeyError(metric)


def nearest_similarity(train_smiles: list[str], test_smiles: list[str]) -> tuple[np.ndarray, int, int]:
    generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    train_fingerprints = []
    invalid_train = 0
    for value in train_smiles:
        molecule = Chem.MolFromSmiles(str(value))
        if molecule is None:
            invalid_train += 1
            continue
        train_fingerprints.append(generator.GetFingerprint(molecule))
    if not train_fingerprints:
        raise ValueError("No fingerprintable training molecules")
    similarities = []
    invalid_test = 0
    for value in test_smiles:
        molecule = Chem.MolFromSmiles(str(value))
        if molecule is None:
            similarities.append(math.nan)
            invalid_test += 1
            continue
        fingerprint = generator.GetFingerprint(molecule)
        similarities.append(max(DataStructs.BulkTanimotoSimilarity(fingerprint, train_fingerprints)))
    return np.asarray(similarities, dtype=float), invalid_train, invalid_test


def build_source_data() -> tuple[pd.DataFrame, dict]:
    scorecard = json.loads(SCORECARD.read_text(encoding="utf-8"))
    if not scorecard["official_results_complete"] or len(scorecard["tasks"]) != 22:
        raise ValueError("The v2.7 official scorecard is not complete for 22 tasks")
    group = admet_group(path=str(ROOT / "data"))
    records: list[dict] = []
    task_audit: list[dict] = []
    RDLogger.DisableLog("rdApp.warning")
    RDLogger.DisableLog("rdApp.error")

    for task_record in scorecard["tasks"]:
        task = task_record["task"]
        metric = task_record["metric"]
        task_type = CLASSIFICATION if metric in {"auroc", "auprc"} else REGRESSION
        payload = group.get(task)
        train = payload["train_val"].reset_index(drop=True)
        test = payload["test"].reset_index(drop=True)
        target = test["Y"].to_numpy(dtype=float)
        result_path = ROOT / scorecard["provenance"][task]["source"]
        final_dir = result_path.parent
        seed_predictions = []
        reproduced_scores = []
        prediction_hashes = {}
        for seed in SEEDS:
            prediction_path = final_dir / f"predictions_seed_{seed}.npz"
            if not prediction_path.is_file():
                raise FileNotFoundError(prediction_path)
            with np.load(prediction_path) as archive:
                if len(archive.files) != 1:
                    raise ValueError(f"Unexpected prediction keys for {task}, seed {seed}")
                prediction = np.asarray(archive[archive.files[0]], dtype=float)
            if prediction.shape != target.shape or not np.isfinite(prediction).all():
                raise ValueError(f"Official prediction alignment failed for {task}, seed {seed}")
            seed_predictions.append(prediction)
            reproduced_scores.append(official_metric(target, prediction, metric))
            prediction_hashes[str(seed)] = sha256(prediction_path)

        reproduced_mean = float(np.mean(reproduced_scores))
        if abs(reproduced_mean - float(task_record["score"])) > 0.0011:
            raise ValueError(
                f"Official metric did not reproduce for {task}: {reproduced_mean} vs {task_record['score']}"
            )
        seed_losses = np.vstack(
            [point_loss(target, prediction, metric) for prediction in seed_predictions]
        )
        mean_loss = np.mean(seed_losses, axis=0)
        similarity, invalid_train, invalid_test = nearest_similarity(
            train["Drug"].astype(str).tolist(), test["Drug"].astype(str).tolist()
        )
        fingerprintable = np.isfinite(similarity)
        if not fingerprintable.any():
            raise ValueError(f"No fingerprintable official test rows for {task}")
        reference_loss = float(np.mean(mean_loss[fingerprintable]))
        normalized_loss = mean_loss / reference_loss
        ranked_similarity = percentile_rank(similarity[fingerprintable])
        quartile = np.minimum((ranked_similarity * 4).astype(int), 3) + 1

        train_ids = {canonical_identity(value) for value in train["Drug"]}
        test_ids = {canonical_identity(value) for value in test["Drug"]}
        train_scaffolds = {scaffold(value) for value in train["Drug"]}
        test_scaffolds = {scaffold(value) for value in test["Drug"]}
        task_audit.append(
            {
                "task": task,
                "metric": metric,
                "task_type": task_type,
                "n_train_val": len(train),
                "n_official_test": len(test),
                "n_similarity_analyzed": int(fingerprintable.sum()),
                "invalid_train_smiles_excluded_from_similarity_reference": invalid_train,
                "invalid_test_smiles_excluded_from_similarity_analysis": invalid_test,
                "canonical_identity_overlap_count": len((train_ids & test_ids) - {""}),
                "scaffold_overlap_count": len((train_scaffolds & test_scaffolds) - {""}),
                "official_score_reported": float(task_record["score"]),
                "official_score_reproduced": reproduced_mean,
                "official_score_seed_values": reproduced_scores,
                "result_path": result_path.relative_to(ROOT).as_posix(),
                "result_sha256": sha256(result_path),
                "prediction_sha256": prediction_hashes,
            }
        )

        analyzed_indexes = np.flatnonzero(fingerprintable)
        for local_index, test_index in enumerate(analyzed_indexes):
            records.append(
                {
                    "task": task,
                    "task_type": task_type,
                    "metric": metric,
                    "test_index": int(test_index),
                    "nearest_train_tanimoto": float(similarity[test_index]),
                    "similarity_quartile": int(quartile[local_index]),
                    "mean_point_loss_across_5_seeds": float(mean_loss[test_index]),
                    "task_normalized_loss": float(normalized_loss[test_index]),
                }
            )

    frame = pd.DataFrame.from_records(records)
    audit = {
        "version": "official-holdout-applicability-v1",
        "analysis_role": "post_freeze_official_holdout_diagnostic",
        "official_test_loaded": True,
        "selection_or_freeze_changed": False,
        "task_count": len(task_audit),
        "seed_count": len(SEEDS),
        "seeds": list(SEEDS),
        "similarity": "maximum Morgan radius-2, 2,048-bit Tanimoto to full train_val",
        "classification_loss": "per-molecule Brier loss, averaged across five frozen seeds",
        "regression_loss": "absolute residual for MAE tasks and absolute percentile-rank residual for Spearman tasks, averaged across five frozen seeds",
        "normalization": "each task divided by its mean loss over fingerprintable official-test molecules",
        "invalid_smiles_policy": "unparseable test molecules are retained in official metric reproduction but excluded from similarity analysis because similarity is undefined",
        "task_audit": task_audit,
    }
    return frame, audit


def endpoint_summaries(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    coverage_records = []
    quartile_records = []
    gain_records = []
    for (task, task_type), task_frame in frame.groupby(["task", "task_type"], sort=False):
        ordered = task_frame.sort_values("nearest_train_tanimoto", ascending=False, kind="stable")
        values = ordered["task_normalized_loss"].to_numpy(dtype=float)
        for coverage in COVERAGES:
            count = max(1, int(math.ceil(len(values) * coverage)))
            coverage_records.append(
                {
                    "task": task,
                    "task_type": task_type,
                    "coverage": float(coverage),
                    "normalized_error": float(np.mean(values[:count])),
                }
            )
        for quartile, subset in task_frame.groupby("similarity_quartile"):
            quartile_records.append(
                {
                    "task": task,
                    "task_type": task_type,
                    "similarity_quartile": int(quartile),
                    "normalized_error": float(subset["task_normalized_loss"].mean()),
                }
            )
        trusted = task_frame.loc[task_frame["similarity_quartile"] == 4, "task_normalized_loss"]
        gain_records.append(
            {
                "task": task,
                "task_type": task_type,
                "trusted_region_error_reduction_pct": float(100.0 * (1.0 - trusted.mean())),
            }
        )
    return (
        pd.DataFrame.from_records(coverage_records),
        pd.DataFrame.from_records(quartile_records),
        pd.DataFrame.from_records(gain_records),
    )


def bootstrap_median_interval(values: np.ndarray, seed: int) -> tuple[float, float, float]:
    values = np.asarray(values, dtype=float)
    generator = np.random.RandomState(seed)
    indexes = generator.randint(0, len(values), size=(10000, len(values)))
    medians = np.median(values[indexes], axis=1)
    return (
        float(np.median(values)),
        float(np.quantile(medians, 0.025)),
        float(np.quantile(medians, 0.975)),
    )


def style_axis(axis: plt.Axes) -> None:
    axis.spines["left"].set_color(TEXT)
    axis.spines["bottom"].set_color(TEXT)
    axis.spines["left"].set_linewidth(0.85)
    axis.spines["bottom"].set_linewidth(0.85)
    axis.tick_params(colors=MUTED, length=3, width=0.7)
    axis.set_axisbelow(True)
    axis.grid(axis="y", color=GRID, linewidth=0.7, alpha=0.8)


def add_footer(figure: plt.Figure) -> None:
    figure.text(
        0.99,
        0.018,
        "Post-freeze official hold-out analysis · 22 endpoints · 5 frozen seeds",
        ha="right",
        va="bottom",
        fontsize=6.3,
        color=MUTED,
    )


def save_figure(figure: plt.Figure, name: str) -> None:
    prefix = OUTPUT_DIR / name
    figure.savefig(prefix.with_suffix(".png"), dpi=300)
    figure.savefig(prefix.with_suffix(".pdf"))
    figure.savefig(prefix.with_suffix(".svg"))
    figure.savefig(prefix.with_suffix(".tiff"), dpi=600, pil_kwargs={"compression": "tiff_lzw"})
    plt.close(figure)


def plot_coverage(coverage: pd.DataFrame) -> None:
    figure, axis = plt.subplots(figsize=(WIDTH_MM / MM_PER_INCH, HEIGHT_MM / MM_PER_INCH))
    figure.subplots_adjust(left=0.105, right=0.975, bottom=0.20, top=0.84)
    axis.axhspan(0, 1, color=PALE_YELLOW, alpha=0.36, linewidth=0, zorder=0)
    axis.axhline(1, color=TEXT, linewidth=0.9, linestyle=(0, (4, 3)), zorder=1)
    colors = {CLASSIFICATION: CORAL, REGRESSION: TEAL}
    for type_index, task_type in enumerate((CLASSIFICATION, REGRESSION)):
        medians, lows, highs = [], [], []
        for coverage_value in COVERAGES:
            values = coverage.loc[
                (coverage["task_type"] == task_type)
                & np.isclose(coverage["coverage"], coverage_value),
                "normalized_error",
            ].to_numpy(dtype=float)
            median, low, high = bootstrap_median_interval(values, seed=4100 + type_index * 100 + int(coverage_value * 100))
            medians.append(median)
            lows.append(low)
            highs.append(high)
        x = COVERAGES * 100
        axis.fill_between(x, lows, highs, color=colors[task_type], alpha=0.14, linewidth=0)
        axis.plot(x, medians, color=colors[task_type], linewidth=2.25, zorder=3)
        axis.scatter(x[::3], np.asarray(medians)[::3], s=20, color=colors[task_type], edgecolor="white", linewidth=0.6, zorder=4)
    axis.set_xlim(24, 101)
    axis.set_ylim(0.35, 1.13)
    axis.set_xticks([25, 40, 60, 80, 100])
    axis.set_yticks([0.4, 0.6, 0.8, 1.0], ["40", "60", "80", "100"])
    axis.set_xlabel("Hold-out coverage retained (%)", labelpad=8, color=TEXT)
    axis.set_ylabel("Normalized prediction error (%)", labelpad=8, color=TEXT)
    axis.set_title("Confidence filtering concentrates reliable predictions", loc="left", fontweight="bold", color=TEXT, pad=12)
    axis.text(0.73, 1.045, "Classification", transform=axis.transAxes, ha="left", va="center", fontsize=7.4, color=CORAL, fontweight="bold")
    axis.text(0.87, 1.045, "Regression", transform=axis.transAxes, ha="left", va="center", fontsize=7.4, color=TEAL, fontweight="bold")
    style_axis(axis)
    add_footer(figure)
    save_figure(figure, "scheme_1_coverage_reliability")


def plot_dose_response(quartiles: pd.DataFrame) -> None:
    figure, axis = plt.subplots(figsize=(WIDTH_MM / MM_PER_INCH, HEIGHT_MM / MM_PER_INCH))
    figure.subplots_adjust(left=0.105, right=0.975, bottom=0.20, top=0.84)
    colors = {CLASSIFICATION: CORAL, REGRESSION: TEAL}
    x = np.arange(1, 5)
    for type_index, task_type in enumerate((CLASSIFICATION, REGRESSION)):
        medians, lows, highs = [], [], []
        for quartile in x:
            values = quartiles.loc[
                (quartiles["task_type"] == task_type)
                & (quartiles["similarity_quartile"] == quartile),
                "normalized_error",
            ].to_numpy(dtype=float)
            median, low, high = bootstrap_median_interval(values, seed=5200 + type_index * 100 + quartile)
            medians.append(median)
            lows.append(low)
            highs.append(high)
        axis.fill_between(x, lows, highs, color=colors[task_type], alpha=0.14, linewidth=0)
        axis.plot(x, medians, color=colors[task_type], linewidth=2.25, zorder=3)
        axis.scatter(x, medians, s=42, color=colors[task_type], edgecolor="white", linewidth=0.8, zorder=4)
    axis.axhline(1, color=TEXT, linewidth=0.9, linestyle=(0, (4, 3)), zorder=1)
    axis.set_xlim(0.75, 4.25)
    axis.set_ylim(0.35, 1.55)
    axis.set_xticks(x, ["Q1\nLowest", "Q2", "Q3", "Q4\nHighest"])
    axis.set_yticks([0.5, 0.75, 1.0, 1.25, 1.5], ["50", "75", "100", "125", "150"])
    axis.set_xlabel("Nearest-training chemical similarity", labelpad=8, color=TEXT)
    axis.set_ylabel("Normalized prediction error (%)", labelpad=8, color=TEXT)
    axis.set_title("Prediction error falls toward familiar chemical space", loc="left", fontweight="bold", color=TEXT, pad=12)
    axis.text(0.73, 1.045, "Classification", transform=axis.transAxes, ha="left", va="center", fontsize=7.4, color=CORAL, fontweight="bold")
    axis.text(0.87, 1.045, "Regression", transform=axis.transAxes, ha="left", va="center", fontsize=7.4, color=TEAL, fontweight="bold")
    style_axis(axis)
    add_footer(figure)
    save_figure(figure, "scheme_2_similarity_error_dose_response")


def plot_gain_distribution(gains: pd.DataFrame) -> None:
    figure, axis = plt.subplots(figsize=(WIDTH_MM / MM_PER_INCH, HEIGHT_MM / MM_PER_INCH))
    figure.subplots_adjust(left=0.15, right=0.975, bottom=0.20, top=0.84)
    axis.axvspan(0, 75, color=PALE_YELLOW, alpha=0.40, linewidth=0, zorder=0)
    axis.axvspan(-30, 0, color=PALE_GREY, alpha=0.72, linewidth=0, zorder=0)
    axis.axvline(0, color=TEXT, linewidth=0.9, zorder=1)
    colors = {CLASSIFICATION: CORAL, REGRESSION: TEAL}
    positions = {CLASSIFICATION: 1.0, REGRESSION: 0.0}
    for task_type in (CLASSIFICATION, REGRESSION):
        values = gains.loc[
            gains["task_type"] == task_type, "trusted_region_error_reduction_pct"
        ].to_numpy(dtype=float)
        position = positions[task_type]
        violin = axis.violinplot(
            values,
            positions=[position],
            vert=False,
            widths=0.50,
            showmeans=False,
            showmedians=False,
            showextrema=False,
            bw_method=0.45,
        )
        body = violin["bodies"][0]
        body.set_facecolor(colors[task_type])
        body.set_edgecolor("none")
        body.set_alpha(0.14)
        jitter = np.linspace(-0.105, 0.105, len(values))
        point_colors = [colors[task_type] if value > 0 else NEUTRAL for value in values]
        axis.scatter(values, position + jitter, s=35, color=point_colors, edgecolor="white", linewidth=0.65, zorder=3)
        median = float(np.median(values))
        q25, q75 = np.quantile(values, [0.25, 0.75])
        axis.plot([q25, q75], [position, position], color=TEXT, linewidth=2.3, zorder=4)
        axis.scatter([median], [position], marker="D", s=62, color=colors[task_type], edgecolor="white", linewidth=0.9, zorder=5)
        improved = int(np.sum(values > 0))
        axis.text(
            72,
            position,
            f"{improved}/{len(values)}",
            ha="right",
            va="center",
            fontsize=8,
            fontweight="bold",
            color=colors[task_type],
        )
    axis.set_xlim(-30, 75)
    axis.set_ylim(-0.55, 1.55)
    axis.set_yticks([0, 1], [REGRESSION, CLASSIFICATION])
    axis.set_xticks([-25, 0, 25, 50, 75])
    axis.set_xlabel("Error reduction in the highest-similarity hold-out quartile (%)", labelpad=8, color=TEXT)
    axis.set_title("Trusted-region gains generalize across endpoints", loc="left", fontweight="bold", color=TEXT, pad=12)
    axis.grid(axis="x", color=GRID, linewidth=0.7, alpha=0.8)
    axis.grid(axis="y", visible=False)
    axis.spines["left"].set_visible(False)
    axis.spines["bottom"].set_color(TEXT)
    axis.tick_params(axis="y", length=0, pad=9, colors=TEXT)
    axis.tick_params(axis="x", colors=MUTED, length=3, width=0.7)
    add_footer(figure)
    save_figure(figure, "scheme_3_trusted_region_gain_distribution")


def panel_label(axis: plt.Axes, label: str) -> None:
    axis.text(
        -0.075,
        1.075,
        label,
        transform=axis.transAxes,
        ha="right",
        va="bottom",
        fontsize=9.2,
        fontweight="bold",
        color=TEXT,
        clip_on=False,
    )


def plot_combined(coverage: pd.DataFrame, gains: pd.DataFrame) -> None:
    figure, (risk_axis, gain_axis) = plt.subplots(
        1,
        2,
        figsize=(COMBINED_WIDTH_MM / MM_PER_INCH, COMBINED_HEIGHT_MM / MM_PER_INCH),
        gridspec_kw={"width_ratios": [1.14, 1.0]},
    )
    figure.subplots_adjust(left=0.065, right=0.985, bottom=0.22, top=0.82, wspace=0.24)
    colors = {CLASSIFICATION: CORAL, REGRESSION: TEAL}

    # Panel a: selective risk versus retained coverage.
    risk_axis.axhspan(0.54, 1.0, color="#FBF7EA", alpha=0.72, linewidth=0, zorder=0)
    risk_axis.axhline(1.0, color=TEXT, linewidth=0.9, linestyle=(0, (4, 3)), zorder=1)
    for type_index, task_type in enumerate((CLASSIFICATION, REGRESSION)):
        medians, lows, highs = [], [], []
        for coverage_value in COVERAGES:
            values = coverage.loc[
                (coverage["task_type"] == task_type)
                & np.isclose(coverage["coverage"], coverage_value),
                "normalized_error",
            ].to_numpy(dtype=float)
            median, low, high = bootstrap_median_interval(
                values, seed=7100 + type_index * 100 + int(coverage_value * 100)
            )
            medians.append(median)
            lows.append(low)
            highs.append(high)
        x = COVERAGES * 100
        risk_axis.fill_between(x, lows, highs, color=colors[task_type], alpha=0.12, linewidth=0)
        risk_axis.plot(x, medians, color=colors[task_type], linewidth=2.25, zorder=3)
        risk_axis.scatter(x[::3], np.asarray(medians)[::3], s=19, color=colors[task_type], edgecolor="white", linewidth=0.6, zorder=4)
    risk_axis.set_xlim(24, 101)
    risk_axis.set_ylim(0.54, 1.075)
    risk_axis.set_xticks([25, 40, 60, 80, 100])
    risk_axis.set_yticks([0.6, 0.8, 1.0], ["60", "80", "100"])
    risk_axis.set_xlabel("Hold-out coverage retained (%)", labelpad=7, color=TEXT)
    risk_axis.set_ylabel("Normalized prediction error (%)", labelpad=7, color=TEXT)
    risk_axis.set_title("Confidence filtering concentrates reliable predictions", loc="left", fontweight="bold", color=TEXT, pad=10)
    risk_axis.text(0.70, 0.965, "Classification", transform=risk_axis.transAxes, ha="left", va="center", fontsize=7.2, color=CORAL, fontweight="bold")
    risk_axis.text(0.87, 0.965, "Regression", transform=risk_axis.transAxes, ha="left", va="center", fontsize=7.2, color=TEAL, fontweight="bold")
    style_axis(risk_axis)
    panel_label(risk_axis, "a")

    # Panel b: endpoint-level gains in the highest-similarity quartile.
    gain_axis.axvspan(-10, 0, color="#F0F3F5", linewidth=0, zorder=0)
    gain_axis.axvspan(0, 70, color="#FBF7EA", linewidth=0, zorder=0)
    gain_axis.axvline(0, color=TEXT, linewidth=0.9, zorder=1)
    positions = {CLASSIFICATION: 1.0, REGRESSION: 0.0}
    for task_type in (CLASSIFICATION, REGRESSION):
        values = gains.loc[
            gains["task_type"] == task_type, "trusted_region_error_reduction_pct"
        ].to_numpy(dtype=float)
        position = positions[task_type]
        violin = gain_axis.violinplot(
            values,
            positions=[position],
            vert=False,
            widths=0.48,
            showmeans=False,
            showmedians=False,
            showextrema=False,
            bw_method=0.45,
        )
        body = violin["bodies"][0]
        body.set_facecolor(colors[task_type])
        body.set_edgecolor("none")
        body.set_alpha(0.12)
        jitter = np.linspace(-0.10, 0.10, len(values))
        point_colors = [colors[task_type] if value > 0 else NEUTRAL for value in values]
        gain_axis.scatter(values, position + jitter, s=31, color=point_colors, edgecolor="white", linewidth=0.65, zorder=3)
        median = float(np.median(values))
        q25, q75 = np.quantile(values, [0.25, 0.75])
        gain_axis.plot([q25, q75], [position, position], color=TEXT, linewidth=2.15, zorder=4)
        gain_axis.scatter([median], [position], marker="D", s=58, color=colors[task_type], edgecolor="white", linewidth=0.9, zorder=5)
        improved = int(np.sum(values > 0))
        gain_axis.text(67.5, position, f"{improved}/{len(values)}", ha="right", va="center", fontsize=8.2, fontweight="bold", color=colors[task_type])
    gain_axis.set_xlim(-10, 70)
    gain_axis.set_ylim(-0.48, 1.48)
    gain_axis.set_yticks([0, 1], [REGRESSION, CLASSIFICATION])
    gain_axis.set_xticks([0, 20, 40, 60])
    gain_axis.set_xlabel("Error reduction at 25% trusted coverage (%)", labelpad=7, color=TEXT)
    gain_axis.set_title("Trusted-region gains across 21 of 22 endpoints", loc="left", fontweight="bold", color=TEXT, pad=10)
    gain_axis.grid(axis="x", color=GRID, linewidth=0.7, alpha=0.8)
    gain_axis.grid(axis="y", visible=False)
    gain_axis.spines["left"].set_visible(False)
    gain_axis.spines["bottom"].set_color(TEXT)
    gain_axis.spines["bottom"].set_linewidth(0.85)
    gain_axis.tick_params(axis="y", length=0, pad=8, colors=TEXT)
    gain_axis.tick_params(axis="x", colors=MUTED, length=3, width=0.7)
    panel_label(gain_axis, "b")

    figure.text(
        0.985,
        0.025,
        "Post-freeze official hold-out · 22 endpoints · 5 frozen seeds",
        ha="right",
        va="bottom",
        fontsize=6.3,
        color=MUTED,
    )
    save_figure(figure, "combined_coverage_and_trusted_region_gain")


def plot_split_combined(
    coverage: pd.DataFrame,
    gains: pd.DataFrame,
    task_type: str,
) -> None:
    color = CORAL if task_type == CLASSIFICATION else TEAL
    type_slug = task_type.lower()
    task_count = int(gains["task_type"].eq(task_type).sum())
    figure, (risk_axis, gain_axis) = plt.subplots(
        1,
        2,
        figsize=(260 / MM_PER_INCH, 78 / MM_PER_INCH),
        gridspec_kw={"width_ratios": [1.25, 0.92]},
    )
    figure.subplots_adjust(left=0.075, right=0.985, bottom=0.23, top=0.79, wspace=0.27)

    # Panel a: one task type, one selective-risk curve.
    risk_axis.axhspan(0.55, 1.0, color="#FBF7EA", alpha=0.72, linewidth=0, zorder=0)
    risk_axis.axhline(1.0, color=TEXT, linewidth=0.9, linestyle=(0, (4, 3)), zorder=1)
    medians, lows, highs = [], [], []
    type_index = 0 if task_type == CLASSIFICATION else 1
    for coverage_value in COVERAGES:
        values = coverage.loc[
            (coverage["task_type"] == task_type)
            & np.isclose(coverage["coverage"], coverage_value),
            "normalized_error",
        ].to_numpy(dtype=float)
        median, low, high = bootstrap_median_interval(
            values, seed=8100 + type_index * 100 + int(coverage_value * 100)
        )
        medians.append(median)
        lows.append(low)
        highs.append(high)
    x = COVERAGES * 100
    risk_axis.fill_between(x, lows, highs, color=color, alpha=0.14, linewidth=0)
    risk_axis.plot(x, medians, color=color, linewidth=2.35, zorder=3)
    risk_axis.scatter(
        x[::3],
        np.asarray(medians)[::3],
        s=24,
        color=color,
        edgecolor="white",
        linewidth=0.7,
        zorder=4,
    )
    risk_axis.set_xlim(24, 101)
    risk_axis.set_ylim(0.55, 1.075)
    risk_axis.set_xticks([25, 40, 60, 80, 100])
    risk_axis.set_yticks([0.6, 0.8, 1.0], ["60", "80", "100"])
    risk_axis.set_xlabel("Hold-out coverage retained (%)", labelpad=7, color=TEXT)
    risk_axis.set_ylabel("Normalized prediction error (%)", labelpad=7, color=TEXT)
    risk_axis.set_title("Confidence filtering concentrates reliable predictions", loc="left", fontweight="bold", color=TEXT, pad=10)
    style_axis(risk_axis)
    panel_label(risk_axis, "a")

    # Panel b: endpoint distribution for the same task type.
    values = gains.loc[
        gains["task_type"] == task_type, "trusted_region_error_reduction_pct"
    ].to_numpy(dtype=float)
    x_min = -10
    x_max = 70 if task_type == CLASSIFICATION else 30
    gain_axis.axvspan(x_min, 0, color="#F0F3F5", linewidth=0, zorder=0)
    gain_axis.axvspan(0, x_max, color="#FBF7EA", linewidth=0, zorder=0)
    gain_axis.axvline(0, color=TEXT, linewidth=0.9, zorder=1)
    violin = gain_axis.violinplot(
        values,
        positions=[0],
        vert=False,
        widths=0.62,
        showmeans=False,
        showmedians=False,
        showextrema=False,
        bw_method=0.45,
    )
    body = violin["bodies"][0]
    body.set_facecolor(color)
    body.set_edgecolor("none")
    body.set_alpha(0.14)
    jitter = np.linspace(-0.13, 0.13, len(values))
    point_colors = [color if value > 0 else NEUTRAL for value in values]
    gain_axis.scatter(
        values,
        jitter,
        s=39,
        color=point_colors,
        edgecolor="white",
        linewidth=0.7,
        zorder=3,
    )
    median = float(np.median(values))
    q25, q75 = np.quantile(values, [0.25, 0.75])
    gain_axis.plot([q25, q75], [0, 0], color=TEXT, linewidth=2.25, zorder=4)
    gain_axis.scatter(
        [median],
        [0],
        marker="D",
        s=70,
        color=color,
        edgecolor="white",
        linewidth=1.0,
        zorder=5,
    )
    improved = int(np.sum(values > 0))
    gain_axis.set_xlim(x_min, x_max)
    gain_axis.set_ylim(-0.55, 0.55)
    gain_axis.set_yticks([])
    gain_axis.set_xticks([-5, 0, 20, 40, 60] if task_type == CLASSIFICATION else [-5, 0, 10, 20, 30])
    gain_axis.set_xlabel("Error reduction at 25% trusted coverage (%)", labelpad=7, color=TEXT)
    gain_axis.set_title(
        f"Trusted-region gains · {improved}/{task_count} improved",
        loc="left",
        fontweight="bold",
        color=TEXT,
        pad=10,
    )
    gain_axis.grid(axis="x", color=GRID, linewidth=0.7, alpha=0.8)
    gain_axis.spines["left"].set_visible(False)
    gain_axis.spines["bottom"].set_color(TEXT)
    gain_axis.spines["bottom"].set_linewidth(0.85)
    gain_axis.tick_params(axis="x", colors=MUTED, length=3, width=0.7)
    panel_label(gain_axis, "b")

    figure.text(
        0.075,
        0.965,
        task_type,
        ha="left",
        va="top",
        fontsize=11.3,
        fontweight="bold",
        color=color,
    )
    figure.text(
        0.985,
        0.025,
        f"Post-freeze official hold-out · {task_count} endpoints · 5 frozen seeds",
        ha="right",
        va="bottom",
        fontsize=6.3,
        color=MUTED,
    )
    save_figure(figure, f"split_{type_slug}_coverage_and_trusted_region_gain")


def plot_white_coverage(coverage: pd.DataFrame) -> None:
    figure, axis = plt.subplots(
        figsize=(183 * WHITE_CANVAS_SCALE * 0.5 / MM_PER_INCH, 80 * WHITE_CANVAS_SCALE / MM_PER_INCH),
        facecolor="white",
    )
    figure.subplots_adjust(left=0.13, right=0.95, bottom=0.28, top=0.55)
    axis.set_facecolor("white")
    axis.axhline(1.0, color=TEXT, linewidth=0.9, linestyle=(0, (4, 3)), zorder=1)
    colors = {CLASSIFICATION: CORAL, REGRESSION: TEAL}
    for type_index, task_type in enumerate((CLASSIFICATION, REGRESSION)):
        medians, lows, highs = [], [], []
        for coverage_value in COVERAGES:
            values = coverage.loc[
                (coverage["task_type"] == task_type)
                & np.isclose(coverage["coverage"], coverage_value),
                "normalized_error",
            ].to_numpy(dtype=float)
            median, low, high = bootstrap_median_interval(
                values, seed=9100 + type_index * 100 + int(coverage_value * 100)
            )
            medians.append(median)
            lows.append(low)
            highs.append(high)
        x = COVERAGES * 100
        axis.fill_between(x, lows, highs, color=colors[task_type], alpha=0.12, linewidth=0)
        axis.plot(
            x,
            medians,
            color=colors[task_type],
            linewidth=2.3 * WHITE_LINE_SCALE,
            solid_capstyle="round",
            solid_joinstyle="round",
            zorder=3,
        )
        axis.scatter(
            x[::3],
            np.asarray(medians)[::3],
            s=24,
            color=colors[task_type],
            edgecolor="white",
            linewidth=0.7,
            zorder=4,
        )
    axis.set_xlim(24, 101)
    axis.set_ylim(0.55, 1.075)
    axis.set_xticks([25, 40, 60, 80, 100])
    axis.set_yticks([0.6, 0.8, 1.0], ["60", "80", "100"])
    axis.set_xlabel(
        "Hold-out coverage retained (%)",
        labelpad=10,
        color=TEXT,
        fontsize=WHITE_LABEL_PT * COVERAGE_TEXT_SCALE,
    )
    figure.text(
        0.13,
        0.97,
        "Confidence filtering concentrates\nreliable predictions",
        ha="left",
        va="top",
        fontweight="bold",
        color=TEXT,
        fontsize=WHITE_TITLE_PT * COVERAGE_TEXT_SCALE,
        linespacing=1.16,
    )
    figure.text(
        0.24,
        0.655,
        "Classification",
        ha="left",
        va="center",
        fontsize=WHITE_DIRECT_LABEL_PT * COVERAGE_TEXT_SCALE,
        color=CORAL,
        fontweight="bold",
    )
    figure.text(
        0.67,
        0.655,
        "Regression",
        ha="left",
        va="center",
        fontsize=WHITE_DIRECT_LABEL_PT * COVERAGE_TEXT_SCALE,
        color=TEAL,
        fontweight="bold",
    )
    style_axis(axis)
    axis.spines["left"].set_visible(False)
    axis.spines["bottom"].set_color("#000000")
    axis.spines["bottom"].set_linewidth(2.2)
    axis.tick_params(
        axis="y",
        colors=MUTED,
        length=0,
        width=0,
        labelsize=WHITE_TICK_PT * COVERAGE_TEXT_SCALE,
        pad=10,
    )
    axis.tick_params(
        axis="x",
        color="#000000",
        labelcolor=MUTED,
        length=6,
        width=2.0,
        labelsize=WHITE_TICK_PT * COVERAGE_TEXT_SCALE,
        pad=10,
    )
    figure.text(
        0.99,
        0.018,
        "Post-freeze official hold-out analysis\n22 endpoints · 5 frozen seeds",
        ha="right",
        va="bottom",
        fontsize=WHITE_FOOTER_PT * COVERAGE_TEXT_SCALE,
        color=MUTED,
    )
    save_figure(figure, "white_coverage_reliability_combined")


def plot_white_gain_combined(gains: pd.DataFrame) -> None:
    figure, axis = plt.subplots(
        figsize=(183 * WHITE_CANVAS_SCALE / MM_PER_INCH, 72 * WHITE_CANVAS_SCALE / MM_PER_INCH),
        facecolor="white",
    )
    figure.subplots_adjust(left=0.065, right=0.975, bottom=0.34, top=0.94)
    axis.set_facecolor("white")
    axis.axvline(0, color=TEXT, linewidth=0.9, linestyle=(0, (4, 3)), zorder=1)

    for task_type, position, color in (
        (CLASSIFICATION, 1.0, CORAL),
        (REGRESSION, 0.0, TEAL),
    ):
        values = gains.loc[
            gains["task_type"] == task_type,
            "trusted_region_error_reduction_pct",
        ].to_numpy(dtype=float)
        violin = axis.violinplot(
            values,
            positions=[position],
            vert=False,
            widths=0.60,
            showmeans=False,
            showmedians=False,
            showextrema=False,
            bw_method=0.45,
        )
        body = violin["bodies"][0]
        body.set_facecolor(color)
        body.set_edgecolor("none")
        body.set_alpha(0.13)
        jitter = position + np.linspace(-0.13, 0.13, len(values))
        point_colors = [color if value > 0 else NEUTRAL for value in values]
        axis.scatter(
            values,
            jitter,
            s=44,
            color=point_colors,
            edgecolor="white",
            linewidth=0.75,
            zorder=3,
        )
        median = float(np.median(values))
        q25, q75 = np.quantile(values, [0.25, 0.75])
        axis.plot(
            [q25, q75],
            [position, position],
            color=color,
            linewidth=2.3 * WHITE_LINE_SCALE,
            solid_capstyle="round",
            zorder=4,
        )
        axis.scatter(
            [median],
            [position],
            marker="D",
            s=76,
            color=color,
            edgecolor="white",
            linewidth=1.0,
            zorder=5,
        )

    axis.set_xlim(-10, 70)
    axis.set_ylim(-0.45, 1.45)
    axis.set_yticks([])
    axis.set_xticks([-5, 0, 20, 40, 60])
    axis.set_xlabel(
        "Error reduction at 25% trusted coverage (%)",
        labelpad=32,
        color=TEXT,
        fontsize=WHITE_LABEL_PT * GAIN_TEXT_SCALE,
    )
    axis.grid(axis="x", color=GRID, linewidth=0.7, alpha=0.75)
    axis.grid(axis="y", visible=False)
    axis.spines["left"].set_visible(False)
    axis.spines["bottom"].set_color("#000000")
    axis.spines["bottom"].set_linewidth(4.0)
    axis.tick_params(
        axis="x",
        color="#000000",
        labelcolor=MUTED,
        length=6,
        width=2.0,
        labelsize=WHITE_TICK_PT * GAIN_TEXT_SCALE,
        pad=18,
    )
    save_figure(figure, "white_classification_regression_trusted_region_gain")


def main() -> None:
    configure_style()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    frame, audit = build_source_data()
    coverage, quartiles, gains = endpoint_summaries(frame)
    frame.to_csv(SOURCE_DATA, index=False)
    summary = {
        **audit,
        "source_data": SOURCE_DATA.relative_to(ROOT).as_posix(),
        "classification_endpoints_improved_in_highest_similarity_quartile": int(
            ((gains["task_type"] == CLASSIFICATION) & (gains["trusted_region_error_reduction_pct"] > 0)).sum()
        ),
        "regression_endpoints_improved_in_highest_similarity_quartile": int(
            ((gains["task_type"] == REGRESSION) & (gains["trusted_region_error_reduction_pct"] > 0)).sum()
        ),
        "classification_median_error_reduction_pct": float(
            gains.loc[gains["task_type"] == CLASSIFICATION, "trusted_region_error_reduction_pct"].median()
        ),
        "regression_median_error_reduction_pct": float(
            gains.loc[gains["task_type"] == REGRESSION, "trusted_region_error_reduction_pct"].median()
        ),
    }
    SUMMARY_JSON.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    plot_coverage(coverage)
    plot_dose_response(quartiles)
    plot_gain_distribution(gains)
    plot_combined(coverage, gains)
    plot_split_combined(coverage, gains, CLASSIFICATION)
    plot_split_combined(coverage, gains, REGRESSION)
    plot_white_coverage(coverage)
    plot_white_gain_combined(gains)
    print(json.dumps({key: value for key, value in summary.items() if key.endswith("pct") or "improved" in key}, sort_keys=True))


if __name__ == "__main__":
    main()
