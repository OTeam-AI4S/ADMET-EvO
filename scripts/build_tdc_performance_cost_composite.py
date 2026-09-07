#!/usr/bin/env python3
"""Assemble a layout draft for the TDC benchmark and efficiency evidence.

The upper-left panel is intentionally reserved for a future visualization of
ADMET-EvO's task-level values. The current TDC comparison occupies the upper
right, and the performance/cost evidence spans the bottom row.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")

import matplotlib.font_manager as fm
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_COMPARISON = ROOT / "outputs" / "figures" / "tdc_admetevo_maplight_comparison_polished.png"
DEFAULT_REPORT = ROOT / "runs" / "v27-router-risk-guard" / "report.json"
DEFAULT_TASK_DATA = ROOT / "reports" / "model_comparison_tdc22_8models_2026-08-25.csv"
DEFAULT_APPLICABILITY = ROOT / "reports" / "validation_calibration_residuals_v27.json"
DEFAULT_OUTPUT = ROOT / "outputs" / "figures" / "tdc_performance_cost_composite"
FONT_REGULAR = ROOT / "assets" / "fonts" / "msyh.ttc"
FONT_BOLD = ROOT / "assets" / "fonts" / "msyhbd.ttc"

MM_PER_INCH = 25.4
FIGURE_WIDTH_MM = 280
FIGURE_HEIGHT_MM = 270
MARGIN = -0.005

ADMET_COLOR = "#CF6559"
MAPLIGHT_COLOR = "#188977"
PALE_YELLOW = "#FFF2C9"
PALE_GREEN = "#E4F1ED"
PALE_GREY = "#EEF2F5"
NEUTRAL_BAR = "#B7C5CE"
GRID = "#D7DEE4"
TEXT = "#26333E"
MUTED = "#657483"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--comparison", type=Path, default=DEFAULT_COMPARISON)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--task-data", type=Path, default=DEFAULT_TASK_DATA)
    parser.add_argument("--applicability", type=Path, default=DEFAULT_APPLICABILITY)
    parser.add_argument("--output-prefix", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def configure_style() -> None:
    for font_path in (FONT_REGULAR, FONT_BOLD):
        if not font_path.exists():
            raise FileNotFoundError(f"Required Microsoft YaHei font is missing: {font_path}")
        fm.fontManager.addfont(font_path)

    resolved = Path(fm.findfont(fm.FontProperties(family="Microsoft YaHei"))).resolve()
    if resolved not in {FONT_REGULAR.resolve(), FONT_BOLD.resolve()}:
        raise RuntimeError(f"Microsoft YaHei did not resolve to the project font: {resolved}")

    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Microsoft YaHei", "Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "font.size": 8,
            "axes.titlesize": 10,
            "axes.labelsize": 8.5,
            "xtick.labelsize": 7.5,
            "ytick.labelsize": 8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 0.8,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.facecolor": "white",
        }
    )


def load_efficiency_stats(path: Path) -> dict[str, float]:
    report = json.loads(path.read_text(encoding="utf-8"))
    if report["task_count"] != 22 or report["fresh_validation_seeds"] != [4, 5]:
        raise ValueError("Efficiency report does not match the frozen 22-task fresh-seed analysis")

    score_seconds = 0.0
    aware_seconds = 0.0
    within_margin = 0
    for task in report["tasks"]:
        score_seconds += float(task["summaries"]["score"]["elapsed_seconds"])
        aware_seconds += float(task["summaries"]["metric_risk_guard"]["elapsed_seconds"])
        within_margin += float(task["guard_minus_score_oriented_gain"]) >= MARGIN

    score_hours = score_seconds / 3600.0
    aware_hours = aware_seconds / 3600.0
    ratio = aware_hours / score_hours
    frozen_ratio = float(report["claim_gate"]["aggregate_guard_cost_ratio_vs_score"])
    if not np.isclose(ratio, frozen_ratio, rtol=0, atol=1e-12):
        raise ValueError("Aggregate fitting-time ratio does not reproduce the frozen report")

    ci_low, ci_high = report["guard_minus_score"]["bootstrap_mean_95_ci"]
    return {
        "score_hours": score_hours,
        "aware_hours": aware_hours,
        "ratio": ratio,
        "reduction": 100.0 * (1.0 - ratio),
        "mean": float(report["guard_minus_score"]["mean"]),
        "ci_low": float(ci_low),
        "ci_high": float(ci_high),
        "within": float(within_margin),
    }


def write_reserved_panel_data(source_path: Path, output_path: Path) -> None:
    frame = pd.read_csv(source_path)
    required = ["Task", "Metric", "Direction", "ADMET-EvO"]
    missing = set(required).difference(frame.columns)
    if missing:
        raise ValueError(f"Task table is missing fields: {sorted(missing)}")
    if len(frame) != 22:
        raise ValueError(f"Expected 22 tasks, found {len(frame)}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame[required].to_csv(output_path, index=False)


def load_applicability(path: Path) -> dict[str, list[dict[str, float | str]]]:
    audit = json.loads(path.read_text(encoding="utf-8"))
    if audit["task_count"] != 22 or audit["classification_task_count"] != 13 or audit["regression_task_count"] != 9:
        raise ValueError("Applicability audit does not match the frozen 13-classification/9-regression panel")
    if audit["analysis_role"] != "post_selection_descriptive_validation_audit":
        raise ValueError("Unexpected applicability-analysis role")

    display = {
        "AMES": "AMES",
        "BBB_Martins": "BBB",
        "Bioavailability_Ma": "Bioavail.",
        "CYP2C9_Substrate_CarbonMangels": "2C9 Sub.",
        "CYP2C9_Veith": "2C9 Veith",
        "CYP2D6_Substrate_CarbonMangels": "2D6 Sub.",
        "CYP2D6_Veith": "2D6 Veith",
        "CYP3A4_Substrate_CarbonMangels": "3A4 Sub.",
        "CYP3A4_Veith": "3A4 Veith",
        "Caco2_Wang": "Caco2",
        "Clearance_Hepatocyte_AZ": "Hep. CL",
        "Clearance_Microsome_AZ": "Mic. CL",
        "DILI": "DILI",
        "HIA_Hou": "HIA",
        "Half_Life_Obach": "Half-Life",
        "LD50_Zhu": "LD50",
        "Lipophilicity_AstraZeneca": "Lipophilicity",
        "PPBR_AZ": "PPBR",
        "Pgp_Broccatelli": "Pgp",
        "Solubility_AqSolDB": "Solubility",
        "VDss_Lombardo": "VDss",
        "hERG": "hERG",
    }
    groups: dict[str, list[dict[str, float | str]]] = {"classification": [], "regression": []}
    for row in audit["tasks"]:
        if row["identity_overlap_count"] != 0 or row["train_valid_scaffold_overlap_count"] != 0:
            raise ValueError("Applicability audit contains train-validation overlap")
        space = row["chemical_space"]
        if row["analysis_type"] == "classification_calibration":
            group = "classification"
            overall = float(row["calibration"]["brier"])
            high = float(space["high_similarity_brier"])
        else:
            group = "regression"
            overall = float(row["residual"]["mean_absolute_error"])
            high = float(space["high_similarity_mean_absolute_error"])
        if overall <= 0 or high <= 0:
            raise ValueError("Applicability-domain error summaries must be positive")
        groups[group].append(
            {
                "task": row["task"],
                "label": display[row["task"]],
                "overall_error": overall,
                "trusted_error": high,
                "error_reduction": 100.0 * (1.0 - high / overall),
            }
        )
    for rows in groups.values():
        rows.sort(key=lambda item: float(item["error_reduction"]))
    return groups


def panel_label(axis: plt.Axes, label: str) -> None:
    axis.text(
        -0.03,
        1.015,
        label,
        transform=axis.transAxes,
        ha="right",
        va="bottom",
        fontsize=13,
        fontweight="bold",
        color=TEXT,
        clip_on=False,
    )


def reserved_panel(axis: plt.Axes) -> None:
    axis.set_xlim(0, 1)
    axis.set_ylim(0, 1)
    axis.set_xticks([])
    axis.set_yticks([])
    for spine in axis.spines.values():
        spine.set_visible(True)
        spine.set_color(GRID)
        spine.set_linewidth(1.1)
        spine.set_linestyle((0, (4, 4)))
    axis.set_facecolor("#FCFDFE")
    axis.set_title(
        "ADMET-EvO task-level performance",
        loc="left",
        fontsize=11,
        fontweight="bold",
        color=TEXT,
        pad=9,
    )
    axis.text(
        0.5,
        0.53,
        "Reserved for 22-task values",
        ha="center",
        va="center",
        fontsize=9,
        color=MUTED,
    )
    axis.text(
        0.5,
        0.47,
        "AUROC / AUPRC / MAE / Spearman",
        ha="center",
        va="center",
        fontsize=7.4,
        color="#8A97A2",
    )
    panel_label(axis, "a")


def comparison_panel(axis: plt.Axes, image_path: Path) -> None:
    image = mpimg.imread(image_path)
    axis.imshow(image, interpolation="lanczos")
    axis.set_axis_off()
    panel_label(axis, "b")


def noninferiority_panel(axis: plt.Axes, stats: dict[str, float]) -> None:
    x_min, x_max = -0.0060, 0.0014
    axis.set_facecolor("white")
    axis.axvspan(MARGIN, x_max, color=PALE_GREEN, linewidth=0, zorder=0)
    axis.axvline(MARGIN, color=ADMET_COLOR, linestyle=(0, (4, 3)), linewidth=1.25, zorder=2)
    axis.axvline(0.0, color="#7E8B96", linewidth=0.9, zorder=2)

    mean = stats["mean"]
    low = stats["ci_low"]
    high = stats["ci_high"]
    axis.errorbar(
        mean,
        0.48,
        xerr=np.array([[mean - low], [high - mean]]),
        fmt="o",
        color=MAPLIGHT_COLOR,
        ecolor=MAPLIGHT_COLOR,
        elinewidth=2.35,
        capsize=5,
        capthick=1.15,
        markersize=8.5,
        markeredgecolor="white",
        markeredgewidth=1.0,
        zorder=4,
    )
    axis.text(
        -0.00455,
        0.84,
        "95% bootstrap CI",
        ha="left",
        va="center",
        fontsize=7.35,
        color=TEXT,
    )
    axis.text(
        -0.00455,
        0.70,
        "[−0.00148, 0.00059]",
        ha="left",
        va="center",
        fontsize=7.8,
        color=MAPLIGHT_COLOR,
        fontweight="bold",
    )
    axis.text(
        -0.00018,
        0.64,
        "mean −0.00035",
        ha="right",
        va="bottom",
        fontsize=7.35,
        color=TEXT,
    )
    axis.text(
        -0.00455,
        0.18,
        "margin −0.005",
        ha="left",
        va="center",
        fontsize=7.35,
        color=ADMET_COLOR,
    )
    axis.text(
        -0.00025,
        0.91,
        f"{int(stats['within'])}/22 within margin",
        ha="right",
        va="top",
        fontsize=7.2,
        color=MUTED,
    )

    axis.set_xlim(x_min, x_max)
    axis.set_ylim(0, 1)
    axis.set_yticks([])
    axis.set_xticks([MARGIN, -0.003, 0.0], ["−0.005", "−0.003", "0"])
    axis.tick_params(axis="x", labelsize=7.7, colors=MUTED)
    axis.set_xlabel("Mean oriented difference", labelpad=8, fontsize=7.8, color=TEXT)
    axis.spines["left"].set_visible(False)
    axis.spines["bottom"].set_color(TEXT)
    axis.spines["bottom"].set_linewidth(0.9)
    axis.set_title("Practical non-inferiority", loc="left", fontsize=9.2, fontweight="bold", color=TEXT, pad=8)
    panel_label(axis, "c")


def cost_panel(axis: plt.Axes, stats: dict[str, float]) -> None:
    values = [stats["score_hours"], stats["aware_hours"]]
    positions = [1, 0]
    colors = [NEUTRAL_BAR, ADMET_COLOR]
    axis.set_facecolor("white")
    axis.barh(positions, values, height=0.34, color=colors, edgecolor="none", zorder=3)
    axis.set_yticks(positions, ["Score-only", "Compute-aware"])
    axis.set_xlim(0, 2.68)
    axis.set_xticks(np.arange(0, 2.6, 0.5))
    for tick in np.arange(0.5, 2.6, 0.5):
        axis.axvspan(tick - 0.002, tick + 0.002, color=GRID, linewidth=0, zorder=0)
    axis.tick_params(axis="x", labelsize=7.7, colors=MUTED)
    axis.tick_params(axis="y", labelsize=7.8, length=0, pad=8, colors=TEXT)
    axis.spines["left"].set_visible(False)
    axis.spines["bottom"].set_color(TEXT)
    axis.spines["bottom"].set_linewidth(0.9)
    axis.set_xlabel("Selected-recipe fitting time (h)", labelpad=8, fontsize=7.8, color=TEXT)
    axis.set_title(
        f"Fitting cost falls by {stats['reduction']:.1f}%",
        loc="left",
        fontsize=9.2,
        fontweight="bold",
        color=TEXT,
        pad=10,
    )

    axis.text(values[0] + 0.055, positions[0], f"{values[0]:.3f} h", va="center", ha="left", fontsize=7.35, color=TEXT, fontweight="bold")
    axis.text(values[1] + 0.055, positions[1], f"{values[1]:.3f} h", va="center", ha="left", fontsize=7.35, color=ADMET_COLOR, fontweight="bold")
    axis.text(
        1.48,
        0.05,
        f"{stats['ratio']:.3f}× of score-only",
        ha="center",
        va="center",
        fontsize=7.35,
        color=MAPLIGHT_COLOR,
        fontweight="bold",
    )
    panel_label(axis, "d")


def applicability_panel(
    axis: plt.Axes,
    groups: dict[str, list[dict[str, float | str]]],
) -> None:
    class_rows = groups["classification"]
    reg_rows = groups["regression"]
    class_values = np.asarray([float(row["error_reduction"]) for row in class_rows])
    reg_values = np.asarray([float(row["error_reduction"]) for row in reg_rows])
    all_values = np.concatenate([class_values, reg_values])
    improved = int(np.sum(all_values > 0))

    axis.set_xlim(-42, 103)
    axis.set_ylim(-0.48, 1.50)
    axis.axvspan(0, 103, color=PALE_YELLOW, alpha=0.48, linewidth=0, zorder=0)
    axis.axvspan(-42, 0, color=PALE_GREY, alpha=0.58, linewidth=0, zorder=0)
    axis.axvline(0, color=TEXT, linewidth=1.0, zorder=2)
    for tick in [-25, 25, 50, 75, 100]:
        axis.axvline(tick, color=GRID, linewidth=0.75, zorder=1)

    row_specs = [
        (class_rows, class_values, 1.0, "Classification", 13),
        (reg_rows, reg_values, 0.0, "Regression", 9),
    ]
    for rows, values, center_y, label, total in row_specs:
        jitter = np.linspace(-0.15, 0.15, len(values))
        colors = [ADMET_COLOR if value > 0 else NEUTRAL_BAR for value in values]
        axis.scatter(
            values,
            center_y + jitter,
            s=39,
            c=colors,
            edgecolors="white",
            linewidths=0.7,
            zorder=4,
        )
        median = float(np.median(values))
        q25, q75 = np.quantile(values, [0.25, 0.75])
        axis.plot([q25, q75], [center_y, center_y], color=TEXT, linewidth=2.2, zorder=5)
        axis.scatter(
            [median],
            [center_y],
            s=76,
            marker="D",
            color=ADMET_COLOR,
            edgecolors="white",
            linewidths=1.0,
            zorder=6,
        )
        improved_count = int(np.sum(values > 0))
        axis.text(
            -40.0,
            center_y,
            f"{label}\n{improved_count}/{total} improved",
            ha="left",
            va="center",
            fontsize=7.6,
            color=TEXT,
            fontweight="bold",
            linespacing=1.35,
        )
        axis.text(
            54.0,
            center_y + 0.29,
            f"median {median:.1f}% lower error",
            ha="left",
            va="center",
            fontsize=7.2,
            color=ADMET_COLOR,
            fontweight="bold",
        )

    axis.set_xticks([-25, 0, 25, 50, 75, 100])
    axis.tick_params(axis="x", labelsize=7.2, colors=MUTED)
    axis.set_yticks([])
    axis.set_xlabel(
        "Error reduction in the similarity-defined trusted region vs overall validation (%)",
        fontsize=7.6,
        color=TEXT,
        labelpad=6,
    )
    axis.spines[["top", "right", "left"]].set_visible(False)
    axis.spines["bottom"].set_color(TEXT)
    axis.spines["bottom"].set_linewidth(0.9)
    panel_label(axis, "e")
    axis.text(
        0,
        1.12,
        "Similarity-guided applicability filtering improves predictive reliability",
        transform=axis.transAxes,
        ha="left",
        va="bottom",
        fontsize=9.2,
        fontweight="bold",
        color=TEXT,
        clip_on=False,
    )
    axis.text(
        1.0,
        1.12,
        f"{improved}/22 endpoints improved in the trusted region",
        transform=axis.transAxes,
        ha="right",
        va="bottom",
        fontsize=7.2,
        color=MUTED,
        clip_on=False,
    )
    axis.text(
        0,
        1.035,
        "Highest nearest-training-similarity quartile vs all validation molecules; dots, endpoints; diamond and line, median and IQR",
        transform=axis.transAxes,
        ha="left",
        va="bottom",
        fontsize=6.9,
        color=MUTED,
        clip_on=False,
    )


def build_figure(
    comparison_path: Path,
    stats: dict[str, float],
    applicability: dict[str, list[dict[str, float | str]]],
) -> plt.Figure:
    configure_style()
    figure = plt.figure(
        figsize=(FIGURE_WIDTH_MM / MM_PER_INCH, FIGURE_HEIGHT_MM / MM_PER_INCH),
        facecolor="white",
    )
    outer = figure.add_gridspec(
        3,
        2,
        width_ratios=[0.34, 0.66],
        height_ratios=[0.63, 0.13, 0.24],
        left=0.045,
        right=0.985,
        bottom=0.09,
        top=0.965,
        wspace=0.10,
        hspace=0.38,
    )
    reserved = figure.add_subplot(outer[0, 0])
    comparison = figure.add_subplot(outer[0, 1])
    middle = outer[1, :].subgridspec(1, 2, width_ratios=[1.08, 1.0], wspace=0.29)
    noninferiority = figure.add_subplot(middle[0, 0])
    cost = figure.add_subplot(middle[0, 1])
    applicability_axis = figure.add_subplot(outer[2, :])

    reserved_panel(reserved)
    comparison_panel(comparison, comparison_path)
    noninferiority_panel(noninferiority, stats)
    cost_panel(cost, stats)
    applicability_panel(applicability_axis, applicability)

    figure.text(
        0.985,
        0.014,
        "Panels c–d: fresh validation, development seeds 1–3 and evaluation seeds 4–5. Panel e: trusted region = highest nearest-training-similarity quartile; post-selection diagnostic on one saved scaffold split per task; no identity or scaffold overlap.",
        ha="right",
        va="bottom",
        fontsize=7,
        color=MUTED,
    )
    return figure


def save_figure(figure: plt.Figure, prefix: Path) -> None:
    prefix.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(prefix.with_suffix(".png"), dpi=300)
    figure.savefig(prefix.with_suffix(".pdf"))
    figure.savefig(prefix.with_suffix(".svg"))
    figure.savefig(
        prefix.with_suffix(".tiff"),
        dpi=600,
        pil_kwargs={"compression": "tiff_lzw"},
    )


def main() -> None:
    args = parse_args()
    stats = load_efficiency_stats(args.report)
    applicability = load_applicability(args.applicability)
    reserved_data = args.output_prefix.parent / "admetevo_22task_values_for_reserved_panel.csv"
    write_reserved_panel_data(args.task_data, reserved_data)
    figure = build_figure(args.comparison, stats, applicability)
    save_figure(figure, args.output_prefix)
    plt.close(figure)
    print(args.output_prefix)


if __name__ == "__main__":
    main()
