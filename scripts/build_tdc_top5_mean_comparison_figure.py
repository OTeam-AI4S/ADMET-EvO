#!/usr/bin/env python3
"""Build Figure 2A using a fixed cohort of five globally leading TDC models.

Eligible comparators must report all 22 endpoints. The five models with the
lowest mean rank across those endpoints are selected once, then their scores
are averaged within every endpoint. Corrected Bioavailability_Ma values are
used wherever the supplied rerun table contains the selected model.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")

import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SNAPSHOT = ROOT / "reports" / "leaderboard_snapshot_2026-08-17.json"
DEFAULT_COMPARISON = ROOT / "reports" / "model_comparison_tdc22_8models_2026-08-25.csv"
DEFAULT_CHEMPROP_BIO = ROOT / "runs" / "bioavailability-chemprop-rdkit-tdc5" / "summary.json"
DEFAULT_OUTPUT = ROOT / "outputs" / "figures" / "tdc_admetevo_fixed_top5_models_mean_comparison"
FONT_REGULAR = ROOT / "assets" / "fonts" / "msyh.ttc"
FONT_BOLD = ROOT / "assets" / "fonts" / "msyhbd.ttc"

ADMET = "ADMET-EvO"
BIO_TASK = "Bioavailability_Ma"
BIO_EXTERNAL_COLUMNS = [
    "MiniMol",
    "MolE",
    "MapLight_author_published",
    "MapLight+GNN_author_published",
    "AttentiveFP",
    "AttrMasking",
    "ContextPred",
]
BIO_COLUMN_BY_MODEL = {
    "MapLight + GNN": "MapLight+GNN_author_published",
    "MapLight": "MapLight_author_published",
    "ContextPred": "ContextPred",
    "AttrMasking": "AttrMasking",
}

DISPLAY_NAMES = {
    "AMES": "AMES",
    "BBB_Martins": "BBB Martins",
    "Bioavailability_Ma": "Bioavailability Ma*",
    "CYP2C9_Substrate_CarbonMangels": "CYP2C9 Substrate",
    "CYP2C9_Veith": "CYP2C9 Veith",
    "CYP2D6_Substrate_CarbonMangels": "CYP2D6 Substrate",
    "CYP2D6_Veith": "CYP2D6 Veith",
    "CYP3A4_Substrate_CarbonMangels": "CYP3A4 Substrate",
    "CYP3A4_Veith": "CYP3A4 Veith",
    "Caco2_Wang": "Caco2 Wang",
    "Clearance_Hepatocyte_AZ": "Clearance Hepatocyte",
    "Clearance_Microsome_AZ": "Clearance Microsome",
    "DILI": "DILI",
    "HIA_Hou": "HIA Hou",
    "Half_Life_Obach": "Half-Life Obach",
    "LD50_Zhu": "LD50 Zhu",
    "Lipophilicity_AstraZeneca": "Lipophilicity AZ",
    "PPBR_AZ": "PPBR AZ",
    "Pgp_Broccatelli": "Pgp Broccatelli",
    "Solubility_AqSolDB": "Solubility AqSolDB",
    "VDss_Lombardo": "VDss Lombardo",
    "hERG": "hERG",
}

# Colors sampled from the supplied Figure 2A raster.
CORAL = "#D55C53"
CORAL_STRONG = "#B9423C"
TEAL = "#238A78"
TEXT = "#25313D"
MUTED = "#657483"
GRID = "#D8E2E8"
ROW = "#F4F7F8"
ADMET_CELL = "#F8DCD8"
TOP5_CELL = "#DDEFEA"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--snapshot", type=Path, default=DEFAULT_SNAPSHOT)
    parser.add_argument("--comparison", type=Path, default=DEFAULT_COMPARISON)
    parser.add_argument("--chemprop-bio", type=Path, default=DEFAULT_CHEMPROP_BIO)
    parser.add_argument("--output-prefix", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def configure_font() -> None:
    for font_path in (FONT_REGULAR, FONT_BOLD):
        if not font_path.is_file():
            raise FileNotFoundError(font_path)
        fm.fontManager.addfont(font_path)
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Microsoft YaHei", "Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 7.0,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "axes.linewidth": 0.7,
            "savefig.facecolor": "white",
        }
    )


def is_admetevo(model: str) -> bool:
    normalized = "".join(character.lower() for character in model if character.isalnum())
    return normalized == "admetevo"


def select_fixed_cohort(pages: dict) -> tuple[list[str], dict[str, float]]:
    task_count = len(DISPLAY_NAMES)
    ranks: dict[str, dict[str, float]] = {}
    for task, page in pages.items():
        for entry in page.get("entries", []):
            model = str(entry["model"])
            if not is_admetevo(model):
                ranks.setdefault(model, {})[task] = float(entry["rank"])
    complete = {
        model: float(np.mean(list(task_ranks.values())))
        for model, task_ranks in ranks.items()
        if len(task_ranks) == task_count
    }
    if len(complete) < 5:
        raise ValueError("Fewer than five external models report all 22 tasks")
    cohort = sorted(complete, key=lambda model: (complete[model], model))[:5]
    return cohort, complete


def load_data(snapshot_path: Path, comparison_path: Path, chemprop_bio_path: Path) -> tuple[pd.DataFrame, str]:
    snapshot = json.loads(snapshot_path.read_text(encoding="utf-8"))
    comparison = pd.read_csv(comparison_path)
    chemprop_bio = json.loads(chemprop_bio_path.read_text(encoding="utf-8"))
    if chemprop_bio.get("task") != BIO_TASK or chemprop_bio.get("model") != "Chemprop-RDKit":
        raise ValueError("Unexpected Chemprop-RDKit Bioavailability rerun summary")
    if chemprop_bio.get("protocol", {}).get("seeds") != [1, 2, 3, 4, 5]:
        raise ValueError("Chemprop-RDKit rerun does not contain the required TDC seeds 1-5")
    chemprop_bio_mean = float(chemprop_bio["mean_auroc"])
    required = {"Task", "Metric", "Direction", ADMET, *BIO_EXTERNAL_COLUMNS}
    if missing := required.difference(comparison.columns):
        raise ValueError(f"Comparison table is missing fields: {sorted(missing)}")
    if len(comparison) != 22 or set(comparison["Task"]) != set(DISPLAY_NAMES):
        raise ValueError("Comparison table does not contain the expected TDC ADMET-22 tasks")

    pages = snapshot.get("pages", {})
    if set(pages) != set(DISPLAY_NAMES):
        raise ValueError("Leaderboard snapshot does not contain the expected 22 tasks")
    cohort, complete_mean_ranks = select_fixed_cohort(pages)
    comparison_by_task = comparison.set_index("Task")

    records: list[dict] = []
    for _, source_row in comparison.iterrows():
        task = str(source_row["Task"])
        direction = str(source_row["Direction"])
        if direction not in {"maximize", "minimize"}:
            raise ValueError(f"Unknown direction for {task}: {direction}")

        page = pages[task]
        if page.get("direction") != direction:
            raise ValueError(f"Direction mismatch for {task}")
        by_model = {str(entry["model"]): entry for entry in page.get("entries", [])}
        selected: list[tuple[str, float]] = []
        override_models: list[str] = []
        for model in cohort:
            if model not in by_model:
                raise ValueError(f"Selected model {model} is missing from {task}")
            value = float(by_model[model]["mean"])
            if task == BIO_TASK:
                if model == "Chemprop-RDKit":
                    value = chemprop_bio_mean
                    override_models.append(model)
                elif model in BIO_COLUMN_BY_MODEL:
                    value = float(comparison_by_task.loc[task, BIO_COLUMN_BY_MODEL[model]])
                    override_models.append(model)
            selected.append((model, value))
        bio_override = bool(override_models)
        source = (
            "Bio rerun overrides for " + "; ".join(override_models)
            if bio_override
            else "TDC leaderboard snapshot (2026-08-17)"
        )

        mean = float(np.mean([value for _, value in selected]))
        score = float(source_row[ADMET])
        relative = (
            (score - mean) / mean * 100.0
            if direction == "maximize"
            else (mean - score) / mean * 100.0
        )
        record = {
            "Task": task,
            "TaskLabel": DISPLAY_NAMES[task],
            "Metric": str(source_row["Metric"]),
            "Direction": direction,
            "ADMET-EvO": score,
            "Top5Mean": mean,
            "RelativeDifference": relative,
            "ADMETBetter": relative >= 0,
            "BioavailabilityOverride": bio_override,
            "ComparatorSource": source,
        }
        for cohort_rank, (model, value) in enumerate(selected, start=1):
            record[f"Cohort{cohort_rank}Model"] = model
            record[f"Cohort{cohort_rank}MeanRank"] = complete_mean_ranks[model]
            record[f"Cohort{cohort_rank}Value"] = value
        records.append(record)

    frame = pd.DataFrame(records)
    return frame, str(snapshot.get("captured_at", "2026-08-17"))


def clean_axis(axis: plt.Axes) -> None:
    for spine in axis.spines.values():
        spine.set_visible(False)
    axis.tick_params(axis="both", length=0, colors=MUTED)


def add_rows(axis: plt.Axes, count: int) -> None:
    for index in range(count):
        if index % 2 == 1:
            axis.axhspan(index - 0.5, index + 0.5, color=ROW, linewidth=0, zorder=0)


def format_relative(value: float) -> str:
    """Keep the original one-decimal style without displaying a real lead as +0.0%."""
    if 0 < abs(value) < 0.05:
        return f"{value:+.2f}%"
    return f"{value:+.1f}%"


def build_figure(frame: pd.DataFrame) -> plt.Figure:
    configure_font()
    figure = plt.figure(figsize=(183 / 25.4, 90.5 / 25.4), facecolor="white")
    grid = figure.add_gridspec(
        1,
        4,
        width_ratios=[8.00, 1.20, 1.15, 1.31],
        left=0.185,
        right=0.995,
        bottom=0.028,
        top=0.925,
        wspace=0.055,
    )
    difference_axis = figure.add_subplot(grid[0, 0])
    outlier_axis = figure.add_subplot(grid[0, 1], sharey=difference_axis)
    admet_axis = figure.add_subplot(grid[0, 2], sharey=difference_axis)
    top5_axis = figure.add_subplot(grid[0, 3], sharey=difference_axis)
    axes = (difference_axis, outlier_axis, admet_axis, top5_axis)
    y = np.arange(len(frame), dtype=float)

    for axis in axes:
        add_rows(axis, len(frame))
        clean_axis(axis)
        axis.set_ylim(len(frame) - 0.5, -0.5)

    difference_axis.set_xlim(-4.2, 31.0)
    difference_axis.set_xticks([0, 10, 20, 30])
    difference_axis.set_xticklabels(["0", "+10", "+20", "+30"])
    difference_axis.xaxis.tick_top()
    difference_axis.tick_params(axis="x", labelsize=6.8, pad=5)
    for tick in (10, 20, 30):
        difference_axis.axvline(tick, color=GRID, linewidth=0.65, zorder=0.5)
    difference_axis.axvline(0, color="#52616C", linewidth=0.9, zorder=2)
    difference_axis.set_yticks(y)
    difference_axis.set_yticklabels(frame["TaskLabel"], fontsize=6.8, color=TEXT)
    difference_axis.tick_params(axis="y", pad=8)

    outlier_axis.set_xlim(74, 85)
    outlier_axis.set_xticks([80])
    outlier_axis.set_xticklabels(["+80"])
    outlier_axis.xaxis.tick_top()
    outlier_axis.tick_params(axis="x", labelsize=6.8, pad=5)
    outlier_axis.axvline(80, color=GRID, linewidth=0.65, zorder=0.5)
    outlier_axis.tick_params(axis="y", labelleft=False)

    slash_style = dict(color=MUTED, linewidth=0.8, clip_on=False)
    difference_axis.plot((0.985, 1.005), (-0.012, 0.012), transform=difference_axis.transAxes, **slash_style)
    difference_axis.plot((0.985, 1.005), (0.988, 1.012), transform=difference_axis.transAxes, **slash_style)
    outlier_axis.plot((-0.025, 0.015), (-0.012, 0.012), transform=outlier_axis.transAxes, **slash_style)
    outlier_axis.plot((-0.025, 0.015), (0.988, 1.012), transform=outlier_axis.transAxes, **slash_style)

    for label, better in zip(difference_axis.get_yticklabels(), frame["ADMETBetter"], strict=True):
        label.set_fontweight("bold" if bool(better) else "normal")

    for yi, relative in zip(y, frame["RelativeDifference"], strict=True):
        if relative < 0:
            color = TEAL
        elif relative >= 8:
            color = CORAL_STRONG
        else:
            color = CORAL
        main_value = min(relative, 30.0) if relative >= 0 else relative
        difference_axis.barh(yi, main_value, height=0.54, color=color, edgecolor="none", zorder=2.5)
        if relative > 30:
            outlier_axis.barh(
                yi + 0.055,
                relative - 74,
                left=74,
                height=0.54,
                color=color,
                edgecolor="none",
                zorder=2.5,
            )
            outlier_axis.text(
                relative - 0.10,
                yi + 0.10,
                format_relative(relative),
                ha="right",
                va="center",
                fontsize=6.45,
                fontweight="bold",
                color="white",
                zorder=3,
                clip_on=False,
            )
            continue
        if relative >= 8:
            x = relative - 0.22
            horizontal_alignment = "right"
            text_color = "white"
        elif relative >= 0:
            x = relative + 0.25
            horizontal_alignment = "left"
            text_color = TEXT
        else:
            x = relative - 0.25
            horizontal_alignment = "right"
            text_color = TEAL
        label_y = yi + 0.055 if relative >= 8 else yi
        difference_axis.text(
            x,
            label_y,
            format_relative(relative),
            ha=horizontal_alignment,
            va="center",
            fontsize=6.45,
            fontweight="bold",
            color=text_color,
            zorder=3,
        )

    for value_axis in (admet_axis, top5_axis):
        value_axis.set_xlim(0, 1)
        value_axis.set_xticks([])
        value_axis.tick_params(axis="y", labelleft=False)
        value_axis.axvline(0, color=GRID, linewidth=0.65, zorder=1)
    top5_axis.axvline(1, color=GRID, linewidth=0.65, zorder=1)

    admet_axis.set_title("ADMET-EvO", fontsize=6.7, fontweight="bold", color=CORAL_STRONG, pad=10)
    top5_axis.set_title(
        "TDC Top-5\nmodels mean",
        fontsize=6.5,
        fontweight="bold",
        color=MUTED,
        pad=4,
        linespacing=1.30,
    )

    for yi, row in frame.iterrows():
        better = bool(row["ADMETBetter"])
        if better:
            admet_axis.barh(yi, 0.68, left=0.16, height=0.56, color=ADMET_CELL, edgecolor="none", zorder=1.5)
        else:
            top5_axis.barh(yi, 0.68, left=0.16, height=0.56, color=TOP5_CELL, edgecolor="none", zorder=1.5)
        admet_suffix = "*" if row["Task"] == BIO_TASK else ""
        admet_axis.text(
            0.5,
            yi + 0.055,
            f"{row[ADMET]:.3f}{admet_suffix}",
            ha="center",
            va="center",
            fontsize=6.45,
            color=CORAL_STRONG if better else TEXT,
            fontweight="bold" if better else "normal",
            zorder=2,
        )
        top5_axis.text(
            0.5,
            yi + 0.055,
            f"{row['Top5Mean']:.3f}",
            ha="center",
            va="center",
            fontsize=6.45,
            color=TEAL if not better else TEXT,
            fontweight="bold" if not better else "normal",
            zorder=2,
        )
    return figure


def save_outputs(figure: plt.Figure, frame: pd.DataFrame, prefix: Path, captured: str) -> None:
    prefix.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(prefix.with_suffix(".png"), dpi=300)
    figure.savefig(prefix.with_suffix(".pdf"))
    figure.savefig(prefix.with_suffix(".svg"))
    figure.savefig(prefix.with_suffix(".tiff"), dpi=600, pil_kwargs={"compression": "tiff_lzw"})
    source_path = prefix.with_name(prefix.name + "_source_data").with_suffix(".csv")
    frame.drop(columns=["TaskLabel", "ADMETBetter"]).to_csv(source_path, index=False, float_format="%.6f")
    notes_path = prefix.with_name(prefix.name + "_notes").with_suffix(".md")
    notes_path.write_text(
        "\n".join(
            [
                "# Figure 2A comparator notes",
                "",
                f"- TDC snapshot captured at: `{captured}`.",
                "- Eligibility: external models with reported results on all 22 endpoints.",
                "- Fixed cohort selected by lowest mean rank across the 22 endpoints: MapLight + GNN (3.364), MapLight (4.727), Chemprop-RDKit (9.318), ContextPred (10.545), and AttrMasking (10.773).",
                "- Comparator: within-endpoint arithmetic mean of those same five fixed models.",
                "- Bioavailability Ma: corrected values replace the corrupted public page for all five cohort models. Chemprop-RDKit was rerun on compute-0003 with Chemprop 1.6.1, RDKit 2D normalized features and TDC scaffold seeds 1-5 (0.682 +/- 0.026 AUROC).",
                "- Positive relative difference indicates better ADMET-EvO performance; MAE tasks are direction-inverted.",
                "- A broken axis displays the Half-Life outlier while preserving resolution for the other endpoints (range: "
                f"{frame['RelativeDifference'].min():+.1f}% to {frame['RelativeDifference'].max():+.1f}%).",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def main() -> None:
    args = parse_args()
    frame, captured = load_data(args.snapshot, args.comparison, args.chemprop_bio)
    figure = build_figure(frame)
    save_outputs(figure, frame, args.output_prefix, captured)
    plt.close(figure)
    print(args.output_prefix)


if __name__ == "__main__":
    main()
