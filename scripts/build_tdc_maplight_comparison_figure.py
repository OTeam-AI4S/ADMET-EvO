#!/usr/bin/env python3
"""Build the polished TDC-22 ADMET-EvO versus MapLight+GNN figure."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")

import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ROOT / "reports" / "model_comparison_tdc22_8models_2026-08-25.csv"
DEFAULT_OUTPUT = ROOT / "outputs" / "figures" / "tdc_admetevo_maplight_comparison_polished"
FONT_REGULAR = ROOT / "assets" / "fonts" / "msyh.ttc"
FONT_BOLD = ROOT / "assets" / "fonts" / "msyhbd.ttc"

ADMET = "ADMET-EvO"
MAPLIGHT = "MapLight+GNN_author_published"
MODEL_COLUMNS = [
    ADMET,
    "MiniMol",
    "MolE",
    "MapLight_author_published",
    MAPLIGHT,
    "AttentiveFP",
    "AttrMasking",
    "ContextPred",
]

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

# The method colors are the swapped colors requested from the source artwork.
ADMET_COLOR = "#CF6559"
MAPLIGHT_COLOR = "#188977"
RANK_FIRST = "#E48A0A"
RANK_TOP3 = "#2B70A2"
RANK_OTHER = "#9BA8B3"
TEXT = "#26333E"
MUTED = "#5F6D7A"
GRID = "#D7DEE4"
ROW_MAPLIGHT_STRONG = "#EEF2F5"
ROW_ADMET_LEAD = "#FFF2C9"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-prefix", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def configure_font() -> None:
    for font_path in (FONT_REGULAR, FONT_BOLD):
        if not font_path.exists():
            raise FileNotFoundError(f"Required Microsoft YaHei font is missing: {font_path}")
        fm.fontManager.addfont(font_path)

    matched = Path(fm.findfont(fm.FontProperties(family="Microsoft YaHei"))).resolve()
    if matched not in {FONT_REGULAR.resolve(), FONT_BOLD.resolve()}:
        raise RuntimeError(f"Microsoft YaHei did not resolve to the project font: {matched}")

    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Microsoft YaHei", "Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "font.size": 8.0,
            "axes.linewidth": 0.7,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "savefig.facecolor": "white",
        }
    )


def load_data(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    required = {"Task", "Direction", ADMET, MAPLIGHT, *MODEL_COLUMNS}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    if len(frame) != 22 or set(frame["Task"]) != set(DISPLAY_NAMES):
        raise ValueError("Input does not match the frozen TDC-22 task set")
    if frame[MODEL_COLUMNS].isna().any().any():
        raise ValueError("Model comparison contains missing values")

    maximize = frame["Direction"].eq("maximize")
    minimize = frame["Direction"].eq("minimize")
    if not (maximize | minimize).all():
        raise ValueError("Direction must be 'maximize' or 'minimize'")

    oriented = frame[MODEL_COLUMNS].mul(np.where(maximize, 1.0, -1.0), axis=0)
    frame = frame.copy()
    frame["Rank"] = oriented.rank(axis=1, ascending=False, method="min")[ADMET].astype(int)
    frame["RelativeDifference"] = np.where(
        maximize,
        (frame[ADMET] - frame[MAPLIGHT]) / frame[MAPLIGHT] * 100.0,
        (frame[MAPLIGHT] - frame[ADMET]) / frame[MAPLIGHT] * 100.0,
    )
    frame["ADMETBetter"] = frame["RelativeDifference"] > 0
    frame["TaskLabel"] = frame["Task"].map(DISPLAY_NAMES)
    return frame


def clean_axis(axis: plt.Axes) -> None:
    for spine in axis.spines.values():
        spine.set_visible(False)
    axis.tick_params(axis="both", length=0, colors=MUTED, labelsize=7.7)


def add_row_backgrounds(axis: plt.Axes, frame: pd.DataFrame) -> None:
    for index, row in frame.iterrows():
        if bool(row["ADMETBetter"]):
            color = ROW_ADMET_LEAD
        elif float(row["RelativeDifference"]) < -6.0:
            color = ROW_MAPLIGHT_STRONG
        else:
            continue
        axis.axhspan(index - 0.5, index + 0.5, color=color, linewidth=0, zorder=0)


def rank_color(rank: int) -> str:
    if rank == 1:
        return RANK_FIRST
    if rank <= 3:
        return RANK_TOP3
    return RANK_OTHER


def build_figure(frame: pd.DataFrame) -> plt.Figure:
    configure_font()

    # Nature-style double-column width with extra height for larger type.
    figure = plt.figure(figsize=(183 / 25.4, 164 / 25.4), facecolor="white")
    grid = figure.add_gridspec(
        1,
        4,
        width_ratios=[2.65, 4.08, 0.86, 0.90],
        left=0.235,
        right=0.985,
        bottom=0.045,
        top=0.92,
        wspace=0.14,
    )
    rank_ax = figure.add_subplot(grid[0, 0])
    diff_ax = figure.add_subplot(grid[0, 1], sharey=rank_ax)
    admet_ax = figure.add_subplot(grid[0, 2], sharey=rank_ax)
    maplight_ax = figure.add_subplot(grid[0, 3], sharey=rank_ax)
    axes = [rank_ax, diff_ax, admet_ax, maplight_ax]

    y = np.arange(len(frame), dtype=float)
    for axis in axes:
        add_row_backgrounds(axis, frame)
        clean_axis(axis)
        axis.set_ylim(len(frame) - 0.5, -0.5)

    rank_ax.set_xlim(0.65, 8.35)
    rank_ax.set_xticks(np.arange(1, 9))
    rank_ax.xaxis.tick_top()
    rank_ax.tick_params(axis="x", pad=5)
    rank_ax.set_title("Rank (lower is better)", fontsize=9.2, color=MUTED, pad=10)
    for tick in np.arange(1, 9):
        rank_ax.axvspan(tick - 0.012, tick + 0.012, color=GRID, linewidth=0, zorder=0.5)
    rank_ax.set_yticks(y)
    rank_ax.set_yticklabels(frame["TaskLabel"], fontsize=7.8, color=TEXT)
    for label, admet_better in zip(rank_ax.get_yticklabels(), frame["ADMETBetter"], strict=True):
        label.set_fontweight("bold" if bool(admet_better) else "normal")
    rank_ax.tick_params(axis="y", pad=8)

    for yi, rank in zip(y, frame["Rank"], strict=True):
        line_end = max(1.0, float(rank) - 0.23)
        if line_end > 1.0:
            rank_ax.barh(
                yi,
                line_end - 1.0,
                left=1.0,
                height=0.055,
                color="#C9D2D9",
                edgecolor="none",
                zorder=2,
            )
        rank_ax.scatter(
            [rank],
            [yi],
            s=84,
            color=rank_color(int(rank)),
            edgecolor="white",
            linewidth=0.9,
            zorder=3,
        )
        rank_ax.text(
            rank,
            yi,
            str(rank),
            ha="center",
            va="center",
            fontsize=7.0,
            fontweight="bold",
            color="white",
            zorder=4,
        )

    diff_ax.set_xlim(-10.5, 30.0)
    diff_ax.set_xticks([-10, 0, 10, 20])
    diff_ax.set_xticklabels(["−10", "0", "+10", "+20"])
    diff_ax.xaxis.tick_top()
    diff_ax.tick_params(axis="x", pad=5)
    diff_ax.set_title(
        "Direction-adjusted relative difference (%)",
        fontsize=9.2,
        color=MUTED,
        pad=10,
    )
    for tick in (-10, 10, 20):
        diff_ax.axvspan(tick - 0.035, tick + 0.035, color=GRID, linewidth=0, zorder=0.5)
    diff_ax.axvline(0, color=TEXT, linewidth=1.0, zorder=2)
    diff_ax.tick_params(axis="y", labelleft=False)

    for yi, value in zip(y, frame["RelativeDifference"], strict=True):
        color = ADMET_COLOR if value >= 0 else MAPLIGHT_COLOR
        diff_ax.barh(yi, value, height=0.54, color=color, edgecolor="none", zorder=2.5)
        if value >= 0:
            x, alignment = value + 0.8, "left"
        else:
            x, alignment = value - 0.8, "right"
        diff_ax.text(
            x,
            yi,
            f"{value:+.1f}%",
            ha=alignment,
            va="center",
            fontsize=7.35,
            color=TEXT,
            fontweight="bold" if value > 0 else "normal",
        )

    for value_axis in (admet_ax, maplight_ax):
        value_axis.set_xlim(0, 1)
        value_axis.set_xticks([])
        value_axis.tick_params(axis="y", labelleft=False)

    admet_ax.set_title(
        "ADMET-\nEvO",
        fontsize=8.2,
        fontweight="bold",
        color=ADMET_COLOR,
        pad=9,
        linespacing=1.18,
    )
    maplight_ax.set_title(
        "MapLight\n+GNN",
        fontsize=8.2,
        fontweight="bold",
        color=MAPLIGHT_COLOR,
        pad=9,
        linespacing=1.18,
    )

    for yi, row in frame.iterrows():
        admet_text = f"{row[ADMET]:.3f}"
        if row["Task"] == "Bioavailability_Ma":
            admet_text += "*"
        admet_ax.text(
            0.5,
            yi,
            admet_text,
            ha="center",
            va="center",
            fontsize=7.35,
            color=TEXT,
            fontweight="bold" if bool(row["ADMETBetter"]) else "normal",
        )
        maplight_ax.text(
            0.5,
            yi,
            f"{row[MAPLIGHT]:.3f}",
            ha="center",
            va="center",
            fontsize=7.35,
            color=TEXT,
            fontweight="normal" if bool(row["ADMETBetter"]) else "bold",
        )

    return figure


def save_figure(figure: plt.Figure, prefix: Path) -> None:
    prefix.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(prefix.with_suffix(".png"), dpi=300, bbox_inches="tight", pad_inches=0.03)
    figure.savefig(prefix.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.03)
    figure.savefig(prefix.with_suffix(".svg"), bbox_inches="tight", pad_inches=0.03)
    figure.savefig(
        prefix.with_suffix(".tiff"),
        dpi=600,
        bbox_inches="tight",
        pad_inches=0.03,
        pil_kwargs={"compression": "tiff_lzw"},
    )


def main() -> None:
    args = parse_args()
    frame = load_data(args.input)
    figure = build_figure(frame)
    save_figure(figure, args.output_prefix)
    plt.close(figure)
    print(args.output_prefix)


if __name__ == "__main__":
    main()
