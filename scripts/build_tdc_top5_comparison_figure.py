#!/usr/bin/env python3
"""Build ADMET-EvO versus historical TDC fifth-place threshold across 22 tasks."""

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
DEFAULT_INPUT = ROOT / "reports" / "goal_scorecard_v27.json"
DEFAULT_OUTPUT = ROOT / "outputs" / "figures" / "tdc_admetevo_top5_comparison_polished"
FONT_REGULAR = ROOT / "assets" / "fonts" / "msyh.ttc"
FONT_BOLD = ROOT / "assets" / "fonts" / "msyhbd.ttc"

ADMET_COLOR = "#C95F55"
TOP5_COLOR = "#178778"
TEXT = "#26333E"
MUTED = "#657483"
GRID = "#D7DEE4"
ROW_TOP5_STRONG = "#EEF2F5"
ROW_ADMET_LEAD = "#FFF2C9"

DISPLAY_NAMES = {
    "AMES": "AMES",
    "BBB_Martins": "BBB Martins",
    "Bioavailability_Ma": "Bioavailability Ma",
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-prefix", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def configure_font() -> None:
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
            "font.size": 8,
            "axes.linewidth": 0.7,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "savefig.facecolor": "white",
        }
    )


def load_data(path: Path) -> tuple[pd.DataFrame, str]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if len(payload.get("tasks", [])) != 22 or not payload.get("official_results_complete"):
        raise ValueError("Expected the complete frozen v2.7 TDC-22 scorecard")
    frame = pd.DataFrame(payload["tasks"])
    required = {"task", "metric", "direction", "score", "top5_threshold"}
    if missing := required.difference(frame.columns):
        raise ValueError(f"Scorecard is missing fields: {sorted(missing)}")
    if set(frame["task"]) != set(DISPLAY_NAMES):
        raise ValueError("Scorecard task set does not match TDC ADMET-22")
    if not frame["direction"].isin(["maximize", "minimize"]).all():
        raise ValueError("Unknown metric direction")
    frame = frame.copy()
    frame["RelativeDifference"] = np.where(
        frame["direction"].eq("maximize"),
        (frame["score"] - frame["top5_threshold"]) / frame["top5_threshold"] * 100.0,
        (frame["top5_threshold"] - frame["score"]) / frame["top5_threshold"] * 100.0,
    )
    frame["ADMETBetter"] = frame["RelativeDifference"] >= 0
    frame["TaskLabel"] = frame["task"].map(DISPLAY_NAMES)
    captured = str(payload.get("leaderboard_captured_at", "historical snapshot"))
    return frame, captured


def clean_axis(axis: plt.Axes) -> None:
    for spine in axis.spines.values():
        spine.set_visible(False)
    axis.tick_params(axis="both", length=0, colors=MUTED, labelsize=7.7)


def add_row_backgrounds(axis: plt.Axes, frame: pd.DataFrame) -> None:
    for index, row in frame.iterrows():
        if bool(row["ADMETBetter"]):
            color = ROW_ADMET_LEAD
        elif float(row["RelativeDifference"]) < -6.0:
            color = ROW_TOP5_STRONG
        else:
            continue
        axis.axhspan(index - 0.5, index + 0.5, color=color, linewidth=0, zorder=0)


def build_figure(frame: pd.DataFrame, captured: str) -> plt.Figure:
    configure_font()
    figure = plt.figure(figsize=(183 / 25.4, 164 / 25.4), facecolor="white")
    grid = figure.add_gridspec(
        1,
        3,
        width_ratios=[6.30, 0.88, 1.02],
        left=0.235,
        right=0.988,
        bottom=0.055,
        top=0.915,
        wspace=0.12,
    )
    difference_axis = figure.add_subplot(grid[0, 0])
    admet_axis = figure.add_subplot(grid[0, 1], sharey=difference_axis)
    top5_axis = figure.add_subplot(grid[0, 2], sharey=difference_axis)
    axes = (difference_axis, admet_axis, top5_axis)
    y = np.arange(len(frame), dtype=float)
    for axis in axes:
        add_row_backgrounds(axis, frame)
        clean_axis(axis)
        axis.set_ylim(len(frame) - 0.5, -0.5)

    difference_axis.set_xlim(-27, 27)
    difference_axis.set_xticks([-20, -10, 0, 10, 20])
    difference_axis.set_xticklabels(["−20", "−10", "0", "+10", "+20"])
    difference_axis.xaxis.tick_top()
    difference_axis.tick_params(axis="x", pad=5)
    difference_axis.set_title(
        "Direction-adjusted relative difference vs TDC No. 5 (%)",
        fontsize=9.2,
        color=MUTED,
        pad=10,
    )
    for tick in (-20, -10, 10, 20):
        difference_axis.axvspan(tick - 0.035, tick + 0.035, color=GRID, linewidth=0, zorder=0.5)
    difference_axis.axvline(0, color=TEXT, linewidth=1.0, zorder=2)
    difference_axis.set_yticks(y)
    difference_axis.set_yticklabels(frame["TaskLabel"], fontsize=7.9, color=TEXT)
    difference_axis.tick_params(axis="y", pad=9)
    for label, is_better in zip(difference_axis.get_yticklabels(), frame["ADMETBetter"], strict=True):
        label.set_fontweight("bold" if bool(is_better) else "normal")

    for yi, value in zip(y, frame["RelativeDifference"], strict=True):
        color = ADMET_COLOR if value >= 0 else TOP5_COLOR
        difference_axis.barh(yi, value, height=0.52, color=color, edgecolor="none", zorder=2.5)
        if value >= 0:
            x, alignment = value + 0.75, "left"
        else:
            x, alignment = value - 0.75, "right"
        difference_axis.text(
            x,
            yi,
            f"{value:+.1f}%",
            ha=alignment,
            va="center",
            fontsize=7.35,
            color=TEXT,
            fontweight="bold" if value >= 0 else "normal",
        )

    for value_axis in (admet_axis, top5_axis):
        value_axis.set_xlim(0, 1)
        value_axis.set_xticks([])
        value_axis.tick_params(axis="y", labelleft=False)
    admet_axis.set_title("ADMET-\nEvO", fontsize=8.2, fontweight="bold", color=ADMET_COLOR, pad=9, linespacing=1.18)
    top5_axis.set_title("TDC\nNo. 5", fontsize=8.2, fontweight="bold", color=TOP5_COLOR, pad=9, linespacing=1.18)

    for yi, row in frame.iterrows():
        admet_axis.text(
            0.5,
            yi,
            f"{row['score']:.3f}",
            ha="center",
            va="center",
            fontsize=7.4,
            color=TEXT,
            fontweight="bold" if bool(row["ADMETBetter"]) else "normal",
        )
        top5_axis.text(
            0.5,
            yi,
            f"{row['top5_threshold']:.3f}",
            ha="center",
            va="center",
            fontsize=7.4,
            color=TEXT,
            fontweight="normal" if bool(row["ADMETBetter"]) else "bold",
        )

    achieved = int(frame["ADMETBetter"].sum())
    figure.text(
        0.988,
        0.975,
        f"{achieved}/22 tasks meet or exceed the historical fifth-place score",
        ha="right",
        va="top",
        fontsize=7.2,
        color=MUTED,
    )
    figure.text(
        0.988,
        0.018,
        f"Historical TDC leaderboard snapshot: {captured}",
        ha="right",
        va="bottom",
        fontsize=6.4,
        color=MUTED,
    )
    return figure


def save_figure(figure: plt.Figure, prefix: Path) -> None:
    prefix.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(prefix.with_suffix(".png"), dpi=300)
    figure.savefig(prefix.with_suffix(".pdf"))
    figure.savefig(prefix.with_suffix(".svg"))
    figure.savefig(prefix.with_suffix(".tiff"), dpi=600, pil_kwargs={"compression": "tiff_lzw"})


def main() -> None:
    args = parse_args()
    frame, captured = load_data(args.input)
    figure = build_figure(frame, captured)
    save_figure(figure, args.output_prefix)
    plt.close(figure)
    print(args.output_prefix)


if __name__ == "__main__":
    main()
