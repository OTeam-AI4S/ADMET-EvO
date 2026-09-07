"""Build the Nature-style performance--cost figure from frozen fresh-seed results.

The plotting backend is intentionally Python/matplotlib only.  The script reads
the frozen router report, validates the quantities used in the manuscript, and
exports a reproducible source-data table together with vector and raster figures.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/admetevo-matplotlib")

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter, FixedLocator


MM_PER_INCH = 25.4
FIGURE_WIDTH_MM = 183
FIGURE_HEIGHT_MM = 112
MARGIN = -0.005

DARK = "#263238"
GREY = "#7B8790"
LIGHT_GREY = "#D7DEE2"
TEAL = "#16817A"
TEAL_LIGHT = "#DCEFEB"
RED = "#C44E52"
ORANGE = "#D98E2B"
BLUE_GREY = "#667C8B"


DISPLAY_NAMES = {
    "AMES": "AMES",
    "BBB_Martins": "BBB",
    "Bioavailability_Ma": "Bioavailability",
    "CYP2C9_Substrate_CarbonMangels": "CYP2C9 substrate",
    "CYP2C9_Veith": "CYP2C9 Veith",
    "CYP2D6_Substrate_CarbonMangels": "CYP2D6 substrate",
    "CYP2D6_Veith": "CYP2D6 Veith",
    "CYP3A4_Substrate_CarbonMangels": "CYP3A4 substrate",
    "CYP3A4_Veith": "CYP3A4 Veith",
    "Caco2_Wang": "Caco-2",
    "Clearance_Hepatocyte_AZ": "Hepatocyte clearance",
    "Clearance_Microsome_AZ": "Microsome clearance",
    "DILI": "DILI",
    "HIA_Hou": "HIA",
    "Half_Life_Obach": "Half-life",
    "LD50_Zhu": "LD50",
    "Lipophilicity_AstraZeneca": "Lipophilicity",
    "PPBR_AZ": "PPBR",
    "Pgp_Broccatelli": "P-gp",
    "Solubility_AqSolDB": "Solubility",
    "VDss_Lombardo": "VDss",
    "hERG": "hERG",
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_report(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _extract_rows(report: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for item in report["tasks"]:
        score = item["summaries"]["score"]
        aware = item["summaries"]["metric_risk_guard"]
        score_seconds = float(score["elapsed_seconds"])
        aware_seconds = float(aware["elapsed_seconds"])
        ratio = aware_seconds / score_seconds
        gain = float(item["guard_minus_score_oriented_gain"])
        rows.append(
            {
                "task": item["task"],
                "display_name": DISPLAY_NAMES[item["task"]],
                "metric": item["metric"],
                "score_only_mean": float(score["mean"]),
                "compute_aware_mean": float(aware["mean"]),
                "oriented_difference": gain,
                "score_only_seconds": score_seconds,
                "compute_aware_seconds": aware_seconds,
                "cost_ratio": ratio,
                "cost_change_percent": 100.0 * (ratio - 1.0),
                "within_margin": gain >= MARGIN,
                "fresh_seed_count": len(score["scores"]),
            }
        )
    return rows


def _validate(report: dict[str, Any], rows: list[dict[str, Any]]) -> dict[str, float]:
    if len(rows) != 22 or report["task_count"] != 22:
        raise ValueError("Expected all 22 frozen endpoints")
    if report["fresh_validation_seeds"] != [4, 5]:
        raise ValueError("Expected fresh evaluation seeds 4--5")
    if set(DISPLAY_NAMES) != {row["task"] for row in rows}:
        raise ValueError("Display-name mapping does not match the frozen endpoint set")
    if any(row["score_only_seconds"] <= 0 for row in rows):
        raise ValueError("All score-only elapsed times must be positive")
    cost_ratios = np.asarray([row["cost_ratio"] for row in rows], dtype=float)
    if any(row["compute_aware_seconds"] <= 0 for row in rows) or np.any(cost_ratios <= 0):
        raise ValueError("All compute-aware elapsed times and log-scale ratios must be positive")
    if any(row["fresh_seed_count"] != 2 for row in rows):
        raise ValueError("Each policy summary must contain two fresh-seed scores")

    score_hours = sum(row["score_only_seconds"] for row in rows) / 3600.0
    aware_hours = sum(row["compute_aware_seconds"] for row in rows) / 3600.0
    ratio = aware_hours / score_hours
    mean_gain = float(np.mean([row["oriented_difference"] for row in rows]))
    within = sum(row["within_margin"] for row in rows)
    ci_low, ci_high = map(float, report["guard_minus_score"]["bootstrap_mean_95_ci"])

    expected_ratio = float(report["claim_gate"]["aggregate_guard_cost_ratio_vs_score"])
    expected_mean = float(report["guard_minus_score"]["mean"])
    if not math.isclose(ratio, expected_ratio, rel_tol=0, abs_tol=1e-12):
        raise ValueError("Aggregate cost ratio does not reproduce the frozen report")
    if not math.isclose(mean_gain, expected_mean, rel_tol=0, abs_tol=1e-12):
        raise ValueError("Mean oriented difference does not reproduce the frozen report")
    if ci_low <= MARGIN:
        raise ValueError("The reported confidence interval does not clear the margin")

    return {
        "score_hours": score_hours,
        "aware_hours": aware_hours,
        "ratio": ratio,
        "reduction_percent": 100.0 * (1.0 - ratio),
        "mean_gain": mean_gain,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "within": float(within),
    }


def _style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "font.size": 6.5,
            "axes.titlesize": 7.2,
            "axes.labelsize": 6.5,
            "xtick.labelsize": 5.8,
            "ytick.labelsize": 5.8,
            "legend.fontsize": 5.7,
            "axes.linewidth": 0.7,
            "axes.edgecolor": DARK,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "xtick.major.width": 0.6,
            "ytick.major.width": 0.6,
            "xtick.major.size": 2.5,
            "ytick.major.size": 2.5,
            "legend.frameon": False,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def _minus(value: str) -> str:
    return value.replace("-", "−")


def _x_ratio(value: float, _position: float) -> str:
    if value >= 1:
        return f"{value:g}×"
    return f"{value:.2g}×"


def _y_difference(value: float, _position: float) -> str:
    if abs(value) < 5e-7:
        return "0"
    return _minus(f"{value:.3f}")


def _panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(
        -0.12,
        1.045,
        label,
        transform=ax.transAxes,
        fontsize=8,
        fontweight="bold",
        va="bottom",
        ha="left",
        color="black",
        clip_on=False,
    )


def _hero_panel(ax: plt.Axes, rows: list[dict[str, Any]]) -> None:
    x_min, x_max = 0.018, 11.0
    y_min, y_max = -0.0105, 0.0052
    ax.axvspan(x_min, 1.0, color=TEAL_LIGHT, alpha=0.62, linewidth=0, zorder=0)
    ax.axhspan(MARGIN, y_max, color=TEAL, alpha=0.045, linewidth=0, zorder=0)
    ax.axvline(1.0, color=GREY, linestyle=(0, (2.2, 2.2)), linewidth=0.8, zorder=1)
    ax.axhline(0.0, color=DARK, linewidth=0.8, zorder=1)
    ax.axhline(MARGIN, color=RED, linestyle=(0, (4, 2.5)), linewidth=0.85, zorder=1)

    ordinary = [r for r in rows if r["within_margin"] and r["cost_ratio"] <= 1.0]
    regressions = [r for r in rows if not r["within_margin"]]
    overspend = [r for r in rows if r["cost_ratio"] > 1.0]

    ax.scatter(
        [r["cost_ratio"] for r in ordinary],
        [r["oriented_difference"] for r in ordinary],
        s=27,
        marker="o",
        facecolor=TEAL,
        edgecolor="white",
        linewidth=0.6,
        alpha=0.92,
        zorder=3,
    )
    ax.scatter(
        [r["cost_ratio"] for r in regressions],
        [r["oriented_difference"] for r in regressions],
        s=36,
        marker="v",
        facecolor=RED,
        edgecolor="white",
        linewidth=0.6,
        zorder=4,
    )
    ax.scatter(
        [r["cost_ratio"] for r in overspend],
        [r["oriented_difference"] for r in overspend],
        s=42,
        marker="D",
        facecolor=ORANGE,
        edgecolor="white",
        linewidth=0.6,
        zorder=5,
    )

    by_task = {row["task"]: row for row in rows}
    annotations = {
        "LD50_Zhu": (8, 8, "left"),
        "BBB_Martins": (8, 5, "left"),
        "CYP2D6_Substrate_CarbonMangels": (-8, -18, "right"),
        "CYP2C9_Veith": (-8, -22, "right"),
    }
    for task, (dx, dy, horizontal_alignment) in annotations.items():
        row = by_task[task]
        detail = row["display_name"]
        if task == "CYP2C9_Veith":
            detail += "\n+0.00190; 7.68× time"
        elif task == "LD50_Zhu":
            detail += "\n−0.00124; 0.027× time"
        ax.annotate(
            detail,
            (row["cost_ratio"], row["oriented_difference"]),
            xytext=(dx, dy),
            textcoords="offset points",
            fontsize=5.5,
            color=DARK,
            ha=horizontal_alignment,
            va="center",
            linespacing=1.1,
            arrowprops={"arrowstyle": "-", "color": GREY, "lw": 0.55, "shrinkA": 2, "shrinkB": 3},
            zorder=6,
        )

    exact_ties = [
        row
        for row in rows
        if abs(row["oriented_difference"]) < 1e-15 and abs(row["cost_ratio"] - 1.0) < 1e-15
    ]
    ax.annotate(
        f"{len(exact_ties)} endpoints unchanged",
        (1.0, 0.0),
        xytext=(-47, 18),
        textcoords="offset points",
        fontsize=5.5,
        color=BLUE_GREY,
        ha="right",
        va="bottom",
        arrowprops={"arrowstyle": "-", "color": BLUE_GREY, "lw": 0.55, "shrinkA": 2, "shrinkB": 3},
        zorder=6,
    )

    ax.text(0.026, 0.00455, "lower measured time", fontsize=5.5, color=TEAL, va="top")
    ax.text(9.7, MARGIN + 0.00025, "practical margin", fontsize=5.4, color=RED, ha="right", va="bottom")

    ax.set_xscale("log")
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.xaxis.set_major_locator(FixedLocator([0.03, 0.1, 0.3, 1.0, 3.0, 10.0]))
    ax.xaxis.set_major_formatter(FuncFormatter(_x_ratio))
    ax.xaxis.set_minor_locator(FixedLocator([]))
    ax.yaxis.set_major_locator(FixedLocator([-0.010, -0.005, 0.0, 0.005]))
    ax.yaxis.set_major_formatter(FuncFormatter(_y_difference))
    ax.grid(axis="y", color=LIGHT_GREY, linewidth=0.5, zorder=0)
    ax.set_xlabel("Selected-recipe fitting-time ratio\n(compute-aware / score-only; log scale)")
    ax.set_ylabel("Oriented performance difference\n(compute-aware − score-only)")
    ax.set_title("Endpoint-level performance–cost trade-off", loc="left", fontweight="bold", pad=5)

    handles = [
        Line2D([0], [0], marker="o", linestyle="none", markerfacecolor=TEAL, markeredgecolor="white", markersize=5.1, label="Within margin"),
        Line2D([0], [0], marker="v", linestyle="none", markerfacecolor=RED, markeredgecolor="white", markersize=5.6, label="Below margin"),
        Line2D([0], [0], marker="D", linestyle="none", markerfacecolor=ORANGE, markeredgecolor="white", markersize=5.0, label="Higher cost"),
    ]
    ax.legend(handles=handles, loc="lower left", bbox_to_anchor=(0.0, 0.015), ncol=3, handletextpad=0.35, columnspacing=0.8)
    _panel_label(ax, "a")


def _aggregate_panel(ax: plt.Axes, stats: dict[str, float]) -> None:
    values = [stats["score_hours"], stats["aware_hours"]]
    bars = ax.bar([0, 1], values, width=0.62, color=[GREY, TEAL], edgecolor="none", zorder=3)
    ax.set_xticks([0, 1], ["Score-only", "Compute-aware"])
    ax.set_ylabel("Measured fitting time (h)")
    ax.set_ylim(0, 2.72)
    ax.set_yticks([0, 1, 2])
    ax.grid(axis="y", color=LIGHT_GREY, linewidth=0.5, zorder=0)
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)
    for bar, value in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + 0.09,
            f"{value:.3f} h",
            ha="center",
            va="bottom",
            fontsize=6.1,
            color=DARK,
            fontweight="bold",
        )
    ax.text(
        0.96,
        0.88,
        f"{stats['reduction_percent']:.1f}% lower",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=7.1,
        color=TEAL,
        fontweight="bold",
    )
    ax.text(
        0.98,
        0.63,
        "cost ratio 0.278×",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=5.7,
        color=DARK,
    )
    ax.set_title("Aggregate measured time", loc="left", fontweight="bold", pad=5)
    _panel_label(ax, "b")


def _noninferiority_panel(ax: plt.Axes, stats: dict[str, float]) -> None:
    x_min, x_max = -0.0062, 0.0018
    ax.axvspan(MARGIN, x_max, color=TEAL_LIGHT, alpha=0.62, linewidth=0, zorder=0)
    ax.axvline(MARGIN, color=RED, linestyle=(0, (4, 2.5)), linewidth=0.85, zorder=1)
    ax.axvline(0.0, color=GREY, linewidth=0.7, zorder=1)
    mean = stats["mean_gain"]
    low = stats["ci_low"]
    high = stats["ci_high"]
    ax.errorbar(
        mean,
        0.49,
        xerr=np.array([[mean - low], [high - mean]]),
        fmt="o",
        color=TEAL,
        ecolor=TEAL,
        elinewidth=2.0,
        capsize=4,
        capthick=1.0,
        markersize=5.4,
        markeredgecolor="white",
        markeredgewidth=0.6,
        zorder=3,
    )
    ax.text(0.22, 0.91, "95% bootstrap CI", transform=ax.transAxes, ha="left", va="top", fontsize=5.7, color=DARK)
    ax.text(0.22, 0.78, "[−0.00148, 0.00059]", transform=ax.transAxes, ha="left", va="top", fontsize=6.1, color=TEAL, fontweight="bold")
    ax.text(0.72, 0.58, "mean −0.00035", transform=ax.transAxes, ha="right", va="bottom", fontsize=5.7, color=DARK)
    ax.text(0.22, 0.12, "margin −0.005", transform=ax.transAxes, ha="left", va="bottom", fontsize=5.35, color=RED)
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(0, 1)
    ax.set_yticks([])
    ax.xaxis.set_major_locator(FixedLocator([MARGIN, -0.0025, 0.0]))
    ax.xaxis.set_major_formatter(FuncFormatter(_y_difference))
    ax.set_xlabel("Mean oriented difference")
    ax.spines["left"].set_visible(False)
    ax.set_title("Practical noninferiority", loc="left", fontweight="bold", pad=5)
    _panel_label(ax, "c")


def _build_figure(rows: list[dict[str, Any]], stats: dict[str, float]) -> plt.Figure:
    _style()
    fig = plt.figure(figsize=(FIGURE_WIDTH_MM / MM_PER_INCH, FIGURE_HEIGHT_MM / MM_PER_INCH))
    grid = fig.add_gridspec(
        2,
        2,
        width_ratios=[2.35, 1.0],
        height_ratios=[1.02, 0.98],
        left=0.095,
        right=0.985,
        bottom=0.135,
        top=0.945,
        wspace=0.38,
        hspace=0.55,
    )
    hero = fig.add_subplot(grid[:, 0])
    aggregate = fig.add_subplot(grid[0, 1])
    noninferiority = fig.add_subplot(grid[1, 1])
    _hero_panel(hero, rows)
    _aggregate_panel(aggregate, stats)
    _noninferiority_panel(noninferiority, stats)
    fig.text(
        0.985,
        0.025,
        "Development seeds 1–3; fresh evaluation seeds 4–5; n = 22 endpoints",
        ha="right",
        va="bottom",
        fontsize=5.5,
        color=GREY,
    )
    return fig


def _write_source_data(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = [
        "task",
        "display_name",
        "metric",
        "score_only_mean",
        "compute_aware_mean",
        "oriented_difference",
        "score_only_seconds",
        "compute_aware_seconds",
        "cost_ratio",
        "cost_change_percent",
        "within_margin",
        "fresh_seed_count",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _write_legend(path: Path, stats: dict[str, float]) -> None:
    path.write_text(
        """# Figure legend and manuscript wording

## Suggested Results text

Across 22 endpoints evaluated on fresh scaffold seeds, compute-aware selection reduced the aggregate measured fitting time of the selected recipes from 2.247 h to 0.625 h (72.2% lower; 0.278× the score-only cost), while the mean oriented performance difference relative to score-only selection was −0.00035 (95% endpoint-bootstrap CI, −0.00148 to 0.00059). The confidence interval remained above the prespecified practical-noninferiority margin of −0.005, and 20 of 22 endpoints were within this margin. Efficiency was not uniform across endpoints: the compute-aware choice for CYP2C9 Veith increased AUPRC by 0.00190 but required 7.68-fold more fitting time, whereas LD50 accounted for the largest saving. These results support aggregate, panel-level efficiency rather than per-endpoint cost dominance.

## Suggested figure legend

**Figure 3 | Compute-aware selection preserves practical performance while reducing aggregate fitting time.** **a,** Oriented performance difference between compute-aware and score-only selection plotted against the ratio of measured selected-recipe fitting times for 22 ADMET endpoints. Positive differences favour compute-aware selection; values below one on the horizontal axis indicate lower measured time. The dashed red line marks the prespecified practical-noninferiority margin (−0.005). Thirteen coincident endpoints had identical selected recipes and are shown at a ratio of one and a difference of zero. Red triangles denote the two endpoints below the margin; the orange diamond denotes the CYP2C9 Veith overspend boundary case. **b,** Aggregate measured fitting time across endpoints for score-only and compute-aware selection. **c,** Mean oriented endpoint difference and 95% endpoint-bootstrap confidence interval; the interval remained above the noninferiority margin. Candidate selection used development seeds 1–3 and evaluation used fresh scaffold seeds 4–5. Times sum the measured fitting time of selected recipes across the two fresh seeds; n = 22 endpoints.

## CYP2C9 percentage wording

The CYP2C9 Veith cost ratio is 267.115 / 34.780 = 7.680. Therefore its **cost increase** is (7.680 − 1) × 100% = **+668.0%**. The equivalent “saving” expression, (1 − 7.680) × 100%, is **−668.0%**. It is not −688%; that value is an arithmetic or transcription error. For the paper, “7.68-fold more fitting time” is clearer than a negative saving percentage.
""",
        encoding="utf-8",
    )


def _write_qa_notes(path: Path, report_path: Path, rows: list[dict[str, Any]], stats: dict[str, float]) -> None:
    cyp = next(row for row in rows if row["task"] == "CYP2C9_Veith")
    notes = f"""# Figure QA notes

## Figure contract

- Core conclusion: compute-aware selection retains practical endpoint-level performance while reducing aggregate selected-recipe fitting time.
- Results-level question: does compute-aware selection reduce measured fitting cost without a practically material mean performance loss relative to score-only selection?
- Archetype: asymmetric quantitative grid with a hero endpoint-level trade-off panel.
- Backend: Python/matplotlib only.
- Final size: {FIGURE_WIDTH_MM} × {FIGURE_HEIGHT_MM} mm.
- Panel a: primary endpoint-level comparison and explicit boundary cases.
- Panel b: aggregate cost evidence.
- Panel c: noninferiority summary and uncertainty.
- Reviewer risk: the aggregate advantage is driven strongly by LD50 and is not uniform; CYP2C9 Veith costs {cyp['cost_ratio']:.2f}× more, and two endpoints fall below the margin. The frozen report's broader preregistered claim gate is false, so the figure supports only the score-only comparison stated here.

## Data integrity

- Source report SHA-256: `{_sha256(report_path)}`.
- Observations included: {len(rows)}/{len(rows)} endpoints; no exclusions or imputation.
- Development seeds: 1–3; fresh evaluation seeds: 4–5.
- Replicate unit for panel a: endpoint summaries over two fresh seeds.
- Oriented difference: compute-aware minus score-only, sign-adjusted so positive is better for every metric.
- Aggregate time: sum of measured selected-recipe fitting seconds over all endpoints and both fresh seeds.
- Center in panel c: arithmetic mean across 22 endpoint differences.
- Interval in panel c: 95% endpoint-bootstrap confidence interval from the frozen report, [{stats['ci_low']:.12f}, {stats['ci_high']:.12f}].
- Prespecified practical-noninferiority margin: {MARGIN}.
- Endpoints within margin: {int(stats['within'])}/22.
- No hypothesis test or multiple-comparison correction is claimed in this figure.
- Image-integrity notes: not applicable; all panels are vector quantitative graphics.

## Panel-by-panel audit

| Panel | Unique claim | Center/summary | Spread/interval | Replicate unit | Labels/legend | Pass |
|---|---|---|---|---|---|---|
| a | Endpoint effects are heterogeneous and expose savings, regressions and overspend | Raw endpoint summaries | None; each point is one endpoint | Endpoint | Four boundary cases and 13 coincident points directly labelled | Pending rendered QA |
| b | Aggregate measured time is 72.2% lower | Sum across endpoints | None; descriptive measured totals | Endpoint × fresh seed | Direct bar values | Pending rendered QA |
| c | Mean loss is above the practical margin | Mean across endpoints | 95% endpoint-bootstrap CI | Endpoint | Margin, zero and CI defined | Pending rendered QA |
"""
    path.write_text(notes, encoding="utf-8")


def build(repo_root: Path, output_dir: Path) -> list[Path]:
    root = repo_root.resolve()
    output = output_dir if output_dir.is_absolute() else root / output_dir
    output.mkdir(parents=True, exist_ok=True)
    report_path = root / "runs/v27-router-risk-guard/report.json"
    report = _load_report(report_path)
    rows = _extract_rows(report)
    stats = _validate(report, rows)

    source_path = output / "source_data_figure3_performance_cost.csv"
    legend_path = output / "figure3_performance_cost_nature_legend.md"
    qa_path = output / "figure3_performance_cost_nature_qa.md"
    _write_source_data(source_path, rows)
    _write_legend(legend_path, stats)
    _write_qa_notes(qa_path, report_path, rows, stats)

    fig = _build_figure(rows, stats)
    stem = output / "figure3_performance_cost_nature"
    svg_path = stem.with_suffix(".svg")
    pdf_path = stem.with_suffix(".pdf")
    tiff_path = stem.with_suffix(".tiff")
    png_path = stem.with_suffix(".png")
    fig.savefig(svg_path)
    fig.savefig(pdf_path)
    fig.savefig(tiff_path, dpi=600, pil_kwargs={"compression": "tiff_lzw"})
    fig.savefig(png_path, dpi=600)
    paths = [source_path, legend_path, qa_path, svg_path, pdf_path, tiff_path, png_path]
    plt.close(fig)
    return paths


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    parser.add_argument("--output-dir", type=Path, default=Path("paper/figures"))
    args = parser.parse_args()
    paths = build(args.repo_root, args.output_dir)
    print(json.dumps({"outputs": [str(path) for path in paths]}, indent=2))


if __name__ == "__main__":
    main()
