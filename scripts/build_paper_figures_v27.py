"""Build publication figures from the frozen v2.7 evidence bundle."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


NAVY = "#234E70"
TEAL = "#16817A"
GOLD = "#D99A20"
RED = "#C84C4C"
BLUE = "#4C78A8"
GRAY = "#68727D"
LIGHT = "#F4F6F8"
DARK = "#17212B"
METRIC_COLORS = {
    "auroc": "#4C78A8",
    "auprc": "#16817A",
    "mae": "#D99A20",
    "spearman": "#B05A8C",
}

SOURCE_FILES = {
    "scorecard": "reports/goal_scorecard_v27.json",
    "router_fresh": "runs/v27-router-risk-guard/report.json",
    "chemical_space": "reports/validation_calibration_residuals_v27.json",
    "manuscript": "MANUSCRIPT_V24.md",
    "router_source": "admet_agent/evidence_router.py",
    "guard_source": "admet_agent/guard.py",
    "executor_source": "admet_agent/frozen_executor.py",
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.titlesize": 11,
            "axes.labelsize": 9,
            "axes.edgecolor": "#B7BEC7",
            "axes.linewidth": 0.7,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "svg.hashsalt": "admet-evo-v27",
        }
    )


def _save(fig: plt.Figure, output_dir: Path, stem: str, title: str) -> list[Path]:
    paths = []
    for extension in ("svg", "png"):
        path = output_dir / f"{stem}.{extension}"
        metadata = (
            {"Date": None, "Creator": "ADMET-EvO v2.7", "Title": title}
            if extension == "svg"
            else {"Software": "ADMET-EvO v2.7", "Title": title}
        )
        fig.savefig(
            path,
            dpi=300 if extension == "png" else None,
            bbox_inches="tight",
            metadata=metadata,
        )
        paths.append(path)
    plt.close(fig)
    return paths


def _box(
    ax: plt.Axes,
    xy: tuple[float, float],
    width: float,
    height: float,
    text: str,
    *,
    color: str,
    subtitle: str | None = None,
    title_size: float = 8.3,
) -> None:
    x, y = xy
    patch = FancyBboxPatch(
        (x, y),
        width,
        height,
        boxstyle="round,pad=0.012,rounding_size=0.018",
        linewidth=1.2,
        edgecolor=color,
        facecolor=mpl.colors.to_rgba(color, 0.09),
    )
    ax.add_patch(patch)
    ax.text(x + width / 2, y + height * 0.62, text, ha="center", va="center", color=DARK, weight="bold", fontsize=title_size, linespacing=0.95)
    if subtitle:
        ax.text(x + width / 2, y + height * 0.23, subtitle, ha="center", va="center", color=GRAY, fontsize=6.7)


def _arrow(ax: plt.Axes, start: tuple[float, float], end: tuple[float, float], *, color: str = GRAY) -> None:
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=12,
            linewidth=1.1,
            color=color,
            shrinkA=2,
            shrinkB=2,
        )
    )


def figure_architecture(output_dir: Path) -> tuple[list[Path], dict[str, Any]]:
    fig, ax = plt.subplots(figsize=(8.2, 4.8))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.text(0.02, 0.95, "ADMET-EvO: evidence acquisition with a freeze-locked evaluation boundary", fontsize=12, weight="bold", color=DARK)

    ax.add_patch(FancyBboxPatch((0.015, 0.18), 0.735, 0.68, boxstyle="round,pad=0.01", facecolor="#F7FAFC", edgecolor="#CFD8E3", linewidth=1.0))
    ax.text(0.035, 0.82, "DISCOVERY — train/validation only", color=NAVY, weight="bold", fontsize=9)
    ax.add_patch(FancyBboxPatch((0.77, 0.18), 0.215, 0.68, boxstyle="round,pad=0.01", facecolor="#FFF8EC", edgecolor="#E8C77B", linewidth=1.0))
    ax.text(0.79, 0.82, "FROZEN EVALUATION", color="#865D0B", weight="bold", fontsize=9)

    boxes = [
        ((0.04, 0.55), "Task profile", "scaffold / metric / size", NAVY),
        ((0.22, 0.55), "Bounded\nplanner", "registered candidates", NAVY),
        ((0.40, 0.55), "Executor", "deterministic + hashed", NAVY),
        ((0.58, 0.55), "Evidence\nledger", "successes + failures", TEAL),
    ]
    width, height = 0.145, 0.17
    for xy, title, subtitle, color in boxes:
        _box(ax, xy, width, height, title, color=color, subtitle=subtitle)
    for left, right in zip(boxes[:-1], boxes[1:]):
        _arrow(ax, (left[0][0] + width, left[0][1] + height / 2), (right[0][0], right[0][1] + height / 2))

    _box(ax, (0.26, 0.28), 0.22, 0.16, "Cost-aware evidence router", color=TEAL, subtitle="score − uncertainty − cost")
    _box(ax, (0.55, 0.28), 0.16, 0.16, "Freeze lock", color=GOLD, subtitle="recipe + SHA-256")
    _arrow(ax, (0.655, 0.55), (0.45, 0.44), color=TEAL)
    _arrow(ax, (0.48, 0.36), (0.55, 0.36), color=GOLD)
    ax.text(0.13, 0.32, "stop 2D  ·  acquire 3D\nacquire target view  ·  fuse\nback off", ha="center", va="center", fontsize=7.5, color=GRAY)
    _arrow(ax, (0.71, 0.36), (0.80, 0.36), color=GOLD)
    _box(ax, (0.80, 0.28), 0.15, 0.16, "One-shot official", color=GOLD, subtitle="five seeds; no tuning")
    ax.plot([0.76, 0.76], [0.20, 0.83], linestyle=(0, (4, 3)), color=RED, linewidth=1.5)
    ax.text(0.755, 0.49, "  test firewall", rotation=90, va="center", ha="right", color=RED, fontsize=8, weight="bold")
    ax.text(0.50, 0.10, "Official results cannot mutate a finalist in the same freeze lineage", ha="center", color=RED, fontsize=8.5, weight="bold")
    paths = _save(fig, output_dir, "figure1_architecture", "ADMET-EvO architecture")
    return paths, {
        "claim": "Discovery is restricted to train/validation evidence and separated from one-shot official evaluation by a content-addressed freeze.",
        "official_test_available_to_discovery": False,
    }


def _short_task(value: str) -> str:
    value = value.replace("_Substrate_CarbonMangels", " substrate")
    suffixes = (
        ("_AstraZeneca", " AZ"),
        ("_Broccatelli", ""),
        ("_Lombardo", ""),
        ("_Obach", ""),
        ("_Martins", ""),
        ("_Wang", ""),
        ("_Zhu", ""),
        ("_Hou", ""),
        ("_Ma", ""),
    )
    for suffix, replacement in suffixes:
        if value.endswith(suffix):
            value = value[: -len(suffix)] + replacement
            break
    return value.replace("_", " ")


def figure_ranks(scorecard: dict[str, Any], output_dir: Path) -> tuple[list[Path], dict[str, Any]]:
    tasks = sorted(scorecard["tasks"], key=lambda row: (row["estimated_rank"], row["task"]))
    fig, ax = plt.subplots(figsize=(7.7, 7.2))
    ax.axvspan(0.5, 5.5, color=mpl.colors.to_rgba(TEAL, 0.09), zorder=0)
    ax.axvspan(5.5, 10.5, color=mpl.colors.to_rgba(GOLD, 0.11), zorder=0)
    ax.axvspan(10.5, 22.5, color=mpl.colors.to_rgba(RED, 0.07), zorder=0)
    y = np.arange(len(tasks))
    ranks = np.asarray([row["estimated_rank"] for row in tasks])
    colors = [TEAL if rank <= 5 else GOLD if rank <= 10 else RED for rank in ranks]
    ax.hlines(y, 1, ranks, color=[mpl.colors.to_rgba(color, 0.45) for color in colors], linewidth=1.8)
    ax.scatter(ranks, y, c=colors, s=34, edgecolor="white", linewidth=0.6, zorder=3)
    for yi, row in zip(y, tasks):
        ax.text(row["estimated_rank"] + 0.35, yi, str(row["estimated_rank"]), va="center", fontsize=7.2, color=DARK)
    ax.set_yticks(y, [_short_task(row["task"]) for row in tasks])
    ax.invert_yaxis()
    ax.set_xlim(0.5, 22.5)
    ax.set_xticks([1, 5, 10, 15, 20])
    ax.set_xlabel("Estimated leaderboard rank (lower is better)")
    ax.set_title("Frozen v2.7 coverage on the dated 22-task TDC snapshot", loc="left", weight="bold")
    ax.grid(axis="x", color="#D9DEE5", linewidth=0.6)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.text(0.02, -0.085, "18/22 Top-5", transform=ax.transAxes, color=TEAL, weight="bold")
    ax.text(0.26, -0.085, "21/22 Top-10", transform=ax.transAxes, color=GOLD, weight="bold")
    ax.text(0.52, -0.085, "Bio rank remains conservative pending authoritative TDC correction", transform=ax.transAxes, color=RED, fontsize=7.4)
    paths = _save(fig, output_dir, "figure2_leaderboard_ranks", "Frozen v2.7 leaderboard ranks")
    return paths, {
        "task_count": len(tasks),
        "top5": sum(row["estimated_rank"] <= 5 for row in tasks),
        "top10": sum(row["estimated_rank"] <= 10 for row in tasks),
        "conservative_deficit_task": next(row["task"] for row in tasks if row["estimated_rank"] > 10),
        "snapshot": scorecard["leaderboard_captured_at"],
    }


def figure_router_tradeoff(router: dict[str, Any], output_dir: Path) -> tuple[list[Path], dict[str, Any]]:
    rows = []
    for row in router["tasks"]:
        guard = row["summaries"]["metric_risk_guard"]
        score_policy = row["summaries"]["score"]
        rows.append(
            {
                "task": row["task"],
                "metric": row["metric"],
                "gain": row["guard_minus_score_oriented_gain"],
                "cost_ratio": guard["elapsed_seconds"] / score_policy["elapsed_seconds"],
                "score_seconds": score_policy["elapsed_seconds"],
            }
        )
    fig, (ax, summary_ax) = plt.subplots(1, 2, figsize=(8.4, 4.3), gridspec_kw={"width_ratios": [3.2, 1.15]})
    for metric, color in METRIC_COLORS.items():
        subset = [row for row in rows if row["metric"] == metric]
        ax.scatter(
            [row["cost_ratio"] for row in subset],
            [row["gain"] for row in subset],
            s=[25 + 8 * math.log1p(row["score_seconds"] / 60) for row in subset],
            c=color,
            alpha=0.85,
            edgecolor="white",
            linewidth=0.6,
            label=metric.upper(),
        )
    ax.axhline(0, color=DARK, linewidth=0.9)
    ax.axhline(MATERIAL_THRESHOLD := 0.005, color=GRAY, linestyle="--", linewidth=0.7)
    ax.axhline(-MATERIAL_THRESHOLD, color=GRAY, linestyle="--", linewidth=0.7)
    ax.axvline(1, color=GRAY, linestyle=":", linewidth=0.8)
    ax.set_xscale("log")
    positive_ratios = [row["cost_ratio"] for row in rows if row["cost_ratio"] > 0]
    ax.set_xlim(min(positive_ratios) * 0.72, max(positive_ratios) * 1.35)
    ax.set_xlabel("Measured fit-time ratio: guarded / score")
    ax.set_ylabel("Oriented validation gain vs score")
    ax.set_title("Fresh scaffold seeds 4–5: compute–accuracy trade-off", loc="left", weight="bold")
    ax.grid(color="#E2E6EB", linewidth=0.6)
    ax.legend(frameon=False, ncol=2, loc="lower left")
    for row in rows:
        if abs(row["gain"]) >= MATERIAL_THRESHOLD or row["cost_ratio"] < 0.05:
            ax.annotate(_short_task(row["task"]), (row["cost_ratio"], row["gain"]), xytext=(3, 3), textcoords="offset points", fontsize=6.7, color=DARK)

    score_hours = sum(row["summaries"]["score"]["elapsed_seconds"] for row in router["tasks"]) / 3600
    guard_hours = sum(row["summaries"]["metric_risk_guard"]["elapsed_seconds"] for row in router["tasks"]) / 3600
    summary_ax.bar([0, 1], [score_hours, guard_hours], color=[GRAY, TEAL], width=0.62)
    summary_ax.set_xticks([0, 1], ["Score", "Guarded"])
    summary_ax.set_ylabel("Measured fit hours")
    summary_ax.set_title("Aggregate", weight="bold")
    summary_ax.set_ylim(0, score_hours * 1.12)
    summary_ax.spines[["top", "right"]].set_visible(False)
    summary_ax.grid(axis="y", color="#E2E6EB", linewidth=0.6)
    for index, value in enumerate((score_hours, guard_hours)):
        summary_ax.text(index, value + score_hours * 0.035, f"{value:.3f} h", ha="center", fontsize=8, weight="bold")
    note_box = {"boxstyle": "round,pad=0.25", "facecolor": "white", "edgecolor": "none", "alpha": 0.90}
    summary_ax.text(0.5, 0.59, f"ratio {router['claim_gate']['aggregate_guard_cost_ratio_vs_score']:.3f}", transform=summary_ax.transAxes, ha="center", color=TEAL, weight="bold", fontsize=9.5, bbox=note_box)
    summary_ax.text(0.5, 0.49, "safety gate: FAIL", transform=summary_ax.transAxes, ha="center", color=RED, weight="bold", fontsize=9.5, bbox=note_box)
    summary_ax.text(0.5, 0.40, "guard − unguarded = 0", transform=summary_ax.transAxes, ha="center", color=GRAY, fontsize=7.1, bbox=note_box)
    fig.subplots_adjust(wspace=0.34)
    paths = _save(fig, output_dir, "figure3_router_tradeoff", "Fresh router compute accuracy trade-off")
    return paths, {
        "task_count": len(rows),
        "fresh_validation_seeds": router["fresh_validation_seeds"],
        "score_fit_hours": score_hours,
        "guard_fit_hours": guard_hours,
        "aggregate_cost_ratio": router["claim_gate"]["aggregate_guard_cost_ratio_vs_score"],
        "guard_minus_unguarded_mean": router["primary_guard_minus_unguarded"]["mean"],
        "material_regressions_vs_score": router["guard_material_regressions_vs_score"],
        "claim_gate_passed": router["claim_gate"]["passed"],
    }


def figure_chemical_space(audit: dict[str, Any], output_dir: Path) -> tuple[list[Path], dict[str, Any]]:
    classification = []
    regression = []
    for row in audit["tasks"]:
        space = row["chemical_space"]
        if row["analysis_type"] == "classification_calibration":
            ratio = space["low_similarity_brier"] / space["high_similarity_brier"]
            classification.append((row["task"], math.log2(ratio)))
        else:
            ratio = space["low_similarity_mean_absolute_error"] / space["high_similarity_mean_absolute_error"]
            regression.append((row["task"], math.log2(ratio)))
    classification.sort(key=lambda item: item[1])
    regression.sort(key=lambda item: item[1])
    fig, axes = plt.subplots(1, 2, figsize=(8.5, 5.0), gridspec_kw={"width_ratios": [13, 9]})
    for ax, rows, title in (
        (axes[0], classification, "Classification: Brier error"),
        (axes[1], regression, "Regression: absolute residual"),
    ):
        y = np.arange(len(rows))
        values = [value for _, value in rows]
        colors = [RED if value > 0 else BLUE for value in values]
        ax.barh(y, values, color=colors, alpha=0.88, height=0.66)
        ax.set_yticks(y, [_short_task(task) for task, _ in rows])
        ax.axvline(0, color=DARK, linewidth=0.9)
        ax.grid(axis="x", color="#E2E6EB", linewidth=0.6)
        ax.spines[["top", "right", "left"]].set_visible(False)
        ax.set_title(title, loc="left", weight="bold")
        ax.set_xlabel("log₂(low-similarity error / high-similarity error)")
    fig.suptitle("Post-selection scaffold audit: error inflation away from training chemistry", x=0.04, ha="left", fontsize=12, weight="bold", color=DARK)
    fig.text(0.04, 0.015, "Positive values indicate worse error in the lowest nearest-train-similarity quartile. Diagnostic only; one saved split per task.", color=GRAY, fontsize=7.6)
    fig.subplots_adjust(top=0.88, bottom=0.15, wspace=0.52)
    paths = _save(fig, output_dir, "figure4_chemical_space_risk", "Chemical space error inflation")
    return paths, {
        "classification_tasks": len(classification),
        "classification_worse_low_similarity": sum(value > 0 for _, value in classification),
        "regression_tasks": len(regression),
        "regression_worse_low_similarity": sum(value > 0 for _, value in regression),
        "identity_overlap_total": sum(row["identity_overlap_count"] for row in audit["tasks"]),
        "scaffold_overlap_total": sum(row["train_valid_scaffold_overlap_count"] for row in audit["tasks"]),
        "analysis_role": audit["analysis_role"],
    }


def captions(data: dict[str, Any]) -> str:
    rank = data["figure2_leaderboard_ranks"]
    router = data["figure3_router_tradeoff"]
    chemistry = data["figure4_chemical_space_risk"]
    return "\n".join(
        [
            "# Main-text figure captions",
            "",
            "**Figure 1 | Protocol-safe evidence acquisition.** The planner can only propose registered candidates, the deterministic executor records successes and failures, and the router may acquire, fuse, or reject evidence. A content-addressed freeze separates train/validation discovery from one-shot official evaluation; official outcomes cannot mutate the finalist in the same lineage.",
            "",
            f"**Figure 2 | Conservative coverage of the dated 22-task TDC ADMET snapshot.** The frozen v2.7 system has {rank['top5']}/22 estimated Top-5 and {rank['top10']}/22 Top-10 results. Bioavailability is retained at its displayed conservative rank pending authoritative correction of the disputed page.",
            "",
            f"**Figure 3 | Fresh-seed compute–accuracy trade-off.** Candidate choices were locked using development seeds 1–3 and evaluated on default scaffold seeds {router['fresh_validation_seeds'][0]}–{router['fresh_validation_seeds'][1]}. The guarded policy used {router['guard_fit_hours']:.3f} measured fit hours versus {router['score_fit_hours']:.3f} for score selection (ratio {router['aggregate_cost_ratio']:.3f}), but guard-minus-unguarded gain was {router['guard_minus_unguarded_mean']:.3f} and the preregistered safety gate failed. Point size reflects score-policy fit time; dashed horizontal lines mark ±0.005 native-metric change.",
            "",
            f"**Figure 4 | Chemical-space error inflation.** In a post-selection diagnostic on one saved scaffold split per task, low-similarity molecules had worse error on {chemistry['classification_worse_low_similarity']}/{chemistry['classification_tasks']} classification and {chemistry['regression_worse_low_similarity']}/{chemistry['regression_tasks']} regression endpoints. Error ratios are shown on a log2 scale to compare endpoints with different native units. Identity and scaffold overlap were both zero; the analysis is diagnostic rather than a prospective performance estimate.",
            "",
        ]
    )


def build(repo_root: Path, output_dir: Path) -> dict[str, Any]:
    root = repo_root.resolve()
    output = output_dir if output_dir.is_absolute() else root / output_dir
    output.mkdir(parents=True, exist_ok=True)
    _style()
    scorecard = _load(root / SOURCE_FILES["scorecard"])
    router = _load(root / SOURCE_FILES["router_fresh"])
    chemistry = _load(root / SOURCE_FILES["chemical_space"])
    generated: dict[str, list[Path]] = {}
    figure_data: dict[str, Any] = {}
    generated["figure1_architecture"], figure_data["figure1_architecture"] = figure_architecture(output)
    generated["figure2_leaderboard_ranks"], figure_data["figure2_leaderboard_ranks"] = figure_ranks(scorecard, output)
    generated["figure3_router_tradeoff"], figure_data["figure3_router_tradeoff"] = figure_router_tradeoff(router, output)
    generated["figure4_chemical_space_risk"], figure_data["figure4_chemical_space_risk"] = figure_chemical_space(chemistry, output)
    caption_path = output / "captions.md"
    caption_path.write_text(captions(figure_data), encoding="utf-8")
    outputs = {
        path.relative_to(root).as_posix(): {"sha256": _sha256(path), "bytes": path.stat().st_size}
        for paths in generated.values()
        for path in paths
    }
    outputs[caption_path.relative_to(root).as_posix()] = {"sha256": _sha256(caption_path), "bytes": caption_path.stat().st_size}
    return {
        "version": "v2.7",
        "generated_on": "2026-08-17",
        "figure_count": 4,
        "formats": ["svg", "png"],
        "official_test_loaded": False,
        "leaderboard_used_for_selection": False,
        "selection_or_freeze_changed": False,
        "sources": {
            name: {"path": path, "sha256": _sha256(root / path)}
            for name, path in SOURCE_FILES.items()
        },
        "figure_data": figure_data,
        "outputs": outputs,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    parser.add_argument("--output-dir", type=Path, default=Path("paper/figures"))
    args = parser.parse_args()
    manifest = build(args.repo_root, args.output_dir)
    root = args.repo_root.resolve()
    output = args.output_dir if args.output_dir.is_absolute() else root / args.output_dir
    manifest_path = output / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({"figure_count": manifest["figure_count"], "output_count": len(manifest["outputs"]), "manifest_sha256": _sha256(manifest_path)}, sort_keys=True))


if __name__ == "__main__":
    main()
