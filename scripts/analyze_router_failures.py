"""Validation-only error and tail-risk analysis for the evidence router."""

from __future__ import annotations

import argparse
import json
import math
import statistics
from collections import Counter, defaultdict
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import spearmanr


def bootstrap_interval(
    values: list[float],
    statistic: Callable[[list[float]], float],
    *,
    samples: int = 10_000,
    seed: int = 24,
) -> list[float]:
    if not values:
        raise ValueError("Bootstrap requires at least one value")
    generator = np.random.default_rng(seed)
    array = np.asarray(values, dtype=float)
    estimates = np.empty(samples, dtype=float)
    for index in range(samples):
        draw = generator.choice(array, size=len(array), replace=True).tolist()
        estimates[index] = statistic(draw)
    return [float(value) for value in np.quantile(estimates, [0.025, 0.975])]


def _correlation(left: list[float], right: list[float]) -> dict[str, Any]:
    if len(set(left)) < 2 or len(set(right)) < 2:
        return {"spearman_rho": None, "p_value": None, "defined": False}
    statistic, p_value = spearmanr(left, right)
    return {
        "spearman_rho": float(statistic),
        "p_value": float(p_value),
        "defined": True,
    }


def _action_label(counts: dict[str, int]) -> str:
    return "+".join(
        f"{action}:{count}" for action, count in sorted(counts.items())
    )


def _group_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    gains = [row["oriented_gain_vs_score"] for row in rows]
    costs = [row["selection_cost_ratio_vs_score"] for row in rows]
    return {
        "tasks": len(rows),
        "mean_oriented_gain_vs_score": statistics.mean(gains),
        "median_cost_ratio_vs_score": statistics.median(costs),
        "improved": sum(value > 1e-12 for value in gains),
        "tied": sum(abs(value) <= 1e-12 for value in gains),
        "worse": sum(value < -1e-12 for value in gains),
    }


def analyze(
    loso: dict[str, Any],
    profiles: dict[str, dict[str, Any]],
    *,
    material_threshold: float = 0.005,
) -> dict[str, Any]:
    if loso.get("official_test_loaded") is not False:
        raise ValueError("Router analysis requires an explicit no-test report")
    rows = []
    for task in loso["tasks"]:
        name = task["task"]
        if name not in profiles:
            raise ValueError(f"Missing validation-only profile for {name}")
        profile = profiles[name]
        selected = task["looto_cost_aware"]
        gain = float(selected["oriented_gain_vs_score"])
        rows.append(
            {
                "task": name,
                "task_type": profile["task_type"],
                "metric": task["metric"],
                "n_train": int(profile["n_train"]),
                "scaffold_density": (
                    float(profile["unique_scaffolds_train"])
                    / max(int(profile["n_train"]), 1)
                ),
                "score_heldout_std": float(task["score"]["std"]),
                "oriented_gain_vs_score": gain,
                "selection_cost_ratio_vs_score": float(
                    selected["selection_cost_ratio_vs_score"]
                ),
                "score_actions": _action_label(task["score"]["action_counts"]),
                "selected_actions": _action_label(selected["action_counts"]),
                "selected_cost_penalty": float(selected["selected_cost_penalty"]),
            }
        )
    gains = [row["oriented_gain_vs_score"] for row in rows]
    costs = [row["selection_cost_ratio_vs_score"] for row in rows]
    material_regressions = sorted(
        (row for row in rows if row["oriented_gain_vs_score"] < -material_threshold),
        key=lambda row: row["oriented_gain_vs_score"],
    )
    material_improvements = sorted(
        (row for row in rows if row["oriented_gain_vs_score"] > material_threshold),
        key=lambda row: row["oriented_gain_vs_score"],
        reverse=True,
    )
    by_metric: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_type: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_metric[row["metric"]].append(row)
        by_type[row["task_type"]].append(row)
    return {
        "protocol": (
            "post-hoc descriptive analysis of the preregistered validation-only "
            "LOOTO router report; no policy is selected from this analysis"
        ),
        "official_test_loaded": False,
        "leaderboard_used": False,
        "task_count": len(rows),
        "material_threshold_native_metric": material_threshold,
        "aggregate": {
            "mean_oriented_gain_vs_score": statistics.mean(gains),
            "mean_gain_bootstrap_95_ci": bootstrap_interval(gains, statistics.mean),
            "median_task_cost_ratio_vs_score": statistics.median(costs),
            "median_cost_bootstrap_95_ci": bootstrap_interval(
                costs, statistics.median, seed=25
            ),
            "outcomes": dict(
                sorted(
                    Counter(
                        "improved"
                        if value > 1e-12
                        else "worse"
                        if value < -1e-12
                        else "tied"
                        for value in gains
                    ).items()
                )
            ),
            "material_regressions": len(material_regressions),
            "material_improvements": len(material_improvements),
        },
        "subgroups": {
            "metric": {
                key: _group_summary(value) for key, value in sorted(by_metric.items())
            },
            "task_type": {
                key: _group_summary(value) for key, value in sorted(by_type.items())
            },
        },
        "correlations": {
            "log_n_train_vs_gain": _correlation(
                [math.log1p(row["n_train"]) for row in rows], gains
            ),
            "scaffold_density_vs_gain": _correlation(
                [row["scaffold_density"] for row in rows], gains
            ),
            "score_heldout_std_vs_gain": _correlation(
                [row["score_heldout_std"] for row in rows], gains
            ),
        },
        "material_regressions": material_regressions,
        "material_improvements": material_improvements,
        "tasks": sorted(rows, key=lambda row: row["task"].lower()),
        "interpretation_limit": (
            "Gains are oriented native-metric differences across heterogeneous "
            "endpoints; subgroup and correlation analyses are descriptive, not a "
            "new model-selection signal."
        ),
    }


def markdown(report: dict[str, Any]) -> str:
    aggregate = report["aggregate"]
    lines = [
        "# Evidence-router validation error analysis",
        "",
        report["protocol"] + ". Official test and leaderboard data were not loaded.",
        "",
        "## Aggregate tail risk",
        "",
        (
            f"- Mean oriented gain versus score selection: "
            f"{aggregate['mean_oriented_gain_vs_score']:.4f} "
            f"(task bootstrap 95% CI "
            f"[{aggregate['mean_gain_bootstrap_95_ci'][0]:.4f}, "
            f"{aggregate['mean_gain_bootstrap_95_ci'][1]:.4f}])."
        ),
        (
            f"- Median per-task cost ratio: "
            f"{aggregate['median_task_cost_ratio_vs_score']:.3f} "
            f"(bootstrap 95% CI "
            f"[{aggregate['median_cost_bootstrap_95_ci'][0]:.3f}, "
            f"{aggregate['median_cost_bootstrap_95_ci'][1]:.3f}])."
        ),
        (
            f"- Material changes at |native-metric gain| > "
            f"{report['material_threshold_native_metric']:.3f}: "
            f"{aggregate['material_improvements']} improvements and "
            f"{aggregate['material_regressions']} regressions."
        ),
        "",
        "## Material regressions",
        "",
        "| Task | Metric | Gain | Cost ratio | Score action | Selected action |",
        "|---|---:|---:|---:|---|---|",
    ]
    for row in report["material_regressions"]:
        lines.append(
            f"| {row['task']} | {row['metric']} | "
            f"{row['oriented_gain_vs_score']:.4f} | "
            f"{row['selection_cost_ratio_vs_score']:.3f} | "
            f"{row['score_actions']} | {row['selected_actions']} |"
        )
    lines.extend(["", "## Metric subgroups", ""])
    lines.extend(
        [
            "| Metric | Tasks | Mean gain | Improved/tied/worse | Median cost ratio |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for metric, item in report["subgroups"]["metric"].items():
        lines.append(
            f"| {metric} | {item['tasks']} | "
            f"{item['mean_oriented_gain_vs_score']:.4f} | "
            f"{item['improved']}/{item['tied']}/{item['worse']} | "
            f"{item['median_cost_ratio_vs_score']:.3f} |"
        )
    lines.extend(
        [
            "",
            report["interpretation_limit"],
            "",
            (
                "The largest tail losses should be reported rather than hidden by "
                "aggregate compute savings. In v2.4 this analysis is diagnostic only; "
                "it does not retroactively alter any frozen policy or recipe."
            ),
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input", type=Path, default=Path("runs/v24-router-loso/report.json")
    )
    parser.add_argument(
        "--search-root", type=Path, default=Path("runs/leaderboard-v21")
    )
    parser.add_argument(
        "--output", type=Path, default=Path("runs/v24-router-error-analysis/report")
    )
    args = parser.parse_args()
    loso = json.loads(args.input.read_text(encoding="utf-8"))
    profiles = {}
    sources = []
    for path in sorted(args.search_root.glob("tasks/*/tasks/*/search_seed_1.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        profiles[payload["profile"]["task"]] = payload["profile"]
        sources.append(path.as_posix())
    report = analyze(loso, profiles)
    report["sources"] = [args.input.as_posix(), *sources]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.with_suffix(".json").write_text(
        json.dumps(report, indent=2, sort_keys=True), encoding="utf-8"
    )
    args.output.with_suffix(".md").write_text(markdown(report), encoding="utf-8")
    print(json.dumps(report["aggregate"], sort_keys=True))


if __name__ == "__main__":
    main()
