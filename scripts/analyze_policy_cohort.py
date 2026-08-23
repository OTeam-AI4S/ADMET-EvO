"""Analyze a complete multi-arm discovery cohort without test access."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import binomtest

from admet_agent.tasks import resolve_task


def _accounted_fit_seconds(row: dict[str, Any]) -> float:
    """Read current collector fields while retaining pre-v4 audit fixtures."""

    return float(
        row.get(
            "accounted_fit_seconds",
            row.get("fit_seconds", row.get("compute_seconds", float("nan"))),
        )
    )


def _accounted_unique_configurations(row: dict[str, Any]) -> int:
    return int(
        row.get(
            "accounted_unique_configurations",
            row.get("unique_configurations", row.get("actions_attempted", -1)),
        )
    )


def _bootstrap(values: np.ndarray, *, seed: int, replicates: int) -> list[float]:
    if not len(values):
        return [float("nan"), float("nan")]
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(values), size=(replicates, len(values)))
    means = values[indices].mean(axis=1)
    return [float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))]


def analyze(
    payload: dict[str, Any],
    *,
    reference_arm: str,
    seed: int = 20260821,
    bootstrap_replicates: int = 10000,
) -> dict[str, Any]:
    if bootstrap_replicates < 1000:
        raise ValueError("At least 1000 bootstrap replicates are required")
    rows = payload.get("results")
    if not isinstance(rows, list) or not rows:
        raise ValueError("Collector payload must contain non-empty results")
    protocol_hashes = {row.get("protocol_sha256") for row in rows}
    if len(protocol_hashes) != 1 or not next(iter(protocol_hashes)):
        raise ValueError("Every result must share one protocol hash")
    indexed: dict[str, dict[str, dict[str, Any]]] = {}
    for row in rows:
        arm = str(row["arm"])
        task = resolve_task(str(row["task"])).name
        if task in indexed.setdefault(arm, {}):
            raise ValueError(f"Duplicate policy result for {arm}/{task}")
        for field in ("validation_score", "validation_std"):
            if not math.isfinite(float(row[field])):
                raise ValueError(f"Non-finite {field} for {arm}/{task}")
        if not math.isfinite(_accounted_fit_seconds(row)):
            raise ValueError(f"Non-finite accounted fit seconds for {arm}/{task}")
        if _accounted_unique_configurations(row) < 0:
            raise ValueError(f"Missing configuration accounting for {arm}/{task}")
        indexed[arm][task] = row
    if reference_arm not in indexed:
        raise ValueError(f"Reference arm {reference_arm!r} is absent")
    task_sets = {arm: set(values) for arm, values in indexed.items()}
    expected = set(indexed[reference_arm])
    mismatched = {arm: sorted(tasks ^ expected) for arm, tasks in task_sets.items() if tasks != expected}
    if mismatched:
        raise ValueError(f"All arms must cover the identical cohort: {mismatched}")
    tasks = sorted(expected)

    per_task: dict[str, Any] = {}
    ranks: dict[str, list[float]] = {arm: [] for arm in indexed}
    firsts = {arm: 0 for arm in indexed}
    for task_name in tasks:
        task = resolve_task(task_name)
        direction = 1.0 if task.maximize else -1.0
        oriented = {
            arm: direction * float(values[task_name]["validation_score"])
            for arm, values in indexed.items()
        }
        ordered_scores = sorted(set(oriented.values()), reverse=True)
        task_ranks = {
            arm: float(1 + ordered_scores.index(value)) for arm, value in oriented.items()
        }
        best = max(oriented.values())
        winners = sorted(arm for arm, value in oriented.items() if abs(value - best) <= 1e-12)
        for arm, rank in task_ranks.items():
            ranks[arm].append(rank)
            firsts[arm] += int(rank == 1.0)
        per_task[task_name] = {
            "metric": task.metric,
            "maximize": task.maximize,
            "scores": {arm: indexed[arm][task_name]["validation_score"] for arm in sorted(indexed)},
            "ranks": {arm: task_ranks[arm] for arm in sorted(indexed)},
            "rank_1_arms": winners,
        }

    arm_summary = {
        arm: {
            "mean_rank": float(np.mean(values)),
            "median_rank": float(np.median(values)),
            "rank_1_tasks": firsts[arm],
            "total_fit_seconds": float(
                sum(_accounted_fit_seconds(indexed[arm][task]) for task in tasks)
            ),
            "total_unique_configurations": int(
                sum(
                    _accounted_unique_configurations(indexed[arm][task])
                    for task in tasks
                )
            ),
        }
        for arm, values in sorted(ranks.items())
    }

    comparisons: dict[str, Any] = {}
    for arm in sorted(indexed):
        if arm == reference_arm:
            continue
        deltas = []
        standardized = []
        config_matches = []
        for task_name in tasks:
            task = resolve_task(task_name)
            direction = 1.0 if task.maximize else -1.0
            reference = indexed[reference_arm][task_name]
            control = indexed[arm][task_name]
            delta = direction * (
                float(reference["validation_score"]) - float(control["validation_score"])
            )
            uncertainty = max(
                0.02,
                float(np.hypot(reference["validation_std"], control["validation_std"])),
            )
            deltas.append(delta)
            standardized.append(delta / uncertainty)
            config_matches.append(
                _accounted_unique_configurations(reference)
                == _accounted_unique_configurations(control)
            )
        values = np.asarray(deltas, dtype=float)
        scaled = np.asarray(standardized, dtype=float)
        wins = int(np.sum(values > 1e-12))
        ties = int(np.sum(np.abs(values) <= 1e-12))
        losses = int(np.sum(values < -1e-12))
        non_ties = wins + losses
        comparisons[arm] = {
            "estimand": f"{reference_arm}_minus_{arm}_oriented_validation_metric",
            "tasks": len(tasks),
            "mean_oriented_delta": float(np.mean(values)),
            "median_oriented_delta": float(np.median(values)),
            "bootstrap_95_ci": _bootstrap(values, seed=seed, replicates=bootstrap_replicates),
            "mean_uncertainty_standardized_delta": float(np.mean(scaled)),
            "standardized_bootstrap_95_ci": _bootstrap(
                scaled, seed=seed + 1, replicates=bootstrap_replicates
            ),
            "wins": wins,
            "ties": ties,
            "losses": losses,
            "one_sided_exact_sign_p": (
                float(binomtest(wins, non_ties, 0.5, alternative="greater").pvalue)
                if non_ties
                else 1.0
            ),
            "per_task_action_budget_parity": bool(all(config_matches)),
            "inferential_claim_allowed": False,
        }

    return {
        "analysis_version": "policy-cohort-audit-v1",
        "protocol_sha256": next(iter(protocol_hashes)),
        "reference_arm": reference_arm,
        "cohort_tasks": tasks,
        "cohort_size": len(tasks),
        "evidence_scope": "discovery_development_only" if len(tasks) < 22 else "full_discovery_suite",
        "test_loaded": False,
        "leaderboard_loaded": False,
        "per_task": per_task,
        "arm_summary": arm_summary,
        "comparisons": comparisons,
        "interpretation_guard": (
            "Development-cohort statistics localize failures and rank controls; "
            "they do not support superiority or generalization claims."
            if len(tasks) < 22
            else "Full-suite discovery evidence still does not replace prospective blind validation."
        ),
    }


def render_markdown(result: dict[str, Any]) -> str:
    lines = [
        "# Multi-arm discovery cohort audit",
        "",
        f"Reference arm: `{result['reference_arm']}`  ",
        f"Scope: `{result['evidence_scope']}`  ",
        f"Tasks: {result['cohort_size']}  ",
        "Test loaded: **no**; leaderboard loaded: **no**.",
        "",
        "## Arm ranking and resource summary",
        "",
        "| Arm | Mean rank | Median rank | Rank-1 tasks | Configurations | Fit seconds |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for arm, row in result["arm_summary"].items():
        lines.append(
            f"| {arm} | {row['mean_rank']:.3f} | {row['median_rank']:.3f} | "
            f"{row['rank_1_tasks']} | {row['total_unique_configurations']} | "
            f"{row['total_fit_seconds']:.1f} |"
        )
    lines.extend(
        [
            "",
            f"## Pairwise differences from `{result['reference_arm']}`",
            "",
            "| Comparator | Mean oriented delta | 95% bootstrap CI | W/T/L | Sign p | Action parity |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for arm, row in result["comparisons"].items():
        low, high = row["bootstrap_95_ci"]
        lines.append(
            f"| {arm} | {row['mean_oriented_delta']:+.6f} | "
            f"[{low:+.6f}, {high:+.6f}] | "
            f"{row['wins']}/{row['ties']}/{row['losses']} | "
            f"{row['one_sided_exact_sign_p']:.4g} | "
            f"{'yes' if row['per_task_action_budget_parity'] else 'no'} |"
        )
    arms = sorted(result["arm_summary"])
    lines.extend(
        [
            "",
            "## Per-task scores and ranks",
            "",
            "| Task | Metric | "
            + " | ".join(f"{arm} score (rank)" for arm in arms)
            + " |",
            "|---|---|" + "---:|" * len(arms),
        ]
    )
    for task, row in result["per_task"].items():
        cells = [
            f"{float(row['scores'][arm]):.6f} ({row['ranks'][arm]:.0f})"
            for arm in arms
        ]
        lines.append(f"| {task} | {row['metric']} | " + " | ".join(cells) + " |")
    lines.extend(["", "## Interpretation guard", "", result["interpretation_guard"], ""])
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--reference-arm", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--markdown", type=Path)
    parser.add_argument("--seed", type=int, default=20260821)
    parser.add_argument("--bootstrap-replicates", type=int, default=10000)
    args = parser.parse_args()
    result = analyze(
        json.loads(args.input.read_text(encoding="utf-8")),
        reference_arm=args.reference_arm,
        seed=args.seed,
        bootstrap_replicates=args.bootstrap_replicates,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    if args.markdown is not None:
        args.markdown.parent.mkdir(parents=True, exist_ok=True)
        args.markdown.write_text(render_markdown(result), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
