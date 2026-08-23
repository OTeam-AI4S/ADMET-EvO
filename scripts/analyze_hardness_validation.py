"""Validate HardnessCard stability and outcome associations without model fitting.

The input contains repeated, discovery-only HardnessCards.  Optional outcome
associations are accepted only from discovery data, or from a post-freeze
artifact that explicitly states that selection was not reopened.  This keeps a
descriptive paper analysis from becoming an unrecorded selection channel.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import spearmanr


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _icc_absolute_agreement(matrix: np.ndarray) -> float:
    """Two-way random-effects, absolute-agreement, single-measure ICC(2,1)."""
    matrix = np.asarray(matrix, dtype=float)
    if matrix.ndim != 2 or min(matrix.shape) < 2 or not np.isfinite(matrix).all():
        raise ValueError("ICC requires a finite task-by-seed matrix of at least 2x2")
    tasks, seeds = matrix.shape
    grand = float(matrix.mean())
    task_means = matrix.mean(axis=1)
    seed_means = matrix.mean(axis=0)
    ms_task = seeds * float(np.square(task_means - grand).sum()) / (tasks - 1)
    ms_seed = tasks * float(np.square(seed_means - grand).sum()) / (seeds - 1)
    residual = matrix - task_means[:, None] - seed_means[None, :] + grand
    ms_error = float(np.square(residual).sum()) / ((tasks - 1) * (seeds - 1))
    denominator = (
        ms_task
        + (seeds - 1) * ms_error
        + seeds * (ms_seed - ms_error) / tasks
    )
    return float((ms_task - ms_error) / denominator) if denominator else 1.0


def _fleiss_kappa(labels: list[list[str]]) -> float:
    if not labels or len(labels[0]) < 2:
        raise ValueError("Fleiss kappa requires at least two ratings per task")
    raters = len(labels[0])
    if any(len(row) != raters for row in labels):
        raise ValueError("Every task must have the same number of ratings")
    categories = sorted({value for row in labels for value in row})
    counts = np.asarray(
        [[row.count(category) for category in categories] for row in labels],
        dtype=float,
    )
    observed = float(np.mean((np.square(counts).sum(axis=1) - raters) / (raters * (raters - 1))))
    proportions = counts.sum(axis=0) / counts.sum()
    expected = float(np.square(proportions).sum())
    return float((observed - expected) / (1.0 - expected)) if expected < 1.0 else 1.0


def _pairwise_spearman(matrix: np.ndarray) -> dict[str, float]:
    values = []
    for left, right in itertools.combinations(range(matrix.shape[1]), 2):
        correlation = float(spearmanr(matrix[:, left], matrix[:, right]).statistic)
        values.append(0.0 if math.isnan(correlation) else correlation)
    return {"mean": float(np.mean(values)), "minimum": float(np.min(values))}


def _mean_pairwise_agreement(values: list[list[Any]]) -> float:
    agreements = []
    for row in values:
        agreements.extend(left == right for left, right in itertools.combinations(row, 2))
    return float(np.mean(agreements))


def _mean_pairwise_jaccard(values: list[list[set[str]]]) -> float:
    scores = []
    for row in values:
        for left, right in itertools.combinations(row, 2):
            union = left | right
            scores.append(len(left & right) / len(union) if union else 1.0)
    return float(np.mean(scores))


def _validate_card_payload(payload: dict[str, Any]) -> tuple[list[dict], list[str], list[int]]:
    if payload.get("test_loaded") is not False or payload.get("discovery_only") is not True:
        raise RuntimeError("Hardness stability input must be explicitly discovery-only")
    cards = payload.get("cards")
    if not isinstance(cards, list) or not cards:
        raise ValueError("Hardness input requires a non-empty cards list")
    keys = [(str(card["task"]), int(card["seed"])) for card in cards]
    if len(keys) != len(set(keys)):
        raise ValueError("Duplicate task/seed HardnessCard")
    tasks = sorted({task for task, _ in keys})
    seeds = sorted({seed for _, seed in keys})
    if len(tasks) < 2 or len(seeds) < 2:
        raise ValueError("Stability analysis requires at least two tasks and two seeds")
    expected = {(task, seed) for task in tasks for seed in seeds}
    if set(keys) != expected:
        raise ValueError("HardnessCard task-by-seed grid must be complete")
    return cards, tasks, seeds


def _validate_outcomes(payload: dict[str, Any]) -> list[dict]:
    discovery_safe = (
        payload.get("discovery_only") is True and payload.get("test_loaded") is False
    )
    frozen_safe = (
        payload.get("finalist_frozen_before_outcomes") is True
        and payload.get("selection_reopened") is False
    )
    if not (discovery_safe or frozen_safe):
        raise RuntimeError(
            "Outcomes must be discovery-only or explicitly post-freeze with no reopened selection"
        )
    rows = payload.get("rows")
    if not isinstance(rows, list) or not rows:
        raise ValueError("Outcome input requires a non-empty rows list")
    tasks = [str(row["task"]) for row in rows]
    if len(tasks) != len(set(tasks)):
        raise ValueError("Outcome rows must contain one record per task")
    return rows


def _permutation_associations(
    card_means: dict[str, dict[str, float]],
    outcomes: list[dict],
    *,
    permutations: int,
    seed: int,
) -> list[dict]:
    if permutations < 100:
        raise ValueError("At least 100 task-label permutations are required")
    by_task = {str(row["task"]): row for row in outcomes}
    tasks = sorted(set(card_means) & set(by_task))
    if len(tasks) < 4:
        raise ValueError("Outcome association requires at least four matched tasks")
    dimensions = sorted(next(iter(card_means.values())))
    outcome_names = sorted(
        key
        for key in by_task[tasks[0]]
        if key != "task"
        and all(isinstance(by_task[task].get(key), (int, float)) for task in tasks)
    )
    generator = np.random.default_rng(seed)
    rows = []
    for dimension in dimensions:
        x = np.asarray([card_means[task][dimension] for task in tasks], dtype=float)
        for outcome in outcome_names:
            y = np.asarray([by_task[task][outcome] for task in tasks], dtype=float)
            observed = float(spearmanr(x, y).statistic)
            observed = 0.0 if math.isnan(observed) else observed
            null = []
            for _ in range(permutations):
                value = float(spearmanr(x, generator.permutation(y)).statistic)
                null.append(0.0 if math.isnan(value) else value)
            p_value = (1 + sum(abs(value) >= abs(observed) for value in null)) / (
                permutations + 1
            )
            rows.append(
                {
                    "dimension": dimension,
                    "outcome": outcome,
                    "tasks": len(tasks),
                    "spearman": observed,
                    "permutation_p_two_sided": float(p_value),
                }
            )
    # Benjamini-Hochberg across this explicitly exploratory association family.
    order = sorted(range(len(rows)), key=lambda index: rows[index]["permutation_p_two_sided"])
    adjusted = [1.0] * len(rows)
    running = 1.0
    for reverse_rank, index in enumerate(reversed(order), start=1):
        rank = len(rows) - reverse_rank + 1
        value = rows[index]["permutation_p_two_sided"] * len(rows) / rank
        running = min(running, value)
        adjusted[index] = float(min(1.0, running))
    for row, value in zip(rows, adjusted, strict=True):
        row["bh_q"] = value
    return rows


def analyze(
    hardness_path: Path,
    *,
    outcomes_path: Path | None = None,
    permutations: int = 10_000,
    seed: int = 20260821,
) -> dict[str, Any]:
    hardness_path = hardness_path.resolve()
    payload = json.loads(hardness_path.read_text(encoding="utf-8"))
    cards, tasks, seeds = _validate_card_payload(payload)
    lookup = {(str(card["task"]), int(card["seed"])): card for card in cards}
    dimensions = ["overall"] + sorted(
        set.intersection(
            *[
                set(card.get("dimensions", {}))
                for card in cards
            ]
        )
    )
    stability = {}
    card_means = {task: {} for task in tasks}
    for dimension in dimensions:
        matrix = np.asarray(
            [
                [
                    float(lookup[(task, value)]["overall"])
                    if dimension == "overall"
                    else float(lookup[(task, value)]["dimensions"][dimension]["severity"])
                    for value in seeds
                ]
                for task in tasks
            ],
            dtype=float,
        )
        task_sd = np.std(matrix, axis=1, ddof=1)
        stability[dimension] = {
            "icc_2_1_absolute_agreement": _icc_absolute_agreement(matrix),
            "pairwise_seed_spearman": _pairwise_spearman(matrix),
            "median_within_task_sd": float(np.median(task_sd)),
            "maximum_within_task_sd": float(np.max(task_sd)),
        }
        for index, task in enumerate(tasks):
            card_means[task][dimension] = float(matrix[index].mean())
    dominant = [
        [str(lookup[(task, value)]["dominant_factors"][0]) for value in seeds]
        for task in tasks
    ]
    actions = [
        [set(map(str, lookup[(task, value)].get("recommended_actions", []))) for value in seeds]
        for task in tasks
    ]
    output: dict[str, Any] = {
        "protocol": "hardness_card_validation_v1",
        "hardness_path": str(hardness_path),
        "hardness_sha256": _sha256(hardness_path),
        "discovery_only": True,
        "test_loaded": False,
        "tasks": len(tasks),
        "seeds": seeds,
        "dimensions": dimensions,
        "stability": stability,
        "dominant_factor": {
            "pairwise_agreement": _mean_pairwise_agreement(dominant),
            "fleiss_kappa": _fleiss_kappa(dominant),
        },
        "recommended_action_pairwise_jaccard": _mean_pairwise_jaccard(actions),
        "outcome_associations": [],
        "claim_boundary": (
            "Stability is descriptive. Outcome associations are exploratory and do not "
            "authorize model selection or reopening a frozen finalist."
        ),
    }
    if outcomes_path is not None:
        outcomes_path = outcomes_path.resolve()
        outcome_payload = json.loads(outcomes_path.read_text(encoding="utf-8"))
        outcome_rows = _validate_outcomes(outcome_payload)
        output["outcomes_path"] = str(outcomes_path)
        output["outcomes_sha256"] = _sha256(outcomes_path)
        output["outcome_scope"] = (
            "discovery_only"
            if outcome_payload.get("discovery_only") is True
            else "post_freeze_no_reselection"
        )
        output["outcome_associations"] = _permutation_associations(
            card_means,
            outcome_rows,
            permutations=permutations,
            seed=seed,
        )
    return output


def render_markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# HardnessCard stability and validity audit",
        "",
        f"- Tasks: {payload['tasks']}",
        f"- Profiling seeds: {', '.join(map(str, payload['seeds']))}",
        "- Discovery-only profiling: true",
        "- Test loaded: false",
        "",
        "## Seed stability",
        "",
        "| Dimension | ICC(2,1) | Mean seed Spearman | Min seed Spearman | Median within-task SD |",
        "|---|---:|---:|---:|---:|",
    ]
    for name, values in payload["stability"].items():
        lines.append(
            f"| {name} | {values['icc_2_1_absolute_agreement']:.3f} | "
            f"{values['pairwise_seed_spearman']['mean']:.3f} | "
            f"{values['pairwise_seed_spearman']['minimum']:.3f} | "
            f"{values['median_within_task_sd']:.4f} |"
        )
    lines.extend(
        [
            "",
            (
                f"Dominant-factor agreement: "
                f"{payload['dominant_factor']['pairwise_agreement']:.3f}; "
                f"Fleiss kappa: {payload['dominant_factor']['fleiss_kappa']:.3f}."
            ),
            f"Recommended-action Jaccard: {payload['recommended_action_pairwise_jaccard']:.3f}.",
        ]
    )
    if payload["outcome_associations"]:
        lines.extend(
            [
                "",
                "## Exploratory outcome associations",
                "",
                "| Dimension | Outcome | Spearman | Permutation p | BH q |",
                "|---|---|---:|---:|---:|",
            ]
        )
        for row in payload["outcome_associations"]:
            lines.append(
                f"| {row['dimension']} | {row['outcome']} | {row['spearman']:.3f} | "
                f"{row['permutation_p_two_sided']:.4f} | {row['bh_q']:.4f} |"
            )
    lines.extend(["", f"Claim boundary: {payload['claim_boundary']}", ""])
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--hardness", type=Path, required=True)
    parser.add_argument("--outcomes", type=Path)
    parser.add_argument("--permutations", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=20260821)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--markdown", type=Path)
    args = parser.parse_args()
    result = analyze(
        args.hardness,
        outcomes_path=args.outcomes,
        permutations=args.permutations,
        seed=args.seed,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    markdown = args.markdown or args.output.with_suffix(".md")
    markdown.write_text(render_markdown(result), encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
