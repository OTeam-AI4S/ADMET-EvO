#!/usr/bin/env python3
"""Summarize paired ADMET ablations with metric-direction-aware improvements."""

from __future__ import annotations

import argparse
import json
import math
import random
import statistics
from pathlib import Path
from typing import Any


def _finite_scores(value: Any, label: str) -> list[float]:
    if not isinstance(value, list) or len(value) < 2:
        raise ValueError(f"{label} scores must be a list with at least two values")
    scores: list[float] = []
    for index, raw in enumerate(value):
        if isinstance(raw, bool) or not isinstance(raw, (int, float)):
            raise TypeError(f"{label} score {index} is not numeric")
        score = float(raw)
        if not math.isfinite(score):
            raise ValueError(f"{label} score {index} is not finite")
        scores.append(score)
    return scores


def _percentile(sorted_values: list[float], probability: float) -> float:
    position = probability * (len(sorted_values) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return sorted_values[lower]
    weight = position - lower
    return sorted_values[lower] * (1 - weight) + sorted_values[upper] * weight


def _bootstrap_ci(
    differences: list[float], samples: int, rng: random.Random
) -> list[float]:
    size = len(differences)
    means = [
        statistics.fmean(rng.choice(differences) for _ in range(size))
        for _ in range(samples)
    ]
    means.sort()
    return [_percentile(means, 0.025), _percentile(means, 0.975)]


def _sign_flip_pvalue(differences: list[float]) -> float:
    observed = abs(statistics.fmean(differences))
    size = len(differences)
    if size <= 20:
        extreme = 0
        total = 1 << size
        for mask in range(total):
            value = statistics.fmean(
                difference if mask & (1 << index) else -difference
                for index, difference in enumerate(differences)
            )
            if abs(value) >= observed - 1e-15:
                extreme += 1
        return extreme / total

    rng = random.Random(0)
    samples = 100_000
    extreme = 0
    for _ in range(samples):
        value = statistics.fmean(
            difference if rng.random() < 0.5 else -difference
            for difference in differences
        )
        if abs(value) >= observed - 1e-15:
            extreme += 1
    return (extreme + 1) / (samples + 1)


def _holm_adjust(pvalues: list[float]) -> list[float]:
    adjusted = [0.0] * len(pvalues)
    running = 0.0
    for rank, index in enumerate(sorted(range(len(pvalues)), key=pvalues.__getitem__)):
        value = min(1.0, (len(pvalues) - rank) * pvalues[index])
        running = max(running, value)
        adjusted[index] = running
    return adjusted


def _cost_delta(candidate: dict[str, Any], baseline: dict[str, Any]) -> dict[str, float]:
    candidate_cost = candidate.get("cost", {})
    baseline_cost = baseline.get("cost", {})
    if not isinstance(candidate_cost, dict) or not isinstance(baseline_cost, dict):
        raise TypeError("cost must be an object when provided")
    keys = sorted(set(candidate_cost) | set(baseline_cost))
    deltas: dict[str, float] = {}
    for key in keys:
        candidate_value = candidate_cost.get(key, 0.0)
        baseline_value = baseline_cost.get(key, 0.0)
        if isinstance(candidate_value, bool) or not isinstance(candidate_value, (int, float)):
            raise TypeError(f"candidate cost {key} is not numeric")
        if isinstance(baseline_value, bool) or not isinstance(baseline_value, (int, float)):
            raise TypeError(f"baseline cost {key} is not numeric")
        deltas[key] = float(candidate_value) - float(baseline_value)
    return deltas


def summarize(
    payload: dict[str, Any], bootstrap_samples: int, random_seed: int
) -> dict[str, Any]:
    metric = payload.get("metric")
    if not isinstance(metric, str) or not metric.strip():
        raise ValueError("metric must be a non-empty string")
    higher_is_better = payload.get("higher_is_better")
    if not isinstance(higher_is_better, bool):
        raise TypeError("higher_is_better must be boolean")
    baseline = payload.get("baseline")
    candidates = payload.get("candidates")
    if not isinstance(baseline, dict):
        raise TypeError("baseline must be an object")
    if not isinstance(candidates, list) or not candidates:
        raise ValueError("candidates must be a non-empty list")

    baseline_name = baseline.get("name")
    if not isinstance(baseline_name, str) or not baseline_name.strip():
        raise ValueError("baseline name must be a non-empty string")
    baseline_scores = _finite_scores(baseline.get("scores"), baseline_name)
    seeds = payload.get("seeds")
    if seeds is not None:
        if not isinstance(seeds, list) or len(seeds) != len(baseline_scores):
            raise ValueError("seeds must match the paired score count")
        if len({str(seed) for seed in seeds}) != len(seeds):
            raise ValueError("seeds must be unique")

    direction = 1.0 if higher_is_better else -1.0
    seen_names = {baseline_name}
    records: list[dict[str, Any]] = []
    raw_pvalues: list[float] = []
    for candidate in candidates:
        if not isinstance(candidate, dict):
            raise TypeError("each candidate must be an object")
        name = candidate.get("name")
        if not isinstance(name, str) or not name.strip():
            raise ValueError("candidate name must be a non-empty string")
        if name in seen_names:
            raise ValueError(f"duplicate method name: {name}")
        seen_names.add(name)
        scores = _finite_scores(candidate.get("scores"), name)
        if len(scores) != len(baseline_scores):
            raise ValueError(f"{name} scores do not match the baseline length")
        differences = [
            direction * (candidate_score - baseline_score)
            for baseline_score, candidate_score in zip(
                baseline_scores, scores, strict=True
            )
        ]
        rng = random.Random(f"{random_seed}:{name}")
        ci = _bootstrap_ci(differences, bootstrap_samples, rng)
        pvalue = _sign_flip_pvalue(differences)
        raw_pvalues.append(pvalue)
        wins = sum(value > 1e-12 for value in differences)
        ties = sum(abs(value) <= 1e-12 for value in differences)
        losses = len(differences) - wins - ties
        records.append(
            {
                "name": name,
                "scores": scores,
                "mean": statistics.fmean(scores),
                "sample_std": statistics.stdev(scores),
                "paired_improvements": differences,
                "mean_improvement": statistics.fmean(differences),
                "median_improvement": statistics.median(differences),
                "bootstrap_95_ci": ci,
                "wins_ties_losses": [wins, ties, losses],
                "sign_flip_pvalue_two_sided": pvalue,
                "cost_delta": _cost_delta(candidate, baseline),
            }
        )

    for record, adjusted in zip(records, _holm_adjust(raw_pvalues), strict=True):
        record["holm_adjusted_pvalue"] = adjusted
        lower, upper = record["bootstrap_95_ci"]
        if lower > 0:
            record["evidence_label"] = "positive_interval"
        elif upper < 0:
            record["evidence_label"] = "negative_interval"
        elif record["mean_improvement"] > 0:
            record["evidence_label"] = "promising_uncertain"
        elif record["mean_improvement"] < 0:
            record["evidence_label"] = "regressing_uncertain"
        else:
            record["evidence_label"] = "tie"

    return {
        "metric": metric,
        "higher_is_better": higher_is_better,
        "positive_improvement_means_better": True,
        "paired_count": len(baseline_scores),
        "seeds": seeds,
        "baseline": {
            "name": baseline_name,
            "scores": baseline_scores,
            "mean": statistics.fmean(baseline_scores),
            "sample_std": statistics.stdev(baseline_scores),
            "cost": baseline.get("cost", {}),
        },
        "candidates": sorted(
            records, key=lambda record: record["mean_improvement"], reverse=True
        ),
        "protocol": payload.get("protocol", {}),
        "method": {
            "bootstrap_samples": bootstrap_samples,
            "bootstrap_random_seed": random_seed,
            "confidence_interval": "paired percentile bootstrap",
            "pvalue": "exact paired sign-flip for <=20 pairs; Monte Carlo otherwise",
            "multiple_comparisons": "Holm adjustment across candidates",
        },
        "warning": (
            "Five paired seeds are preliminary: the minimum two-sided exact "
            "sign-flip p-value is 0.0625. Use practical thresholds and a "
            "preregistered confirmation split before promotion."
            if len(baseline_scores) == 5
            else None
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="Paired ablation JSON")
    parser.add_argument("--output", type=Path, help="Write summary JSON")
    parser.add_argument("--bootstrap-samples", type=int, default=20_000)
    parser.add_argument("--random-seed", type=int, default=20260813)
    args = parser.parse_args()
    if args.bootstrap_samples < 1_000:
        parser.error("--bootstrap-samples must be at least 1000")
    payload = json.loads(args.input.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError("input JSON must be an object")
    result = summarize(payload, args.bootstrap_samples, args.random_seed)
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")


if __name__ == "__main__":
    main()
