"""Validation-only controls for negative memory and cost-aware routing.

This script consumes only the completed three-seed validation search.  It
compares full outcome memory with a survivorship-biased winner-only memory and
compares the selected candidate with a maximum-entropy random policy at exactly
matched expected training cost.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import wilcoxon
from tdc.benchmark_group import admet_group

from admet_agent.evidence_router import EvidenceRouter
from admet_agent.schemas import RunResult
from scripts.evaluate_router_context_ablation import PENALTIES, choose_penalty
from scripts.evaluate_router_loso import _candidate_evidence, replay_loso


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _softmax_tilt(values: np.ndarray, coefficient: float) -> np.ndarray:
    logits = coefficient * values
    logits -= np.max(logits)
    weights = np.exp(logits)
    return weights / weights.sum()


def maximum_entropy_weights(
    costs: list[float] | np.ndarray,
    target_cost: float,
    *,
    tolerance: float = 1e-12,
) -> np.ndarray:
    """Maximum-entropy weights with an exact expected-cost constraint."""
    raw = np.asarray(costs, dtype=float)
    if raw.ndim != 1 or raw.size == 0 or not np.all(np.isfinite(raw)):
        raise ValueError("Costs must be a non-empty finite vector")
    if np.any(raw < 0) or not math.isfinite(target_cost):
        raise ValueError("Costs and target must be finite and non-negative")
    minimum, maximum = float(raw.min()), float(raw.max())
    scale = max(1.0, abs(target_cost), abs(minimum), abs(maximum))
    if target_cost < minimum - tolerance * scale or target_cost > maximum + tolerance * scale:
        raise ValueError("Target cost lies outside the candidate cost range")
    if math.isclose(minimum, maximum, rel_tol=0.0, abs_tol=tolerance * scale):
        return np.full(raw.size, 1.0 / raw.size)
    if math.isclose(target_cost, minimum, rel_tol=0.0, abs_tol=tolerance * scale):
        mask = np.isclose(raw, minimum, rtol=0.0, atol=tolerance * scale)
        return mask.astype(float) / mask.sum()
    if math.isclose(target_cost, maximum, rel_tol=0.0, abs_tol=tolerance * scale):
        mask = np.isclose(raw, maximum, rtol=0.0, atol=tolerance * scale)
        return mask.astype(float) / mask.sum()

    normalized = (raw - minimum) / (maximum - minimum)
    target = (target_cost - minimum) / (maximum - minimum)
    low, high = -1024.0, 1024.0
    weights = np.full(raw.size, 1.0 / raw.size)
    for _ in range(256):
        middle = (low + high) / 2.0
        weights = _softmax_tilt(normalized, middle)
        expectation = float(np.dot(weights, normalized))
        if expectation < target:
            low = middle
        else:
            high = middle
    weights = _softmax_tilt(normalized, (low + high) / 2.0)
    error = abs(float(np.dot(weights, raw)) - target_cost) / scale
    if error > 1e-10:
        raise ValueError(f"Failed to match expected cost: relative error {error}")
    return weights


def _policy_gain_and_cost(grid: dict[float, dict[str, Any]], penalty: float) -> tuple[float, float]:
    policies = grid[penalty]["policies"]
    selected = policies["cost_aware_lcb"]
    lcb = policies["lcb"]
    gain = float(selected["oriented_gain_vs_score"] - lcb["oriented_gain_vs_score"])
    cost = float(selected["selection_seconds"] / lcb["selection_seconds"])
    return gain, cost


def task_winner_penalty(
    row: dict[str, Any], penalties: tuple[float, ...] = PENALTIES
) -> dict[str, float]:
    candidates = []
    for penalty in penalties:
        gain, cost = _policy_gain_and_cost(row["grid"], penalty)
        candidates.append(( -gain, cost, penalty, gain))
    _, cost, penalty, gain = min(candidates)
    return {"penalty": float(penalty), "gain_vs_lcb": gain, "cost_ratio_vs_lcb": cost}


def choose_winner_only_penalty(
    rows: list[dict[str, Any]],
    evidence_indices: list[int],
    penalties: tuple[float, ...] = PENALTIES,
) -> tuple[float, dict[str, Any]]:
    """Choose from one retained winner per evidence task."""
    retained = [task_winner_penalty(rows[index], penalties) for index in evidence_indices]
    votes = Counter(record["penalty"] for record in retained)
    gains: dict[float, list[float]] = defaultdict(list)
    costs: dict[float, list[float]] = defaultdict(list)
    for record in retained:
        gains[record["penalty"]].append(record["gain_vs_lcb"])
        costs[record["penalty"]].append(record["cost_ratio_vs_lcb"])
    selected = min(
        votes,
        key=lambda penalty: (
            -votes[penalty],
            -statistics.mean(gains[penalty]),
            statistics.mean(costs[penalty]),
            penalty,
        ),
    )
    return float(selected), {
        "votes": {str(key): value for key, value in sorted(votes.items())},
        "retained_records": retained,
    }


def bootstrap_interval(
    values: list[float], *, seed: int = 20260817, replicates: int = 10_000
) -> list[float]:
    array = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(array), size=(replicates, len(array)))
    means = array[indices].mean(axis=1)
    return [float(value) for value in np.quantile(means, [0.025, 0.975])]


def paired_summary(values: list[float]) -> dict[str, Any]:
    if any(abs(value) > 1e-12 for value in values):
        statistic, p_value = wilcoxon(values, alternative="two-sided")
    else:
        statistic, p_value = 0.0, 1.0
    return {
        "mean": statistics.mean(values),
        "median": statistics.median(values),
        "bootstrap_mean_95_ci": bootstrap_interval(values),
        "improved": sum(value > 1e-12 for value in values),
        "tied": sum(abs(value) <= 1e-12 for value in values),
        "worse": sum(value < -1e-12 for value in values),
        "wilcoxon_statistic": float(statistic),
        "wilcoxon_p_value": float(p_value),
    }


def _training_candidates(row: dict[str, Any], heldout: int) -> list[RunResult]:
    templates = row["templates"]
    scores = row["seed_scores"]
    costs = row["seed_costs"]
    all_seeds = sorted(next(iter(scores.values())))
    training = [seed for seed in all_seeds if seed != heldout]
    return [
        RunResult(
            **{
                **template.to_dict(),
                "representations": tuple(template.representations),
                "validation_score": statistics.mean(scores[run_id][seed] for seed in training),
                "validation_std": statistics.pstdev(scores[run_id][seed] for seed in training),
                "elapsed_seconds": sum(costs[run_id][seed] for seed in training),
            }
        )
        for run_id, template in templates.items()
    ]


def evaluate_memory(rows: list[dict[str, Any]]) -> dict[str, Any]:
    decisions = []
    gains = []
    full_seconds = 0.0
    winner_seconds = 0.0
    for heldout in range(len(rows)):
        evidence = [index for index in range(len(rows)) if index != heldout]
        full_penalty = choose_penalty(rows, evidence)
        winner_penalty, winner_audit = choose_winner_only_penalty(rows, evidence)
        full = rows[heldout]["grid"][full_penalty]["policies"]["cost_aware_lcb"]
        winner = rows[heldout]["grid"][winner_penalty]["policies"]["cost_aware_lcb"]
        gain = float(full["oriented_gain_vs_score"] - winner["oriented_gain_vs_score"])
        gains.append(gain)
        full_seconds += float(full["selection_seconds"])
        winner_seconds += float(winner["selection_seconds"])
        decisions.append(
            {
                "task": rows[heldout]["profile"]["task"],
                "metric": rows[heldout]["profile"]["metric"],
                "full_memory_penalty": full_penalty,
                "winner_only_penalty": winner_penalty,
                "full_memory_mean": float(full["mean"]),
                "winner_only_mean": float(winner["mean"]),
                "full_minus_winner_oriented_gain": gain,
                "full_to_winner_cost_ratio": float(
                    full["selection_seconds"] / winner["selection_seconds"]
                ),
                "winner_only_votes": winner_audit["votes"],
            }
        )
    summary = paired_summary(gains)
    return {
        "comparison": "full_outcome_memory_minus_winner_only_memory",
        "summary": summary,
        "aggregate_full_to_winner_cost_ratio": full_seconds / winner_seconds,
        "claim_gate": {
            "required_positive_mean": True,
            "required_task_wins": 12,
            "passed": summary["mean"] > 0 and summary["improved"] >= 12,
        },
        "decisions": decisions,
    }


def evaluate_matched_random(
    rows: list[dict[str, Any]], memory_report: dict[str, Any]
) -> dict[str, Any]:
    selected_penalties = {
        row["task"]: float(row["full_memory_penalty"])
        for row in memory_report["decisions"]
    }
    task_records = []
    task_gains = []
    all_cost_errors = []
    for row in rows:
        task = row["profile"]["task"]
        metric = row["profile"]["metric"]
        direction = -1.0 if metric == "mae" else 1.0
        penalty = selected_penalties[task]
        heldout_records = []
        for heldout in (2, 3):
            candidates = _training_candidates(row, heldout)
            by_id = {candidate.run_id: candidate for candidate in candidates}
            router = EvidenceRouter(uncertainty_penalty=0.5, cost_penalty=penalty)
            decision = router.select(row["profile_object"], candidates)
            selected = by_id[decision["selected_run_id"]]
            candidate_ids = list(by_id)
            costs = np.asarray([by_id[run_id].elapsed_seconds for run_id in candidate_ids])
            weights = maximum_entropy_weights(costs, selected.elapsed_seconds)
            heldout_scores = np.asarray(
                [row["seed_scores"][run_id][heldout] for run_id in candidate_ids],
                dtype=float,
            )
            expected_cost = float(np.dot(weights, costs))
            expected_score = float(np.dot(weights, heldout_scores))
            selected_score = float(row["seed_scores"][selected.run_id][heldout])
            cost_error = abs(expected_cost - selected.elapsed_seconds) / max(
                1.0, selected.elapsed_seconds
            )
            all_cost_errors.append(cost_error)
            entropy = -float(np.sum(weights[weights > 0] * np.log(weights[weights > 0])))
            heldout_records.append(
                {
                    "heldout_seed": heldout,
                    "selected_run_id": selected.run_id,
                    "selected_action": decision["selected_action"],
                    "selected_training_seconds": selected.elapsed_seconds,
                    "random_expected_training_seconds": expected_cost,
                    "relative_cost_error": cost_error,
                    "selected_heldout_score": selected_score,
                    "random_expected_heldout_score": expected_score,
                    "selected_minus_random_oriented_gain": direction
                    * (selected_score - expected_score),
                    "random_effective_candidate_count": math.exp(entropy),
                    "candidate_count": len(candidate_ids),
                }
            )
        task_gain = statistics.mean(
            item["selected_minus_random_oriented_gain"] for item in heldout_records
        )
        task_gains.append(task_gain)
        task_records.append(
            {
                "task": task,
                "metric": metric,
                "selected_cost_penalty": penalty,
                "selected_minus_random_oriented_gain": task_gain,
                "heldout": heldout_records,
            }
        )
    summary = paired_summary(task_gains)
    return {
        "comparison": "full_memory_agent_minus_maximum_entropy_matched_cost_random",
        "summary": summary,
        "heldout_decisions": len(rows) * 2,
        "maximum_relative_cost_error": max(all_cost_errors),
        "claim_gate": {
            "required_positive_mean": True,
            "required_task_wins": 12,
            "passed": summary["mean"] > 0 and summary["improved"] >= 12,
        },
        "tasks": task_records,
    }


def build_report(root: Path, data_dir: Path, preregistration_path: Path) -> dict[str, Any]:
    preregistration = json.loads(preregistration_path.read_text(encoding="utf-8"))
    if any(
        preregistration[field] is not False
        for field in ("official_test_loaded", "official_results_used", "leaderboard_used")
    ):
        raise ValueError("Preregistration does not declare a complete firewall")
    paths = sorted(root.glob("tasks/*/tasks/*/search_seed_1.json"))
    if len(paths) != 22:
        raise ValueError(f"Expected 22 search summaries, found {len(paths)}")
    group = admet_group(path=str(data_dir))
    rows = []
    for path in paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        profile, templates, seed_scores, seed_costs = _candidate_evidence(
            payload, path, group
        )
        rows.append(
            {
                "profile": payload["profile"],
                "profile_object": profile,
                "source": path.as_posix(),
                "templates": templates,
                "seed_scores": seed_scores,
                "seed_costs": seed_costs,
                "grid": {
                    penalty: replay_loso(
                        profile,
                        templates,
                        seed_scores,
                        seed_costs,
                        cost_penalty=penalty,
                    )
                    for penalty in PENALTIES
                },
            }
        )
    memory = evaluate_memory(rows)
    matched_random = evaluate_matched_random(rows, memory)
    return {
        "version": "v27-router-controls-1",
        "protocol": "preregistered retrospective validation-only LOOTO controls",
        "preregistration": preregistration_path.as_posix(),
        "preregistration_sha256": _sha256(preregistration_path),
        "official_test_loaded": False,
        "official_results_used": False,
        "leaderboard_used": False,
        "task_count": len(rows),
        "memory_ablation": memory,
        "matched_cost_random": matched_random,
        "sources": [row["source"] for row in rows],
    }


def markdown(report: dict[str, Any]) -> str:
    memory = report["memory_ablation"]
    random = report["matched_cost_random"]
    lines = [
        "# Router memory and matched-cost random controls",
        "",
        report["protocol"] + ". Official test, official results, and leaderboard data were not loaded.",
        "",
        "| Control | Mean oriented gain | 95% task-bootstrap CI | Improved/tied/worse | Wilcoxon p | Claim gate |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for item in (memory, random):
        summary = item["summary"]
        lines.append(
            f"| {item['comparison']} | {summary['mean']:.5f} | "
            f"[{summary['bootstrap_mean_95_ci'][0]:.5f}, {summary['bootstrap_mean_95_ci'][1]:.5f}] | "
            f"{summary['improved']}/{summary['tied']}/{summary['worse']} | "
            f"{summary['wilcoxon_p_value']:.4f} | {item['claim_gate']['passed']} |"
        )
    lines.extend(
        [
            "",
            f"Full/winner-only aggregate selected-cost ratio: {memory['aggregate_full_to_winner_cost_ratio']:.3f}.",
            "",
            f"Maximum matched-random relative cost error: {random['maximum_relative_cost_error']:.3e}.",
            "",
            "The comparison is retrospective and validation-only. A failed gate is retained as negative evidence and cannot be rewritten as support for the corresponding claim.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("runs/leaderboard-v21"))
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument(
        "--preregistration",
        type=Path,
        default=Path("runs/v27-router-controls/preregistration.json"),
    )
    parser.add_argument(
        "--output", type=Path, default=Path("runs/v27-router-controls/report")
    )
    args = parser.parse_args()
    report = build_report(args.root, args.data_dir, args.preregistration)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.with_suffix(".json").write_text(
        json.dumps(report, indent=2, sort_keys=True), encoding="utf-8"
    )
    args.output.with_suffix(".md").write_text(markdown(report), encoding="utf-8")
    print(
        json.dumps(
            {
                "memory": report["memory_ablation"]["summary"],
                "memory_gate": report["memory_ablation"]["claim_gate"]["passed"],
                "matched_random": report["matched_cost_random"]["summary"],
                "matched_random_gate": report["matched_cost_random"]["claim_gate"]["passed"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
