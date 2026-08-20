"""Leave-one-task-out ablation of context-conditioned cost penalties.

The candidate evidence and held-out scores come from validation seeds only.
Official test data and leaderboard positions are never loaded.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import wilcoxon
from tdc.benchmark_group import admet_group

from scripts.evaluate_router_loso import _candidate_evidence, replay_loso

PENALTIES = (0.0, 0.0001, 0.00025, 0.0005, 0.001, 0.002, 0.005, 0.01)
CONTEXT_MODES = ("global", "structural", "full")


def profile_vector(profile: dict[str, Any], mode: str) -> list[float]:
    """Encode only train/validation task-profile fields."""
    if mode not in {"structural", "full"}:
        raise ValueError(f"Unsupported context mode: {mode}")
    n_train = max(int(profile["n_train"]), 1)
    structural = [
        math.log1p(n_train),
        math.log1p(int(profile["n_valid"])),
        float(profile["unique_scaffolds_train"]) / n_train,
        float(profile["unseen_valid_scaffold_rate"]),
        float(profile["invalid_smiles_train"]) / n_train,
    ]
    if mode == "structural":
        return structural
    metrics = ("auroc", "auprc", "mae", "spearman")
    positive_rate = profile.get("positive_rate_train")
    target_mean = float(profile["target_mean"])
    target_std = float(profile["target_std"])
    return [
        *structural,
        float(profile["task_type"] == "classification"),
        *(float(profile["metric"] == metric) for metric in metrics),
        0.5 if positive_rate is None else float(positive_rate),
        math.copysign(math.log1p(abs(target_mean)), target_mean),
        math.log1p(abs(target_std)),
    ]


def nearest_tasks(
    rows: list[dict[str, Any]],
    heldout: int,
    mode: str,
    neighbor_count: int,
) -> list[int]:
    training = [index for index in range(len(rows)) if index != heldout]
    if mode == "global":
        return training
    matrix = np.asarray(
        [profile_vector(rows[index]["profile"], mode) for index in training],
        dtype=float,
    )
    target = np.asarray(profile_vector(rows[heldout]["profile"], mode), dtype=float)
    scale = np.std(matrix, axis=0)
    scale[scale < 1e-9] = 1.0
    distances = np.sqrt(np.sum(((matrix - target) / scale) ** 2, axis=1))
    count = min(neighbor_count, len(training))
    return [training[index] for index in np.argsort(distances)[:count]]


def choose_penalty(
    rows: list[dict[str, Any]],
    evidence_indices: list[int],
    *,
    penalties: tuple[float, ...] = PENALTIES,
    gain_tolerance: float = 0.0005,
) -> float:
    """Choose the cheapest penalty whose neighbor gain is within tolerance."""
    eligible = []
    for penalty in penalties:
        gains = []
        cost_ratios = []
        for index in evidence_indices:
            policies = rows[index]["grid"][penalty]["policies"]
            selected = policies["cost_aware_lcb"]
            lcb = policies["lcb"]
            gains.append(
                selected["oriented_gain_vs_score"]
                - lcb["oriented_gain_vs_score"]
            )
            cost_ratios.append(
                selected["selection_seconds"] / lcb["selection_seconds"]
            )
        mean_gain = statistics.mean(gains)
        if mean_gain >= -gain_tolerance:
            eligible.append(
                (statistics.mean(cost_ratios), penalty, mean_gain)
            )
    return min(eligible)[1] if eligible else 0.0


def _paired_summary(values: list[float]) -> dict[str, Any]:
    if any(abs(value) > 1e-12 for value in values):
        statistic, p_value = wilcoxon(values, alternative="two-sided")
    else:
        statistic, p_value = 0.0, 1.0
    return {
        "mean": statistics.mean(values),
        "median": statistics.median(values),
        "improved": sum(value > 1e-12 for value in values),
        "tied": sum(abs(value) <= 1e-12 for value in values),
        "worse": sum(value < -1e-12 for value in values),
        "wilcoxon_statistic": float(statistic),
        "wilcoxon_p_value": float(p_value),
    }


def evaluate_modes(
    rows: list[dict[str, Any]],
    *,
    penalties: tuple[float, ...] = PENALTIES,
    neighbor_count: int = 5,
    gain_tolerance: float = 0.0005,
) -> dict[str, Any]:
    output = {}
    for mode in CONTEXT_MODES:
        decisions = []
        gains = []
        selected_seconds = 0.0
        score_seconds = 0.0
        task_cost_ratios = []
        for heldout in range(len(rows)):
            neighbors = nearest_tasks(rows, heldout, mode, neighbor_count)
            penalty = choose_penalty(
                rows,
                neighbors,
                penalties=penalties,
                gain_tolerance=gain_tolerance,
            )
            policies = rows[heldout]["grid"][penalty]["policies"]
            selected = policies["cost_aware_lcb"]
            score_policy = policies["score"]
            gain = float(selected["oriented_gain_vs_score"])
            ratio = selected["selection_seconds"] / score_policy["selection_seconds"]
            gains.append(gain)
            task_cost_ratios.append(ratio)
            selected_seconds += selected["selection_seconds"]
            score_seconds += score_policy["selection_seconds"]
            decisions.append(
                {
                    "task": rows[heldout]["profile"]["task"],
                    "selected_cost_penalty": penalty,
                    "evidence_tasks": [
                        rows[index]["profile"]["task"] for index in neighbors
                    ],
                    "oriented_gain_vs_score": gain,
                    "selection_cost_ratio_vs_score": ratio,
                }
            )
        output[mode] = {
            "gain_vs_score": _paired_summary(gains),
            "mean_task_cost_ratio_vs_score": statistics.mean(task_cost_ratios),
            "aggregate_selection_cost_ratio_vs_score": (
                selected_seconds / score_seconds
            ),
            "selected_cost_penalties": dict(
                sorted(Counter(str(row["selected_cost_penalty"]) for row in decisions).items())
            ),
            "decisions": decisions,
        }
    return output


def build_report(root: Path, data_dir: Path) -> dict[str, Any]:
    paths = sorted(root.glob("tasks/*/tasks/*/search_seed_1.json"))
    if len(paths) != 22:
        raise ValueError(f"Expected 22 search summaries, found {len(paths)}")
    group = admet_group(path=str(data_dir))
    rows = []
    for path in paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        evidence = _candidate_evidence(payload, path, group)
        rows.append(
            {
                "profile": payload["profile"],
                "source": path.as_posix(),
                "grid": {
                    penalty: replay_loso(*evidence, cost_penalty=penalty)
                    for penalty in PENALTIES
                },
            }
        )
    return {
        "protocol": (
            "leave-one-task-out cost-penalty selection; candidate set generated "
            "on validation seed 1; router evaluated on validation seeds 2 and 3"
        ),
        "official_test_loaded": False,
        "leaderboard_used": False,
        "task_count": len(rows),
        "neighbor_count": 5,
        "gain_tolerance_vs_lcb": 0.0005,
        "penalties": list(PENALTIES),
        "modes": evaluate_modes(rows),
        "sources": [row["source"] for row in rows],
    }


def markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Context-conditioned evidence-router ablation",
        "",
        report["protocol"] + ". Official test and leaderboard data were not loaded.",
        "",
        "| Context | Mean gain vs score | Improved/tied/worse | Wilcoxon p | Aggregate cost ratio |",
        "|---|---:|---:|---:|---:|",
    ]
    for mode in CONTEXT_MODES:
        result = report["modes"][mode]
        gain = result["gain_vs_score"]
        lines.append(
            f"| {mode} | {gain['mean']:.4f} | "
            f"{gain['improved']}/{gain['tied']}/{gain['worse']} | "
            f"{gain['wilcoxon_p_value']:.4f} | "
            f"{result['aggregate_selection_cost_ratio_vs_score']:.3f} |"
        )
    lines.extend(
        [
            "",
            (
                "The contextual variants use five nearest training tasks after feature "
                "standardization. The global condition uses all other tasks. Context is "
                "promoted only if it improves held-out performance without erasing the "
                "global policy's cost advantage; otherwise it is retained as a negative "
                "result and the validated router remains metric-aware but non-contextual."
            ),
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("runs/leaderboard-v21"))
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument(
        "--output", type=Path, default=Path("runs/v24-router-context/report")
    )
    args = parser.parse_args()
    report = build_report(args.root, args.data_dir)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.with_suffix(".json").write_text(
        json.dumps(report, indent=2, sort_keys=True), encoding="utf-8"
    )
    args.output.with_suffix(".md").write_text(markdown(report), encoding="utf-8")
    print(json.dumps({key: value["gain_vs_score"] for key, value in report["modes"].items()}))


if __name__ == "__main__":
    main()
