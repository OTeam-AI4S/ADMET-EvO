"""Cross-seed ablation of metric-aware, reliability-penalized expert routing."""

from __future__ import annotations

import argparse
import itertools
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.stats import rankdata
from tdc.benchmark_group import admet_group

from admet_agent.runner import _columns
from admet_agent.selection import score
from admet_agent.tasks import resolve_task

BASE_EXPERTS = (
    "2d_lightgbm",
    "2d_extra_trees",
    "2d_svm",
    "frozen_unimol2_lightgbm",
    "frozen_unimol2_extra_trees",
    "hybrid_unimol2_2d_lightgbm",
    "finetuned_unimol2",
)


@dataclass(frozen=True)
class Candidate:
    members: tuple[str, ...]
    weights: tuple[float, ...]
    aggregation: str

    @property
    def name(self) -> str:
        if len(self.members) == 1:
            return self.members[0]
        joined = "+".join(
            f"{member}@{weight:.2f}"
            for member, weight in zip(self.members, self.weights, strict=True)
        )
        return f"{self.aggregation}({joined})"


def _prediction_path(root: Path, shard_root: Path, task: str, seed: int) -> Path:
    primary = root / task / f"seed_{seed}" / "validation_predictions.npz"
    if primary.exists():
        return primary
    matches = list(
        shard_root.glob(f"*seed{seed}/{task}/seed_{seed}/validation_predictions.npz")
    )
    if len(matches) != 1:
        raise FileNotFoundError(f"Expected one prediction file for {task} seed {seed}")
    return matches[0]


def _candidates(metric: str) -> list[Candidate]:
    candidates = [Candidate((name,), (1.0,), "raw") for name in BASE_EXPERTS]
    aggregations = ("raw",) if metric == "mae" else ("raw", "rank")
    for left, right in itertools.combinations(BASE_EXPERTS, 2):
        for left_weight in (0.25, 0.5, 0.75):
            weights = (left_weight, 1.0 - left_weight)
            candidates.extend(
                Candidate((left, right), weights, aggregation)
                for aggregation in aggregations
            )
    for members in itertools.combinations(BASE_EXPERTS, 3):
        candidates.extend(
            Candidate(members, (1 / 3, 1 / 3, 1 / 3), aggregation)
            for aggregation in aggregations
        )
    return candidates


def _predict(candidate: Candidate, predictions: dict[str, np.ndarray]) -> np.ndarray:
    values = []
    for member in candidate.members:
        current = predictions[member]
        if candidate.aggregation == "rank":
            current = rankdata(current, method="average") / len(current)
        values.append(current)
    return np.sum(
        [weight * value for weight, value in zip(candidate.weights, values, strict=True)],
        axis=0,
    )


def _utility(values: list[float], maximize: bool, penalty: float) -> float:
    array = np.asarray(values)
    direction = 1 if maximize else -1
    return float(direction * np.mean(array) - penalty * np.std(array))


def _route_loso(
    candidate_scores: dict[str, dict[int, float]],
    candidates: list[Candidate],
    maximize: bool,
    penalty: float,
) -> dict:
    selected = {}
    heldout_scores = []
    seeds = sorted(next(iter(candidate_scores.values())))
    for heldout in seeds:
        training_seeds = [seed for seed in seeds if seed != heldout]
        winner = max(
            candidates,
            key=lambda candidate: _utility(
                [candidate_scores[candidate.name][seed] for seed in training_seeds],
                maximize,
                penalty,
            ),
        )
        selected[str(heldout)] = winner.name
        heldout_scores.append(candidate_scores[winner.name][heldout])
    return {
        "mean": float(np.mean(heldout_scores)),
        "std": float(np.std(heldout_scores)),
        "selected": selected,
        "scores": heldout_scores,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("runs/unimol2-pilot-v22"))
    parser.add_argument(
        "--shard-root", type=Path, default=Path("runs/unimol2-pilot-v22-shards")
    )
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    args = parser.parse_args()

    group = admet_group(path=str(args.data_dir))
    report = {"protocol": "leave-one-outer-seed-out router ablation", "tasks": {}}
    for result_path in sorted(args.root.glob("*/results.json")):
        result = json.loads(result_path.read_text(encoding="utf-8"))
        task = resolve_task(result["task"])
        candidates = _candidates(task.metric)
        candidate_scores = {candidate.name: {} for candidate in candidates}
        base_scores = {name: {} for name in BASE_EXPERTS}
        for seed in range(1, 6):
            _, valid = group.get_train_valid_split(
                benchmark=task.name, split_type="default", seed=seed
            )
            _, y_col = _columns(valid)
            y_valid = valid[y_col].to_numpy()
            with np.load(
                _prediction_path(args.root, args.shard_root, task.name, seed)
            ) as loaded:
                predictions = {name: loaded[name] for name in BASE_EXPERTS}
            for candidate in candidates:
                value = score(y_valid, _predict(candidate, predictions), task.metric)
                candidate_scores[candidate.name][seed] = value
                if len(candidate.members) == 1:
                    base_scores[candidate.name][seed] = value

        direction = 1 if task.maximize else -1
        best_single = max(
            BASE_EXPERTS,
            key=lambda name: direction * np.mean(list(base_scores[name].values())),
        )
        best_candidate = max(
            candidates,
            key=lambda item: direction
            * np.mean(list(candidate_scores[item.name].values())),
        )
        oracle = []
        for seed in range(1, 6):
            oracle.append(
                max(base_scores[name][seed] for name in BASE_EXPERTS)
                if task.maximize
                else min(base_scores[name][seed] for name in BASE_EXPERTS)
            )
        task_report = {
            "metric": task.metric,
            "candidate_count": len(candidates),
            "reported_champion": result["champion"],
            "reported_champion_mean": result["summary"][result["champion"]]["mean"],
            "best_single": best_single,
            "best_single_mean": float(np.mean(list(base_scores[best_single].values()))),
            "best_blend_in_sample": best_candidate.name,
            "best_blend_in_sample_mean": float(
                np.mean(list(candidate_scores[best_candidate.name].values()))
            ),
            "single_oracle_mean": float(np.mean(oracle)),
            "loso": {},
        }
        single_candidates = [candidate for candidate in candidates if len(candidate.members) == 1]
        for penalty in (0.0, 0.25, 0.5, 1.0):
            task_report["loso"][f"single_lcb_{penalty}"] = _route_loso(
                candidate_scores, single_candidates, task.maximize, penalty
            )
            task_report["loso"][f"blend_lcb_{penalty}"] = _route_loso(
                candidate_scores, candidates, task.maximize, penalty
            )
        report["tasks"][task.name] = task_report

    output = args.root / "router_ablation.json"
    output.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
