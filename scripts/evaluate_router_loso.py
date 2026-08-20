"""Held-out-seed evaluation of finalist routing on existing validation artifacts."""

from __future__ import annotations

import argparse
import json
import statistics
from collections import Counter
from pathlib import Path

import numpy as np
from scipy.stats import wilcoxon
from tdc.benchmark_group import admet_group

from admet_agent.evidence_router import EvidenceRouter
from admet_agent.runner import _columns
from admet_agent.schemas import RunResult, TaskProfile
from admet_agent.selection import score
from admet_agent.tasks import resolve_task


def _result(payload: dict) -> RunResult:
    return RunResult(
        **{
            **payload,
            "representations": tuple(payload.get("representations", ())),
        }
    )


def _signature(result: RunResult) -> tuple:
    return (
        result.model,
        tuple(result.representations),
        json.dumps(result.hyperparameters, sort_keys=True),
    )


def replay_loso(
    profile: TaskProfile,
    templates: dict[str, RunResult],
    seed_scores: dict[str, dict[int, float]],
    seed_costs: dict[str, dict[int, float]],
    *,
    heldout_seeds: tuple[int, ...] = (2, 3),
    cost_penalty: float = 0.01,
) -> dict:
    all_seeds = sorted(next(iter(seed_scores.values())))
    policies = {
        "score": EvidenceRouter(uncertainty_penalty=0.0, cost_penalty=0.0),
        "lcb": EvidenceRouter(uncertainty_penalty=0.5, cost_penalty=0.0),
        "cost_aware_lcb": EvidenceRouter(
            uncertainty_penalty=0.5, cost_penalty=cost_penalty
        ),
    }
    records = {name: [] for name in policies}
    for heldout in heldout_seeds:
        training = [seed for seed in all_seeds if seed != heldout]
        candidates = []
        for run_id, template in templates.items():
            values = [seed_scores[run_id][seed] for seed in training]
            candidates.append(
                RunResult(
                    **{
                        **template.to_dict(),
                        "representations": tuple(template.representations),
                        "validation_score": statistics.mean(values),
                        "validation_std": statistics.pstdev(values),
                        "elapsed_seconds": sum(
                            seed_costs[run_id][seed] for seed in training
                        ),
                    }
                )
            )
        by_id = {candidate.run_id: candidate for candidate in candidates}
        for name, router in policies.items():
            decision = router.select(profile, candidates)
            selected_id = decision["selected_run_id"]
            selected = by_id[selected_id]
            records[name].append(
                {
                    "heldout_seed": heldout,
                    "selected_run_id": selected_id,
                    "selected_action": decision["selected_action"],
                    "heldout_score": seed_scores[selected_id][heldout],
                    "selection_seconds": selected.elapsed_seconds,
                }
            )
    direction = 1.0 if profile.metric != "mae" else -1.0
    summaries = {}
    for name, values in records.items():
        scores = [item["heldout_score"] for item in values]
        summaries[name] = {
            "mean": statistics.mean(scores),
            "std": statistics.pstdev(scores),
            "scores": scores,
            "selection_seconds": sum(item["selection_seconds"] for item in values),
            "action_counts": dict(
                sorted(Counter(item["selected_action"] for item in values).items())
            ),
            "decisions": values,
        }
    score_mean = summaries["score"]["mean"]
    for name in ("lcb", "cost_aware_lcb"):
        summaries[name]["oriented_gain_vs_score"] = direction * (
            summaries[name]["mean"] - score_mean
        )
        summaries[name]["selection_cost_ratio_vs_score"] = (
            summaries[name]["selection_seconds"]
            / summaries["score"]["selection_seconds"]
        )
    return {
        "task": profile.task,
        "metric": profile.metric,
        "candidate_generation_seed": 1,
        "heldout_router_seeds": list(heldout_seeds),
        "cost_penalty": cost_penalty,
        "policies": summaries,
    }


def _candidate_evidence(
    payload: dict,
    summary_path: Path,
    group,
) -> tuple[TaskProfile, dict, dict, dict]:
    profile = TaskProfile(**payload["profile"])
    task = resolve_task(profile.task)
    seeds = [int(seed) for seed in payload["validation_seeds"]]
    raw = [_result(item) for item in payload["results"]]
    templates = {
        item["run_id"]: _result(item)
        for item in [*payload["robust_results"], *payload["ensemble_results"]]
    }
    seed_scores = {}
    seed_costs = {}
    for item in map(_result, payload["robust_results"]):
        matches = {
            candidate.seed: candidate
            for candidate in raw
            if candidate.status == "completed"
            and _signature(candidate) == _signature(item)
        }
        if set(matches) < set(seeds):
            raise ValueError(f"Incomplete seed evidence for {profile.task}/{item.run_id}")
        seed_scores[item.run_id] = {
            seed: matches[seed].validation_score for seed in seeds
        }
        seed_costs[item.run_id] = {
            seed: matches[seed].elapsed_seconds for seed in seeds
        }

    run_dir = summary_path.parents[2]
    valid_targets = {}
    for seed in seeds:
        _, valid = group.get_train_valid_split(
            benchmark=task.name, split_type="default", seed=seed
        )
        _, y_col = _columns(valid)
        valid_targets[seed] = valid[y_col].to_numpy()
    for item in map(_result, payload["ensemble_results"]):
        members = item.hyperparameters.get("members", [])
        path = run_dir / "artifacts" / item.run_id / "validation_predictions.npz"
        with np.load(path, allow_pickle=False) as loaded:
            predictions = {seed: loaded[f"seed_{seed}"] for seed in seeds}
        seed_scores[item.run_id] = {
            seed: score(valid_targets[seed], predictions[seed], task.metric)
            for seed in seeds
        }
        seed_costs[item.run_id] = {
            seed: sum(seed_costs[member][seed] for member in members)
            for seed in seeds
        }
    return profile, templates, seed_scores, seed_costs


def _paired_summary(gains: list[float]) -> dict:
    nonzero = [value for value in gains if abs(value) > 1e-12]
    if nonzero:
        statistic, p_value = wilcoxon(gains, alternative="two-sided")
    else:
        statistic, p_value = 0.0, 1.0
    return {
        "mean": statistics.mean(gains),
        "median": statistics.median(gains),
        "improved": sum(value > 1e-12 for value in gains),
        "tied": sum(abs(value) <= 1e-12 for value in gains),
        "worse": sum(value < -1e-12 for value in gains),
        "wilcoxon_statistic": float(statistic),
        "wilcoxon_p_value": float(p_value),
    }


def build_report(root: Path, data_dir: Path) -> dict:
    paths = sorted(root.glob("tasks/*/tasks/*/search_seed_1.json"))
    if len(paths) != 22:
        raise ValueError(f"Expected 22 search summaries, found {len(paths)}")
    group = admet_group(path=str(data_dir))
    penalties = (0.0, 0.0001, 0.00025, 0.0005, 0.001, 0.002, 0.005, 0.01)
    grids = []
    for path in paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        evidence = _candidate_evidence(payload, path, group)
        grids.append(
            {
                penalty: replay_loso(*evidence, cost_penalty=penalty)
                for penalty in penalties
            }
        )

    sweep = {}
    for penalty in penalties:
        policies = [grid[penalty]["policies"] for grid in grids]
        gains_score = [item["cost_aware_lcb"]["oriented_gain_vs_score"] for item in policies]
        gains_lcb = [
            item["cost_aware_lcb"]["oriented_gain_vs_score"]
            - item["lcb"]["oriented_gain_vs_score"]
            for item in policies
        ]
        sweep[str(penalty)] = {
            "vs_score": _paired_summary(gains_score),
            "vs_lcb": _paired_summary(gains_lcb),
            "selection_cost_ratio_vs_lcb": (
                sum(item["cost_aware_lcb"]["selection_seconds"] for item in policies)
                / sum(item["lcb"]["selection_seconds"] for item in policies)
            ),
        }

    tolerance = 0.0005
    tasks = []
    selected_penalties = Counter()
    for heldout_index, grid in enumerate(grids):
        eligible = []
        for penalty in penalties:
            others = [item for index, item in enumerate(grids) if index != heldout_index]
            mean_gain_vs_lcb = statistics.mean(
                item[penalty]["policies"]["cost_aware_lcb"]["oriented_gain_vs_score"]
                - item[penalty]["policies"]["lcb"]["oriented_gain_vs_score"]
                for item in others
            )
            if mean_gain_vs_lcb < -tolerance:
                continue
            cost_ratio = (
                sum(
                    item[penalty]["policies"]["cost_aware_lcb"]["selection_seconds"]
                    for item in others
                )
                / sum(
                    item[penalty]["policies"]["lcb"]["selection_seconds"]
                    for item in others
                )
            )
            eligible.append((cost_ratio, penalty, mean_gain_vs_lcb))
        _, selected_penalty, training_gain = min(eligible)
        selected_penalties[str(selected_penalty)] += 1
        base = grid[0.0]
        selected = grid[selected_penalty]["policies"]["cost_aware_lcb"]
        lcb = base["policies"]["lcb"]
        tasks.append(
            {
                "task": base["task"],
                "metric": base["metric"],
                "score": base["policies"]["score"],
                "lcb": lcb,
                "looto_cost_aware": {
                    **selected,
                    "selected_cost_penalty": selected_penalty,
                    "training_tasks_mean_gain_vs_lcb": training_gain,
                    "oriented_gain_vs_lcb": (
                        selected["oriented_gain_vs_score"]
                        - lcb["oriented_gain_vs_score"]
                    ),
                },
            }
        )

    lcb_gains = [task["lcb"]["oriented_gain_vs_score"] for task in tasks]
    adaptive_score_gains = [
        task["looto_cost_aware"]["oriented_gain_vs_score"] for task in tasks
    ]
    adaptive_lcb_gains = [
        task["looto_cost_aware"]["oriented_gain_vs_lcb"] for task in tasks
    ]
    aggregate = {
        "lcb_vs_score": _paired_summary(lcb_gains),
        "looto_cost_aware_vs_score": _paired_summary(adaptive_score_gains),
        "looto_cost_aware_vs_lcb": _paired_summary(adaptive_lcb_gains),
        "looto_selection_cost_ratio_vs_score": (
            sum(task["looto_cost_aware"]["selection_seconds"] for task in tasks)
            / sum(task["score"]["selection_seconds"] for task in tasks)
        ),
        "looto_selection_cost_ratio_vs_lcb": (
            sum(task["looto_cost_aware"]["selection_seconds"] for task in tasks)
            / sum(task["lcb"]["selection_seconds"] for task in tasks)
        ),
        "selected_cost_penalties": dict(sorted(selected_penalties.items())),
    }
    return {
        "protocol": (
            "candidate set generated on seed 1; router trained on two seeds and "
            "evaluated on held-out seed 2 or 3"
        ),
        "official_test_loaded": False,
        "task_count": len(tasks),
        "heldout_decisions": len(tasks) * 2,
        "looto_gain_tolerance_vs_lcb": tolerance,
        "aggregate": aggregate,
        "penalty_sweep": sweep,
        "tasks": tasks,
    }


def markdown(report: dict) -> str:
    lines = [
        "# Evidence-router held-out-seed evaluation",
        "",
        report["protocol"] + ". Official test data were not loaded.",
        "",
        "| Task | Metric | Score | LCB gain | LOOTO cost gain | Lambda | Cost ratio |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for task in sorted(report["tasks"], key=lambda item: item["task"].lower()):
        selected = task["looto_cost_aware"]
        lines.append(
            f"| {task['task']} | {task['metric']} | {task['score']['mean']:.4f} | "
            f"{task['lcb']['oriented_gain_vs_score']:.4f} | "
            f"{selected['oriented_gain_vs_score']:.4f} | "
            f"{selected['selected_cost_penalty']:.5f} | "
            f"{selected['selection_cost_ratio_vs_score']:.3f} |"
        )
    lines.extend(["", "## Aggregate", ""])
    for name in ("lcb_vs_score", "looto_cost_aware_vs_score", "looto_cost_aware_vs_lcb"):
        item = report["aggregate"][name]
        lines.append(
            f"- {name}: mean oriented gain {item['mean']:.4f}; improved/tied/worse "
            f"{item['improved']}/{item['tied']}/{item['worse']}; "
            f"Wilcoxon p={item['wilcoxon_p_value']:.4f}."
        )
    lines.append(
        "- LOOTO cost ratio versus score/LCB: "
        f"{report['aggregate']['looto_selection_cost_ratio_vs_score']:.3f}/"
        f"{report['aggregate']['looto_selection_cost_ratio_vs_lcb']:.3f}."
    )
    lines.extend(
        [
            "",
            (
                "The evaluation isolates router selection on seeds 2-3 and chooses each "
                "task's cost penalty using the other 21 tasks. It does not prove "
                "generalization to the official test split or to a newly generated "
                "candidate set."
            ),
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root", type=Path, default=Path("runs/leaderboard-v21")
    )
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument(
        "--output", type=Path, default=Path("runs/v24-router-loso/report")
    )
    args = parser.parse_args()
    report = build_report(args.root, args.data_dir)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.with_suffix(".json").write_text(
        json.dumps(report, indent=2, sort_keys=True), encoding="utf-8"
    )
    args.output.with_suffix(".md").write_text(markdown(report), encoding="utf-8")
    print(json.dumps(report["aggregate"], sort_keys=True))


if __name__ == "__main__":
    main()
