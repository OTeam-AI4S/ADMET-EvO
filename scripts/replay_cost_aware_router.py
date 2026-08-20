"""Replay three finalist policies on the completed 22-task validation search."""

from __future__ import annotations

import argparse
import json
import statistics
from collections import Counter
from dataclasses import replace
from pathlib import Path

from admet_agent.evidence_router import EvidenceRouter
from admet_agent.schemas import RunResult, TaskProfile


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


def reconstruct_candidates(payload: dict) -> list[RunResult]:
    """Recover cumulative per-recipe time omitted by the v2.1 summaries."""
    seeds = [int(seed) for seed in payload["validation_seeds"]]
    raw = [_result(item) for item in payload["results"]]
    robust = [_result(item) for item in payload["robust_results"]]
    costs = {}
    reconstructed = []
    for candidate in robust:
        matches = {
            item.seed: item
            for item in raw
            if item.status == "completed" and _signature(item) == _signature(candidate)
        }
        missing = set(seeds) - set(matches)
        if missing:
            raise ValueError(
                f"{candidate.task}/{candidate.run_id} is missing seeds {sorted(missing)}"
            )
        cumulative = sum(matches[seed].elapsed_seconds for seed in seeds)
        costs[candidate.run_id] = cumulative
        reconstructed.append(replace(candidate, elapsed_seconds=cumulative))

    for candidate in map(_result, payload["ensemble_results"]):
        members = candidate.hyperparameters.get("members", [])
        if len(members) < 2 or any(member not in costs for member in members):
            raise ValueError(f"Cannot reconstruct ensemble cost for {candidate.run_id}")
        cumulative = sum(costs[member] for member in members)
        costs[candidate.run_id] = cumulative
        reconstructed.append(replace(candidate, elapsed_seconds=cumulative))
    return reconstructed


def replay_task(payload: dict) -> dict:
    profile = TaskProfile(**payload["profile"])
    candidates = reconstruct_candidates(payload)
    policies = {
        "score": EvidenceRouter(uncertainty_penalty=0.0, cost_penalty=0.0),
        "lcb": EvidenceRouter(uncertainty_penalty=0.5, cost_penalty=0.0),
        "cost_aware_lcb": EvidenceRouter(
            uncertainty_penalty=0.5, cost_penalty=0.01
        ),
    }
    by_id = {candidate.run_id: candidate for candidate in candidates}
    selections = {}
    for name, router in policies.items():
        audit = router.select(profile, candidates)
        selected = by_id[audit["selected_run_id"]]
        selections[name] = {
            "run_id": selected.run_id,
            "action": audit["selected_action"],
            "validation_score": selected.validation_score,
            "validation_std": selected.validation_std,
            "recipe_seconds": selected.elapsed_seconds,
            "utility": audit["candidates"][0]["utility"],
        }
    direction = 1.0 if profile.metric != "mae" else -1.0
    score_value = selections["score"]["validation_score"]
    for name in ("lcb", "cost_aware_lcb"):
        selections[name]["validation_regret_vs_score"] = direction * (
            score_value - selections[name]["validation_score"]
        )
        selections[name]["recipe_cost_ratio_vs_score"] = (
            selections[name]["recipe_seconds"] / selections["score"]["recipe_seconds"]
        )
    return {
        "task": profile.task,
        "metric": profile.metric,
        "validation_seeds": payload["validation_seeds"],
        "candidate_count": len(candidates),
        "selections": selections,
    }


def build_report(root: Path) -> dict:
    paths = sorted(root.glob("tasks/*/tasks/*/search_seed_1.json"))
    if len(paths) != 22:
        raise ValueError(f"Expected 22 completed search summaries, found {len(paths)}")
    tasks = [replay_task(json.loads(path.read_text(encoding="utf-8"))) for path in paths]
    aggregate = {}
    for policy in ("score", "lcb", "cost_aware_lcb"):
        selections = [task["selections"][policy] for task in tasks]
        aggregate[policy] = {
            "action_counts": dict(sorted(Counter(item["action"] for item in selections).items())),
            "total_selected_recipe_hours": sum(
                item["recipe_seconds"] for item in selections
            )
            / 3600.0,
            "changed_from_score": sum(
                item["run_id"] != task["selections"]["score"]["run_id"]
                for task, item in zip(tasks, selections, strict=True)
            ),
        }
    cost_rows = [task["selections"]["cost_aware_lcb"] for task in tasks]
    aggregate["cost_aware_lcb"].update(
        {
            "total_recipe_cost_ratio_vs_score": (
                aggregate["cost_aware_lcb"]["total_selected_recipe_hours"]
                / aggregate["score"]["total_selected_recipe_hours"]
            ),
            "median_recipe_cost_ratio_vs_score": statistics.median(
                item["recipe_cost_ratio_vs_score"] for item in cost_rows
            ),
            "mean_validation_regret_vs_score": statistics.mean(
                item["validation_regret_vs_score"] for item in cost_rows
            ),
        }
    )
    return {
        "protocol": "replay of completed three-seed validation summaries",
        "official_test_loaded": False,
        "uncertainty_penalty": 0.5,
        "cost_penalty": 0.01,
        "task_count": len(tasks),
        "aggregate": aggregate,
        "tasks": tasks,
    }


def markdown(report: dict) -> str:
    lines = [
        "# Cost-aware evidence-router replay",
        "",
        (
            "Validation-only replay of the completed 22-task v2.1 search. No official "
            "test data or leaderboard scores are inputs to the router."
        ),
        "",
        (
            "| Task | Metric | Score action | LCB action | Cost-aware action | "
            "Cost ratio | Validation regret |"
        ),
        "|---|---:|---|---|---|---:|---:|",
    ]
    for task in sorted(report["tasks"], key=lambda item: item["task"].lower()):
        choices = task["selections"]
        cost = choices["cost_aware_lcb"]
        lines.append(
            f"| {task['task']} | {task['metric']} | {choices['score']['action']} | "
            f"{choices['lcb']['action']} | {cost['action']} | "
            f"{cost['recipe_cost_ratio_vs_score']:.3f} | "
            f"{cost['validation_regret_vs_score']:.4f} |"
        )
    aggregate = report["aggregate"]
    lines.extend(
        [
            "",
            "## Aggregate",
            "",
            f"- Score-selected recipe time: {aggregate['score']['total_selected_recipe_hours']:.2f} h.",
            f"- LCB changes: {aggregate['lcb']['changed_from_score']}/22 tasks.",
            (
                "- Cost-aware LCB changes: "
                f"{aggregate['cost_aware_lcb']['changed_from_score']}/22 tasks; "
                "total selected-recipe cost ratio "
                f"{aggregate['cost_aware_lcb']['total_recipe_cost_ratio_vs_score']:.3f}; "
                "median per-task ratio "
                f"{aggregate['cost_aware_lcb']['median_recipe_cost_ratio_vs_score']:.3f}."
            ),
            (
                "- Mean native-metric validation regret versus score selection: "
                f"{aggregate['cost_aware_lcb']['mean_validation_regret_vs_score']:.4f}."
            ),
            "",
            (
                "This replay measures decision behavior, not held-out generalization. A "
                "leave-one-seed-out evaluation remains required before claiming predictive "
                "gain."
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
    parser.add_argument(
        "--output", type=Path, default=Path("runs/v24-router-replay/report")
    )
    args = parser.parse_args()
    report = build_report(args.root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.with_suffix(".json").write_text(
        json.dumps(report, indent=2, sort_keys=True), encoding="utf-8"
    )
    args.output.with_suffix(".md").write_text(markdown(report), encoding="utf-8")
    print(json.dumps(report["aggregate"], sort_keys=True))


if __name__ == "__main__":
    main()
