"""Discovery-only replay diagnostics for search-process redundancy.

The analysis deliberately consumes only completed validation summaries.  It is
intended to answer whether later primary actions, wide multi-seed promotion, or
ensembling materially changed a frozen discovery decision without touching the
test split.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
import json

from .tasks import resolve_task


PRACTICAL_EFFECT = 0.005


def _oriented(value: float, maximize: bool) -> float:
    return float(value) if maximize else -float(value)


def _completed(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [item for item in items if item.get("status") == "completed"]


def _finalist(summary: dict[str, Any]) -> dict[str, Any]:
    finalist_id = summary.get("finalist_run_id")
    for item in summary.get("robust_results", []) + summary.get(
        "ensemble_results", []
    ):
        if item.get("run_id") == finalist_id:
            return item
    raise ValueError(f"Missing completed finalist {finalist_id!r}")


def _member_ids(result: dict[str, Any]) -> list[str]:
    if result.get("model") != "ensemble":
        return [str(result["run_id"])]
    members = result.get("hyperparameters", {}).get("members", [])
    if not members:
        raise ValueError(f"Ensemble {result.get('run_id')} has no member ids")
    return [str(value) for value in members]


def analyze_search_summary(
    summary: dict[str, Any], practical_effect: float = PRACTICAL_EFFECT
) -> dict[str, Any]:
    """Measure which parts of one frozen validation search affected its finalist."""
    if practical_effect < 0:
        raise ValueError("practical_effect must be non-negative")
    task_name = summary.get("profile", {}).get("task")
    task = resolve_task(str(task_name))
    seeds = [int(value) for value in summary.get("validation_seeds", [1])]
    primary_seed = seeds[0]
    primary = [
        item
        for item in _completed(summary.get("results", []))
        if int(item.get("seed", -1)) == primary_seed
    ]
    promoted = _completed(summary.get("robust_results", []))
    ensembles = _completed(summary.get("ensemble_results", []))
    if not primary or not promoted:
        raise ValueError(f"{task.name} has no completed primary/promoted candidates")

    execution_order = {
        str(item["run_id"]): index for index, item in enumerate(primary, start=1)
    }
    score_order = sorted(
        primary,
        key=lambda item: (
            _oriented(item["validation_score"], task.maximize),
            str(item["run_id"]),
        ),
        reverse=True,
    )
    score_rank = {
        str(item["run_id"]): index for index, item in enumerate(score_order, start=1)
    }
    promoted_order = {
        str(item["run_id"]): index for index, item in enumerate(promoted, start=1)
    }
    primary_by_id = {str(item["run_id"]): item for item in primary}
    primary_oriented = [
        _oriented(item["validation_score"], task.maximize) for item in primary
    ]

    def prefix_best(width: int) -> float:
        return max(primary_oriented[: min(width, len(primary_oriented))])

    first_eight = primary_oriented[: min(8, len(primary_oriented))]
    best_position_in_first_eight = first_eight.index(max(first_eight)) + 1
    gain_actions_5_to_8 = (
        prefix_best(8) - prefix_best(4) if len(primary_oriented) >= 5 else 0.0
    )
    gain_actions_7_to_8 = (
        prefix_best(8) - prefix_best(6) if len(primary_oriented) >= 7 else 0.0
    )

    finalist = _finalist(summary)
    members = _member_ids(finalist)
    missing_members = [member for member in members if member not in primary_by_id]
    if missing_members:
        raise ValueError(
            f"{task.name} finalist members are absent from primary results: {missing_members}"
        )
    unpromoted_members = [member for member in members if member not in promoted_order]
    if unpromoted_members:
        raise ValueError(
            f"{task.name} finalist members are absent from promoted results: "
            f"{unpromoted_members}"
        )

    best_single = max(
        promoted,
        key=lambda item: _oriented(item["validation_score"], task.maximize),
    )
    finalist_oriented = _oriented(finalist["validation_score"], task.maximize)
    best_single_oriented = _oriented(
        best_single["validation_score"], task.maximize
    )
    ensemble_gain = (
        finalist_oriented - best_single_oriented
        if finalist.get("model") == "ensemble"
        else 0.0
    )

    # Replay narrower promotion using the already measured promoted pool.  This
    # is conservative because summaries retain only the top three ensembles.
    promoted_by_primary_score = sorted(
        promoted,
        key=lambda item: (
            _oriented(primary_by_id[str(item["run_id"])]["validation_score"], task.maximize),
            str(item["run_id"]),
        ),
        reverse=True,
    )
    promotion_replay = []
    for width in range(1, len(promoted_by_primary_score) + 1):
        allowed = {
            str(item["run_id"]) for item in promoted_by_primary_score[:width]
        }
        selectable = [
            item for item in promoted if str(item["run_id"]) in allowed
        ]
        selectable.extend(
            item
            for item in ensembles
            if set(_member_ids(item)).issubset(allowed)
        )
        replay_winner = max(
            selectable,
            key=lambda item: _oriented(item["validation_score"], task.maximize),
        )
        regret = finalist_oriented - _oriented(
            replay_winner["validation_score"], task.maximize
        )
        promotion_replay.append(
            {
                "width": width,
                "selected_run_id": replay_winner["run_id"],
                "validation_score": float(replay_winner["validation_score"]),
                "oriented_regret": float(regret),
                "practically_equivalent": regret <= practical_effect,
                "captures_actual_finalist_members": set(members).issubset(allowed),
            }
        )

    action_prefix_replay = []
    for width in range(1, len(primary) + 1):
        allowed = {
            run_id for run_id, order in execution_order.items() if order <= width
        }
        selectable = [
            item for item in promoted if str(item["run_id"]) in allowed
        ]
        selectable.extend(
            item
            for item in ensembles
            if set(_member_ids(item)).issubset(allowed)
        )
        if not selectable:
            continue
        replay_winner = max(
            selectable,
            key=lambda item: _oriented(item["validation_score"], task.maximize),
        )
        regret = finalist_oriented - _oriented(
            replay_winner["validation_score"], task.maximize
        )
        action_prefix_replay.append(
            {
                "width": width,
                "selected_run_id": replay_winner["run_id"],
                "validation_score": float(replay_winner["validation_score"]),
                "oriented_regret": float(regret),
                "practically_equivalent": regret <= practical_effect,
            }
        )

    return {
        "task": task.name,
        "metric": task.metric,
        "maximize": task.maximize,
        "primary_actions": len(primary),
        "promoted_configurations": len(promoted),
        "validation_seeds": seeds,
        "total_recorded_fits": len(_completed(summary.get("results", []))),
        "finalist_run_id": finalist["run_id"],
        "finalist_model": finalist["model"],
        "finalist_score": float(finalist["validation_score"]),
        "finalist_members": members,
        "minimum_execution_prefix_for_finalist": max(
            execution_order[member] for member in members
        ),
        "worst_primary_score_rank_for_finalist": max(
            score_rank[member] for member in members
        ),
        "best_action_position_in_first_8": best_position_in_first_eight,
        "oriented_gain_actions_5_to_8": float(gain_actions_5_to_8),
        "oriented_gain_actions_7_to_8": float(gain_actions_7_to_8),
        "minimum_current_promotion_width_for_finalist": max(
            promoted_order[member] for member in members
        ),
        "best_single_run_id": best_single["run_id"],
        "best_single_score": float(best_single["validation_score"]),
        "ensemble_oriented_gain_over_best_single": float(ensemble_gain),
        "ensemble_material": bool(ensemble_gain > practical_effect),
        "action_prefix_replay": action_prefix_replay,
        "promotion_replay": promotion_replay,
        "replay_scope": "current_promoted_pool_and_summary_top3_ensembles",
        "test_loaded": False,
    }


def analyze_search_run(
    run_dir: Path, practical_effect: float = PRACTICAL_EFFECT
) -> dict[str, Any]:
    paths = sorted(run_dir.glob("tasks/*/search_seed_*.json"))
    tasks = [
        analyze_search_summary(
            json.loads(path.read_text(encoding="utf-8")), practical_effect
        )
        for path in paths
    ]
    widths = sorted(
        {
            item["width"]
            for task in tasks
            for item in task["promotion_replay"]
        }
    )
    promotion_summary = {}
    for width in widths:
        rows = [
            next(
                (row for row in task["promotion_replay"] if row["width"] == width),
                None,
            )
            for task in tasks
        ]
        rows = [row for row in rows if row is not None]
        if not rows:
            continue
        promotion_summary[str(width)] = {
            "tasks_evaluated": len(rows),
            "practically_equivalent_tasks": sum(
                bool(row["practically_equivalent"]) for row in rows
            ),
            "mean_oriented_regret": sum(
                float(row["oriented_regret"]) for row in rows
            )
            / len(rows),
            "max_oriented_regret": max(
                float(row["oriented_regret"]) for row in rows
            ),
        }
    action_widths = sorted(
        {
            item["width"]
            for task in tasks
            for item in task["action_prefix_replay"]
        }
    )
    action_summary = {}
    for width in action_widths:
        rows = [
            next(
                (row for row in task["action_prefix_replay"] if row["width"] == width),
                None,
            )
            for task in tasks
        ]
        rows = [row for row in rows if row is not None]
        if not rows:
            continue
        action_summary[str(width)] = {
            "tasks_evaluated": len(rows),
            "practically_equivalent_tasks": sum(
                bool(row["practically_equivalent"]) for row in rows
            ),
            "mean_oriented_regret": sum(
                float(row["oriented_regret"]) for row in rows
            )
            / len(rows),
            "max_oriented_regret": max(
                float(row["oriented_regret"]) for row in rows
            ),
        }
    return {
        "protocol": "discovery-only-search-redundancy-replay-v1",
        "run_dir": str(run_dir),
        "tasks_complete": len(tasks),
        "practical_effect": practical_effect,
        "tasks_captured_within_8_actions": sum(
            task["minimum_execution_prefix_for_finalist"] <= 8 for task in tasks
        ),
        "tasks_practically_equivalent_with_8_actions": sum(
            any(
                row["width"] == 8 and row["practically_equivalent"]
                for row in task["action_prefix_replay"]
            )
            for task in tasks
        ),
        "tasks_captured_within_4_promotions": sum(
            task["minimum_current_promotion_width_for_finalist"] <= 4
            for task in tasks
        ),
        "ensemble_finalists": sum(
            task["finalist_model"] == "ensemble" for task in tasks
        ),
        "material_ensemble_gains": sum(task["ensemble_material"] for task in tasks),
        "action_width_replay": action_summary,
        "promotion_width_replay": promotion_summary,
        "tasks": tasks,
        "limitations": [
            "Replay uses only candidates already promoted in the frozen run.",
            "Search summaries retain only the top three ensemble candidates.",
            "Validation replay cannot establish validation-to-test transfer.",
        ],
        "test_loaded": False,
    }
