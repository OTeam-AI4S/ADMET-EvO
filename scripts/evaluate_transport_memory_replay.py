"""Measure whether admitted transport priors actually change 22-task decisions."""

from __future__ import annotations

import json
from pathlib import Path

from admet_agent.action_memory import ResearchOutcomeMemory
from admet_agent.hardness import HardnessCard, HardnessDimension
from admet_agent.research_loop import EffectFirstResearchPlanner


def _card(payload: dict) -> HardnessCard:
    return HardnessCard(
        task=payload["task"],
        task_type=payload["task_type"],
        metric=payload["metric"],
        overall=payload["overall"],
        dominant_factors=tuple(payload["dominant_factors"]),
        dimensions={
            name: HardnessDimension(**dimension)
            for name, dimension in payload["dimensions"].items()
        },
        recommended_actions=tuple(payload["recommended_actions"]),
        discovery_only=payload["discovery_only"],
        profiler_version=payload["profiler_version"],
        evidence_sha256=payload["evidence_sha256"],
    )


def main() -> None:
    root = Path(__file__).parents[1]
    history_dir = root / "runs/v3-agent-tdc22-v5/tasks"
    memory = ResearchOutcomeMemory(
        root / "runs/v5-transport-grid/transport_action_outcomes.jsonl"
    )
    planner = EffectFirstResearchPlanner()
    rows = []
    planner_actions = set()
    for path in sorted(history_dir.glob("*/search_seed_1.json")):
        summary = json.loads(path.read_text(encoding="utf-8"))
        card = _card(summary["hardness"])
        category = summary["endpoint_contract"]["category"]
        no_memory = planner.plan(card)
        with_memory = planner.plan(card, action_priors=memory.priors(category))
        no_action = no_memory["selected_action_id"]
        with_action = with_memory["selected_action_id"]
        no_order = [action["action_id"] for action in no_memory["actions"]]
        with_order = [action["action_id"] for action in with_memory["actions"]]
        planner_actions.update(no_order)
        no_utility = next(
            action["utility"]
            for action in no_memory["actions"]
            if action["action_id"] == no_action
        )
        with_utility = next(
            action["utility"]
            for action in with_memory["actions"]
            if action["action_id"] == with_action
        )
        rows.append(
            {
                "task": card.task,
                "no_memory_action": no_action,
                "with_memory_action": with_action,
                "selection_changed": no_action != with_action,
                "ranking_changed": no_order != with_order,
                "selected_utility_delta": with_utility - no_utility,
                "test_loaded": False,
            }
        )
    report = {
        "version": "transport-memory-behavior-replay-v1",
        "task_count": len(rows),
        "test_loaded": False,
        "selection_changed_tasks": sum(row["selection_changed"] for row in rows),
        "utility_changed_tasks": sum(
            abs(row["selected_utility_delta"]) > 1e-12 for row in rows
        ),
        "ranking_changed_tasks": sum(row["ranking_changed"] for row in rows),
        "behavioral_memory_advantage_demonstrated": False,
        "planner_action_coverage": {
            action_id: action_id in planner_actions
            for action_id in (
                "feature-domain-shift",
                "data-quality-filter",
                "model-estimator-family",
            )
        },
        "blockers": [
            "no_selected_action_changes_on_the_22_historical_hardness_cards",
            "global_prior_has_no_hardness_by_action_interaction",
            "eligible_priors_do_not_change_the_top_selected_action",
            "data_quality_filter_prior_is_neutral_after_all_inconclusive_outcomes",
            "no_independent_post_memory_endpoint_cohort_has_been_executed",
        ],
        "next_gate": (
            "Preregister predicted action choices on an independent endpoint cohort, then "
            "execute matched actions before comparing memory versus no-memory regret."
        ),
        "rows": rows,
    }
    (root / "reports/transport_memory_behavior_replay_v1.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    lines = [
        "# Transport-memory behavioral replay",
        "",
        "Status: **NO BEHAVIORAL ADVANTAGE DEMONSTRATED**",
        "",
        f"- Historical hardness cards replayed: {len(rows)}/22",
        f"- Selected actions changed: {report['selection_changed_tasks']}/22",
        f"- Full action rankings changed: {report['ranking_changed_tasks']}/22",
        f"- Selected utilities changed: {report['utility_changed_tasks']}/22",
        "- Test loaded: false",
        "",
        "The prospective grid created valid feature/model priors, but the current planner",
        "does not yet turn them into a different action choice on the 22 stored hardness",
        "cards. Feature/model evidence can change utilities or lower ranks without changing",
        "the top action; the data prior correctly stays neutral.",
        "",
        "Therefore the framework may claim protocol-bound cross-task evidence retention,",
        "not a memory-driven behavioral or predictive advantage. The next valid upgrade is",
        "an independent post-memory endpoint cohort with choices frozen before execution.",
        "",
    ]
    (root / "reports/transport_memory_behavior_replay_v1.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
