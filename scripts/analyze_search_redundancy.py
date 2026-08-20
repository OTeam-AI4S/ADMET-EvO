#!/usr/bin/env python3
"""Write a discovery-only search redundancy replay report."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from admet_agent.redundancy_analysis import analyze_search_run


def _markdown(report: dict) -> str:
    lines = [
        "# Search Redundancy Replay",
        "",
        f"- Completed tasks: {report['tasks_complete']}",
        f"- Finalists captured within 8 actions: "
        f"{report['tasks_captured_within_8_actions']}/{report['tasks_complete']}",
        f"- Practically equivalent within 8 actions: "
        f"{report['tasks_practically_equivalent_with_8_actions']}/"
        f"{report['tasks_complete']}",
        f"- Finalists captured within 4 promoted configurations: "
        f"{report['tasks_captured_within_4_promotions']}/{report['tasks_complete']}",
        f"- Ensemble finalists: {report['ensemble_finalists']}",
        f"- Material ensemble gains (> {report['practical_effect']:.3f}): "
        f"{report['material_ensemble_gains']}",
        "- Test loaded: false",
        "",
        "| Task | Actions | Promotion | Finalist | Min action prefix | "
        "Min promotion width | Ensemble gain |",
        "|---|---:|---:|---|---:|---:|---:|",
    ]
    for task in report["tasks"]:
        lines.append(
            f"| {task['task']} | {task['primary_actions']} | "
            f"{task['promoted_configurations']} | {task['finalist_model']} | "
            f"{task['minimum_execution_prefix_for_finalist']} | "
            f"{task['minimum_current_promotion_width_for_finalist']} | "
            f"{task['ensemble_oriented_gain_over_best_single']:.6f} |"
        )
    lines.extend(["", "## Promotion-width replay", ""])
    lines.append("| Width | Tasks | Practical equivalents | Mean regret | Max regret |")
    lines.append("|---:|---:|---:|---:|---:|")
    for width, row in report["promotion_width_replay"].items():
        lines.append(
            f"| {width} | {row['tasks_evaluated']} | "
            f"{row['practically_equivalent_tasks']} | "
            f"{row['mean_oriented_regret']:.6f} | "
            f"{row['max_oriented_regret']:.6f} |"
        )
    lines.extend(["", "## Action-prefix replay", ""])
    lines.append("| Width | Tasks | Practical equivalents | Mean regret | Max regret |")
    lines.append("|---:|---:|---:|---:|---:|")
    for width, row in report["action_width_replay"].items():
        lines.append(
            f"| {width} | {row['tasks_evaluated']} | "
            f"{row['practically_equivalent_tasks']} | "
            f"{row['mean_oriented_regret']:.6f} | "
            f"{row['max_oriented_regret']:.6f} |"
        )
    lines.extend(["", "## Limitations", ""])
    lines.extend(f"- {item}" for item in report["limitations"])
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-markdown", type=Path, required=True)
    parser.add_argument("--practical-effect", type=float, default=0.005)
    args = parser.parse_args()
    report = analyze_search_run(args.run_dir, args.practical_effect)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_markdown.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(report, indent=2, sort_keys=True), encoding="utf-8"
    )
    args.output_markdown.write_text(_markdown(report), encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items() if key != "tasks"}, indent=2))


if __name__ == "__main__":
    main()
