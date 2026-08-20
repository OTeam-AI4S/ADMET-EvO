"""Suite-level reporting for the fixed 22-task discovery diagnosis."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from statistics import mean, median
from typing import Any

from .tasks import TASKS


def _diagnosis_paths(run_dir: Path, seed: int) -> list[Path]:
    return sorted(run_dir.glob(f"tasks/*/diagnosis_seed_{seed}.json"))


def summarize_diagnoses(run_dir: Path, *, seed: int = 1) -> dict[str, Any]:
    packets = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in _diagnosis_paths(run_dir, seed)
    ]
    expected = {item.name for item in TASKS.values()}
    present = {packet["task"] for packet in packets}
    duplicates = len(packets) != len(present)
    if duplicates:
        raise ValueError("Duplicate diagnosis packets detected")
    dimensions = sorted(
        {
            name
            for packet in packets
            for name in packet["hardness"]["dimensions"]
        }
    )
    dimension_summary = {}
    for name in dimensions:
        rows = [
            (
                packet["task"],
                float(packet["hardness"]["dimensions"][name]["severity"]),
                float(packet["hardness"]["dimensions"][name]["confidence"]),
            )
            for packet in packets
            if name in packet["hardness"]["dimensions"]
        ]
        ordered = sorted(rows, key=lambda item: (-item[1], item[0]))
        dimension_summary[name] = {
            "mean_severity": mean(item[1] for item in rows) if rows else 0.0,
            "median_severity": median(item[1] for item in rows) if rows else 0.0,
            "mean_confidence": mean(item[2] for item in rows) if rows else 0.0,
            "top_tasks": [
                {"task": task, "severity": severity, "confidence": confidence}
                for task, severity, confidence in ordered[:5]
            ],
        }
    action_counts = Counter(
        packet["research_plan"]["selected_action_id"] for packet in packets
    )
    task_rows = []
    for packet in packets:
        hardness = packet["hardness"]
        quality = hardness["dimensions"]["data_quality"]["evidence"]
        shift = hardness["dimensions"]["domain_shift"]
        task_rows.append(
            {
                "task": packet["task"],
                "task_type": packet["endpoint_contract"]["task_type"],
                "metric": packet["endpoint_contract"]["primary_metric"],
                "n_train": packet["profile"]["n_train"],
                "hardness": hardness["overall"],
                "dominant_factors": hardness["dominant_factors"],
                "selected_action": packet["research_plan"]["selected_action_id"],
                "domain_shift": shift["severity"],
                "domain_shift_confidence": shift["confidence"],
                "conflicting_structures": quality["conflicting_structures"],
                "exact_train_valid_overlap_rows": hardness["dimensions"][
                    "split_leakage_risk"
                ]["evidence"]["exact_train_valid_overlap_rows"],
            }
        )
    task_rows.sort(key=lambda item: (-item["hardness"], item["task"]))
    return {
        "protocol": "tdc22_discovery_only_hardness_summary_v1",
        "seed": seed,
        "expected_tasks": 22,
        "observed_tasks": len(present),
        "complete": present == expected,
        "missing_tasks": sorted(expected - present),
        "extra_tasks": sorted(present - expected),
        "mean_hardness": mean(row["hardness"] for row in task_rows) if task_rows else 0.0,
        "median_hardness": median(row["hardness"] for row in task_rows) if task_rows else 0.0,
        "selected_action_counts": dict(sorted(action_counts.items())),
        "dimensions": dimension_summary,
        "tasks": task_rows,
        "test_loaded": False,
    }


def render_markdown(summary: dict[str, Any]) -> str:
    lines = [
        "# TDC-22 Discovery-only Hardness 基线",
        "",
        f"- 完整性：{summary['observed_tasks']}/{summary['expected_tasks']}",
        f"- 平均 Hardness：{summary['mean_hardness']:.3f}",
        f"- 中位 Hardness：{summary['median_hardness']:.3f}",
        "- Test loaded：false",
        "",
        "## 动作路由分布",
        "",
        "| Action | Tasks |",
        "|---|---:|",
    ]
    lines.extend(
        f"| `{name}` | {count} |"
        for name, count in summary["selected_action_counts"].items()
    )
    lines.extend(
        [
            "",
            "## Endpoint 明细",
            "",
            "| Task | Type | Metric | N train | Hardness | Domain shift | Confidence | Selected action | Conflicts | Train-valid overlap |",
            "|---|---|---|---:|---:|---:|---:|---|---:|---:|",
        ]
    )
    for row in summary["tasks"]:
        lines.append(
            f"| {row['task']} | {row['task_type']} | {row['metric']} | "
            f"{row['n_train']} | {row['hardness']:.3f} | {row['domain_shift']:.3f} | "
            f"{row['domain_shift_confidence']:.3f} | `{row['selected_action']}` | "
            f"{row['conflicting_structures']} | {row['exact_train_valid_overlap_rows']} |"
        )
    lines.extend(
        [
            "",
            "## 维度最高的 endpoints",
            "",
        ]
    )
    for name, values in summary["dimensions"].items():
        tasks = ", ".join(
            f"{item['task']} ({item['severity']:.3f})"
            for item in values["top_tasks"]
        )
        lines.append(
            f"- **{name}**：mean={values['mean_severity']:.3f}, "
            f"confidence={values['mean_confidence']:.3f}; {tasks}"
        )
    return "\n".join(lines) + "\n"


def write_suite_report(
    run_dir: Path,
    output_json: Path,
    output_markdown: Path,
    *,
    seed: int = 1,
) -> dict[str, Any]:
    summary = summarize_diagnoses(run_dir, seed=seed)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_markdown.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )
    output_markdown.write_text(render_markdown(summary), encoding="utf-8")
    return summary
