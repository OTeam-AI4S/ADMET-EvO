"""Discovery-only suite reports for multi-seed search runs."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

from .tasks import TASKS


def _finalist(summary: dict[str, Any]) -> dict[str, Any] | None:
    finalist_id = summary.get("finalist_run_id")
    rows = summary.get("robust_results", []) + summary.get("ensemble_results", [])
    return next((item for item in rows if item.get("run_id") == finalist_id), None)


def _legacy_action_delta(summary: dict[str, Any]) -> dict[str, Any] | None:
    """Conservative fallback for runs created before matched evidence existed."""
    if summary.get("research_plan", {}).get("selected_action_id") != "model-imbalance-aware":
        return None
    rows = summary.get("robust_results", [])
    treatment = next(
        (item for item in rows if item.get("model") == "balanced_extra_trees"), None
    )
    control = next((item for item in rows if item.get("model") == "extra_trees"), None)
    if treatment is None or control is None:
        return None
    metric = summary.get("profile", {}).get("metric")
    direction = -1.0 if metric == "mae" else 1.0
    delta = direction * (
        float(treatment["validation_score"]) - float(control["validation_score"])
    )
    return {
        "status": (
            "supported" if delta >= 0.005 else "refuted" if delta <= -0.005 else "inconclusive"
        ),
        "mean_oriented_delta": delta,
        "single_axis_verified": True,
        "source": "legacy_robust_mean_pair",
    }


def summarize_search(run_dir: Path) -> dict[str, Any]:
    paths = sorted(run_dir.glob("tasks/*/search_seed_*.json"))
    rows = []
    for path in paths:
        summary = json.loads(path.read_text(encoding="utf-8"))
        finalist = _finalist(summary)
        if finalist is None:
            continue
        evidence = summary.get("action_evidence") or _legacy_action_delta(summary)
        std = finalist.get("validation_std")
        rows.append(
            {
                "task": summary["profile"]["task"],
                "metric": summary["profile"]["metric"],
                "hardness": summary.get("hardness", {}).get("overall"),
                "selected_action": summary.get("research_plan", {}).get(
                    "selected_action_id"
                ),
                "action_status": evidence.get("status") if evidence else "not_evaluated",
                "action_oriented_delta": (
                    evidence.get("mean_oriented_delta") if evidence else None
                ),
                "finalist_model": finalist["model"],
                "validation_score": finalist["validation_score"],
                "validation_std": std,
                "stability_risk": bool(std is not None and float(std) > 0.05),
                "validation_seeds": summary.get("validation_seeds", []),
                "source": str(path),
            }
        )
    expected = {item.name for item in TASKS.values()}
    present = {item["task"] for item in rows}
    stds = [float(item["validation_std"]) for item in rows if item["validation_std"] is not None]
    action_counts = Counter(item["action_status"] for item in rows)
    return {
        "protocol": "discovery_only_multi_seed_search_report",
        "run_dir": str(run_dir),
        "tasks": rows,
        "task_count": len(rows),
        "complete_tdc22": present == expected,
        "missing_tasks": sorted(expected - present),
        "mean_finalist_validation_std": float(np.mean(stds)) if stds else None,
        "stability_risk_tasks": [item["task"] for item in rows if item["stability_risk"]],
        "action_status_counts": dict(sorted(action_counts.items())),
        "test_loaded": False,
    }


def render_search_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# TDC-22 discovery-only search report",
        "",
        f"- Tasks completed: {report['task_count']}/22",
        f"- Mean finalist validation std: {report['mean_finalist_validation_std']}",
        f"- Stability-risk tasks (>0.05): {len(report['stability_risk_tasks'])}",
        "- Frozen test loaded: no",
        "",
        "| Task | Metric | Hardness | Selected action | Action evidence | Action Δ | Finalist | Validation mean ± std |",
        "|---|---|---:|---|---|---:|---|---:|",
    ]
    for row in report["tasks"]:
        delta = row["action_oriented_delta"]
        delta_text = "—" if delta is None else f"{float(delta):+.4f}"
        std = row["validation_std"]
        score_text = f"{float(row['validation_score']):.4f}"
        if std is not None:
            score_text += f" ± {float(std):.4f}"
        lines.append(
            f"| {row['task']} | {row['metric']} | {float(row['hardness']):.3f} | "
            f"{row['selected_action']} | {row['action_status']} | {delta_text} | "
            f"{row['finalist_model']} | {score_text} |"
        )
    lines.extend(
        [
            "",
            "> This report contains validation evidence only. It must not be described as frozen-test performance.",
            "",
        ]
    )
    return "\n".join(lines)


def write_search_report(
    run_dir: Path,
    output_json: Path,
    output_markdown: Path,
) -> dict[str, Any]:
    report = summarize_search(run_dir)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_markdown.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(
        json.dumps(report, indent=2, sort_keys=True), encoding="utf-8"
    )
    output_markdown.write_text(render_search_markdown(report), encoding="utf-8")
    return report
