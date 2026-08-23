"""Compare the frozen ADMET Agent with fixed models across heterogeneous endpoints."""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
from pathlib import Path


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _agent_better(direction: str, agent_score: float, other_score: float) -> bool:
    return agent_score > other_score if direction == "maximize" else agent_score < other_score


def build(scorecard_path: Path, snapshot_path: Path) -> dict:
    scorecard = json.loads(scorecard_path.read_text(encoding="utf-8"))
    snapshot = json.loads(snapshot_path.read_text(encoding="utf-8"))
    agent_rows = {row["task"]: row for row in scorecard["tasks"]}
    pages = snapshot["pages"]
    if set(agent_rows) != set(pages) or len(pages) != 22:
        raise ValueError("Scorecard and leaderboard must contain the same 22 endpoints")

    complete_models = set.intersection(
        *[{entry["model"] for entry in page["entries"]} for page in pages.values()]
    )
    agent_ranks = []
    agent_percentiles = []
    top1_tasks = []
    per_task = []
    for task, page in pages.items():
        agent = agent_rows[task]
        better = sum(
            _agent_better(page["direction"], entry["mean"], agent["score"])
            for entry in page["entries"]
        )
        joint_rank = better + 1
        if joint_rank != agent["estimated_rank"]:
            raise ValueError(f"Rank mismatch for {task}: {joint_rank}")
        percentile = 1.0 - (joint_rank - 1) / len(page["entries"])
        agent_ranks.append(joint_rank)
        agent_percentiles.append(percentile)
        if joint_rank == 1:
            top1_tasks.append(task)
        per_task.append(
            {
                "task": task,
                "metric": page["metric"],
                "direction": page["direction"],
                "agent_score": agent["score"],
                "agent_std": agent["std"],
                "estimated_joint_rank": joint_rank,
                "leaderboard_entries_before_insertion": len(page["entries"]),
            }
        )

    fixed_rows = []
    for model in sorted(complete_models):
        ranks = []
        percentiles = []
        wins = ties = losses = 0
        for task, page in pages.items():
            agent = agent_rows[task]
            entry = next(item for item in page["entries"] if item["model"] == model)
            agent_wins = _agent_better(
                page["direction"], agent["score"], entry["mean"]
            )
            model_wins = _agent_better(
                page["direction"], entry["mean"], agent["score"]
            )
            wins += int(agent_wins)
            losses += int(model_wins)
            ties += int(not agent_wins and not model_wins)
            joint_rank = entry["rank"] + int(agent_wins)
            ranks.append(joint_rank)
            percentiles.append(1.0 - (joint_rank - 1) / len(page["entries"]))
        fixed_rows.append(
            {
                "model": model,
                "coverage": len(pages),
                "joint_mean_rank": statistics.mean(ranks),
                "mean_rank_percentile": statistics.mean(percentiles),
                "agent_pairwise_wins": wins,
                "ties": ties,
                "agent_pairwise_losses": losses,
            }
        )
    fixed_rows.sort(key=lambda row: (row["joint_mean_rank"], row["model"]))
    agent_mean_rank = statistics.mean(agent_ranks)
    agent_mean_percentile = statistics.mean(agent_percentiles)
    surpassed = [row for row in fixed_rows if agent_mean_rank < row["joint_mean_rank"]]
    not_surpassed = [row for row in fixed_rows if agent_mean_rank >= row["joint_mean_rank"]]

    return {
        "protocol": "heterogeneous-endpoint-joint-rank-comparison-v1",
        "leaderboard_captured_at": snapshot["captured_at"],
        "endpoint_count": len(pages),
        "agent": {
            "estimated_top1_count": len(top1_tasks),
            "estimated_top1_tasks": top1_tasks,
            "joint_mean_rank": agent_mean_rank,
            "median_rank": statistics.median(agent_ranks),
            "mean_rank_percentile": agent_mean_percentile,
            "top5_count": sum(rank <= 5 for rank in agent_ranks),
            "top10_count": sum(rank <= 10 for rank in agent_ranks),
        },
        "complete_coverage_fixed_model_count": len(fixed_rows),
        "fixed_model_ranking": fixed_rows,
        "fixed_models_surpassed_by_agent_mean_rank": len(surpassed),
        "fixed_models_not_surpassed_by_agent_mean_rank": [
            row["model"] for row in not_surpassed
        ],
        "agent_position_among_complete_fixed_models_plus_agent": 1
        + sum(row["joint_mean_rank"] < agent_mean_rank for row in fixed_rows),
        "per_task": per_task,
        "test_loaded_during_analysis": False,
        "claim_boundary": [
            "Ranks are estimated by inserting frozen Agent scores into a captured public leaderboard; they are not official submitted ranks.",
            "Raw metrics are not averaged because AUROC, AUPRC, Spearman and MAE have different scales and directions.",
            "Only exact model labels present on all 22 captured pages are eligible for the complete-coverage comparison.",
            "Mean rank is descriptive and does not by itself establish statistical superiority or equal compute budgets.",
        ],
        "source_hashes": {
            str(scorecard_path): _sha256(scorecard_path),
            str(snapshot_path): _sha256(snapshot_path),
        },
    }


def render(payload: dict) -> str:
    agent = payload["agent"]
    lines = [
        "# ADMET Agent 跨 22 endpoints 的固定模型比较",
        "",
        f"榜单快照：`{payload['leaderboard_captured_at']}`",
        "",
        "## 主要结果",
        "",
        f"- 估算 Top-1：{agent['estimated_top1_count']}/22（{', '.join(agent['estimated_top1_tasks'])}）",
        f"- Top-5 / Top-10：{agent['top5_count']}/22 / {agent['top10_count']}/22",
        f"- 联合平均名次：{agent['joint_mean_rank']:.3f}；中位名次：{agent['median_rank']:.1f}",
        f"- 平均 rank percentile：{agent['mean_rank_percentile']:.3f}",
        f"- 在 Agent + {payload['complete_coverage_fixed_model_count']} 个完整覆盖固定模型中："
        f"第 {payload['agent_position_among_complete_fixed_models_plus_agent']} 名",
        f"- 平均名次超过：{payload['fixed_models_surpassed_by_agent_mean_rank']}/"
        f"{payload['complete_coverage_fixed_model_count']} 个固定模型",
        f"- 尚未超过：{', '.join(payload['fixed_models_not_surpassed_by_agent_mean_rank'])}",
        "",
        "## 完整覆盖固定模型",
        "",
        "| 模型 | 22-task 联合平均名次 | 平均 rank percentile | Agent 对它 W/T/L |",
        "|---|---:|---:|---:|",
    ]
    for row in payload["fixed_model_ranking"]:
        lines.append(
            f"| {row['model']} | {row['joint_mean_rank']:.3f} | "
            f"{row['mean_rank_percentile']:.3f} | "
            f"{row['agent_pairwise_wins']}/{row['ties']}/{row['agent_pairwise_losses']} |"
        )
    lines.extend(
        [
            "",
            "## 解释边界",
            "",
            *[f"- {item}" for item in payload["claim_boundary"]],
            "",
            "因此可表述为：Agent 已超过绝大多数完整覆盖的单一固定模型，但当前",
            "证据不能表述为‘22-task 平均超过所有模型’；MapLight + GNN 的联合",
            "平均名次仍更好。",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--scorecard", type=Path, default=Path("reports/goal_scorecard_v27.json")
    )
    parser.add_argument(
        "--snapshot",
        type=Path,
        default=Path("reports/leaderboard_snapshot_2026-08-17.json"),
    )
    parser.add_argument(
        "--output-prefix",
        type=Path,
        default=Path("reports/cross_endpoint_fixed_model_comparison_v1"),
    )
    args = parser.parse_args()
    payload = build(args.scorecard, args.snapshot)
    args.output_prefix.with_suffix(".json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    args.output_prefix.with_suffix(".md").write_text(render(payload), encoding="utf-8")


if __name__ == "__main__":
    main()
