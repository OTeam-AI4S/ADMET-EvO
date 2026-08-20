"""Build the 22-task leaderboard scorecard used by the active research goal."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

try:
    from scripts.compare_leaderboards import build as build_comparison
except ModuleNotFoundError:  # Direct execution from the scripts directory.
    from compare_leaderboards import build as build_comparison


def _threshold(entries: list[dict], rank: int) -> float | None:
    if len(entries) < rank:
        return None
    return float(entries[rank - 1]["mean"])


def _improvement_needed(
    score: float, threshold: float | None, direction: str
) -> float | None:
    if threshold is None:
        return None
    if direction == "minimize":
        return max(0.0, score - threshold)
    return max(0.0, threshold - score)


def build_scorecard(
    comparison: dict,
    snapshot: dict,
    *,
    required_top5: int = 15,
    required_top10: int = 22,
) -> dict:
    """Combine frozen results and leaderboard thresholds into goal evidence."""
    boards = snapshot["pages"]
    rows = []
    for result in comparison["tasks"]:
        board = boards[result["task"]]
        top5 = _threshold(board["entries"], 5)
        top10 = _threshold(board["entries"], 10)
        rank = int(result["estimated_rank"])
        if rank > 10:
            priority = "rescue_top10"
        elif rank > 5:
            priority = "promote_top5"
        else:
            priority = "hold_top5"
        rows.append(
            {
                "task": result["task"],
                "metric": result["metric"],
                "direction": board["direction"],
                "score": float(result["mean"]),
                "std": float(result["std"]),
                "estimated_rank": rank,
                "tie_worst_rank": int(result["tie_worst_rank"]),
                "top5_threshold": top5,
                "top10_threshold": top10,
                "improvement_to_top5": _improvement_needed(
                    float(result["mean"]), top5, board["direction"]
                ),
                "improvement_to_top10": _improvement_needed(
                    float(result["mean"]), top10, board["direction"]
                ),
                "priority": priority,
                "finalist": result["finalist"],
            }
        )

    top5_count = sum(row["estimated_rank"] <= 5 for row in rows)
    top10_count = sum(row["estimated_rank"] <= 10 for row in rows)
    rescue = sorted(
        row["task"] for row in rows if row["priority"] == "rescue_top10"
    )
    promotion = sorted(
        (row for row in rows if row["priority"] == "promote_top5"),
        key=lambda row: (row["estimated_rank"], row["improvement_to_top5"]),
    )
    return {
        "leaderboard_captured_at": snapshot["captured_at"],
        "source": snapshot["source"],
        "goal": {
            "total_tasks": len(rows),
            "required_top5": required_top5,
            "required_top10": required_top10,
        },
        "status": {
            "top5": top5_count,
            "top10": top10_count,
            "top5_deficit": max(0, required_top5 - top5_count),
            "top10_deficit": max(0, required_top10 - top10_count),
            "goal_met": top5_count >= required_top5
            and top10_count >= required_top10,
        },
        "rescue_top10": rescue,
        "promotion_candidates": [row["task"] for row in promotion],
        "tasks": rows,
    }


def markdown(scorecard: dict) -> str:
    status = scorecard["status"]
    goal = scorecard["goal"]
    lines = [
        "# ADMET-EvO leaderboard goal scorecard",
        "",
        f"Leaderboard snapshot: {scorecard['leaderboard_captured_at']}",
        "",
        (
            f"Current: **{status['top5']}/{goal['total_tasks']} Top-5**, "
            f"**{status['top10']}/{goal['total_tasks']} Top-10**. "
            f"Target: at least {goal['required_top5']} Top-5 and "
            f"{goal['required_top10']} Top-10."
        ),
        "",
        f"Goal met: **{'yes' if status['goal_met'] else 'no'}**",
        "",
        (
            "| Priority | Task | Metric | Score | Rank | Top-5 threshold | "
            "Gap to Top-5 | Top-10 threshold | Gap to Top-10 |"
        ),
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    order = {"rescue_top10": 0, "promote_top5": 1, "hold_top5": 2}
    for row in sorted(
        scorecard["tasks"],
        key=lambda item: (order[item["priority"]], item["estimated_rank"], item["task"]),
    ):
        def value(item: float | None) -> str:
            return "n/a" if item is None else f"{item:.4f}"

        lines.append(
            f"| {row['priority']} | {row['task']} | {row['metric']} | "
            f"{row['score']:.4f} | {row['estimated_rank']} | "
            f"{value(row['top5_threshold'])} | {value(row['improvement_to_top5'])} | "
            f"{value(row['top10_threshold'])} | {value(row['improvement_to_top10'])} |"
        )
    lines.extend(
        [
            "",
            (
                "A gap is expressed in the native metric and is always non-negative: "
                "higher-is-better metrics must gain that amount; MAE must decrease by it."
            ),
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, default=Path("runs/leaderboard-v21"))
    parser.add_argument(
        "--snapshot",
        type=Path,
        default=Path("reports/leaderboard_snapshot_2026-08-13.json"),
    )
    parser.add_argument(
        "--output", type=Path, default=Path("reports/goal_scorecard")
    )
    args = parser.parse_args()

    comparison = build_comparison(args.run_root, args.snapshot)
    snapshot = json.loads(args.snapshot.read_text(encoding="utf-8"))
    scorecard = build_scorecard(comparison, snapshot)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.with_suffix(".json").write_text(
        json.dumps(scorecard, indent=2, sort_keys=True), encoding="utf-8"
    )
    args.output.with_suffix(".md").write_text(markdown(scorecard), encoding="utf-8")
    print(json.dumps(scorecard["status"], sort_keys=True))


if __name__ == "__main__":
    main()
