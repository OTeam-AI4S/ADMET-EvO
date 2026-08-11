"""Compare completed ADMET runs with a frozen official leaderboard snapshot."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path


def expected_rank(score: float, entries: list[dict], direction: str) -> tuple[int, int]:
    if direction == "minimize":
        better = sum(item["mean"] < score for item in entries)
        no_worse = sum(item["mean"] <= score for item in entries)
    else:
        better = sum(item["mean"] > score for item in entries)
        no_worse = sum(item["mean"] >= score for item in entries)
    return better + 1, no_worse + 1


def finalist_label(finalist: dict) -> str:
    if finalist.get("kind") == "ensemble":
        names = [item["model"] for item in finalist["members"]]
        return "ensemble(" + "+".join(names) + ")"
    return finalist.get("model", "unknown")


def build(root: Path, snapshot_path: Path) -> dict:
    snapshot = json.loads(snapshot_path.read_text())
    rows = []
    missing = []
    for task, board in snapshot["pages"].items():
        result_path = root / "tasks" / task / "final" / "results.json"
        if not result_path.exists():
            missing.append(task)
            continue
        payload = json.loads(result_path.read_text())
        values = next(iter(payload["results"].values()))
        mean, std = float(values[0]), float(values[1])
        rank_best, rank_worst = expected_rank(mean, board["entries"], board["direction"])
        finalist = payload["finalists"][task]
        rows.append(
            {
                "task": task,
                "metric": board["metric"],
                "mean": mean,
                "std": std,
                "estimated_rank": rank_best,
                "tie_worst_rank": rank_worst,
                "existing_entries": len(board["entries"]),
                "leader_score": board["entries"][0]["mean"],
                "finalist": finalist_label(finalist),
                "url": board["url"],
            }
        )
    ranks = [row["estimated_rank"] for row in rows]
    return {
        "leaderboard_captured_at": snapshot["captured_at"],
        "completed": len(rows),
        "missing": missing,
        "summary": {
            "top_1": sum(rank == 1 for rank in ranks),
            "top_3": sum(rank <= 3 for rank in ranks),
            "top_5": sum(rank <= 5 for rank in ranks),
            "top_10": sum(rank <= 10 for rank in ranks),
            "median_rank": statistics.median(ranks) if ranks else None,
            "mean_rank": statistics.mean(ranks) if ranks else None,
        },
        "tasks": sorted(rows, key=lambda row: row["task"].lower()),
    }


def markdown(report: dict) -> str:
    lines = [
        "# ADMET Agent V2.1 leaderboard estimate",
        "",
        f"Leaderboard snapshot: {report['leaderboard_captured_at']}",
        "",
        f"Completed: {report['completed']}/22",
        "",
        "| Task | Metric | Agent | Estimated rank | Finalist |",
        "|---|---:|---:|---:|---|",
    ]
    for row in report["tasks"]:
        rank = str(row["estimated_rank"])
        if row["tie_worst_rank"] != row["estimated_rank"]:
            rank += f"-{row['tie_worst_rank']}"
        lines.append(
            f"| {row['task']} | {row['metric']} | "
            f"{row['mean']:.3f} +/- {row['std']:.3f} | "
            f"{rank}/{row['existing_entries'] + 1} | {row['finalist']} |"
        )
    if report["missing"]:
        lines.extend(["", "Missing: " + ", ".join(report["missing"])])
    lines.extend(["", "Summary: " + json.dumps(report["summary"], sort_keys=True)])
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("runs/leaderboard-v21"))
    args = parser.parse_args()
    report = build(args.root, args.root / "leaderboard_snapshot.json")
    (args.root / "comparison.json").write_text(json.dumps(report, indent=2, sort_keys=True))
    (args.root / "comparison.md").write_text(markdown(report))
    print(json.dumps({"completed": report["completed"], **report["summary"]}))


if __name__ == "__main__":
    main()
