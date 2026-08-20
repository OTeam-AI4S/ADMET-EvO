from scripts.build_goal_scorecard import build_scorecard


def _entries(values):
    return [{"mean": value} for value in values]


def test_scorecard_tracks_both_goal_gates_and_metric_directions():
    comparison = {
        "tasks": [
            {
                "task": "Strong",
                "metric": "auroc",
                "mean": 0.90,
                "std": 0.01,
                "estimated_rank": 2,
                "tie_worst_rank": 2,
                "finalist": "model-a",
            },
            {
                "task": "Promote",
                "metric": "mae",
                "mean": 0.70,
                "std": 0.02,
                "estimated_rank": 7,
                "tie_worst_rank": 7,
                "finalist": "model-b",
            },
            {
                "task": "Rescue",
                "metric": "auprc",
                "mean": 0.40,
                "std": 0.03,
                "estimated_rank": 12,
                "tie_worst_rank": 12,
                "finalist": "model-c",
            },
        ]
    }
    snapshot = {
        "captured_at": "2026-08-13T00:00:00Z",
        "source": "test",
        "pages": {
            "Strong": {
                "direction": "maximize",
                "entries": _entries([0.95, 0.91, 0.89, 0.88, 0.87] + [0.8] * 5),
            },
            "Promote": {
                "direction": "minimize",
                "entries": _entries([0.4, 0.45, 0.5, 0.55, 0.6] + [0.75] * 5),
            },
            "Rescue": {
                "direction": "maximize",
                "entries": _entries([0.8, 0.75, 0.7, 0.65, 0.6] + [0.5] * 5),
            },
        },
    }

    result = build_scorecard(
        comparison, snapshot, required_top5=2, required_top10=3
    )

    assert result["status"] == {
        "top5": 1,
        "top10": 2,
        "top5_deficit": 1,
        "top10_deficit": 1,
        "goal_met": False,
    }
    by_task = {row["task"]: row for row in result["tasks"]}
    assert by_task["Strong"]["priority"] == "hold_top5"
    assert by_task["Promote"]["priority"] == "promote_top5"
    assert abs(by_task["Promote"]["improvement_to_top5"] - 0.1) < 1e-12
    assert by_task["Rescue"]["priority"] == "rescue_top10"
    assert abs(by_task["Rescue"]["improvement_to_top10"] - 0.1) < 1e-12
