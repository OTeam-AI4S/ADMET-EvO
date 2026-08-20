import pytest

from scripts.analyze_router_failures import analyze, bootstrap_interval


def _task(name, metric, gain, cost):
    return {
        "task": name,
        "metric": metric,
        "score": {
            "std": 0.1,
            "action_counts": {"fuse": 2},
        },
        "looto_cost_aware": {
            "oriented_gain_vs_score": gain,
            "selection_cost_ratio_vs_score": cost,
            "selected_cost_penalty": 0.005,
            "action_counts": {"stop_2d": 2},
        },
    }


def _profile(name):
    return {
        "task": name,
        "task_type": "classification",
        "n_train": 100,
        "unique_scaffolds_train": 50,
    }


def test_bootstrap_interval_is_deterministic():
    first = bootstrap_interval([1.0, 2.0, 3.0], sum, samples=100, seed=7)
    second = bootstrap_interval([1.0, 2.0, 3.0], sum, samples=100, seed=7)
    assert first == second


def test_failure_analysis_keeps_material_tail_losses_visible():
    loso = {
        "official_test_loaded": False,
        "tasks": [
            _task("A", "auroc", -0.01, 0.2),
            _task("B", "auprc", 0.02, 0.5),
            _task("C", "auroc", 0.0, 1.0),
        ],
    }
    report = analyze(loso, {name: _profile(name) for name in ("A", "B", "C")})
    assert [row["task"] for row in report["material_regressions"]] == ["A"]
    assert [row["task"] for row in report["material_improvements"]] == ["B"]
    assert report["official_test_loaded"] is False
    assert report["leaderboard_used"] is False


def test_failure_analysis_fail_closes_without_no_test_declaration():
    with pytest.raises(ValueError, match="explicit no-test"):
        analyze({"official_test_loaded": True, "tasks": []}, {})
