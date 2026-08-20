import importlib.util
from pathlib import Path

import numpy as np


SCRIPT = Path(__file__).parents[1] / "scripts/ablate_router_memory_and_random.py"


def _load_script():
    spec = importlib.util.spec_from_file_location("router_controls", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_maximum_entropy_random_matches_target_cost_exactly():
    module = _load_script()
    costs = np.asarray([1.0, 2.0, 5.0])
    weights = module.maximum_entropy_weights(costs, 2.5)
    assert np.isclose(weights.sum(), 1.0)
    assert np.isclose(np.dot(weights, costs), 2.5, rtol=0.0, atol=1e-10)
    assert np.all(weights > 0)


def _grid(gains):
    grid = {}
    for penalty, (gain, cost) in gains.items():
        grid[penalty] = {
            "policies": {
                "lcb": {
                    "oriented_gain_vs_score": 0.0,
                    "selection_seconds": 100.0,
                },
                "cost_aware_lcb": {
                    "oriented_gain_vs_score": gain,
                    "selection_seconds": 100.0 * cost,
                },
            }
        }
    return {"grid": grid}


def test_winner_only_memory_discards_failures_and_changes_choice():
    module = _load_script()
    penalties = (0.0, 0.01)
    rows = [
        _grid({0.0: (0.0, 1.0), 0.01: (1.0, 0.5)}),
        _grid({0.0: (0.0, 1.0), 0.01: (-2.0, 0.5)}),
    ]
    winner, audit = module.choose_winner_only_penalty(rows, [0, 1], penalties)
    full = module.choose_penalty(
        rows, [0, 1], penalties=penalties, gain_tolerance=0.0005
    )
    assert audit["votes"] == {"0.0": 1, "0.01": 1}
    assert winner == 0.01
    assert full == 0.0


def test_generated_router_controls_are_validation_only_and_cost_matched():
    root = Path(__file__).parents[1]
    report = __import__("json").loads(
        (root / "runs/v27-router-controls/report.json").read_text()
    )
    assert report["task_count"] == 22
    assert report["official_test_loaded"] is False
    assert report["official_results_used"] is False
    assert report["leaderboard_used"] is False
    assert report["matched_cost_random"]["heldout_decisions"] == 44
    assert report["matched_cost_random"]["maximum_relative_cost_error"] <= 1e-9
    for control in ("memory_ablation", "matched_cost_random"):
        summary = report[control]["summary"]
        assert summary["improved"] + summary["tied"] + summary["worse"] == 22
        assert len(summary["bootstrap_mean_95_ci"]) == 2
