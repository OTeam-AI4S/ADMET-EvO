import pytest

from scripts.evaluate_router_context_ablation import (
    choose_penalty,
    evaluate_modes,
    profile_vector,
)


def _profile(task, n_train=100, metric="auroc"):
    return {
        "task": task,
        "task_type": "classification",
        "metric": metric,
        "n_train": n_train,
        "n_valid": 20,
        "positive_rate_train": 0.2,
        "target_mean": 0.2,
        "target_std": 0.4,
        "invalid_smiles_train": 0,
        "unique_scaffolds_train": n_train // 2,
        "unseen_valid_scaffold_rate": 1.0,
    }


def _policy(gain, selected_seconds, score_seconds=100.0):
    return {
        "policies": {
            "score": {"selection_seconds": score_seconds},
            "lcb": {
                "oriented_gain_vs_score": 0.0,
                "selection_seconds": score_seconds,
            },
            "cost_aware_lcb": {
                "oriented_gain_vs_score": gain,
                "selection_seconds": selected_seconds,
            },
        }
    }


def _row(task, n_train, cheap_gain):
    return {
        "profile": _profile(task, n_train),
        "grid": {
            0.0: _policy(0.0, 100.0),
            0.01: _policy(cheap_gain, 10.0),
        },
    }


def test_profile_vector_rejects_unknown_mode():
    with pytest.raises(ValueError, match="Unsupported"):
        profile_vector(_profile("A"), "unknown")


def test_choose_penalty_uses_failures_to_enforce_gain_tolerance():
    rows = [_row("A", 100, -0.01), _row("B", 200, 0.0)]
    assert choose_penalty(rows, [0, 1], penalties=(0.0, 0.01)) == 0.0


def test_context_ablation_is_leave_one_task_out():
    rows = [
        _row("A", 100, 0.0),
        _row("B", 200, 0.0),
        _row("C", 300, 0.0),
    ]
    report = evaluate_modes(rows, penalties=(0.0, 0.01), neighbor_count=1)
    for mode in ("global", "structural", "full"):
        for decision in report[mode]["decisions"]:
            assert decision["task"] not in decision["evidence_tasks"]
        assert report[mode]["aggregate_selection_cost_ratio_vs_score"] == pytest.approx(0.1)
