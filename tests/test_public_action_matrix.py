from __future__ import annotations

import numpy as np
import pytest

from admet_agent.public_action_matrix import evaluate_action, sampled_structural_support


def test_classification_actions_are_oriented_and_cost_audited():
    rng = np.random.default_rng(7)
    x = rng.normal(size=(80, 10))
    y = np.asarray([0, 1] * 40)
    groups = np.asarray([f"g{index // 2}" for index in range(80)])
    baseline = evaluate_action(x[:60], y[:60], groups[:60], x[60:], y[60:], action="baseline", task_type="classification", seed=3, trees=10, workers=1, inner_splits=2)
    nested = evaluate_action(x[:60], y[:60], groups[:60], x[60:], y[60:], action="nested_imbalance", task_type="classification", seed=3, trees=10, workers=1, inner_splits=2)
    assert baseline["fits"] == 1
    assert nested["fits"] == 3
    assert baseline["oriented_score"] == baseline["raw_score"]
    assert nested["outer_validation_labels_used_for_training_or_threshold"] is False


def test_regression_mae_is_minimize_oriented():
    x = np.arange(100, dtype=float).reshape(50, 2)
    y = np.linspace(-1, 1, 50)
    result = evaluate_action(x[:40], y[:40], np.arange(40), x[40:], y[40:], action="baseline", task_type="regression", seed=4, trees=10, workers=1)
    assert result["raw_score"] >= 0
    assert result["oriented_score"] == -result["raw_score"]
    with pytest.raises(ValueError):
        evaluate_action(x[:40], y[:40], np.arange(40), x[40:], y[40:], action="nested_imbalance", task_type="regression", seed=4, trees=10, workers=1)


def test_boosting_and_expanded_automl_are_explicit_controls():
    rng = np.random.default_rng(12)
    x = rng.normal(size=(90, 8))
    y = np.asarray([0, 1] * 45)
    groups = np.asarray([f"g{index // 2}" for index in range(90)])
    anchor = evaluate_action(x[:72], y[:72], groups[:72], x[72:], y[72:], action="boosting_anchor", task_type="classification", seed=2, trees=5, workers=1, inner_splits=2)
    automl = evaluate_action(x[:72], y[:72], groups[:72], x[72:], y[72:], action="automl_expanded", task_type="classification", seed=2, trees=5, workers=1, inner_splits=2)
    assert anchor["fits"] == 1
    assert automl["fits"] == 9
    assert automl["selected_automl_candidate"] in range(4)


def test_structural_support_is_label_blind_deterministic_tanimoto():
    train = np.asarray([[1, 0, 1], [0, 1, 0]], dtype=float)
    valid = np.asarray([[1, 0, 1], [1, 1, 0]], dtype=float)
    first = sampled_structural_support(train, np.asarray(["a", "b"]), valid, np.asarray(["c", "d"]), sample_size=2, seed=9)
    second = sampled_structural_support(train, np.asarray(["a", "b"]), valid, np.asarray(["c", "d"]), sample_size=2, seed=9)
    assert first == second
    assert 0 <= first <= 1
