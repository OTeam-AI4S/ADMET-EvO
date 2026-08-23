from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression

from admet_agent.ecvoi_v5_policy import (
    EligibilityContract,
    allocate_actions,
    prepare_state_actions,
)
from admet_agent.imbalance_action import (
    fit_nested_threshold_action,
    select_mcc_threshold,
)


def _raw():
    stop = {"mean": 0.0, "harm_probability": 0.0, "cost_ratio": 0.0}
    return {
        "a": {
            "stop": stop,
            "cheap_weak": {"mean": 0.004, "harm_probability": 0.0, "cost_ratio": 1.0},
            "useful": {"mean": 0.020, "harm_probability": 0.1, "cost_ratio": 5.0},
        },
        "b": {
            "stop": stop,
            "harmful": {"mean": 0.030, "harm_probability": 0.3, "cost_ratio": 1.0},
            "useful": {"mean": 0.010, "harm_probability": 0.1, "cost_ratio": 5.0},
        },
    }


def test_v5_does_not_fill_budget_with_weak_or_harmful_actions():
    prepared = prepare_state_actions(_raw(), contract=EligibilityContract())
    allocation = allocate_actions(prepared, budget=100.0)
    assert allocation["actions"] == {"a": "useful", "b": "useful"}
    assert prepared["a"]["cheap_weak"]["ineligibility_reasons"] == [
        "below_practical_gain"
    ]
    assert prepared["b"]["harmful"]["ineligibility_reasons"] == [
        "exceeds_harm_probability"
    ]
    assert allocation["unused_budget"] == 90.0


def test_v5_selects_stop_when_no_intervention_is_eligible_or_affordable():
    prepared = prepare_state_actions(_raw(), contract=EligibilityContract())
    allocation = allocate_actions(prepared, budget=0.0)
    assert allocation["actions"] == {"a": "stop", "b": "stop"}
    assert allocation["predicted_effect"] == 0.0


def test_mcc_threshold_is_selected_from_oof_probabilities():
    target = np.array([0, 0, 0, 1, 1])
    probability = np.array([0.05, 0.10, 0.20, 0.30, 0.40])
    threshold = select_mcc_threshold(target, probability)
    assert 0.20 < threshold <= 0.30


def test_nested_threshold_action_covers_rows_with_group_disjoint_inner_folds():
    generator = np.random.default_rng(7)
    groups = np.repeat(np.arange(20), 2)
    x = generator.normal(size=(40, 3))
    target = np.tile([0, 1], 20)

    def factory(seed: int):
        return LogisticRegression(random_state=seed, max_iter=500)

    result = fit_nested_threshold_action(
        x,
        target,
        groups,
        estimator_factory=factory,
        splits=4,
        seed=11,
    )
    assert result.inner_fits == 4
    assert np.isfinite(result.oof_probability).all()
    assert 0.05 <= result.threshold <= 0.95
    assert result.predict(x).shape == (40,)
