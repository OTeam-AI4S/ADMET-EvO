from __future__ import annotations

from admet_agent.ecvoi_v4_budget_policy import (
    allocate_actions,
    oracle_allocation,
    score_allocation,
)


def _predictions():
    return {
        "low": {
            "baseline": {"mean": 0.0, "cost_ratio": 0.0, "eligible": True},
            "feature_rich": {"mean": 0.01, "cost_ratio": 10.0, "eligible": True},
            "imbalance_aware": {"mean": 0.005, "cost_ratio": 1.0, "eligible": True},
        },
        "high": {
            "baseline": {"mean": 0.0, "cost_ratio": 0.0, "eligible": True},
            "feature_rich": {"mean": 0.10, "cost_ratio": 10.0, "eligible": True},
            "imbalance_aware": {"mean": 0.006, "cost_ratio": 1.0, "eligible": True},
        },
    }


def test_exact_allocator_prioritizes_high_value_state_under_budget():
    allocation = allocate_actions(_predictions(), budget=11.0)

    assert allocation["actions"] == {
        "high": "feature_rich",
        "low": "imbalance_aware",
    }
    assert allocation["predicted_cost"] == 11.0


def test_oracle_and_scoring_use_fresh_effects_without_changing_costs():
    predictions = _predictions()
    actual = {
        "low": {"baseline": 0.0, "feature_rich": 0.08, "imbalance_aware": -0.01},
        "high": {"baseline": 0.0, "feature_rich": 0.02, "imbalance_aware": 0.01},
    }

    oracle = oracle_allocation(predictions, actual, budget=11.0)
    scored = score_allocation(
        oracle, actual, predictions, harm_margin=0.005
    )

    assert oracle["actions"] == {
        "high": "imbalance_aware",
        "low": "feature_rich",
    }
    assert scored["total_oriented_effect"] == 0.09
    assert scored["harmful_interventions"] == 0


def test_ineligible_harmful_action_is_never_allocated():
    predictions = _predictions()
    predictions["high"]["feature_rich"]["eligible"] = False

    allocation = allocate_actions(predictions, budget=20.0)

    assert allocation["actions"]["high"] != "feature_rich"
