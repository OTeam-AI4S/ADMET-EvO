from __future__ import annotations

import numpy as np
import pandas as pd

from admet_agent.public_policy_v6 import (
    IdentifiabilityContract,
    cross_source_posterior_v6,
    diagnostic_identifiability,
)
from admet_agent.public_policy_analysis import STATE_FEATURES


def _table(source_count: int, transferable: bool) -> pd.DataFrame:
    rows = []
    for i in range(source_count):
        x = i / max(1, source_count - 1)
        for replicate in range(2):
            effect = (0.20 * x - 0.05) if transferable else (0.05 if i % 2 else -0.05)
            row = {
                "endpoint_id": f"e{i}_{replicate}", "source_id": f"s{i}",
                "task_type": "classification", "effect_feature_rich": effect,
                "effect_nested_imbalance": -effect,
                "task_is_classification": 1.0, "log_train_rows": 4.0 + x,
                "train_minority_fraction_filled": x,
                "minority_fraction_missing": 0.0, "train_group_density": 0.5,
                "structural_support": x,
            }
            rows.append(row)
    return pd.DataFrame(rows)


def test_identifiability_fails_closed_with_too_few_sources():
    result = diagnostic_identifiability(
        _table(4, True), action="feature_rich", contract=IdentifiabilityContract()
    )
    assert result["identifiable"] is False
    assert result["shrinkage_weight"] == 0.0
    assert result["reason"] == "insufficient_training_sources"


def test_transferable_diagnostics_receive_nonzero_bounded_weight():
    result = diagnostic_identifiability(
        _table(9, True), action="feature_rich", contract=IdentifiabilityContract()
    )
    assert result["identifiable"] is True
    assert result["source_win_fraction"] >= 0.6
    assert 0.0 < result["shrinkage_weight"] <= 1.0


def test_contract_rejects_underidentified_configuration():
    try:
        IdentifiabilityContract(min_training_sources=3)
    except ValueError as exc:
        assert "four" in str(exc)
    else:
        raise AssertionError("underidentified contract was accepted")


def test_required_state_features_are_present_in_fixture():
    assert set(STATE_FEATURES).issubset(_table(5, True).columns)


def test_posterior_is_deterministic_and_excludes_held_source():
    table = _table(10, True)
    contract = IdentifiabilityContract()
    first = cross_source_posterior_v6(
        table, 0, action="feature_rich", identifiability_contract=contract,
        bootstraps=32, seed=17, harm_margin=0.005,
    )
    second = cross_source_posterior_v6(
        table, 0, action="feature_rich", identifiability_contract=contract,
        bootstraps=32, seed=17, harm_margin=0.005,
    )
    assert first == second
    assert first["training_sources"] == 9
    assert 0.0 <= first["harm_probability"] <= 1.0
