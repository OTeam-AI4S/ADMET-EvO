import numpy as np
import pandas as pd

from admet_agent.adaptive_external_data import (
    external_mass_capped_weights,
    select_relevance_diverse_binary_coreset,
    select_relevance_diverse_regression_coreset,
)


def _frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Y": [0, 0, 0, 0, 1, 1, 1, 1],
            "standard_smiles": [
                "CCO",
                "CCN",
                "CCC",
                "CCCl",
                "c1ccccc1",
                "c1ccncc1",
                "C1CCCCC1",
                "C1CCNCC1",
            ],
            "standard_inchikey": [f"key-{index}" for index in range(8)],
            "scaffold": ["a", "a", "b", "c", "d", "d", "e", "f"],
        }
    )


def test_coreset_is_deterministic_class_matched_and_diverse():
    relevance = np.asarray([0.9, 0.8, 0.7, 0.6, 0.95, 0.85, 0.75, 0.65])
    first, audit = select_relevance_diverse_binary_coreset(
        _frame(),
        relevance,
        total_rows=4,
        target_positive_rate=0.5,
        minimum_relevance=0.2,
    )
    second, _ = select_relevance_diverse_binary_coreset(
        _frame(),
        relevance,
        total_rows=4,
        target_positive_rate=0.5,
        minimum_relevance=0.2,
    )
    assert first.tolist() == second.tolist()
    assert _frame().iloc[first]["Y"].mean() == 0.5
    assert audit["unique_scaffolds"] == 4


def test_coreset_respects_relevance_floor_and_available_rows():
    selected, audit = select_relevance_diverse_binary_coreset(
        _frame(),
        np.asarray([0.1, 0.1, 0.7, 0.6, 0.1, 0.1, 0.75, 0.65]),
        total_rows=20,
        target_positive_rate=0.5,
        minimum_relevance=0.5,
    )
    assert len(selected) == 4
    assert audit["eligible_rows"] == 4
    assert audit["minimum_train_relevance"] >= 0.5


def test_external_weight_mass_equals_tdc_mass_by_default():
    weights = external_mass_capped_weights(100, 800)
    assert weights[:100].sum() == 100
    assert np.isclose(weights[100:].sum(), 100)
    assert np.isclose(weights[100], 0.125)


def test_regression_coreset_matches_target_range_and_relevance():
    frame = _frame().assign(Y=np.arange(8, dtype=float))
    selected, audit = select_relevance_diverse_regression_coreset(
        frame,
        np.asarray([0.8, 0.7, 0.6, 0.5, 0.8, 0.7, 0.6, 0.5]),
        total_rows=4,
        reference_targets=np.asarray([1.0, 2.0, 5.0, 6.0]),
        minimum_relevance=0.5,
        quantile_bins=2,
    )
    assert len(selected) == 4
    assert audit["minimum_train_relevance"] >= 0.5
    assert 1.0 <= audit["selected_target_median"] <= 6.0
