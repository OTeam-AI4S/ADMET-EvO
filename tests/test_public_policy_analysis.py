from __future__ import annotations

import numpy as np
import pandas as pd

from admet_agent.ecvoi_v5_policy import EligibilityContract
from admet_agent.public_policy_analysis import endpoint_effect_table, select_policy


def _screen() -> pd.DataFrame:
    rows = []
    for endpoint, source, minority, feature, nested in (
        ("e1", "s1", 0.08, 0.00, 0.08),
        ("e2", "s2", 0.10, 0.01, 0.07),
        ("e3", "s3", 0.42, 0.06, -0.01),
        ("e4", "s4", 0.40, 0.05, 0.00),
    ):
        for fold in range(2):
            for action, effect in (("baseline", 0.0), ("feature_rich", feature), ("nested_imbalance", nested), ("boosting_anchor", 0.02), ("automl_expanded", 0.03)):
                rows.append({"endpoint_id": endpoint, "source_id": source, "outer_fold": fold, "action": action, "task_type": "classification", "oriented_score": 0.4 + effect, "train_rows": 100, "train_chemistry_groups": 50, "train_minority_fraction": minority, "median_sampled_validation_to_train_tanimoto": 0.4})
    return pd.DataFrame(rows)


def test_effect_table_and_cross_source_policy_are_deterministic():
    table = endpoint_effect_table(_screen())
    assert len(table) == 4
    assert np.isclose(table.loc[0, "effect_nested_imbalance"], 0.08)
    contract = EligibilityContract(practical_gain=0.005, harm_margin=0.005, harm_probability_limit=0.20)
    first = select_policy(table, mode="full", contract=contract, bootstraps=32, seed=11)
    second = select_policy(table, mode="full", contract=contract, bootstraps=32, seed=11)
    pd.testing.assert_frame_equal(first, second)
    assert set(first["source_id"]) == {"s1", "s2", "s3", "s4"}
    assert (first["realized_fit_cost"] <= 6).all()
