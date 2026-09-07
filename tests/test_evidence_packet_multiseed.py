import numpy as np
import pandas as pd

from admet_agent.evidence_packet_multiseed import (
    effective_external_sample_weights,
    pooled_normalized_improvement,
    quantity_rows,
    reconcile_external_records,
    select_external_positions,
    summarize_grid,
)


def test_cross_paper_conflicts_are_removed_and_duplicates_are_collapsed():
    frame = pd.DataFrame(
        {
            "Drug": ["CCO", "OCC", "CCN"],
            "Y": [0, 0, 1],
            "source_record_id": ["a", "b", "c"],
            "evidence_packet_id": ["paper-a", "paper-b", "paper-b"],
            "evidence_base_weight": [0.25, 0.75, 0.50],
        }
    )
    output, audit = reconcile_external_records(frame, task_type="classification")
    assert len(output) == 2
    assert audit["duplicate_identity_rows_collapsed"] == 1
    ethanol = output.loc[output["Y"] == 0].iloc[0]
    assert ethanol["evidence_base_weight"] == 0.75

    conflict = frame.copy()
    conflict.loc[1, "Y"] = 1
    output, audit = reconcile_external_records(
        conflict, task_type="classification"
    )
    assert output["standard_inchikey"].nunique() == 1
    assert audit["cross_packet_conflicting_identity_count"] == 1
    assert audit["conflicting_identity_rows_removed"] == 2
    assert audit["duplicate_identity_rows_collapsed"] == 0


def test_effective_weights_multiply_packet_reliability():
    frame = pd.DataFrame({"evidence_base_weight": [0.25, 0.75, 0.50]})
    weights = effective_external_sample_weights(
        frame, np.array([2, 0]), multiplier=0.5
    )
    np.testing.assert_allclose(weights, [0.25, 0.125])


def test_quantity_rows_is_bounded_and_deterministic():
    assert quantity_rows(10, 0.25) == 2
    assert quantity_rows(3, 0.25) == 1
    assert quantity_rows(10, 1.0) == 10


def test_binary_subset_is_label_matched_and_validation_blind():
    frame = pd.DataFrame(
        {
            "Drug": ["C", "CC", "CCC", "CCCC", "c1ccccc1", "CCO"],
            "Y": [0, 0, 0, 1, 1, 1],
        }
    )
    first, audit = select_external_positions(
        frame,
        native_targets=np.array([0, 0, 0, 1]),
        task_type="classification",
        total_rows=4,
    )
    second, _ = select_external_positions(
        frame,
        native_targets=np.array([0, 0, 0, 1]),
        task_type="classification",
        total_rows=4,
    )
    assert first.tolist() == second.tolist()
    assert len(first) == 4
    assert audit["selection_uses_validation_labels"] is False
    assert audit["selection_uses_test_labels"] is False


def test_grid_uses_all_seed_mean_and_top3_is_diagnostic_only():
    rows = []
    for seed, delta in zip((1, 2, 3, 4, 5), (0.10, 0.09, 0.08, -0.20, -0.20)):
        rows.append(
            {
                "seed": seed,
                "weight": 1.0,
                "quantity_fraction": 1.0,
                "normalized_improvement": delta,
            }
        )
    for seed, delta in zip((1, 2, 3, 4, 5), (0.03, 0.02, 0.02, 0.01, 0.01)):
        rows.append(
            {
                "seed": seed,
                "weight": 0.5,
                "quantity_fraction": 0.5,
                "normalized_improvement": delta,
            }
        )
    summaries, selection = summarize_grid(rows)
    assert selection["selected"] == {"weight": 0.5, "quantity_fraction": 0.5}
    optimistic = next(item for item in summaries if item["weight"] == 1.0)
    assert optimistic["top3_mean_diagnostic_only"] > 0
    assert optimistic["mean_normalized_improvement"] < 0
    assert optimistic["top3_is_primary_selection_statistic"] is False


def test_pooled_normalization_uses_ratio_of_means_near_zero_seed():
    mean, seed_values = pooled_normalized_improvement(
        "spearman",
        [0.057689, 0.175938, -0.011907],
        [0.102041, 0.194049, -0.045804],
    )
    assert mean > 0
    assert (seed_values > 0).mean() == 2 / 3


def test_one_se_selection_does_not_trade_away_best_mean_seed_stability():
    rows = []
    for seed, delta in zip((1, 2, 3, 4, 5), (0.30, 0.20, 0.10, 0.05, -0.10)):
        rows.append(
            {
                "seed": seed,
                "weight": 1.0,
                "quantity_fraction": 0.5,
                "normalized_improvement": delta,
                "external_to_native_mass_ratio": 0.01,
            }
        )
    for seed, delta in zip((1, 2, 3, 4, 5), (0.20, 0.10, 0.05, -0.01, -0.02)):
        rows.append(
            {
                "seed": seed,
                "weight": 0.25,
                "quantity_fraction": 1.0,
                "normalized_improvement": delta,
                "external_to_native_mass_ratio": 0.001,
            }
        )
    _, selection = summarize_grid(rows)
    assert selection["selected"] == {"weight": 1.0, "quantity_fraction": 0.5}
    assert selection["selection_positive_seed_fraction_floor"] == 0.8
