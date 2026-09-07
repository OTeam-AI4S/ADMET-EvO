import numpy as np

from admet_agent.evidence_packet_v54 import (
    cap_external_weights,
    independent_gate,
    summarize_source_grid,
)


def test_external_mass_cap_keeps_native_strictly_higher_weight():
    values, audit = cap_external_weights(
        np.asarray([0.8, 0.8, 0.8]), native_rows=10, maximum_mass_ratio=0.1
    )
    assert values.sum() == 1.0
    assert values.max() < 1.0
    assert audit["mass_cap_applied"] is True
    assert audit["observed_external_to_native_mass_ratio"] == 0.1


def test_source_grid_uses_all_seeds_and_prefers_low_mass_within_one_se():
    rows = []
    for source, weight, mass, values in (
        ("larger", 0.5, 0.08, [0.020, 0.018, 0.016, 0.014, 0.012]),
        ("smaller", 0.25, 0.02, [0.019, 0.017, 0.015, 0.013, 0.011]),
    ):
        for seed, value in enumerate(values):
            rows.append(
                {
                    "source_set_id": source,
                    "weight": weight,
                    "quantity_fraction": 0.5,
                    "normalized_improvement": value,
                    "external_to_native_mass_ratio": mass,
                    "seed": seed,
                }
            )
    summaries, selection = summarize_source_grid(rows)
    assert len(summaries) == 2
    assert selection["selected"]["source_set_id"] == "smaller"
    assert all(item["top3_is_primary_selection_statistic"] is False for item in summaries)


def test_independent_gate_enforces_effect_and_seed_support():
    passed = independent_gate(
        [0.010, 0.009, 0.008, -0.001],
        minimum_mean=0.005,
        minimum_positive_fraction=0.75,
    )
    failed = independent_gate(
        [0.010, 0.009, -0.001, -0.001],
        minimum_mean=0.0,
        minimum_positive_fraction=0.75,
    )
    assert passed["passed"] is True
    assert failed["passed"] is False
