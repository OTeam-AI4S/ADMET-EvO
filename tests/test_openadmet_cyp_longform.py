import pandas as pd

from scripts.build_openadmet_cyp_longform import (
    _single_observations,
    _wide_observations,
)


def test_wide_observations_preserve_assay_semantics_and_intervals():
    target = "CYP2D6_pIC50_direct_inhibition"
    frame = pd.DataFrame(
        {
            "SMILES": ["CCO"],
            target: [5.5],
            f"{target}_conf_low": [5.2],
            f"{target}_conf_high": [5.8],
            f"{target}_std": [0.15],
        }
    )
    rows = _wide_observations(
        frame,
        source_table="direct",
        targets=(target,),
        assay_arm="direct_inhibition",
        readout="pIC50",
        fidelity="primary_direct",
        task_type="regression",
    )

    assert len(rows) == 1
    row = rows[0]
    assert row["isoform"] == "CYP2D6"
    assert row["endpoint_key"] == (
        "CYP2D6|direct_inhibition|pIC50|primary_direct"
    )
    assert row["credible_low"] == 5.2
    assert row["credible_high"] == 5.8
    assert row["measurement_std"] == 0.15


def test_single_concentration_observation_keeps_condition_and_uncertainty():
    frame = pd.DataFrame(
        {
            "SMILES": ["CCN"],
            "enzyme": ["CYP3A4"],
            "log2fc_estimate": [-1.4],
            "log2fc_std_error": [0.2],
            "concentration_M": [4.95e-5],
        }
    )
    rows = _single_observations(frame)

    assert len(rows) == 1
    row = rows[0]
    assert row["assay_arm"] == "single_concentration_active_preincubation"
    assert row["fidelity"] == "low_single_concentration"
    assert row["target"] == -1.4
    assert row["measurement_std"] == 0.2
    assert row["concentration_M"] == 4.95e-5
