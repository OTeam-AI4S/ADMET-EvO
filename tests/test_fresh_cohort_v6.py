from __future__ import annotations

import pandas as pd

from admet_agent.fresh_cohort_v6 import cap_endpoint, endpoint_admission, standardize_structure_column


def test_standardization_is_label_independent_and_reuses_cache():
    frame = pd.DataFrame({"smiles": ["CC", "CC", "bad"], "Y": [0, 1, 1]})
    cache = {}
    output, audit = standardize_structure_column(frame, smiles_column="smiles", cache=cache)
    assert len(output) == 2
    assert audit["invalid_rows"] == 1
    assert set(output["Y"]) == {0, 1}
    assert len(cache) == 2


def test_classification_cap_keeps_every_minority_identity():
    frame = pd.DataFrame({
        "inchikey": [f"k{i}" for i in range(30050)],
        "target": [1.0] * 25 + [0.0] * 30025,
    })
    capped = cap_endpoint(frame, task_type="classification")
    assert int(capped["target"].sum()) == 25
    assert len(capped) == 30025


def test_admission_fails_closed_for_small_minority():
    frame = pd.DataFrame({"inchikey": [f"k{i}" for i in range(500)], "target": [1] * 39 + [0] * 461})
    passed, reasons = endpoint_admission(frame, task_type="classification", valid_smiles_fraction=1.0)
    assert passed is False
    assert "classification_minority_below_40" in reasons
