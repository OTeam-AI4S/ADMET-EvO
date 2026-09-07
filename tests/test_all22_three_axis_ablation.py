from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from scripts import run_all22_three_axis_ablation as ablation


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "reports/all22_three_axis_ablation_v1/protocol.json"


def test_metric_directions_are_oriented_consistently() -> None:
    assert ablation._normalized_improvement("auroc", 0.8, 0.84) == pytest.approx(0.05)
    assert ablation._normalized_improvement("mae", 0.8, 0.72) == pytest.approx(0.10)
    assert ablation._raw_oriented_delta("spearman", 0.4, 0.5) == pytest.approx(0.1)
    assert ablation._raw_oriented_delta("mae", 0.4, 0.3) == pytest.approx(0.1)


def test_factorial_changes_only_requested_axes() -> None:
    selected = {
        "model": {"M3": {"config": ablation._model_single("extra_trees")}},
        "feature": {"F3": {"config": ["morgan", "rdkit2d"]}},
        "data": {
            "D3": {
                "config": {"enabled": True, "method": "coreset", "ratio": 2.0}
            }
        },
    }
    cells = ablation._factorial_configs(selected)
    assert cells["000"] == {
        "model": ablation.BASELINE_MODEL,
        "feature": ablation.BASELINE_FEATURE,
        "data": ablation.NO_DATA,
    }
    assert cells["100"]["model"]["model"] == "extra_trees"
    assert cells["100"]["feature"] == ablation.BASELINE_FEATURE
    assert cells["010"]["feature"] == ["morgan", "rdkit2d"]
    assert cells["010"]["model"] == ablation.BASELINE_MODEL
    assert cells["001"]["data"]["ratio"] == 2.0
    assert cells["111"]["model"]["model"] == "extra_trees"


def test_protocol_is_hash_bound_and_test_blind() -> None:
    protocol = ablation._load_protocol(PROTOCOL)
    assert protocol["test_labels_loaded"] is False
    assert protocol["selection"]["test_seed_selection_forbidden"] is True
    assert protocol["data_axis"]["safety_filter_always_on"] is True
    assert protocol["factorial"]["cells"] == [
        "000", "100", "010", "001", "110", "101", "011", "111"
    ]


def test_structure_only_fallback_preserves_every_solubility_test_row() -> None:
    path = ROOT / "data/admet_group/solubility_aqsoldb/test.csv"
    raw = pd.read_csv(path, usecols=["Drug"])
    structures = ablation._standardize_structure_frame(raw)
    assert len(structures) == len(raw)
    assert structures["standard_inchikey"].notna().all()
    assert structures.attrs["standardization_fallback_count"] >= 1
    assert structures.attrs["unparseable_structure_count"] == 2
    assert int((~structures["fingerprintable"]).sum()) == 2


@pytest.mark.parametrize(
    ("task", "expected_rejected"),
    [("HIA_Hou", True), ("Half_Life_Obach", False)],
)
def test_same_source_admission_uses_frozen_raw_overlap(
    task: str, expected_rejected: bool
) -> None:
    protocol = ablation._load_protocol(PROTOCOL)
    benchmark = ablation.TASKS[task][0]
    raw = pd.read_csv(
        ROOT / "data/admet_group" / benchmark / "test.csv", usecols=["Drug"]
    )
    structures = ablation._standardize_structure_frame(raw)
    _, audit = ablation._load_external(
        task=task,
        path=ROOT / "data/ADMET_v3_clean_external_candidate_pool.csv",
        protocol=protocol,
        test_structures=structures,
    )
    assert audit["source_rejected_same_source"] is expected_rejected
    assert audit["admitted"] is False
    assert audit["admission_status"] == "fail_closed"
    assert audit["admission_error"] == "protocol_missing_source_admission_binding"
    assert audit["test_structure_columns_loaded"] == ["Drug"]
    assert audit["test_labels_loaded"] is False
