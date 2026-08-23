import math
import json
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.run_openadmet_external_propensity_bridge import (
    _aggregate_endpoint_metrics,
    _endpoint_metrics,
    _excluded_outer_folds,
    _source_fold,
)
from scripts.run_openadmet_external_propensity_bridge_v2 import (
    excluded_outer_folds as excluded_outer_folds_v2,
)
from scripts.freeze_openadmet_propensity_task_type_route import freeze


ROOT = Path(__file__).resolve().parents[1]


def test_source_holdout_fold_is_deterministic_and_group_atomic():
    assert _source_fold("group-a") == _source_fold("group-a")
    assert 0 <= _source_fold("group-a") < 5
    assert len({_source_fold(f"group-{index}") for index in range(30)}) > 1


def test_excluded_outer_fold_parser_fails_closed_on_recorded_values():
    assert _excluded_outer_folds("") == set()
    assert _excluded_outer_folds(np.nan) == set()
    assert _excluded_outer_folds("0;3") == {0, 3}


def test_v2_adapter_accepts_only_integral_csv_float_serialization():
    assert excluded_outer_folds_v2(np.nan) == set()
    assert excluded_outer_folds_v2(2.0) == {2}
    assert excluded_outer_folds_v2("0.0;3") == {0, 3}
    with np.testing.assert_raises_regex(ValueError, "Invalid integral"):
        excluded_outer_folds_v2("2.5")
    with np.testing.assert_raises_regex(ValueError, "outside"):
        excluded_outer_folds_v2("5.0")


def test_v2_protocol_preserves_v1_scientific_design():
    v1 = json.loads(
        (ROOT / "reports/openadmet_external_propensity_protocol_v1.json").read_text()
    )
    v2 = json.loads(
        (ROOT / "reports/openadmet_external_propensity_protocol_v2.json").read_text()
    )
    assert v2["aborted_predecessor_protocol_sha256"] == v1["sha256"]
    assert v2["implementation_source_sha256"] == v1["implementation_source_sha256"]
    assert v2["inputs"] == v1["inputs"]
    assert v2["arms"] == v1["arms"]
    assert v2["outer_folds"] == v1["outer_folds"]
    assert v2["seeds"] == v1["seeds"]
    assert v2["promotion_gate"] == v1["promotion_gate"]
    assert v2["scientific_design_changed_from_predecessor"] is False


def test_training_only_task_type_route_is_frozen_before_blind_evaluation():
    route = freeze(
        ROOT / "reports/openadmet_external_propensity_protocol_v2.json",
        ROOT
        / "runs/openadmet-cyp-external-propensity-v2/external_propensity_result.json",
    )
    assert route["route_frozen"] is True
    assert route["test_labels_loaded"] is False
    assert route["leaderboard_feedback_used"] is False
    assert route["route"] == {
        "primary_direct_regression": "b1_external_propensity",
        "primary_classification": "b1_control",
        "fallback_unseen_endpoint_type": "b1_control",
    }
    assert all(
        row["external_propensity_oriented_gain"] > 0
        for row in route["endpoint_development_effects"]
        if row["task_type"] == "regression"
    )
    assert all(
        row["external_propensity_oriented_gain"] <= 0
        for row in route["endpoint_development_effects"]
        if row["task_type"] == "classification"
    )


def test_classification_metrics_and_macro_aggregation_include_calibration():
    frame = pd.DataFrame(
        {
            "task_type": ["classification"] * 4,
            "target": [0, 0, 1, 1],
            "prediction": [0.1, 0.4, 0.6, 0.9],
        }
    )
    metrics = _endpoint_metrics(frame)
    assert math.isclose(metrics["mcc"], 1.0)
    assert math.isclose(metrics["brier"], 0.085)
    aggregate = _aggregate_endpoint_metrics(
        {
            "classification": metrics,
            "regression": {"st_rae": 0.7, "mae": 0.2},
        }
    )
    assert math.isclose(aggregate["macro_st_rae"], 0.7)
    assert math.isclose(aggregate["macro_mcc"], 1.0)
    assert math.isclose(aggregate["macro_brier"], 0.085)
