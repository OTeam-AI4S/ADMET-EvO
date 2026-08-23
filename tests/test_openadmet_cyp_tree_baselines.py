import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.openadmet_cyp_metrics import soft_threshold_distance
from scripts.run_openadmet_cyp_tree_baselines import _endpoint_metric


def test_soft_threshold_distance_has_zero_error_inside_interval():
    distance = soft_threshold_distance(
        np.asarray([5.5, 7.0, 4.0]),
        np.asarray([5.4, 6.0, 5.0]),
        np.asarray([5.2, 5.5, np.nan]),
        np.asarray([5.8, 6.5, np.nan]),
    )
    np.testing.assert_allclose(distance, [0.0, 0.5, 1.0])


def test_endpoint_metrics_use_evaluated_fold_mean_baseline_and_mcc():
    regression_validation = pd.DataFrame(
        {
            "target": [4.0, 6.0],
            "prediction": [4.0, 6.0],
            "credible_low": [3.9, 5.9],
            "credible_high": [4.1, 6.1],
        }
    )
    regression = _endpoint_metric(
        task_type="regression",
        validation=regression_validation,
    )
    assert regression["st_rae"] == 0.0
    assert regression["mae"] == 0.0

    classification = _endpoint_metric(
        task_type="classification",
        validation=pd.DataFrame(
            {"target": [0, 1, 1, 0], "prediction": [0.1, 0.9, 0.8, 0.2]}
        ),
    )
    assert classification["mcc"] == 1.0

    mean_predictor = regression_validation.copy()
    mean_predictor["prediction"] = mean_predictor["target"].mean()
    baseline = _endpoint_metric(
        task_type="regression", validation=mean_predictor
    )
    assert baseline["st_rae"] == 1.0


def test_st_rae_matches_frozen_official_challenge_implementation():
    source = (
        Path(__file__).resolve().parents[1]
        / "external/openadmet/cyp_challenge_tutorial/evaluation/custom_scoring_functions.py"
    )
    spec = importlib.util.spec_from_file_location("openadmet_official_scoring", source)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    frame = pd.DataFrame(
        {
            "target": [4.0, 5.0, 6.0],
            "prediction": [3.8, 5.1, 6.7],
            "credible_low": [3.9, 4.8, 5.8],
            "credible_high": [4.2, 5.2, 6.2],
        }
    )
    local = _endpoint_metric(task_type="regression", validation=frame)["st_rae"]
    official = module.rae_soft_threshold_absolute_error(
        frame["target"].to_numpy(),
        frame["prediction"].to_numpy(),
        y_true_upper=frame["credible_high"].to_numpy(),
        y_true_lower=frame["credible_low"].to_numpy(),
    )
    assert local == official
