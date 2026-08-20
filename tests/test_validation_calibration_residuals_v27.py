import importlib.util
import json
from pathlib import Path

import numpy as np


SCRIPT = Path(__file__).parents[1] / "scripts/analyze_validation_calibration_residuals_v27.py"


def _load_script():
    spec = importlib.util.spec_from_file_location("validation_audit_v27", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_calibration_and_rank_residual_statistics_are_oriented_correctly():
    module = _load_script()
    classification = module.classification_summary(
        np.asarray([0.0, 1.0]), np.asarray([0.0, 1.0])
    )
    assert classification["brier"] == 0.0
    assert classification["adaptive_ece_10"] == 0.0
    residual, error, unit = module.regression_error(
        np.asarray([10.0, 20.0, 30.0]),
        np.asarray([1.0, 2.0, 3.0]),
        "spearman",
    )
    assert unit == "percentile_rank"
    assert np.allclose(residual, 0.0)
    assert np.allclose(error, 0.0)


def test_persisted_validation_audit_has_complete_firewalled_coverage():
    root = Path(__file__).parents[1]
    report = json.loads(
        (root / "reports/validation_calibration_residuals_v27.json").read_text(
            encoding="utf-8"
        )
    )
    assert report["task_count"] == 22
    assert report["classification_task_count"] + report["regression_task_count"] == 22
    assert report["official_test_loaded"] is False
    assert report["official_results_used"] is False
    assert report["leaderboard_used_for_analysis"] is False
    assert report["selection_or_freeze_changed"] is False
    assert all(row["identity_overlap_count"] == 0 for row in report["tasks"])
    assert all(row["train_valid_scaffold_overlap_count"] == 0 for row in report["tasks"])
