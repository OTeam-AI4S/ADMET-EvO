import importlib.util
from pathlib import Path

import numpy as np
import pytest


SCRIPT = Path(__file__).parents[1] / "scripts/evaluate_maplight_gnn_confirmation.py"


def _load_script():
    spec = importlib.util.spec_from_file_location("maplight_gnn_confirmation", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_score_dispatches_binary_metrics():
    module = _load_script()
    labels = np.asarray([0, 0, 1, 1])
    predictions = np.asarray([0.1, 0.4, 0.35, 0.8])
    assert module._score(labels, predictions, "auroc") == pytest.approx(0.75)
    assert module._score(labels, predictions, "auprc") == pytest.approx(
        0.8333333333333333
    )


def test_preregistration_covers_both_remaining_tasks_and_fixed_gate():
    module = _load_script()
    root = Path(__file__).parents[1]
    upstream = (
        root
        / "vendor/tdc-admet-bench-audit/models/maplight_gnn/upstream/maplight_gnn.py"
    )
    payload = module.preregistration(root, upstream)
    assert set(payload["tasks"]) == {
        "CYP2C9_Substrate_CarbonMangels",
        "Bioavailability_Ma",
    }
    assert payload["fresh_confirmation_seeds"] == [6, 7, 8, 9, 10]
    assert payload["candidate"]["classifier"]["parameters"]["random_strength"] == 2
    assert payload["promotion_gate"]["all_tasks_must_pass"] is True
    assert payload["split_protocol"]["official_test_loaded"] is False
    assert payload["split_protocol"]["test_labels_used"] is False


def test_canonical_identity_collapses_equivalent_smiles():
    module = _load_script()
    assert module._canonical_identity("CCO") == module._canonical_identity("OCC")
