from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from admet_agent.data_axis_v2 import native_training_plan
from scripts import run_all22_three_axis_ablation_v2 as v2


ROOT = Path(__file__).resolve().parents[1]


def _frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "standard_smiles": ["CC", "CC", "CCC", "CCC", "c1ccccc1"],
            "standard_inchikey": ["a", "a", "b", "b", "c"],
            "Y": [0, 0, 0, 1, 1],
        }
    )


def test_classification_curation_collapses_replicates_and_removes_conflicts() -> None:
    positions, targets, weights, audit = native_training_plan(
        _frame(), task_type="classification",
        native_policy="curated", weight_policy="none",
    )
    assert positions.tolist() == [0, 4]
    assert targets.tolist() == [0, 1]
    assert weights.tolist() == pytest.approx([1.0, 1.0])
    assert audit["conflicting_identities_removed"] == 1
    assert audit["heldout_labels_used"] is False


def test_regression_curation_uses_train_only_replicate_median() -> None:
    frame = _frame().assign(Y=[1.0, 3.0, 2.0, 8.0, 5.0])
    positions, targets, _, audit = native_training_plan(
        frame, task_type="regression",
        native_policy="curated", weight_policy="none",
    )
    assert positions.tolist() == [0, 2, 4]
    assert targets.tolist() == pytest.approx([2.0, 5.0, 5.0])
    assert audit["selected_rows"] == 3


@pytest.mark.parametrize("policy", ["scaffold_balance", "target_balance", "combined"])
def test_valuation_weights_are_positive_normalized_and_label_local(policy: str) -> None:
    _, _, weights, audit = native_training_plan(
        _frame().drop_duplicates("standard_inchikey"),
        task_type="classification", native_policy="official", weight_policy=policy,
    )
    assert np.isfinite(weights).all()
    assert (weights > 0).all()
    assert float(weights.mean()) == pytest.approx(1.0)
    assert audit["heldout_labels_used"] is False


def test_variance_penalty_prefers_stable_arm() -> None:
    configs = {"stable": {"x": 1}, "unstable": {"x": 2}}
    rows = []
    for seed, value in enumerate([0.04, 0.04, 0.04, 0.04]):
        rows.append({"arm": "stable", "seed": seed, "score": value, "normalized_improvement": value})
    for seed, value in enumerate([-0.20, 0.30, -0.10, 0.30]):
        rows.append({"arm": "unstable", "seed": seed, "score": value, "normalized_improvement": value})
    arm, _, summary = v2._best_arm_v2(rows, configs, ["stable", "unstable"])
    assert arm == "stable"
    assert summary["selection_rule"] == "mean_minus_0.5_standard_error"


def test_confirmation_route_requires_positive_seed_support() -> None:
    summaries = {}
    for cell in ("000", "100", "010", "001", "110", "101", "011", "111"):
        values = [0.0, 0.0, 0.0, 0.0]
        if cell == "110":
            values = [0.06, 0.05, 0.04, 0.05]
        if cell == "111":
            values = [0.20, -0.10, -0.10, 0.20]
        summaries[f"factorial_{cell}"] = {
            "mean_score": float(np.mean(values)),
            "sample_score_sd": float(np.std(values, ddof=1)),
            "mean_normalized_improvement": float(np.mean(values)),
            "sample_normalized_improvement_sd": float(np.std(values, ddof=1)),
            "positive_seed_fraction": float(np.mean(np.asarray(values) > 0)),
            "seeds": [1, 2, 3, 4],
        }
    payload = {"factorial": {"summaries": summaries}}
    protocol = {
        "selection": {"confirmation_practical_threshold": 0.005, "positive_seed_fraction": 0.75},
        "factorial": {"cells": ["000", "100", "010", "001", "110", "101", "011", "111"]},
    }
    routed = v2._confirmation_route(payload, protocol)
    assert routed["selected_cell"] == "110"
    assert routed["cell_audit"]["111"]["passed"] is False
