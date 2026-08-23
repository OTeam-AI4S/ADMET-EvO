from __future__ import annotations

import copy

import pytest

from scripts.run_minimal_exact_pair_pilot import build_specs, promotion_decision


def protocol():
    return {
        "sha256": "protocol",
        "screen_seeds": [1, 2],
        "promotion_gate": {"minimum_mean_oriented_gain": 0.005},
        "tasks": {
            "CYP2C9_Veith": {
                "representations": ["morgan", "avalon", "erg", "rdkit2d"],
                "control": {"model": "extra_trees", "imbalance": "none", "hypothesis": "control"},
                "treatment": {"model": "extra_trees", "imbalance": "class_weight", "hypothesis": "treatment"},
            },
            "LD50_Zhu": {
                "representations": ["morgan", "avalon", "erg", "rdkit2d"],
                "control": {"model": "lightgbm", "imbalance": "none", "hypothesis": "control"},
                "treatment": {"model": "extra_trees", "imbalance": "none", "hypothesis": "treatment"},
            },
        },
    }


def registry(scores):
    records = {}
    for task, values in scores.items():
        for seed, (control, treatment) in enumerate(values, start=1):
            for role, score in (("control", control), ("treatment", treatment)):
                records[f"{task}|{seed}|{role}"] = {
                    "task": task, "seed": seed, "role": role,
                    "result": {"status": "completed", "validation_score": score},
                }
    return {"records": records}


def test_specs_change_only_the_preregistered_axis():
    p = protocol()
    metabolism = build_specs(p, "CYP2C9_Veith", 1, 4)
    assert metabolism["control"].model == metabolism["treatment"].model
    assert metabolism["control"].representations == metabolism["treatment"].representations
    assert metabolism["control"].imbalance != metabolism["treatment"].imbalance
    toxicity = build_specs(p, "LD50_Zhu", 1, 4)
    assert toxicity["control"].model != toxicity["treatment"].model
    assert toxicity["control"].representations == toxicity["treatment"].representations
    assert toxicity["control"].imbalance == toxicity["treatment"].imbalance


def test_promotion_requires_both_positive_and_minimum_mean_gain():
    decision = promotion_decision(
        registry({
            "CYP2C9_Veith": [(0.70, 0.71), (0.72, 0.73)],
            # MAE: lower treatment is positive; one seed loses.
            "LD50_Zhu": [(0.60, 0.59), (0.60, 0.601)],
        }),
        protocol(),
    )
    assert decision["tasks"]["CYP2C9_Veith"]["promoted"] is True
    assert decision["tasks"]["LD50_Zhu"]["promoted"] is False


def test_promotion_hash_changes_if_screen_result_is_tampered():
    p = protocol()
    first = registry({
        "CYP2C9_Veith": [(0.70, 0.71), (0.72, 0.73)],
        "LD50_Zhu": [(0.60, 0.59), (0.61, 0.60)],
    })
    second = copy.deepcopy(first)
    second["records"]["LD50_Zhu|2|treatment"]["result"]["validation_score"] = 0.62
    assert promotion_decision(first, p)["sha256"] != promotion_decision(second, p)["sha256"]


def test_incomplete_screen_is_rejected():
    with pytest.raises(RuntimeError, match="Screen is incomplete"):
        promotion_decision({"records": {}}, protocol())
