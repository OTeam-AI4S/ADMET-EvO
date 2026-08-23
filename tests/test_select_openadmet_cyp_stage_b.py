import json

import pytest

from scripts.select_openadmet_cyp_stage_b import select


def _result(st_rae, mcc):
    return {
        "training_only": True,
        "blind_test_opened": False,
        "sealed_certification_opened": False,
        "outer_folds": [0, 1, 2, 3, 4],
        "seeds": [1, 2, 3],
        "fold_seed_results": [
            {
                "outer_fold": fold,
                "seed": seed,
                "metrics": {"macro_st_rae": st_rae, "macro_mcc": mcc},
            }
            for fold in range(5)
            for seed in (1, 2, 3)
        ],
    }


def test_noninferior_improving_candidate_is_admitted(tmp_path):
    baseline = tmp_path / "b1.json"
    candidate = tmp_path / "m3.json"
    baseline.write_text(json.dumps(_result(0.8, 0.3)))
    candidate.write_text(json.dumps(_result(0.78, 0.29)))
    result = select(baseline, {"M3": candidate})
    assert result["selected_components"] == ["B1", "M3"]
    assert result["recipe_mode"] == "nested_oof_stack"
    assert result["decisions"]["M3"]["admit_as_stack_component"] is True


def test_material_or_worst_fold_regression_forces_single_b1(tmp_path):
    baseline = tmp_path / "b1.json"
    candidate = tmp_path / "m3.json"
    baseline.write_text(json.dumps(_result(0.8, 0.3)))
    payload = _result(0.78, 0.31)
    for row in payload["fold_seed_results"]:
        if row["outer_fold"] == 4:
            row["metrics"]["macro_mcc"] = 0.15
    candidate.write_text(json.dumps(payload))
    result = select(baseline, {"M3": candidate})
    assert result["selected_components"] == ["B1"]
    assert result["recipe_mode"] == "single_component_B1"
    assert result["decisions"]["M3"]["worst_fold_safe"] is False


def test_incomplete_grid_is_rejected(tmp_path):
    baseline = tmp_path / "b1.json"
    candidate = tmp_path / "m3.json"
    baseline.write_text(json.dumps(_result(0.8, 0.3)))
    payload = _result(0.78, 0.31)
    payload["fold_seed_results"].pop()
    candidate.write_text(json.dumps(payload))
    with pytest.raises(RuntimeError, match="incomplete"):
        select(baseline, {"M3": candidate})
