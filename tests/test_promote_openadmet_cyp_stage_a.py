import json

import pytest

from scripts.promote_openadmet_cyp_stage_a import promote


def _result(kind, st_rae, mcc):
    rows = []
    for fold in (0, 1, 2):
        metrics = {"macro_st_rae": st_rae, "macro_mcc": mcc}
        row = {"outer_fold": fold, "seed": 1}
        if kind == "tree":
            row.update(metrics)
        else:
            row["metrics"] = metrics
        rows.append(row)
    return {
        "training_only": True,
        "blind_test_opened": False,
        "sealed_certification_opened": False,
        "outer_folds": [0, 1, 2],
        "seeds": [1],
        "fold_seed_results": rows,
    }


def test_primary_gain_promotes_noninferior_candidate(tmp_path):
    baseline = tmp_path / "b1.json"
    candidate = tmp_path / "m1.json"
    baseline.write_text(json.dumps(_result("tree", 0.8, 0.4)))
    candidate.write_text(json.dumps(_result("neural", 0.75, 0.39)))
    result = promote(baseline, {"M1": candidate})
    decision = result["decisions"]["M1"]
    assert decision["promote_to_stage_b"] is True
    assert decision["mean_st_rae_gain_vs_b1"] > 0
    assert decision["blind_leaderboard_can_override"] is False


def test_material_primary_regression_stops_candidate_even_with_uq(tmp_path):
    baseline = tmp_path / "b1.json"
    candidate = tmp_path / "m1.json"
    baseline.write_text(json.dumps(_result("tree", 0.8, 0.4)))
    candidate.write_text(json.dumps(_result("neural", 0.9, 0.5)))
    b1_analysis = tmp_path / "b1-analysis.json"
    m1_analysis = tmp_path / "m1-analysis.json"
    b1_analysis.write_text(json.dumps({"endpoints": {"c": {"brier": 0.3, "ece_10": 0.2}}}))
    m1_analysis.write_text(json.dumps({"endpoints": {"c": {"brier": 0.1, "ece_10": 0.05}}}))
    result = promote(
        baseline,
        {"M1": candidate},
        analysis_paths={"B1": b1_analysis, "M1": m1_analysis},
    )
    decision = result["decisions"]["M1"]
    assert decision["uq_gate"]["passed"] is True
    assert decision["regression_noninferior"] is False
    assert decision["promote_to_stage_b"] is False


def test_wrong_stage_a_fold_contract_is_rejected(tmp_path):
    baseline = tmp_path / "b1.json"
    candidate = tmp_path / "m1.json"
    payload = _result("tree", 0.8, 0.4)
    payload["outer_folds"] = [0, 1]
    baseline.write_text(json.dumps(payload))
    candidate.write_text(json.dumps(_result("neural", 0.7, 0.5)))
    with pytest.raises(RuntimeError, match="folds"):
        promote(baseline, {"M1": candidate})


def test_multi_model_cv_analysis_is_resolved_by_exact_label(tmp_path):
    baseline = tmp_path / "b1.json"
    candidate = tmp_path / "m1.json"
    baseline.write_text(json.dumps(_result("tree", 0.8, 0.4)))
    candidate.write_text(json.dumps(_result("neural", 0.8, 0.4)))
    analysis = tmp_path / "analysis.json"
    analysis.write_text(
        json.dumps(
            {
                "models": [
                    {"label": "B1", "endpoints": {"c": {"brier": 0.3, "ece_10": 0.2}}},
                    {"label": "M1", "endpoints": {"c": {"brier": 0.1, "ece_10": 0.1}}},
                ]
            }
        )
    )
    result = promote(
        baseline,
        {"M1": candidate},
        analysis_paths={"B1": analysis, "M1": analysis},
    )
    assert result["decisions"]["M1"]["uq_gate"]["passed"] is True
    assert result["decisions"]["M1"]["promote_to_stage_b"] is True


def test_multi_model_analysis_rejects_missing_or_duplicate_labels(tmp_path):
    baseline = tmp_path / "b1.json"
    candidate = tmp_path / "m1.json"
    baseline.write_text(json.dumps(_result("tree", 0.8, 0.4)))
    candidate.write_text(json.dumps(_result("neural", 0.8, 0.4)))
    analysis = tmp_path / "analysis.json"
    analysis.write_text(json.dumps({"models": [{"label": "B1", "endpoints": {}}]}))
    with pytest.raises(RuntimeError, match="exactly one model labeled M1"):
        promote(
            baseline,
            {"M1": candidate},
            analysis_paths={"B1": analysis, "M1": analysis},
        )
