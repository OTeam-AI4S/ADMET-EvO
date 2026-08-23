import json

import pytest

from scripts.analyze_expert_blind_scores import RUBRIC_MAXIMA, analyze


def _components(score_fraction):
    return {name: maximum * score_fraction for name, maximum in RUBRIC_MAXIMA.items()}


def _payload():
    rows = []
    for case_index in range(4):
        for system, fraction in (("sys-x", 0.9), ("sys-y", 0.5)):
            for scorer, offset in (("r1", 0.0), ("r2", -0.01)):
                rows.append(
                    {
                        "case_id": f"case-{case_index}",
                        "system_code": system,
                        "scorer_id": scorer,
                        "components": _components(fraction + offset),
                        "unsafe_action": False,
                        "catastrophic_violation": False,
                        "elapsed_seconds": 10,
                        "cost_usd": 0.1,
                    }
                )
    return {
        "scoring_completed_before_system_reveal": True,
        "independent_scorers_attested": True,
        "rows": rows,
    }


def test_blind_score_analysis_is_case_paired(tmp_path):
    path = tmp_path / "scores.json"
    path.write_text(json.dumps(_payload()))
    result = analyze(path, primary="sys-x", bootstrap_samples=100, seed=3)
    assert result["systems"]["sys-x"]["mean_score"] > result["systems"]["sys-y"]["mean_score"]
    assert result["contrasts"]["sys-y"]["wins"] == 4
    assert result["contrasts"]["sys-y"]["losses"] == 0
    assert result["inter_rater"]["scorers"] == 2
    assert result["claim_gate"]["zero_primary_catastrophic_violations"] is True


def test_unblinded_scoring_is_rejected(tmp_path):
    payload = _payload()
    payload["scoring_completed_before_system_reveal"] = False
    path = tmp_path / "scores.json"
    path.write_text(json.dumps(payload))
    with pytest.raises(RuntimeError, match="blinded"):
        analyze(path, primary="sys-x", bootstrap_samples=100)


def test_incomplete_rater_grid_is_rejected(tmp_path):
    payload = _payload()
    payload["rows"].pop()
    path = tmp_path / "scores.json"
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="same scorer set"):
        analyze(path, primary="sys-x", bootstrap_samples=100)
