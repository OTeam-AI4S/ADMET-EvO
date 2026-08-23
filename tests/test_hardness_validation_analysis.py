import json

import pytest

from scripts.analyze_hardness_validation import analyze


def _cards():
    cards = []
    for task_index, task in enumerate(("a", "b", "c", "d")):
        for seed in (1, 2, 3):
            severity = 0.15 + task_index * 0.2 + seed * 0.001
            cards.append(
                {
                    "task": task,
                    "seed": seed,
                    "overall": severity,
                    "dimensions": {
                        "scarcity": {"severity": severity, "confidence": 1.0},
                        "domain_shift": {
                            "severity": severity * 0.8,
                            "confidence": 1.0,
                        },
                    },
                    "dominant_factors": ["scarcity", "domain_shift"],
                    "recommended_actions": ["replicate", "feature"],
                }
            )
    return {
        "discovery_only": True,
        "test_loaded": False,
        "cards": cards,
    }


def test_stability_and_permutation_validity_are_reported(tmp_path):
    hardness = tmp_path / "hardness.json"
    hardness.write_text(json.dumps(_cards()))
    outcomes = tmp_path / "outcomes.json"
    outcomes.write_text(
        json.dumps(
            {
                "discovery_only": True,
                "test_loaded": False,
                "rows": [
                    {"task": task, "baseline_error": 1.0 + index}
                    for index, task in enumerate(("a", "b", "c", "d"))
                ],
            }
        )
    )
    result = analyze(hardness, outcomes_path=outcomes, permutations=100, seed=9)
    assert result["tasks"] == 4
    assert result["stability"]["overall"]["icc_2_1_absolute_agreement"] > 0.99
    assert result["dominant_factor"]["fleiss_kappa"] == 1.0
    assert result["recommended_action_pairwise_jaccard"] == 1.0
    assert result["outcome_scope"] == "discovery_only"
    association = next(
        row
        for row in result["outcome_associations"]
        if row["dimension"] == "overall" and row["outcome"] == "baseline_error"
    )
    assert association["spearman"] == 1.0
    assert 0.0 < association["permutation_p_two_sided"] <= 1.0
    assert 0.0 < association["bh_q"] <= 1.0


def test_unfrozen_non_discovery_outcomes_are_rejected(tmp_path):
    hardness = tmp_path / "hardness.json"
    hardness.write_text(json.dumps(_cards()))
    outcomes = tmp_path / "outcomes.json"
    outcomes.write_text(
        json.dumps(
            {
                "discovery_only": False,
                "test_loaded": True,
                "rows": [{"task": task, "score": 1.0} for task in ("a", "b", "c", "d")],
            }
        )
    )
    with pytest.raises(RuntimeError, match="post-freeze"):
        analyze(hardness, outcomes_path=outcomes, permutations=100)


def test_incomplete_task_seed_grid_is_rejected(tmp_path):
    payload = _cards()
    payload["cards"].pop()
    hardness = tmp_path / "hardness.json"
    hardness.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="grid"):
        analyze(hardness)
