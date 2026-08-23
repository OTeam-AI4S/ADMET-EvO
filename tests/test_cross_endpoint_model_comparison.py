from pathlib import Path

from scripts.build_cross_endpoint_model_comparison import build


ROOT = Path(__file__).resolve().parents[1]


def test_frozen_cross_endpoint_comparison_has_expected_claim_boundary():
    payload = build(
        ROOT / "reports/goal_scorecard_v27.json",
        ROOT / "reports/leaderboard_snapshot_2026-08-17.json",
    )
    assert payload["endpoint_count"] == 22
    assert payload["agent"]["estimated_top1_count"] == 5
    assert payload["agent"]["estimated_top1_tasks"] == [
        "AMES",
        "Half_Life_Obach",
        "Lipophilicity_AstraZeneca",
        "PPBR_AZ",
        "Solubility_AqSolDB",
    ]
    assert payload["complete_coverage_fixed_model_count"] == 14
    assert payload["fixed_models_surpassed_by_agent_mean_rank"] == 13
    assert payload["fixed_models_not_surpassed_by_agent_mean_rank"] == [
        "MapLight + GNN"
    ]
    assert payload["agent_position_among_complete_fixed_models_plus_agent"] == 2
    assert payload["test_loaded_during_analysis"] is False
