import pytest

from admet_agent.lean_task_policy import data_regime, route_task, task_family


def test_two_data_regimes_remove_non_actionable_medium_branch():
    assert data_regime(999) == "low_data"
    assert data_regime(1000) == "higher_data"
    assert data_regime(9191) == "higher_data"


@pytest.mark.parametrize(
    ("task", "expected"),
    [
        ("hERG", "binary_auroc"),
        ("CYP2C9_Veith", "imbalanced_binary_auprc"),
        ("LD50_Zhu", "continuous_mae"),
        ("Half_Life_Obach", "rank_pk_spearman"),
    ],
)
def test_four_modeling_families_are_metric_aligned(task, expected):
    assert task_family(task) == expected


def test_qualified_task_reuses_frozen_recipe_without_research():
    route = route_task("hERG", 457)
    assert route.production_action == "reuse_qualified_frozen_recipe_and_stop"
    assert route.research_action == "none"
    assert route.maximum_research_fits == 0
    assert route.reviewer_mode == "single_agent"


def test_low_and_higher_data_change_repetition_not_candidate_width():
    low = route_task("hERG", 457, {"uncertainty_high": True})
    high = route_task("AMES", 5093, {"uncertainty_high": True})
    assert low.maximum_research_configurations == high.maximum_research_configurations == 2
    assert low.confirmation_seeds == (1, 2, 3, 4, 5)
    assert high.confirmation_seeds == (1, 2, 3)
    assert (low.maximum_research_fits, high.maximum_research_fits) == (10, 6)


def test_contract_and_data_problems_do_not_launch_model_search():
    contract = route_task("Caco2_Wang", 637, {"contract_ok": False})
    assert contract.production_action == "quarantine"
    assert contract.maximum_research_fits == 0
    drift = route_task("Caco2_Wang", 637, {"data_drift_or_gap": True})
    assert drift.research_action == "open_data_stewardship_ticket"
    assert drift.reviewer_mode == "targeted_data_review"
    assert drift.maximum_research_fits == 0


def test_router_rejects_test_feedback_and_unbounded_signals():
    with pytest.raises(ValueError, match="Test-like"):
        route_task("hERG", 457, {"test_score": 0.9})
    with pytest.raises(ValueError, match="Unknown"):
        route_task("hERG", 457, {"hardness_score": 0.9})
    with pytest.raises(ValueError, match="boolean"):
        route_task("hERG", 457, {"qualification_ok": "yes"})
