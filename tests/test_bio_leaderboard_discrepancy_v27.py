import importlib.util
from pathlib import Path


SCRIPT = Path(__file__).parents[1] / "scripts/audit_bio_leaderboard_v27.py"


def _load_script():
    spec = importlib.util.spec_from_file_location("bio_leaderboard_audit_v27", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_bio_discrepancy_is_confirmed_without_overriding_official_threshold():
    module = _load_script()
    root = Path(__file__).parents[1]
    report = module.build_audit(
        root, root / "reports/leaderboard_snapshot_2026-08-17.json"
    )

    assert report["assessment"]["status"] == (
        "confirmed_material_leaderboard_discrepancy"
    )
    assert report["cross_page_evidence"]["pgp_containment"] == 1.0
    assert report["cross_page_evidence"]["exact_shared_model_mean_std"] == 16
    assert report["cross_page_evidence"]["jaccard"] == 0.8
    assert report["checks"]["current_snapshot_matches_audit_repository_bio"]
    assert report["checks"]["current_snapshot_matches_audit_repository_pgp"]
    assert report["checks"]["local_bio_exactly_matches_published_reproduction"]
    assert report["checks"][
        "current_bio_minimol_exactly_equals_historical_pgp_minimol"
    ]
    assert report["checks"]["historical_bio_and_pgp_scores_are_distinct"]
    assert report["cross_page_evidence"]["historical_bio_minimol"]["mean"] == 0.689
    assert report["cross_page_evidence"]["historical_pgp_minimol"]["mean"] == 0.942
    assert report["local_reproduction"]["bio"]["mean"] == 0.731
    assert report["local_reproduction"]["bio"]["std"] == 0.007
    assert report["local_reproduction"]["cyp_positive_control"]["absolute_delta"] <= 0.01

    assert report["assessment"]["root_cause_proven"] is False
    assert report["assessment"]["authoritative_correction_obtained"] is False
    assert report["goal_effect"]["threshold_override_allowed"] is False
    assert report["goal_effect"]["displayed_top10_threshold"] == 0.922
    assert report["goal_effect"]["historical_june_2024_rank_upper_bound"] == 5
    assert report["goal_effect"]["historical_reference_is_not_current_rank_evidence"]
    assert report["goal_effect"]["verified_top10_count"] == 21
    assert report["goal_effect"]["goal_met"] is False
    assert report["tdc_public_issue"]["number"] == 394
    assert report["tdc_public_issue"]["state_observed_at_audit"] == "open"
    assert report["tdc_public_issue"]["maintainer_comments_observed_at_audit"] == 0


def test_bio_selection_evidence_remains_test_blind_and_identity_safe():
    module = _load_script()
    root = Path(__file__).parents[1]
    report = module.build_audit(
        root, root / "reports/leaderboard_snapshot_2026-08-17.json"
    )
    local = report["local_reproduction"]
    assert local["official_test_loaded_during_selection"] is False
    assert local["test_labels_used_during_selection"] is False
    assert local["cross_split_identity_overlap"] == 0
    assert report["checks"]["confirmation_never_loaded_official_test"]
    assert report["checks"]["confirmation_has_zero_identity_overlap"]
