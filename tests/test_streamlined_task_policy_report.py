from pathlib import Path

from scripts.build_streamlined_task_policy_report import build


ROOT = Path(__file__).resolve().parents[1]


def test_shadow_replay_quantifies_category_size_simplification():
    payload = build(
        ROOT / "reports/v3_tdc22_hardness_summary.json",
        ROOT / "reports/v4_agent_v5_redundancy_replay.json",
    )
    assert payload["task_count"] == 22
    assert payload["size_counts"] == {"large": 6, "medium": 3, "small": 13}
    assert payload["category_size_matrix"]["metabolism"] == {
        "small": 3,
        "medium": 0,
        "large": 3,
    }
    assert payload["old_v5"]["recorded_fits"] == 789
    assert payload["minimal_policy"]["screen_configurations"] == 60
    assert payload["minimal_policy"]["typical_fit_ceiling"] == 252
    assert (
        payload["minimal_policy"]["historical_two_correction_adjusted_fit_ceiling"]
        == 264
    )
    assert payload["activation_status"].startswith("shadow_only")
    assert payload["training_started"] is False
    assert payload["test_loaded"] is False
