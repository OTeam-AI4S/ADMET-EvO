from pathlib import Path

from scripts.build_conditional_committee_activation_report import build


ROOT = Path(__file__).resolve().parents[1]


def test_historical_replay_is_selective_and_covers_observed_needs():
    payload = build(
        ROOT / "runs/v3-agent-tdc22-v5",
        ROOT / "reports/self_correction_schedule_v1.json",
    )
    assert payload["task_count"] == 22
    assert payload["mode_counts"] == {
        "full_committee": 1,
        "single_agent": 20,
        "targeted_review": 1,
    }
    assert payload["escalated_tasks"] == [
        "CYP2C9_Substrate_CarbonMangels",
        "hERG",
    ]
    assert payload["scheduled_correction_coverage"]["rate"] == 1.0
    assert payload["observed_candidate_divergence_coverage"]["rate"] == 1.0
    assert payload["new_model_fits"] == 0
    assert payload["test_loaded"] is False
