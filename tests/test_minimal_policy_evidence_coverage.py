from pathlib import Path

from scripts.build_minimal_policy_evidence_coverage import build


ROOT = Path(__file__).resolve().parents[1]


def test_existing_evidence_covers_axes_and_three_exact_pairs(monkeypatch):
    monkeypatch.chdir(ROOT)
    payload = build(
        ROOT / "reports/transport_grid_results_v1.json",
        ROOT / "reports/post_memory_confirmation_v1.json",
    )
    assert payload["axis_coverage"] == {"numerator": 5, "denominator": 5}
    assert payload["exact_pair_coverage"] == {"numerator": 3, "denominator": 5}
    assert payload["activation_decision"] == "blocked_pending_two_exact_pair_pilots"
    assert payload["minimum_new_compute"]["initial_screen_fits"] == 8
    assert payload["minimum_new_compute"]["maximum_confirmation_fits"] == 20
    assert payload["minimum_new_compute"]["unimol2_fits"] == 0
    assert payload["training_started"] is False
    assert payload["test_loaded"] is False
