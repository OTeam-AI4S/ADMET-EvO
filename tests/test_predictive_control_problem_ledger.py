from pathlib import Path

from scripts.build_predictive_control_problem_ledger import build


def test_problem_ledger_retains_losses_without_mutating_or_reusing_evaluation():
    payload = build(Path("reports/minimal_predictive_control_v1.json"))
    assert payload["material_loss_count"] == 8
    assert payload["automl_winner_roles_on_losses"] == {"lightgbm": 8}
    assert payload["policy_mutated_after_evaluation"] is False
    assert payload["evaluation_seed_reuse_for_future_confirmation"] is False
    assert len(payload["falsifiable_future_hypotheses"]) == 3
    assert payload["test_loaded"] is False
