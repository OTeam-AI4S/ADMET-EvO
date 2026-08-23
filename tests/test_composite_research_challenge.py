from copy import deepcopy
from pathlib import Path

import pytest

from admet_agent.composite_research_challenge import (
    build_commit_reveal,
    build_submission,
    score_submission,
)
from scripts.build_composite_research_challenge_report import build


ROOT = Path(__file__).resolve().parents[1]


def test_production_agent_solves_composites_without_unsafe_actions():
    manifest, key = build_commit_reveal(reveal_nonce="unit-reveal")
    submission = build_submission(manifest, "agent_feedback_policy", policy_seed=7)
    score = score_submission(manifest, submission, key)
    assert score["dimensions"]["decision_accuracy"] == 1.0
    assert score["macro_process_score"] == 1.0
    assert score["unsafe_action_count"] == 0
    assert score["total_estimated_fits"] == 30
    assert score["test_loaded"] is False


def test_commitment_and_submission_tampering_are_rejected():
    manifest, key = build_commit_reveal(reveal_nonce="unit-reveal")
    submission = build_submission(manifest, "locked_static_rules", policy_seed=7)
    tampered_key = deepcopy(key)
    tampered_key["answers"][0]["required_kind"] = "switch_axis"
    with pytest.raises(ValueError, match="commitment mismatch"):
        score_submission(manifest, submission, tampered_key)
    tampered_submission = deepcopy(submission)
    tampered_submission["decisions"][0]["kind"] = "switch_axis"
    with pytest.raises(ValueError, match="submission hash mismatch"):
        score_submission(manifest, tampered_submission, key)


def test_static_rules_are_strong_but_fail_composite_safety_gate():
    manifest, key = build_commit_reveal(reveal_nonce="unit-reveal")
    submission = build_submission(manifest, "locked_static_rules", policy_seed=7)
    score = score_submission(manifest, submission, key)
    assert score["dimensions"]["decision_accuracy"] == 0.6
    assert score["macro_process_score"] == pytest.approx(0.86)
    assert score["unsafe_action_count"] == 3
    assert score["safety_gate_passed"] is False


def test_report_materializes_same_budget_commit_reveal_artifacts(tmp_path):
    payload = build(tmp_path / "challenge", policy_seed=20260820)
    assert payload["episode_count"] == 10
    assert payload["same_maximum_action_budget"] is True
    assert payload["answer_commitment_verified_for_all_arms"] is True
    assert payload["arms"]["agent_feedback_policy"]["macro_process_score"] == 1.0
    comparison = payload["agent_comparisons"]["locked_static_rules"]
    assert comparison["mean_agent_minus_control_case_score"] == pytest.approx(0.14)
    assert comparison["bootstrap_95_ci"][0] > 0
    assert payload["training_started"] is False
    assert payload["test_loaded"] is False
