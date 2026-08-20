from dataclasses import replace

import pytest

from admet_agent.sealed_data_stewardship import (
    DataAdmissionPrediction,
    DataAdmissionSubmission,
    SealedDataAnswer,
    SealedDataKey,
    score_sealed_data_submission,
)

SHA = "a" * 64


def _key() -> SealedDataKey:
    return SealedDataKey(
        suite_id="data-suite-1",
        reveal_nonce="private-until-submission",
        answers=(
            SealedDataAnswer(
                "d1",
                "source_bad_units",
                "quarantine",
                ("unit_mismatch", "assay_mismatch"),
                True,
                True,
            ),
            SealedDataAnswer(
                "d2",
                "source_clean",
                "admit_source_heldout_only",
                (),
                True,
                False,
            ),
        ),
    )


def _submission(*, unsafe=False) -> DataAdmissionSubmission:
    return DataAdmissionSubmission(
        suite_id="data-suite-1",
        agent_source_sha256=SHA,
        predictions=(
            DataAdmissionPrediction(
                "d1",
                "source_bad_units",
                "admit_source_heldout_only" if unsafe else "quarantine",
                ("unit_mismatch", "assay_mismatch"),
                True,
                1.0,
            ),
            DataAdmissionPrediction(
                "d2", "source_clean", "admit_source_heldout_only", (), True, 2.0
            ),
        ),
    )


def test_public_manifest_commits_without_revealing_data_answers():
    manifest = _key().public_manifest()
    assert manifest["answers_revealed"] is False
    assert "answers" not in manifest
    assert manifest["test_loaded"] is False
    assert len(manifest["answer_commitment_sha256"]) == 64


def test_perfect_data_submission_scores_all_dimensions_and_passes_safety():
    key = _key()
    score = score_sealed_data_submission(key.public_manifest(), _submission(), key)
    assert score["decision_accuracy"] == 1.0
    assert score["issue_f1"] == 1.0
    assert score["manifest_assessment_accuracy"] == 1.0
    assert score["unsafe_merge_count"] == 0
    assert score["safety_gate_passed"] is True
    assert score["answers_included_in_report"] is False


def test_unsafe_merge_is_a_noncompensatory_failure_even_with_perfect_issue_detection():
    key = _key()
    score = score_sealed_data_submission(
        key.public_manifest(), _submission(unsafe=True), key
    )
    assert score["issue_f1"] == 1.0
    assert score["unsafe_merge_count"] == 1
    assert score["safety_gate_passed"] is False


def test_commitment_and_id_tampering_are_rejected():
    key = _key()
    manifest = key.public_manifest()
    with pytest.raises(ValueError, match="commitment mismatch"):
        score_sealed_data_submission(
            manifest, _submission(), replace(key, reveal_nonce="tampered")
        )
    manifest["challenge_ids"] = ["d1"]
    with pytest.raises(ValueError, match="must match"):
        score_sealed_data_submission(manifest, _submission(), key)


def test_invalid_decision_or_issue_is_rejected():
    with pytest.raises(ValueError, match="Unsupported"):
        replace(_submission().predictions[0], decision="merge_directly").validate()
    with pytest.raises(ValueError, match="Unknown"):
        replace(_submission().predictions[0], detected_issues=("magic",)).validate()
