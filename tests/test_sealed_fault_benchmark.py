import pytest

from admet_agent.sealed_fault_benchmark import (
    SEALED_LOCAL_CLAIM_SCOPE,
    FaultPrediction,
    FaultPredictionSubmission,
    SealedFaultAnswer,
    SealedFaultKey,
    score_sealed_submission,
)


def key() -> SealedFaultKey:
    return SealedFaultKey(
        suite_id="sealed-v1",
        answers=(
            SealedFaultAnswer("clean", "HIA_Hou", (), ()),
            SealedFaultAnswer(
                "overlap",
                "HIA_Hou",
                ("exact_split_overlap",),
                ("remove_train_overlap",),
            ),
            SealedFaultAnswer(
                "species",
                "HIA_Hou",
                ("species_mismatch",),
                ("quarantine_source", "request_species_alignment"),
            ),
        ),
        reveal_nonce="private-reveal-nonce",
    )


def submission() -> FaultPredictionSubmission:
    return FaultPredictionSubmission(
        suite_id="sealed-v1",
        detector_source_sha256="a" * 64,
        predictions=(
            FaultPrediction("clean", "HIA_Hou", (), (), 0.1),
            FaultPrediction(
                "overlap",
                "HIA_Hou",
                ("exact_split_overlap",),
                ("remove_train_overlap",),
                0.2,
            ),
            FaultPrediction(
                "species",
                "HIA_Hou",
                ("species_mismatch",),
                ("quarantine_source",),
                0.3,
            ),
        ),
    )


def test_public_manifest_commits_without_revealing_answers():
    manifest = key().public_manifest()
    assert manifest["answers_revealed"] is False
    assert manifest["test_loaded"] is False
    assert "answers" not in manifest
    assert "reveal_nonce" not in manifest
    assert len(manifest["answer_commitment_sha256"]) == 64


def test_perfect_sealed_submission_scores_after_commitment_reveal():
    result = score_sealed_submission(key().public_manifest(), submission(), key())
    assert result["answer_commitment_verified"] is True
    assert result["answers_included_in_report"] is False
    assert result["claim_scope"] == SEALED_LOCAL_CLAIM_SCOPE
    assert result["score"]["micro_f1"] == pytest.approx(1.0)
    assert result["score"]["critical_recall"] == pytest.approx(1.0)
    assert result["score"]["remediation_validity"] == pytest.approx(1.0)
    assert result["test_loaded"] is False


def test_commitment_rejects_answer_tampering():
    original = key()
    manifest = original.public_manifest()
    tampered = SealedFaultKey(
        suite_id=original.suite_id,
        answers=(
            *original.answers[:-1],
            SealedFaultAnswer("species", "HIA_Hou", (), ()),
        ),
        reveal_nonce=original.reveal_nonce,
    )
    with pytest.raises(ValueError, match="commitment mismatch"):
        score_sealed_submission(manifest, submission(), tampered)


def test_wrong_nonce_rejects_reveal():
    original = key()
    wrong_nonce = SealedFaultKey(
        suite_id=original.suite_id,
        answers=original.answers,
        reveal_nonce="different",
    )
    with pytest.raises(ValueError, match="commitment mismatch"):
        score_sealed_submission(original.public_manifest(), submission(), wrong_nonce)


def test_missing_or_extra_prediction_is_rejected():
    original = submission()
    missing = FaultPredictionSubmission(
        suite_id=original.suite_id,
        predictions=original.predictions[:-1],
        detector_source_sha256=original.detector_source_sha256,
    )
    with pytest.raises(ValueError, match="must match exactly"):
        score_sealed_submission(key().public_manifest(), missing, key())


def test_invalid_remediation_lowers_validity_without_revealing_answer():
    original = submission()
    altered = FaultPredictionSubmission(
        suite_id=original.suite_id,
        detector_source_sha256=original.detector_source_sha256,
        predictions=(
            *original.predictions[:-1],
            FaultPrediction(
                "species",
                "HIA_Hou",
                ("species_mismatch",),
                ("merge_without_alignment",),
                0.3,
            ),
        ),
    )
    result = score_sealed_submission(key().public_manifest(), altered, key())
    assert result["score"]["micro_f1"] == pytest.approx(1.0)
    assert result["score"]["remediation_validity"] < 1.0
    assert "expected_faults" not in result


def test_duplicate_prediction_ids_and_unknown_faults_are_rejected():
    duplicate = FaultPredictionSubmission(
        suite_id="sealed-v1",
        detector_source_sha256="a" * 64,
        predictions=(
            FaultPrediction("same", "HIA_Hou", (), (), 0.1),
            FaultPrediction("same", "HIA_Hou", (), (), 0.2),
        ),
    )
    with pytest.raises(ValueError, match="must be unique"):
        duplicate.validate()

    invalid = FaultPrediction("x", "HIA_Hou", ("invented_fault",), (), 0.1)
    with pytest.raises(ValueError, match="Unknown detected"):
        invalid.validate()


def test_suite_task_and_test_declaration_must_match():
    manifest = key().public_manifest()
    manifest["test_loaded"] = True
    with pytest.raises(ValueError, match="test_loaded=false"):
        score_sealed_submission(manifest, submission(), key())

    original = submission()
    wrong_task = FaultPredictionSubmission(
        suite_id=original.suite_id,
        detector_source_sha256=original.detector_source_sha256,
        predictions=(
            FaultPrediction("clean", "AMES", (), (), 0.1),
            *original.predictions[1:],
        ),
    )
    with pytest.raises(ValueError, match="Task mismatch"):
        score_sealed_submission(key().public_manifest(), wrong_task, key())
