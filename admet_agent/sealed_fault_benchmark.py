"""Commit/reveal scoring for sealed scientific-fault challenges.

This module separates public challenge predictions from a private answer key.
It provides tamper-evident local holdout evidence, but deliberately does not
claim that a key created by the same development team is independent blind
validation.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from typing import Any

from .hidden_fault_benchmark import (
    SUPPORTED_FAULTS,
    FaultChallengeResult,
    score_hidden_faults,
)

SEALED_FAULT_PROTOCOL = "sealed-scientific-fault-commit-reveal-v1"
SEALED_LOCAL_CLAIM_SCOPE = "sealed_local_holdout_not_independent_external_validation"


def _canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def _validate_sha256(value: str, name: str) -> None:
    if len(value) != 64 or any(
        character not in "0123456789abcdef" for character in value
    ):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")


def _unique_nonempty(values: tuple[str, ...], name: str) -> None:
    if any(not value for value in values) or len(values) != len(set(values)):
        raise ValueError(f"{name} must contain unique non-empty values")


@dataclass(frozen=True)
class FaultPrediction:
    challenge_id: str
    task: str
    detected_faults: tuple[str, ...]
    remediation_ids: tuple[str, ...]
    detection_seconds: float

    def validate(self) -> None:
        if not self.challenge_id or not self.task:
            raise ValueError("challenge_id and task are required")
        _unique_nonempty(self.detected_faults, "detected_faults")
        _unique_nonempty(self.remediation_ids, "remediation_ids")
        unknown = set(self.detected_faults) - SUPPORTED_FAULTS
        if unknown:
            raise ValueError(f"Unknown detected fault types: {sorted(unknown)}")
        if not math.isfinite(self.detection_seconds) or self.detection_seconds < 0:
            raise ValueError("detection_seconds must be finite and non-negative")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class SealedFaultAnswer:
    challenge_id: str
    task: str
    expected_faults: tuple[str, ...]
    valid_remediation_ids: tuple[str, ...]

    def validate(self) -> None:
        if not self.challenge_id or not self.task:
            raise ValueError("challenge_id and task are required")
        _unique_nonempty(self.expected_faults, "expected_faults")
        _unique_nonempty(self.valid_remediation_ids, "valid_remediation_ids")
        unknown = set(self.expected_faults) - SUPPORTED_FAULTS
        if unknown:
            raise ValueError(f"Unknown expected fault types: {sorted(unknown)}")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class SealedFaultKey:
    suite_id: str
    answers: tuple[SealedFaultAnswer, ...]
    reveal_nonce: str
    protocol: str = SEALED_FAULT_PROTOCOL

    def validate(self) -> None:
        if not self.suite_id or not self.reveal_nonce:
            raise ValueError("suite_id and reveal_nonce are required")
        if self.protocol != SEALED_FAULT_PROTOCOL:
            raise ValueError("Unsupported sealed fault protocol")
        if not self.answers:
            raise ValueError("At least one sealed answer is required")
        for answer in self.answers:
            answer.validate()
        ids = [answer.challenge_id for answer in self.answers]
        if len(ids) != len(set(ids)):
            raise ValueError("Sealed challenge IDs must be unique")

    def commitment_payload(self) -> dict[str, Any]:
        self.validate()
        return {
            "protocol": self.protocol,
            "suite_id": self.suite_id,
            "answers": [answer.to_dict() for answer in self.answers],
            "reveal_nonce": self.reveal_nonce,
        }

    @property
    def answer_commitment_sha256(self) -> str:
        return _canonical_sha256(self.commitment_payload())

    def public_manifest(self) -> dict[str, Any]:
        """Return a manifest that commits to, but does not reveal, labels."""

        return {
            "protocol": self.protocol,
            "suite_id": self.suite_id,
            "challenge_ids": sorted(answer.challenge_id for answer in self.answers),
            "challenge_count": len(self.answers),
            "answer_commitment_sha256": self.answer_commitment_sha256,
            "answers_revealed": False,
            "claim_scope": SEALED_LOCAL_CLAIM_SCOPE,
            "test_loaded": False,
        }


@dataclass(frozen=True)
class FaultPredictionSubmission:
    suite_id: str
    predictions: tuple[FaultPrediction, ...]
    detector_source_sha256: str
    protocol: str = SEALED_FAULT_PROTOCOL

    def validate(self) -> None:
        if not self.suite_id or self.protocol != SEALED_FAULT_PROTOCOL:
            raise ValueError("Valid suite_id and sealed fault protocol are required")
        _validate_sha256(self.detector_source_sha256, "detector_source_sha256")
        if not self.predictions:
            raise ValueError("At least one prediction is required")
        for prediction in self.predictions:
            prediction.validate()
        ids = [prediction.challenge_id for prediction in self.predictions]
        if len(ids) != len(set(ids)):
            raise ValueError("Prediction challenge IDs must be unique")

    def _payload(self) -> dict[str, Any]:
        self.validate()
        return {
            "protocol": self.protocol,
            "suite_id": self.suite_id,
            "predictions": [prediction.to_dict() for prediction in self.predictions],
            "detector_source_sha256": self.detector_source_sha256,
            "test_loaded": False,
        }

    @property
    def submission_sha256(self) -> str:
        return _canonical_sha256(self._payload())

    def to_dict(self) -> dict[str, Any]:
        return {**self._payload(), "submission_sha256": self.submission_sha256}


def score_sealed_submission(
    public_manifest: dict[str, Any],
    submission: FaultPredictionSubmission,
    revealed_key: SealedFaultKey,
) -> dict[str, Any]:
    """Verify commit/reveal integrity and score a frozen prediction submission."""

    revealed_key.validate()
    submission.validate()
    if public_manifest.get("protocol") != SEALED_FAULT_PROTOCOL:
        raise ValueError("Public manifest uses an unsupported protocol")
    if public_manifest.get("test_loaded") is not False:
        raise ValueError("Sealed discovery benchmark must declare test_loaded=false")
    suite_ids = {
        public_manifest.get("suite_id"),
        submission.suite_id,
        revealed_key.suite_id,
    }
    if len(suite_ids) != 1:
        raise ValueError("Suite IDs do not match")
    claimed_commitment = public_manifest.get("answer_commitment_sha256")
    _validate_sha256(str(claimed_commitment), "answer_commitment_sha256")
    if claimed_commitment != revealed_key.answer_commitment_sha256:
        raise ValueError("Sealed answer commitment mismatch")

    answers = {answer.challenge_id: answer for answer in revealed_key.answers}
    predictions = {
        prediction.challenge_id: prediction for prediction in submission.predictions
    }
    public_ids = set(public_manifest.get("challenge_ids", ()))
    if public_ids != set(answers) or set(predictions) != set(answers):
        raise ValueError(
            "Public, submitted, and revealed challenge IDs must match exactly"
        )

    rows = []
    remediation_detail = []
    for challenge_id in sorted(answers):
        answer = answers[challenge_id]
        prediction = predictions[challenge_id]
        if prediction.task != answer.task:
            raise ValueError(f"Task mismatch for sealed challenge {challenge_id}")
        proposed = set(prediction.remediation_ids)
        accepted = set(answer.valid_remediation_ids)
        valid_count = len(proposed & accepted)
        rows.append(
            FaultChallengeResult(
                challenge_id=challenge_id,
                task=answer.task,
                expected_faults=answer.expected_faults,
                detected_faults=prediction.detected_faults,
                detection_seconds=prediction.detection_seconds,
                remediations_proposed=len(proposed),
                valid_remediations=valid_count,
            )
        )
        remediation_detail.append(
            {
                "challenge_id": challenge_id,
                "proposed": len(proposed),
                "valid": valid_count,
                "invalid": len(proposed - accepted),
            }
        )

    score = score_hidden_faults(rows)
    return {
        "protocol": SEALED_FAULT_PROTOCOL,
        "suite_id": revealed_key.suite_id,
        "claim_scope": SEALED_LOCAL_CLAIM_SCOPE,
        "answer_commitment_verified": True,
        "answer_commitment_sha256": revealed_key.answer_commitment_sha256,
        "submission_sha256": submission.submission_sha256,
        "challenge_count": len(rows),
        "score": score,
        "remediation_summary": remediation_detail,
        "answers_included_in_report": False,
        "test_loaded": False,
    }


__all__ = [
    "SEALED_FAULT_PROTOCOL",
    "SEALED_LOCAL_CLAIM_SCOPE",
    "FaultPrediction",
    "FaultPredictionSubmission",
    "SealedFaultAnswer",
    "SealedFaultKey",
    "score_sealed_submission",
]
