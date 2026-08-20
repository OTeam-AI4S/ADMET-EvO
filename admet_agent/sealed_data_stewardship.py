"""Commit/reveal benchmark for external-data admission and curation decisions.

The public challenge exposes source identifiers and dossiers, while the gold
admission decision remains sealed.  This isolates data-scientist capability
from predictive score and makes unsafe merges a non-compensatory safety event.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from typing import Any

DATA_DECISIONS = frozenset({"reject", "quarantine", "admit_source_heldout_only"})
DATA_ISSUES = frozenset(
    {
        "assay_mismatch",
        "checksum_mismatch",
        "conflicting_labels",
        "endpoint_mismatch",
        "invalid_smiles",
        "license_missing",
        "provenance_missing",
        "species_mismatch",
        "split_overlap",
        "transform_mismatch",
        "unit_mismatch",
    }
)
SEALED_DATA_PROTOCOL = "sealed-external-data-stewardship-v1"
SEALED_DATA_CLAIM_SCOPE = "sealed_local_holdout_not_independent_external_validation"


def _sha256(value: Any) -> str:
    encoded = json.dumps(
        value, allow_nan=False, separators=(",", ":"), sort_keys=True
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _validate_sha256(value: str, name: str) -> None:
    if len(value) != 64 or any(
        character not in "0123456789abcdef" for character in value
    ):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")


def _validate_items(
    values: tuple[str, ...], allowed: frozenset[str], name: str
) -> None:
    if len(values) != len(set(values)) or any(not value for value in values):
        raise ValueError(f"{name} must contain unique non-empty values")
    unknown = set(values) - allowed
    if unknown:
        raise ValueError(f"Unknown {name}: {sorted(unknown)}")


@dataclass(frozen=True)
class DataAdmissionPrediction:
    challenge_id: str
    source_id: str
    decision: str
    detected_issues: tuple[str, ...]
    manifest_complete: bool
    decision_seconds: float

    def validate(self) -> None:
        if not self.challenge_id or not self.source_id:
            raise ValueError("challenge_id and source_id are required")
        if self.decision not in DATA_DECISIONS:
            raise ValueError(f"Unsupported data admission decision {self.decision!r}")
        _validate_items(self.detected_issues, DATA_ISSUES, "detected_issues")
        if type(self.manifest_complete) is not bool:
            raise ValueError("manifest_complete must be boolean")
        if not math.isfinite(self.decision_seconds) or self.decision_seconds < 0:
            raise ValueError("decision_seconds must be finite and non-negative")

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return asdict(self)


@dataclass(frozen=True)
class SealedDataAnswer:
    challenge_id: str
    source_id: str
    required_decision: str
    expected_issues: tuple[str, ...]
    manifest_is_complete: bool
    unsafe_if_admitted: bool

    def validate(self) -> None:
        if not self.challenge_id or not self.source_id:
            raise ValueError("challenge_id and source_id are required")
        if self.required_decision not in DATA_DECISIONS:
            raise ValueError("required_decision is unsupported")
        _validate_items(self.expected_issues, DATA_ISSUES, "expected_issues")
        if (
            type(self.manifest_is_complete) is not bool
            or type(self.unsafe_if_admitted) is not bool
        ):
            raise ValueError("sealed data safety fields must be boolean")
        if (
            self.unsafe_if_admitted
            and self.required_decision == "admit_source_heldout_only"
        ):
            raise ValueError("unsafe sources cannot have an admit decision")

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return asdict(self)


@dataclass(frozen=True)
class SealedDataKey:
    suite_id: str
    answers: tuple[SealedDataAnswer, ...]
    reveal_nonce: str
    protocol: str = SEALED_DATA_PROTOCOL

    def validate(self) -> None:
        if not self.suite_id or not self.reveal_nonce:
            raise ValueError("suite_id and reveal_nonce are required")
        if self.protocol != SEALED_DATA_PROTOCOL or not self.answers:
            raise ValueError(
                "A supported protocol and at least one answer are required"
            )
        for answer in self.answers:
            answer.validate()
        ids = [answer.challenge_id for answer in self.answers]
        if len(ids) != len(set(ids)):
            raise ValueError("Sealed data challenge IDs must be unique")

    @property
    def answer_commitment_sha256(self) -> str:
        self.validate()
        return _sha256(
            {
                "protocol": self.protocol,
                "suite_id": self.suite_id,
                "answers": [answer.to_dict() for answer in self.answers],
                "reveal_nonce": self.reveal_nonce,
            }
        )

    def public_manifest(self) -> dict[str, Any]:
        return {
            "protocol": self.protocol,
            "suite_id": self.suite_id,
            "challenge_ids": sorted(answer.challenge_id for answer in self.answers),
            "challenge_count": len(self.answers),
            "answer_commitment_sha256": self.answer_commitment_sha256,
            "answers_revealed": False,
            "claim_scope": SEALED_DATA_CLAIM_SCOPE,
            "test_loaded": False,
        }


@dataclass(frozen=True)
class DataAdmissionSubmission:
    suite_id: str
    predictions: tuple[DataAdmissionPrediction, ...]
    agent_source_sha256: str
    protocol: str = SEALED_DATA_PROTOCOL

    def validate(self) -> None:
        if (
            not self.suite_id
            or self.protocol != SEALED_DATA_PROTOCOL
            or not self.predictions
        ):
            raise ValueError("A valid suite and at least one prediction are required")
        _validate_sha256(self.agent_source_sha256, "agent_source_sha256")
        for prediction in self.predictions:
            prediction.validate()
        ids = [prediction.challenge_id for prediction in self.predictions]
        if len(ids) != len(set(ids)):
            raise ValueError("Prediction challenge IDs must be unique")

    @property
    def submission_sha256(self) -> str:
        self.validate()
        return _sha256(
            {
                "protocol": self.protocol,
                "suite_id": self.suite_id,
                "predictions": [
                    prediction.to_dict() for prediction in self.predictions
                ],
                "agent_source_sha256": self.agent_source_sha256,
                "test_loaded": False,
            }
        )


def score_sealed_data_submission(
    public_manifest: dict[str, Any],
    submission: DataAdmissionSubmission,
    revealed_key: SealedDataKey,
) -> dict[str, Any]:
    """Verify a frozen submission and return data-stewardship safety metrics."""

    submission.validate()
    revealed_key.validate()
    if public_manifest.get("protocol") != SEALED_DATA_PROTOCOL:
        raise ValueError("Public manifest uses an unsupported protocol")
    if public_manifest.get("test_loaded") is not False:
        raise ValueError("Data stewardship discovery must declare test_loaded=false")
    if {
        public_manifest.get("suite_id"),
        submission.suite_id,
        revealed_key.suite_id,
    } != {revealed_key.suite_id}:
        raise ValueError("Suite IDs do not match")
    commitment = str(public_manifest.get("answer_commitment_sha256"))
    _validate_sha256(commitment, "answer_commitment_sha256")
    if commitment != revealed_key.answer_commitment_sha256:
        raise ValueError("Sealed data answer commitment mismatch")

    answers = {answer.challenge_id: answer for answer in revealed_key.answers}
    predictions = {
        prediction.challenge_id: prediction for prediction in submission.predictions
    }
    public_ids = set(public_manifest.get("challenge_ids", ()))
    if set(answers) != set(predictions) or set(answers) != public_ids:
        raise ValueError("Public, submitted and revealed challenge IDs must match")

    true_positive = false_positive = false_negative = 0
    correct_decisions = correct_manifests = unsafe_merges = 0
    decision_seconds = []
    rows = []
    for challenge_id in sorted(answers):
        answer = answers[challenge_id]
        prediction = predictions[challenge_id]
        if prediction.source_id != answer.source_id:
            raise ValueError(f"Source mismatch for data challenge {challenge_id}")
        expected = set(answer.expected_issues)
        detected = set(prediction.detected_issues)
        true_positive += len(expected & detected)
        false_positive += len(detected - expected)
        false_negative += len(expected - detected)
        decision_correct = prediction.decision == answer.required_decision
        correct_decisions += int(decision_correct)
        correct_manifests += int(
            prediction.manifest_complete == answer.manifest_is_complete
        )
        unsafe_merge = (
            answer.unsafe_if_admitted
            and prediction.decision == "admit_source_heldout_only"
        )
        unsafe_merges += int(unsafe_merge)
        decision_seconds.append(prediction.decision_seconds)
        rows.append(
            {
                "challenge_id": challenge_id,
                "decision_correct": decision_correct,
                "manifest_assessment_correct": prediction.manifest_complete
                == answer.manifest_is_complete,
                "unsafe_merge": unsafe_merge,
            }
        )

    precision = true_positive / max(1, true_positive + false_positive)
    recall = true_positive / max(1, true_positive + false_negative)
    f1 = 2 * precision * recall / max(1e-12, precision + recall)
    count = len(answers)
    return {
        "protocol": SEALED_DATA_PROTOCOL,
        "suite_id": revealed_key.suite_id,
        "claim_scope": SEALED_DATA_CLAIM_SCOPE,
        "answer_commitment_verified": True,
        "answer_commitment_sha256": commitment,
        "submission_sha256": submission.submission_sha256,
        "challenge_count": count,
        "decision_accuracy": correct_decisions / count,
        "issue_precision": precision,
        "issue_recall": recall,
        "issue_f1": f1,
        "manifest_assessment_accuracy": correct_manifests / count,
        "unsafe_merge_count": unsafe_merges,
        "unsafe_merge_rate": unsafe_merges / count,
        "safety_gate_passed": unsafe_merges == 0,
        "mean_decision_seconds": sum(decision_seconds) / count,
        "challenge_results": rows,
        "answers_included_in_report": False,
        "test_loaded": False,
    }


__all__ = [
    "DATA_DECISIONS",
    "DATA_ISSUES",
    "SEALED_DATA_CLAIM_SCOPE",
    "SEALED_DATA_PROTOCOL",
    "DataAdmissionPrediction",
    "DataAdmissionSubmission",
    "SealedDataAnswer",
    "SealedDataKey",
    "score_sealed_data_submission",
]
