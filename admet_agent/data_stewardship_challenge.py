"""Deterministic public dossiers and sealed controls for data stewardship."""

from __future__ import annotations

import hashlib
import json
import random
from dataclasses import asdict, dataclass
from typing import Any

from .sealed_data_stewardship import (
    DATA_ISSUES,
    DataAdmissionPrediction,
    DataAdmissionSubmission,
    SealedDataAnswer,
    SealedDataKey,
    score_sealed_data_submission,
)

DATA_CHALLENGE_PROTOCOL = "external-data-dossier-controls-v1"


def _sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


@dataclass(frozen=True)
class PublicDataDossier:
    challenge_id: str
    source_id: str
    endpoint_match: bool = True
    license_present: bool = True
    checksum_valid: bool = True
    provenance_complete: bool = True
    assay_aligned: bool = True
    species_aligned: bool = True
    unit_aligned: bool = True
    transform_aligned: bool = True
    split_overlap: bool = False
    conflicting_labels: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def default_data_dossiers() -> tuple[PublicDataDossier, ...]:
    base = {"challenge_id": "clean", "source_id": "source-clean"}
    values = [PublicDataDossier(**base)]
    variants = (
        ("license", "license_present", False),
        ("checksum", "checksum_valid", False),
        ("provenance", "provenance_complete", False),
        ("endpoint", "endpoint_match", False),
        ("assay", "assay_aligned", False),
        ("species", "species_aligned", False),
        ("unit", "unit_aligned", False),
        ("transform", "transform_aligned", False),
        ("overlap", "split_overlap", True),
        ("conflict", "conflicting_labels", True),
    )
    for name, field, value in variants:
        payload = {
            **base,
            "challenge_id": f"fault-{name}",
            "source_id": f"source-{name}",
            field: value,
        }
        values.append(PublicDataDossier(**payload))
    return tuple(values)


def _issues(dossier: PublicDataDossier) -> tuple[str, ...]:
    values = []
    checks = (
        (not dossier.endpoint_match, "endpoint_mismatch"),
        (not dossier.license_present, "license_missing"),
        (not dossier.checksum_valid, "checksum_mismatch"),
        (not dossier.provenance_complete, "provenance_missing"),
        (not dossier.assay_aligned, "assay_mismatch"),
        (not dossier.species_aligned, "species_mismatch"),
        (not dossier.unit_aligned, "unit_mismatch"),
        (not dossier.transform_aligned, "transform_mismatch"),
        (dossier.split_overlap, "split_overlap"),
        (dossier.conflicting_labels, "conflicting_labels"),
    )
    for present, issue in checks:
        if present:
            values.append(issue)
    return tuple(sorted(values))


def _gold_decision(issues: tuple[str, ...]) -> str:
    if set(issues) & {
        "endpoint_mismatch",
        "license_missing",
        "checksum_mismatch",
        "provenance_missing",
    }:
        return "reject"
    return "quarantine" if issues else "admit_source_heldout_only"


def build_sealed_data_key(
    dossiers: tuple[PublicDataDossier, ...],
    *,
    reveal_nonce: str,
) -> SealedDataKey:
    return SealedDataKey(
        suite_id="tdc22-external-data-dossiers-v1",
        reveal_nonce=reveal_nonce,
        answers=tuple(
            SealedDataAnswer(
                challenge_id=dossier.challenge_id,
                source_id=dossier.source_id,
                required_decision=_gold_decision(_issues(dossier)),
                expected_issues=_issues(dossier),
                manifest_is_complete=(
                    dossier.license_present and dossier.provenance_complete
                ),
                unsafe_if_admitted=bool(_issues(dossier)),
            )
            for dossier in dossiers
        ),
    )


def public_dossier_manifest(
    dossiers: tuple[PublicDataDossier, ...],
    key: SealedDataKey,
) -> dict[str, Any]:
    payload = [dossier.to_dict() for dossier in dossiers]
    return {
        **key.public_manifest(),
        "dossier_protocol": DATA_CHALLENGE_PROTOCOL,
        "dossiers": payload,
        "dossiers_sha256": _sha256(payload),
    }


def _policy_prediction(
    dossier: PublicDataDossier,
    arm: str,
    *,
    rng: random.Random,
) -> DataAdmissionPrediction:
    all_issues = _issues(dossier)
    if arm in {"agent_contract_steward", "complete_rules_steward"}:
        detected = all_issues
        decision = _gold_decision(detected)
        manifest_complete = dossier.license_present and dossier.provenance_complete
    elif arm == "expanded_automl_schema_steward":
        detected = tuple(
            issue
            for issue in all_issues
            if issue in {"checksum_mismatch", "split_overlap", "conflicting_labels"}
        )
        decision = "quarantine" if detected else "admit_source_heldout_only"
        manifest_complete = True
    elif arm == "conventional_automl_direct_fit":
        detected = ()
        decision = "admit_source_heldout_only"
        manifest_complete = True
    elif arm == "seeded_random_admission":
        detected = (rng.choice(sorted(DATA_ISSUES)),)
        decision = rng.choice(["reject", "quarantine", "admit_source_heldout_only"])
        manifest_complete = bool(rng.getrandbits(1))
    else:
        raise ValueError(f"Unknown data-stewardship arm {arm!r}")
    return DataAdmissionPrediction(
        challenge_id=dossier.challenge_id,
        source_id=dossier.source_id,
        decision=decision,
        detected_issues=detected,
        manifest_complete=manifest_complete,
        decision_seconds=0.0,
    )


def compare_data_stewardship_controls(
    *,
    policy_seed: int = 20260820,
) -> dict[str, Any]:
    dossiers = default_data_dossiers()
    key = build_sealed_data_key(dossiers, reveal_nonce="local-fixture-reveal-v1")
    public = public_dossier_manifest(dossiers, key)
    arms = (
        "agent_contract_steward",
        "complete_rules_steward",
        "expanded_automl_schema_steward",
        "conventional_automl_direct_fit",
        "seeded_random_admission",
    )
    scores = {}
    for arm in arms:
        rng = random.Random(f"{policy_seed}:{arm}")
        submission = DataAdmissionSubmission(
            suite_id=key.suite_id,
            predictions=tuple(
                _policy_prediction(dossier, arm, rng=rng) for dossier in dossiers
            ),
            agent_source_sha256=_sha256({"arm": arm, "version": 1}),
        )
        scores[arm] = score_sealed_data_submission(public, submission, key)
    return {
        "protocol": DATA_CHALLENGE_PROTOCOL,
        "claim_scope": "sealed_local_fixture_not_independent_external_validation",
        "public_manifest": {
            key: value for key, value in public.items() if key not in {"dossiers"}
        },
        "policy_seed": policy_seed,
        "arms": scores,
        "test_loaded": False,
        "interpretation": {
            "rule_boundary": (
                "The current Agent policy is equivalent to the complete deterministic "
                "contract-rules policy on this fixture."
            ),
            "upgrade_requirement": (
                "An independently authored challenge must include composite ambiguity "
                "and evidence-dependent remediation choices not reducible to these flags."
            ),
        },
    }


__all__ = [
    "DATA_CHALLENGE_PROTOCOL",
    "PublicDataDossier",
    "build_sealed_data_key",
    "compare_data_stewardship_controls",
    "default_data_dossiers",
    "public_dossier_manifest",
]
