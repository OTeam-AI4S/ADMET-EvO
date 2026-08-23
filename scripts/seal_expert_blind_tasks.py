"""Custodian-side sealing for independently authored blind scientific tasks.

Real hidden packets must be sealed by an external custodian outside the Agent's
workspace.  This utility is provided for that custodian; repository fixtures do
not constitute independent blind tasks.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

PROTOCOL = "admet_expert_blind_scientific_task_v1"
ALLOWED_CATEGORIES = {
    "identity_leakage",
    "duplicate_conflict",
    "unit_or_censoring",
    "endpoint_mixture",
    "source_shift",
    "class_imbalance",
    "activity_cliff",
    "external_contamination",
    "license_or_provenance",
    "repeated_selection",
    "ood_abstention",
    "clean_stop_control",
}
FORBIDDEN_VISIBLE_KEYS = {
    "gold",
    "gold_diagnosis",
    "acceptable_actions",
    "unsafe_actions",
    "hidden_fault",
    "answer",
    "scoring_key",
}


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _visible_keys(value: Any) -> set[str]:
    if isinstance(value, dict):
        return {str(key).lower() for key in value} | set().union(
            *(_visible_keys(item) for item in value.values()),
            set(),
        )
    if isinstance(value, list):
        return set().union(*(_visible_keys(item) for item in value), set())
    return set()


def validate_hidden_packet(packet: dict[str, Any]) -> None:
    required = {
        "protocol",
        "case_id",
        "category",
        "visible_packet",
        "gold_packet",
        "author_attestation",
    }
    missing = required - set(packet)
    if missing:
        raise ValueError(f"Hidden packet missing fields: {sorted(missing)}")
    if packet["protocol"] != PROTOCOL:
        raise ValueError(f"Protocol must be {PROTOCOL}")
    if packet["category"] not in ALLOWED_CATEGORIES:
        raise ValueError("Unknown blind-task category")
    if not isinstance(packet["visible_packet"], dict) or not packet["visible_packet"]:
        raise ValueError("visible_packet must be a non-empty object")
    if not isinstance(packet["gold_packet"], dict) or not packet["gold_packet"]:
        raise ValueError("gold_packet must be a non-empty object")
    leaked = _visible_keys(packet["visible_packet"]) & FORBIDDEN_VISIBLE_KEYS
    if leaked:
        raise ValueError(f"Visible packet contains hidden-answer keys: {sorted(leaked)}")
    attestation = packet["author_attestation"]
    if not isinstance(attestation, dict):
        raise TypeError("author_attestation must be an object")
    required_true = {
        "independent_of_agent_development",
        "did_not_view_agent_outputs",
        "gold_authored_before_release",
    }
    if any(attestation.get(key) is not True for key in required_true):
        raise ValueError("All independence attestations must be explicitly true")
    if not str(attestation.get("author_pseudonym", "")).strip():
        raise ValueError("An author pseudonym is required")


def seal(hidden_path: Path, public_path: Path, commitment_path: Path) -> dict[str, Any]:
    hidden_path = hidden_path.resolve()
    packet = json.loads(hidden_path.read_text(encoding="utf-8"))
    validate_hidden_packet(packet)
    public = {
        "protocol": packet["protocol"],
        "case_id": str(packet["case_id"]),
        "category": packet["category"],
        "visible_packet": packet["visible_packet"],
        "blind": True,
        "gold_released": False,
    }
    commitment = {
        "protocol": packet["protocol"],
        "case_id": str(packet["case_id"]),
        "category": packet["category"],
        "hidden_packet_canonical_sha256": _canonical_sha256(packet),
        "public_packet_canonical_sha256": _canonical_sha256(public),
        "author_pseudonym": packet["author_attestation"]["author_pseudonym"],
        "independence_attested": True,
        "gold_authored_before_release": True,
        "contains_gold": False,
    }
    for path, value in ((public_path, public), (commitment_path, commitment)):
        path.parent.mkdir(parents=True, exist_ok=True)
        rendered = json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False)
        if path.exists():
            existing = json.loads(path.read_text(encoding="utf-8"))
            if _canonical_sha256(existing) != _canonical_sha256(value):
                raise RuntimeError(f"Refusing to overwrite different sealed artifact: {path}")
        else:
            path.write_text(rendered, encoding="utf-8")
    return commitment


def verify_reveal(
    hidden_path: Path,
    public_path: Path,
    commitment_path: Path,
) -> dict[str, Any]:
    hidden = json.loads(hidden_path.read_text(encoding="utf-8"))
    public = json.loads(public_path.read_text(encoding="utf-8"))
    commitment = json.loads(commitment_path.read_text(encoding="utf-8"))
    validate_hidden_packet(hidden)
    checks = {
        "case_id_matches": str(hidden["case_id"]) == str(public.get("case_id")) == str(commitment.get("case_id")),
        "hidden_hash_matches": _canonical_sha256(hidden)
        == commitment.get("hidden_packet_canonical_sha256"),
        "public_hash_matches": _canonical_sha256(public)
        == commitment.get("public_packet_canonical_sha256"),
        "visible_view_matches": hidden["visible_packet"] == public.get("visible_packet"),
        "commitment_contains_no_gold": commitment.get("contains_gold") is False,
    }
    return {
        "protocol": PROTOCOL,
        "case_id": str(hidden["case_id"]),
        "verified": all(checks.values()),
        "checks": checks,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)
    seal_parser = commands.add_parser("seal")
    seal_parser.add_argument("--hidden", type=Path, required=True)
    seal_parser.add_argument("--public", type=Path, required=True)
    seal_parser.add_argument("--commitment", type=Path, required=True)
    verify_parser = commands.add_parser("verify-reveal")
    verify_parser.add_argument("--hidden", type=Path, required=True)
    verify_parser.add_argument("--public", type=Path, required=True)
    verify_parser.add_argument("--commitment", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "seal":
        result = seal(args.hidden, args.public, args.commitment)
    else:
        result = verify_reveal(args.hidden, args.public, args.commitment)
        if not result["verified"]:
            raise SystemExit(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
