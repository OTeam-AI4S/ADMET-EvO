#!/usr/bin/env python3
"""Promote a firewall-passing candidate and bind it into the active registry."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from admet_agent.evidence_packets import sha256_file, validate_evidence_packet


def promote(manifest_path: Path, audit_path: Path, registry_path: Path) -> dict:
    manifest_path = manifest_path.resolve()
    audit_path = audit_path.resolve()
    registry_path = registry_path.resolve()
    packet = validate_evidence_packet(
        manifest_path, require_modeling_ready=False
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    packet_audit = audit.get("audit", {})
    if audit.get("packet_id") != packet.packet_id or audit.get("task") != packet.task:
        raise ValueError("candidate audit does not identify this packet")
    if not audit.get("promotion_eligible") or not packet_audit.get(
        "admitted_for_training"
    ):
        raise ValueError("candidate did not pass the full structural firewall")
    if packet_audit.get("manifest_sha256") != sha256_file(manifest_path):
        raise ValueError("candidate manifest changed after its firewall audit")
    if packet_audit.get("records_sha256") != sha256_file(packet.records_path):
        raise ValueError("candidate records changed after their firewall audit")
    if audit.get("test_labels_loaded") or packet_audit.get("test_labels_loaded"):
        raise ValueError("candidate audit is contaminated by test labels")

    promoted_at = datetime.now(timezone.utc).isoformat()
    review = manifest["review"]
    if review.get("status") != "review_required":
        raise ValueError("only review_required candidates can be promoted")
    review["status"] = "modeling_ready"
    review["promoted_at"] = promoted_at
    review["promotion_audit_path"] = str(audit_path.relative_to(ROOT))
    review["pre_promotion_manifest_sha256"] = packet_audit["manifest_sha256"]
    review["notes"] = (
        f"{review['notes']} Passed the full-development and label-blind test "
        f"structure firewall: {packet_audit['admitted_rows']} admitted rows; "
        f"test skeleton coverage "
        f"{packet_audit['structure_firewall']['external_test_skeleton_coverage_fraction']:.6f}."
    )
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    post_sha = sha256_file(manifest_path)

    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    entries = [
        item
        for item in registry.get("active_packets", [])
        if item.get("packet_id") != packet.packet_id
    ]
    entries.append(
        {
            "manifest_path": str(manifest_path.relative_to(ROOT)),
            "manifest_sha256": post_sha,
            "packet_id": packet.packet_id,
            "task": packet.task,
        }
    )
    registry["active_packets"] = sorted(
        entries, key=lambda item: (item["task"], item["packet_id"])
    )
    registry_path.write_text(json.dumps(registry, indent=2, sort_keys=True) + "\n")

    result = {
        "packet_id": packet.packet_id,
        "task": packet.task,
        "promoted_at": promoted_at,
        "pre_promotion_manifest_sha256": packet_audit["manifest_sha256"],
        "manifest_sha256": post_sha,
        "registry_sha256": sha256_file(registry_path),
        "admitted_rows": packet_audit["admitted_rows"],
        "test_labels_loaded": False,
    }
    promotion_path = manifest_path.parent / "promotion_audit.json"
    promotion_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--candidate-audit", type=Path, required=True)
    parser.add_argument("--registry", type=Path, required=True)
    args = parser.parse_args()
    print(
        json.dumps(
            promote(args.manifest, args.candidate_audit, args.registry),
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
