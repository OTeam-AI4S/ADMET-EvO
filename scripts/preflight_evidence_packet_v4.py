#!/usr/bin/env python3
"""Fail-closed registry preflight for the Evidence Packet v4 data axis."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from admet_agent.evidence_packets import sha256_file, validate_evidence_packet  # noqa: E402


REGISTRY_SCHEMA = "admet-evidence-packet-registry-v4.1"
DEFAULT_REGISTRY = ROOT / "data/evidence_packets_v4/registry.json"
DEFAULT_OUTPUT = ROOT / "reports/evidence_packet_data_axis_v4/preflight.json"


def preflight(registry_path: Path) -> dict[str, Any]:
    registry_path = registry_path.resolve()
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    if registry.get("schema_version") != REGISTRY_SCHEMA:
        raise ValueError("Evidence Packet registry schema mismatch")
    if registry.get("policy", {}).get("legacy_bulk_pool_allowed") is not False:
        raise ValueError("v4 registry must explicitly forbid legacy bulk pools")
    if registry.get("test_labels_loaded") is not False:
        raise ValueError("registry must certify that test labels were not loaded")
    packet_root = (ROOT / registry["policy"]["active_packet_root"]).resolve()
    expected_root = (ROOT / "data/evidence_packets_v4/packets").resolve()
    if packet_root != expected_root:
        raise ValueError("active packet root is not the frozen v4 packet directory")

    seen_ids: set[str] = set()
    seen_manifests: set[Path] = set()
    packets: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []
    for entry in registry.get("active_packets", []):
        manifest_path = (ROOT / str(entry["manifest_path"])).resolve()
        try:
            if not manifest_path.is_relative_to(packet_root):
                raise ValueError("manifest is outside the active packet root")
            if manifest_path in seen_manifests:
                raise ValueError("duplicate manifest registry entry")
            packet = validate_evidence_packet(manifest_path)
            if packet.packet_id != entry.get("packet_id"):
                raise ValueError("registry packet_id does not match manifest")
            if packet.task != entry.get("task"):
                raise ValueError("registry task does not match manifest")
            expected_sha = entry.get("manifest_sha256")
            if expected_sha != sha256_file(manifest_path):
                raise ValueError("registry manifest SHA-256 mismatch")
            if packet.packet_id in seen_ids:
                raise ValueError("duplicate packet_id")
            seen_ids.add(packet.packet_id)
            seen_manifests.add(manifest_path)
            packets.append(
                {
                    "packet_id": packet.packet_id,
                    "task": packet.task,
                    "source_kind": packet.source_kind,
                    "rows": len(packet.records),
                    "manifest_path": str(manifest_path),
                    "manifest_sha256": expected_sha,
                    "records_path": str(packet.records_path),
                    "records_sha256": sha256_file(packet.records_path),
                }
            )
        except Exception as error:
            failures.append(
                {
                    "manifest_path": str(manifest_path),
                    "error": type(error).__name__,
                    "message": str(error),
                }
            )

    status = (
        "failed_closed"
        if failures
        else "ready"
        if packets
        else "blocked_no_modeling_ready_packets"
    )
    return {
        "protocol": "evidence-packet-v4-registry-preflight-v1",
        "status": status,
        "registry_path": str(registry_path),
        "registry_sha256": sha256_file(registry_path),
        "active_packet_count": len(packets),
        "active_record_count": sum(packet["rows"] for packet in packets),
        "packets": packets,
        "failures": failures,
        "formal_experiment_allowed": bool(packets and not failures),
        "legacy_bulk_pool_used": False,
        "quarantined_legacy_source_count": len(
            registry.get("quarantined_legacy_sources", [])
        ),
        "test_labels_loaded": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--require-ready", action="store_true")
    args = parser.parse_args()
    report = preflight(args.registry)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, sort_keys=True))
    if report["status"] == "failed_closed" or (
        args.require_ready and not report["formal_experiment_allowed"]
    ):
        raise SystemExit(2)


if __name__ == "__main__":
    main()
