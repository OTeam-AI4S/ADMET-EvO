#!/usr/bin/env python3
"""Roll back packets whose complete development grid selected no configuration."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from admet_agent.evidence_packets import sha256_file

PROTOCOL = "admet-evidence-packet-multiseed-weight-quantity-v5.1"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--development", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    registry_path = args.registry.resolve()
    development_path = args.development.resolve()
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    development = json.loads(development_path.read_text(encoding="utf-8"))
    selection = development.get("selection", {})
    if not (
        development.get("protocol") == PROTOCOL
        and development.get("phase") == "development"
        and selection.get("status") == "no_robust_positive_configuration"
        and selection.get("selected") is None
        and development.get("safety", {}).get("test_labels_loaded") is False
    ):
        raise ValueError("development artifact is not a v5.1 unselected safe trial")
    if development.get("registry_sha256") != sha256_file(registry_path):
        raise ValueError("registry changed after the development trial")
    targets = set(development.get("packet_ids", []))
    if not targets:
        raise ValueError("development artifact contains no packet IDs")

    active = registry.get("active_packets", [])
    by_id = {item["packet_id"]: item for item in active}
    missing = sorted(targets - set(by_id))
    if missing:
        raise ValueError(f"development-trial packets are not active: {missing}")
    for packet_id in targets:
        entry = by_id[packet_id]
        manifest = ROOT / entry["manifest_path"]
        if sha256_file(manifest) != entry["manifest_sha256"]:
            raise ValueError(f"active manifest hash mismatch for {packet_id}")

    before_sha = sha256_file(registry_path)
    registry["active_packets"] = [
        item for item in active if item["packet_id"] not in targets
    ]
    registry_path.write_text(json.dumps(registry, indent=2, sort_keys=True) + "\n")
    result = {
        "protocol": "admet-evidence-packet-development-rollback-v5.1",
        "rolled_back_at": datetime.now(timezone.utc).isoformat(),
        "registry_path": str(registry_path.relative_to(ROOT)),
        "registry_sha256_before": before_sha,
        "registry_sha256_after": sha256_file(registry_path),
        "development_artifact": str(development_path.relative_to(ROOT)),
        "development_artifact_sha256": sha256_file(development_path),
        "task": development["task"],
        "rolled_back_packet_ids": sorted(targets),
        "reason": "no_robust_positive_configuration_over_all_development_seeds",
        "packet_files_deleted": False,
        "historical_experiment_artifacts_retained": True,
        "test_labels_loaded": False,
    }
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError(output)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
