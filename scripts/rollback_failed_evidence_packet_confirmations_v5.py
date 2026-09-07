#!/usr/bin/env python3
"""Remove confirmation-failed packet trials from the active incumbent registry."""

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


def rollback(registry_path: Path, confirmation_paths: list[Path], output: Path) -> dict:
    registry_path = registry_path.resolve()
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    before_sha = sha256_file(registry_path)
    targets: dict[str, dict] = {}
    confirmations = []
    for path in confirmation_paths:
        path = path.resolve()
        payload = json.loads(path.read_text(encoding="utf-8"))
        summary = payload.get("confirmation_summary", {})
        if summary.get("confirmation_passed") is not False:
            raise ValueError(f"refusing to roll back non-failed confirmation: {path}")
        for packet_id in payload.get("packet_ids", []):
            targets[packet_id] = {
                "task": payload["task"],
                "confirmation_artifact": str(path.relative_to(ROOT)),
                "confirmation_artifact_sha256": sha256_file(path),
                "mean_normalized_improvement": summary.get(
                    "mean_normalized_improvement"
                ),
                "positive_seed_fraction": summary.get("positive_seed_fraction"),
            }
        confirmations.append(str(path.relative_to(ROOT)))
    if not targets:
        raise ValueError("no failed-trial packet IDs found")

    active = registry.get("active_packets", [])
    by_id = {item["packet_id"]: item for item in active}
    missing = sorted(set(targets) - set(by_id))
    if missing:
        raise ValueError(f"failed-trial packets are not active: {missing}")
    for packet_id, entry in by_id.items():
        if packet_id not in targets:
            continue
        manifest = ROOT / entry["manifest_path"]
        if sha256_file(manifest) != entry["manifest_sha256"]:
            raise ValueError(f"active manifest hash mismatch for {packet_id}")

    registry["active_packets"] = [
        item for item in active if item["packet_id"] not in targets
    ]
    registry_path.write_text(json.dumps(registry, indent=2, sort_keys=True) + "\n")
    result = {
        "protocol": "admet-evidence-packet-confirmation-rollback-v5.0",
        "rolled_back_at": datetime.now(timezone.utc).isoformat(),
        "registry_path": str(registry_path.relative_to(ROOT)),
        "registry_sha256_before": before_sha,
        "registry_sha256_after": sha256_file(registry_path),
        "confirmation_artifacts": confirmations,
        "rolled_back_packets": [
            {"packet_id": packet_id, **targets[packet_id]}
            for packet_id in sorted(targets)
        ],
        "packet_files_deleted": False,
        "historical_experiment_artifacts_retained": True,
        "test_labels_loaded": False,
    }
    output = output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--confirmation", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(
        json.dumps(
            rollback(args.registry, args.confirmation, args.output), sort_keys=True
        )
    )


if __name__ == "__main__":
    main()
