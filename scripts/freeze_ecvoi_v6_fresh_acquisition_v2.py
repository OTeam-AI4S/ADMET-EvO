"""Bind replacement downloads and all raw cache bytes before cohort modeling."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    registry_path = Path("reports/ecvoi_v6_fresh_source_registry_v2.json")
    registry = json.loads(registry_path.read_text())
    parent_path = Path("reports/ecvoi_v6_fresh_source_acquisition_v1.json")
    parent = json.loads(parent_path.read_text())
    cache = Path("external/ecvoi_v6_tdc_cache")
    files = {
        str(path): {"sha256": _sha256(path), "bytes": path.stat().st_size}
        for path in sorted(cache.iterdir()) if path.is_file()
    }
    payload = {
        "version": "ecvoi-v6-fresh-source-acquisition-v2",
        "frozen_at": "2026-08-23",
        "source_registry_sha256": registry["sha256"],
        "parent_acquisition_file_sha256": _sha256(parent_path),
        "raw_files": files,
        "files": len(files),
        "bytes": sum(item["bytes"] for item in files.values()),
        "model_actions_run": False,
        "action_outcomes_computed": False,
        "protected_labels_loaded": False,
        "test_or_leaderboard_feedback_used": False,
        "tdc_19_unlocked": False,
    }
    encoded = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    payload["sha256"] = hashlib.sha256(encoded.encode()).hexdigest()
    output = Path("reports/ecvoi_v6_fresh_source_acquisition_v2.json")
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(payload["sha256"])


if __name__ == "__main__":
    main()
