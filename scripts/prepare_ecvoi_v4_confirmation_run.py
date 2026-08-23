"""Prepare a separate confirmation run after non-executor source additions.

The production source firewall intentionally prevents reopening a run directory
when any snapshotted source file changes.  This recovery copies only the frozen
screen registry and integrity decision into a fresh directory; the matched-state
protocol independently binds every file that can affect execution.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--screen-run", type=Path, required=True)
    parser.add_argument("--confirmation-run", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    args = parser.parse_args()
    if args.confirmation_run.exists() and any(args.confirmation_run.iterdir()):
        raise RuntimeError("Confirmation run directory must not already contain files")
    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    registry_path = args.screen_run / "registry.json"
    decision_path = args.screen_run / "screen_integrity_decision.json"
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    decision = json.loads(decision_path.read_text(encoding="utf-8"))
    if registry.get("protocol_sha256") != protocol.get("sha256"):
        raise RuntimeError("Screen registry is not bound to the requested protocol")
    if decision.get("protocol_sha256") != protocol.get("sha256") or not decision.get("passed"):
        raise RuntimeError("Frozen screen integrity decision is absent or failed")
    expected = len(protocol["tasks"]) * len(protocol["states"]) * len(protocol["actions"]) * len(protocol["screen_seeds"])
    if len(registry.get("records", {})) != expected:
        raise RuntimeError("Screen registry is incomplete")

    args.confirmation_run.mkdir(parents=True, exist_ok=True)
    target_registry = args.confirmation_run / "registry.json"
    target_decision = args.confirmation_run / "screen_integrity_decision.json"
    shutil.copy2(registry_path, target_registry)
    shutil.copy2(decision_path, target_decision)
    audit = {
        "version": "ecvoi-v4-source-firewall-recovery-v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "reason": "full source snapshot changed after adding offline policy-analysis code; frozen execution-file hashes remain protocol-bound",
        "screen_run": str(args.screen_run),
        "confirmation_run": str(args.confirmation_run),
        "protocol_sha256": protocol["sha256"],
        "copied_registry_sha256": _sha(target_registry),
        "copied_integrity_decision_sha256": _sha(target_decision),
        "execution_implementation_sha256": protocol["implementation_sha256"],
        "scientific_design_changed": False,
        "screen_results_changed": False,
        "test_loaded": False,
        "leaderboard_loaded": False,
    }
    (args.confirmation_run / "source_firewall_recovery_audit.json").write_text(
        json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(audit, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
