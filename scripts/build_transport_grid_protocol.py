"""Freeze the minimal prospective cross-task transport protocol."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from admet_agent.source_snapshot import production_source_manifest
from admet_agent.transport_protocol import build_transport_protocol


def main() -> None:
    root = Path(__file__).parents[1]
    output = root / "runs/v5-transport-grid/protocol.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    runner_path = root / "scripts/run_transport_grid.py"
    protocol = build_transport_protocol(
        production_source_manifest()["source_sha256"],
        hashlib.sha256(runner_path.read_bytes()).hexdigest(),
    )
    output.write_text(
        json.dumps(protocol, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
