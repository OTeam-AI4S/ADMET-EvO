#!/usr/bin/env python
"""Materialize the 22-task external-data and literature registry."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from admet_agent.task_data_registry import build_task_data_registry


def build(output: Path) -> dict:
    registry = build_task_data_registry()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(registry, indent=2, ensure_ascii=False) + "\n")
    return registry


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("reports/tdc22_data_source_registry_v1.json"),
    )
    args = parser.parse_args()
    registry = build(args.output)
    ready = [row["task"] for row in registry["tasks"] if row["local_direct_ready"]]
    print(json.dumps({"output": str(args.output), "tasks": len(registry["tasks"]), "local_direct_ready": ready}, indent=2))


if __name__ == "__main__":
    main()
