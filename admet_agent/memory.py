"""Append-only experiment memory grounded in actual run artifacts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable


class ExperimentMemory:
    def __init__(self, path: Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def append(self, record: dict[str, Any]) -> None:
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, sort_keys=True) + "\n")

    def records(self, task: str | None = None) -> Iterable[dict[str, Any]]:
        if not self.path.exists():
            return []
        rows = []
        with self.path.open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                row = json.loads(line)
                if task is None or row.get("task", "").lower() == task.lower():
                    rows.append(row)
        return rows
