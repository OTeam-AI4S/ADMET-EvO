#!/usr/bin/env python3
"""Stop a task process once its log contains complete seeds 2-3 coverage."""

from __future__ import annotations

import argparse
import json
import os
import signal
import time
from pathlib import Path


EXPECTED = {(seed, ratio) for seed in (2, 3) for ratio in (0.5, 1.0, 2.0, 4.0, 8.0)}


def completed_keys(path: Path, task: str) -> set[tuple[int, float]]:
    keys: set[tuple[int, float]] = set()
    try:
        fragments = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return keys
    for fragment in fragments:
        start = fragment.find('{"task":')
        if start < 0:
            continue
        try:
            row = json.loads(fragment[start:])
        except json.JSONDecodeError:
            continue
        if row.get("task") == task and "seed" in row and "ratio" in row:
            key = (int(row["seed"]), float(row["ratio"]))
            if key in EXPECTED:
                keys.add(key)
    return keys


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", required=True)
    parser.add_argument("--log", type=Path, required=True)
    parser.add_argument("--pid", type=int, required=True)
    parser.add_argument("--poll-seconds", type=float, default=20.0)
    args = parser.parse_args()

    while completed_keys(args.log, args.task) != EXPECTED:
        time.sleep(args.poll_seconds)
    try:
        os.kill(args.pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    print(json.dumps({"task": args.task, "stopped_pid": args.pid, "coverage_rows": 10}))


if __name__ == "__main__":
    main()
