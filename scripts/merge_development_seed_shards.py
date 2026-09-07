#!/usr/bin/env python3
"""Merge seeds 2-3 from a live development log with a seeds 4-5 shard."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from run_all22_optimal_recipe_adaptive_data import _ratio_selection


EXPECTED_SEEDS = (2, 3, 4, 5)
EXPECTED_RATIOS = (0.5, 1.0, 2.0, 4.0, 8.0)


def _rows_from_log(path: Path, task: str) -> list[dict]:
    rows: list[dict] = []
    for fragment in path.read_text(encoding="utf-8", errors="replace").splitlines():
        start = fragment.find('{"task":')
        if start < 0:
            continue
        try:
            item = json.loads(fragment[start:])
        except json.JSONDecodeError:
            continue
        if (
            item.get("task") == task
            and item.get("seed") in (2, 3)
            and "ratio" in item
            and "oriented_delta" in item
        ):
            item.pop("task", None)
            rows.append(item)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", required=True)
    parser.add_argument("--base-log", type=Path, required=True)
    parser.add_argument("--shard-json", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    shard = json.loads(args.shard_json.read_text(encoding="utf-8"))
    if shard.get("task") != args.task:
        raise RuntimeError("Shard task does not match requested task")

    rows = _rows_from_log(args.base_log, args.task)
    rows.extend(shard.get("per_seed_ratio_model", []))
    keyed = {(int(row["seed"]), float(row["ratio"])): row for row in rows}
    expected = {(seed, ratio) for seed in EXPECTED_SEEDS for ratio in EXPECTED_RATIOS}
    missing = sorted(expected - set(keyed))
    if missing:
        raise RuntimeError(f"Missing seed/ratio rows: {missing}")

    merged_rows = [keyed[key] for key in sorted(expected)]
    summaries, selection = _ratio_selection(merged_rows)
    payload = {
        **shard,
        "status": selection["status"],
        "seeds": list(EXPECTED_SEEDS),
        "ratios": list(EXPECTED_RATIOS),
        "per_seed_ratio_model": merged_rows,
        "ratio_summaries": summaries,
        "selection": selection,
        "seed_shard_merge": {
            "base_log": str(args.base_log.resolve()),
            "shard_json": str(args.shard_json.resolve()),
            "base_seeds": [2, 3],
            "shard_seeds": [4, 5],
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(args.output)
    print(json.dumps({"task": args.task, "rows": len(merged_rows), "output": str(args.output)}))


if __name__ == "__main__":
    main()
