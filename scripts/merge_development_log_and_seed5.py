#!/usr/bin/env python3
"""Merge seeds 2-3 and seed 4 logs with a complete seed 5 JSON shard."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from run_all22_optimal_recipe_adaptive_data import _ratio_selection


RATIOS = (0.5, 1.0, 2.0, 4.0, 8.0)


def rows_from_log(path: Path, task: str, seeds: set[int]) -> list[dict]:
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
            and int(item.get("seed", -1)) in seeds
            and "ratio" in item
            and "oriented_delta" in item
        ):
            item.pop("task", None)
            rows.append(item)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", required=True)
    parser.add_argument("--seed23-log", type=Path, required=True)
    parser.add_argument("--seed4-log", type=Path, required=True)
    parser.add_argument("--seed5-json", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    seed5 = json.loads(args.seed5_json.read_text(encoding="utf-8"))
    if seed5.get("task") != args.task or seed5.get("seeds") != [5]:
        raise RuntimeError("Invalid seed 5 shard")
    rows = rows_from_log(args.seed23_log, args.task, {2, 3})
    rows.extend(rows_from_log(args.seed4_log, args.task, {4}))
    rows.extend(seed5.get("per_seed_ratio_model", []))
    keyed = {(int(row["seed"]), float(row["ratio"])): row for row in rows}
    expected = {(seed, ratio) for seed in (2, 3, 4, 5) for ratio in RATIOS}
    if set(keyed) != expected:
        raise RuntimeError(f"Missing rows: {sorted(expected - set(keyed))}")

    merged_rows = [keyed[key] for key in sorted(expected)]
    summaries, selection = _ratio_selection(merged_rows)
    payload = {
        **seed5,
        "status": selection["status"],
        "seeds": [2, 3, 4, 5],
        "ratios": list(RATIOS),
        "per_seed_ratio_model": merged_rows,
        "ratio_summaries": summaries,
        "selection": selection,
        "seed_shard_merge": {
            "seed23_log": str(args.seed23_log.resolve()),
            "seed4_log": str(args.seed4_log.resolve()),
            "seed5_json": str(args.seed5_json.resolve()),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(args.output)
    print(json.dumps({"task": args.task, "rows": len(merged_rows)}))


if __name__ == "__main__":
    main()
