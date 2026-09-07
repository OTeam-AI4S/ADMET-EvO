#!/usr/bin/env python3
"""Merge complete development seed-shard JSON files into one audited artifact."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from run_all22_optimal_recipe_adaptive_data import _ratio_selection


EXPECTED_SEEDS = (2, 3, 4, 5)
EXPECTED_RATIOS = (0.5, 1.0, 2.0, 4.0, 8.0)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", required=True)
    parser.add_argument("--shards", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    payloads = [json.loads(path.read_text(encoding="utf-8")) for path in args.shards]
    if any(payload.get("task") != args.task for payload in payloads):
        raise RuntimeError("At least one shard task does not match requested task")
    hashes = {payload.get("frozen_mapping_sha256") for payload in payloads}
    protocols = {payload.get("protocol") for payload in payloads}
    if len(hashes) != 1 or len(protocols) != 1:
        raise RuntimeError("Shard protocol or frozen mapping differs")

    rows = [
        row
        for payload in payloads
        for row in payload.get("per_seed_ratio_model", [])
    ]
    keyed = {(int(row["seed"]), float(row["ratio"])): row for row in rows}
    expected = {(seed, ratio) for seed in EXPECTED_SEEDS for ratio in EXPECTED_RATIOS}
    missing = sorted(expected - set(keyed))
    extras = sorted(set(keyed) - expected)
    if missing or extras or len(rows) != len(keyed):
        raise RuntimeError(
            f"Invalid shard coverage: missing={missing}, extras={extras}, "
            f"rows={len(rows)}, unique={len(keyed)}"
        )

    merged_rows = [keyed[key] for key in sorted(expected)]
    summaries, selection = _ratio_selection(merged_rows)
    payload = {
        **payloads[0],
        "status": selection["status"],
        "seeds": list(EXPECTED_SEEDS),
        "ratios": list(EXPECTED_RATIOS),
        "per_seed_ratio_model": merged_rows,
        "ratio_summaries": summaries,
        "selection": selection,
        "seed_shard_merge": {
            "shards": [str(path.resolve()) for path in args.shards],
            "coverage": {str(path): payload.get("seeds") for path, payload in zip(args.shards, payloads)},
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(args.output)
    print(json.dumps({"task": args.task, "rows": len(merged_rows), "output": str(args.output)}))


if __name__ == "__main__":
    main()
