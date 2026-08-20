"""Merge independently executed Uni-Mol2 pilot seeds without changing scores."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def _read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--primary", type=Path, required=True)
    parser.add_argument("--shard", type=Path, action="append", default=[])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    result = _read(args.primary)
    for path in args.shard:
        shard = _read(path)
        if (shard["task"], shard["metric"]) != (result["task"], result["metric"]):
            raise ValueError(f"Incompatible shard: {path}")
        for seed, seed_result in shard["seeds"].items():
            if seed in result["seeds"]:
                raise ValueError(f"Duplicate seed {seed} in {path}")
            result["seeds"][seed] = seed_result

    result["seeds"] = dict(sorted(result["seeds"].items(), key=lambda item: int(item[0])))
    if len(result["seeds"]) != 5:
        raise ValueError(f"Expected five seeds, found {len(result['seeds'])}")
    names = sorted(
        set.intersection(*(set(value["scores"]) for value in result["seeds"].values()))
    )
    result["summary"] = {
        name: {
            "mean": float(np.mean([value["scores"][name] for value in result["seeds"].values()])),
            "std": float(np.std([value["scores"][name] for value in result["seeds"].values()])),
        }
        for name in names
    }
    direction = -1 if result["metric"] == "mae" else 1
    result["champion"] = max(
        names, key=lambda name: direction * result["summary"][name]["mean"]
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
