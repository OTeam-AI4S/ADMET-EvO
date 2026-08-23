"""Build repeated discovery-only HardnessCards without fitting predictive models."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import pandas as pd
from tdc.benchmark_group import admet_group

from admet_agent.hardness import profile_hardness
from admet_agent.tasks import TASKS, resolve_task


def _frame_sha256(frame: pd.DataFrame, smiles_col: str, target_col: str) -> str:
    selected = frame[[smiles_col, target_col]].copy()
    values = pd.util.hash_pandas_object(selected, index=True).to_numpy().tobytes()
    return hashlib.sha256(values).hexdigest()


def _columns(frame: pd.DataFrame) -> tuple[str, str]:
    smiles = "Drug" if "Drug" in frame else "SMILES"
    if smiles not in frame or "Y" not in frame:
        raise ValueError(f"Unexpected discovery split columns: {list(frame.columns)}")
    return smiles, "Y"


def build(
    group: Any,
    task_names: Iterable[str],
    seeds: Iterable[int],
    *,
    sample_limit: int = 2000,
) -> dict:
    names = list(dict.fromkeys(task_names))
    seed_values = list(dict.fromkeys(int(value) for value in seeds))
    if not names or not seed_values:
        raise ValueError("At least one task and profiling seed are required")
    if any(value < 0 for value in seed_values):
        raise ValueError("Profiling seeds must be non-negative")
    cards = []
    splits = []
    for task_name in names:
        spec = resolve_task(task_name)
        for seed in seed_values:
            train, valid = group.get_train_valid_split(
                benchmark=spec.name,
                split_type="default",
                seed=seed,
            )
            smiles_col, target_col = _columns(train)
            valid_smiles_col, valid_target_col = _columns(valid)
            if (smiles_col, target_col) != (valid_smiles_col, valid_target_col):
                raise ValueError("Train and validation schemas disagree")
            card = profile_hardness(
                train,
                valid,
                spec,
                smiles_col,
                target_col,
                sample_limit=sample_limit,
                seed=seed,
            ).to_dict()
            cards.append({"seed": seed, **card})
            splits.append(
                {
                    "task": spec.name,
                    "seed": seed,
                    "split_type": "TDC default train/validation",
                    "train_rows": len(train),
                    "valid_rows": len(valid),
                    "train_sha256": _frame_sha256(train, smiles_col, target_col),
                    "valid_sha256": _frame_sha256(valid, smiles_col, target_col),
                }
            )
    cards.sort(key=lambda row: (row["task"], row["seed"]))
    splits.sort(key=lambda row: (row["task"], row["seed"]))
    return {
        "protocol": "tdc22_discovery_only_repeated_hardness_v1",
        "discovery_only": True,
        "test_loaded": False,
        "model_fits": 0,
        "sample_limit": sample_limit,
        "tasks": names,
        "seeds": seed_values,
        "cards": cards,
        "split_manifest": splits,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--task", action="append")
    parser.add_argument("--seeds", default="1,2,3,4,5")
    parser.add_argument("--sample-limit", type=int, default=2000)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("reports/tdc22_repeated_hardness_cards_v1.json"),
    )
    args = parser.parse_args()
    seeds = [int(value) for value in args.seeds.split(",") if value.strip()]
    task_names = args.task or [spec.name for spec in TASKS.values()]
    payload = build(
        admet_group(path=str(args.data_dir)),
        task_names,
        seeds,
        sample_limit=args.sample_limit,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(
        json.dumps(
            {
                "output": str(args.output.resolve()),
                "tasks": len(task_names),
                "seeds": seeds,
                "model_fits": 0,
                "test_loaded": False,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
