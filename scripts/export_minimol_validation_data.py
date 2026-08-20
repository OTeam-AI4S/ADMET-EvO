"""Export train/validation-only TDC splits for isolated MiniMol featurization."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from tdc.benchmark_group import admet_group

from admet_agent.runner import _columns
from admet_agent.tasks import resolve_task


def build_export(group: object, task_name: str, seeds: list[int]) -> tuple[dict, dict]:
    task = resolve_task(task_name)
    smiles: list[str] = []
    index: dict[str, int] = {}
    splits = {}

    def ids(values: list[str]) -> list[int]:
        result = []
        for value in values:
            value = str(value)
            if value not in index:
                index[value] = len(smiles)
                smiles.append(value)
            result.append(index[value])
        return result

    for seed in seeds:
        train, valid = group.get_train_valid_split(
            benchmark=task.name, split_type="default", seed=seed
        )
        smiles_col, y_col = _columns(train)
        splits[str(seed)] = {
            "train_ids": ids(train[smiles_col].astype(str).tolist()),
            "train_y": [float(value) for value in train[y_col]],
            "valid_ids": ids(valid[smiles_col].astype(str).tolist()),
            "valid_y": [float(value) for value in valid[y_col]],
        }
    smiles_payload = {
        "task": task.name,
        "smiles": smiles,
        "official_test_loaded": False,
        "contains_labels": False,
    }
    split_payload = {
        "task": task.name,
        "metric": task.metric,
        "seeds": seeds,
        "split_type": "TDC default train/validation scaffold split",
        "splits": splits,
        "official_test_loaded": False,
        "smiles_sha256": hashlib.sha256(
            json.dumps(smiles_payload, sort_keys=True).encode()
        ).hexdigest(),
    }
    return smiles_payload, split_payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", default="CYP2C9_Substrate_CarbonMangels")
    parser.add_argument("--seeds", default="1,2,3,4,5")
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument(
        "--output-dir", type=Path, default=Path("runs/v24-minimol-validation")
    )
    args = parser.parse_args()
    seeds = [int(value) for value in args.seeds.split(",")]
    group = admet_group(path=str(args.data_dir))
    smiles, splits = build_export(group, args.task, seeds)
    output = args.output_dir / resolve_task(args.task).name
    output.mkdir(parents=True, exist_ok=True)
    (output / "smiles.json").write_text(
        json.dumps(smiles, indent=2, sort_keys=True), encoding="utf-8"
    )
    (output / "splits.json").write_text(
        json.dumps(splits, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "task": splits["task"],
                "unique_smiles": len(smiles["smiles"]),
                "seeds": seeds,
                "official_test_loaded": False,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
