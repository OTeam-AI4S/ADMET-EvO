"""Run one Uni-Mol2 scale/seed ablation on a frozen outer validation split."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from run_unimol2_pilot import _fit_unimol2
from tdc.benchmark_group import admet_group

from admet_agent.runner import _columns
from admet_agent.selection import score
from admet_agent.tasks import resolve_task

TASKS = ("DILI", "Lipophilicity_AstraZeneca", "hERG")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", required=True, choices=TASKS)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--model-size", default="164m")
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--inner-folds", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument(
        "--output-dir", type=Path, default=Path("runs/unimol2-scale-ablation")
    )
    args = parser.parse_args()

    task = resolve_task(args.task)
    train, valid = admet_group(path=str(args.data_dir)).get_train_valid_split(
        benchmark=task.name, split_type="default", seed=args.seed
    )
    smiles_col, y_col = _columns(train)
    run_dir = (
        args.output_dir
        / args.model_size
        / task.name
        / f"lr_{args.learning_rate:g}"
        / f"seed_{args.seed}"
    )
    oof, predictions, elapsed = _fit_unimol2(
        task,
        train,
        valid,
        smiles_col,
        y_col,
        run_dir / "model",
        args.epochs,
        args.inner_folds,
        args.batch_size,
        model_size=args.model_size,
        learning_rate=args.learning_rate,
    )
    value = score(valid[y_col].to_numpy(), predictions, task.metric)
    run_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        run_dir / "predictions.npz", oof=oof, validation=predictions
    )
    payload = {
        "task": task.name,
        "metric": task.metric,
        "seed": args.seed,
        "model_size": args.model_size,
        "learning_rate": args.learning_rate,
        "epochs": args.epochs,
        "inner_folds": args.inner_folds,
        "score": value,
        "elapsed_seconds": elapsed,
        "test_labels_used": False,
    }
    (run_dir / "results.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
