#!/usr/bin/env python3
"""Reproduce the TDC Bioavailability Chemprop-RDKit benchmark on a compute node."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata as metadata
import json
import os
import platform
import shutil
import socket
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score
from tdc.utils.split import create_scaffold_split


ROOT = Path(__file__).resolve().parents[1]
TRAIN_VAL = ROOT / "data" / "admet_group" / "bioavailability_ma" / "train_val.csv"
TEST = ROOT / "data" / "admet_group" / "bioavailability_ma" / "test.csv"
DEFAULT_OUTPUT = ROOT / "runs" / "bioavailability-chemprop-rdkit-tdc5"
CHEMPROP = ROOT / ".venvs" / "chemprop" / "bin" / "chemprop_train"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--ensemble-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def version(package: str) -> str | None:
    try:
        return metadata.version(package)
    except metadata.PackageNotFoundError:
        return None


def prepare_split(train_val: pd.DataFrame, test: pd.DataFrame, seed_dir: Path, seed: int) -> tuple[Path, Path, Path]:
    split = create_scaffold_split(train_val, seed, frac=[0.875, 0.125, 0.0], entity="Drug")
    train = split["train"][["Drug", "Y"]].rename(columns={"Drug": "smiles"})
    valid = split["valid"][["Drug", "Y"]].rename(columns={"Drug": "smiles"})
    heldout = test[["Drug", "Y"]].rename(columns={"Drug": "smiles"})
    paths = seed_dir / "train.csv", seed_dir / "valid.csv", seed_dir / "test.csv"
    for frame, path in zip((train, valid, heldout), paths, strict=True):
        frame.to_csv(path, index=False)
    return paths


def load_score(prediction_path: Path, test_path: Path) -> float:
    predictions = pd.read_csv(prediction_path)
    test = pd.read_csv(test_path)
    prediction_columns = [column for column in predictions.columns if column not in {"smiles"}]
    if "Y" in predictions.columns:
        prediction_column = "Y"
    elif len(prediction_columns) == 1:
        prediction_column = prediction_columns[0]
    else:
        raise ValueError(f"Cannot identify prediction column in {prediction_path}: {prediction_columns}")
    if len(predictions) != len(test):
        raise ValueError("Prediction count does not match the fixed TDC test set")
    return float(roc_auc_score(test["Y"].to_numpy(), predictions[prediction_column].to_numpy()))


def main() -> None:
    args = parse_args()
    if socket.gethostname().startswith("control-"):
        raise RuntimeError("Training is forbidden on the storage/control node; submit through Slurm")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA compute node is required")
    if not CHEMPROP.is_file():
        raise FileNotFoundError(CHEMPROP)

    train_val = pd.read_csv(TRAIN_VAL)
    test = pd.read_csv(TEST)
    if set(train_val.columns) != {"Drug_ID", "Drug", "Y"} or set(test.columns) != {"Drug_ID", "Drug", "Y"}:
        raise ValueError("Unexpected TDC Bioavailability input schema")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    scores: dict[str, float] = {}
    commands: dict[str, list[str]] = {}
    for seed in args.seeds:
        seed_dir = args.output_dir / f"seed_{seed}"
        if seed_dir.exists() and args.force:
            shutil.rmtree(seed_dir)
        seed_dir.mkdir(parents=True, exist_ok=True)
        train_path, valid_path, test_path = prepare_split(train_val, test, seed_dir, seed)
        model_dir = seed_dir / "model"
        prediction_path = model_dir / "test_preds.csv"
        command = [
            str(CHEMPROP),
            "--data_path", str(train_path),
            "--separate_val_path", str(valid_path),
            "--separate_test_path", str(test_path),
            "--smiles_columns", "smiles",
            "--target_columns", "Y",
            "--dataset_type", "classification",
            "--metric", "auc",
            "--features_generator", "rdkit_2d_normalized",
            "--no_features_scaling",
            "--epochs", str(args.epochs),
            "--ensemble_size", str(args.ensemble_size),
            "--seed", str(seed),
            "--pytorch_seed", str(seed),
            "--num_workers", str(args.num_workers),
            "--save_dir", str(model_dir),
            "--save_preds",
            "--quiet",
        ]
        commands[str(seed)] = command
        if not prediction_path.is_file():
            subprocess.run(command, cwd=ROOT, check=True)
        scores[str(seed)] = load_score(prediction_path, test_path)
        print(f"seed={seed} test_auroc={scores[str(seed)]:.9f}", flush=True)

    values = np.array([scores[str(seed)] for seed in args.seeds], dtype=float)
    report = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "task": "Bioavailability_Ma",
        "model": "Chemprop-RDKit",
        "protocol": {
            "tdc_split": "scaffold; train/validation fractions 0.875/0.125 from frozen train_val.csv",
            "test": "frozen TDC test.csv; never used for fitting or model selection",
            "seeds": args.seeds,
            "features_generator": "rdkit_2d_normalized",
            "features_scaling": False,
            "epochs": args.epochs,
            "ensemble_size_per_seed": args.ensemble_size,
            "chemprop_other_hyperparameters": "Chemprop 1.6.1 defaults",
        },
        "scores": scores,
        "mean_auroc": float(values.mean()),
        "std_auroc_population": float(values.std(ddof=0)),
        "mean_rounded_tdc": round(float(values.mean()), 3),
        "std_rounded_tdc": round(float(values.std(ddof=0)), 3),
        "n_test": int(len(test)),
        "test_positive_fraction": float(test["Y"].mean()),
        "input_sha256": {str(TRAIN_VAL): sha256(TRAIN_VAL), str(TEST): sha256(TEST)},
        "runtime": {
            "host": socket.gethostname(),
            "platform": platform.platform(),
            "python": sys.version,
            "executable": sys.executable,
            "cuda_device": torch.cuda.get_device_name(0),
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
            "versions": {
                package: version(package)
                for package in ["chemprop", "descriptastorus", "numpy", "pandas", "rdkit", "scikit-learn", "torch", "PyTDC"]
            },
        },
        "commands": commands,
    }
    report_path = args.output_dir / "summary.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"summary": str(report_path), "mean_auroc": report["mean_auroc"], "std": report["std_auroc_population"]}, indent=2))


if __name__ == "__main__":
    main()
