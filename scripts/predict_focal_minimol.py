"""Isolated inference-time trainer for the frozen 16-head focal MiniMol expert."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from scripts.evaluate_minimol_rank_heads import LossConfig, _train_head


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run(input_path: Path, manifest_path: Path) -> tuple[np.ndarray, dict]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected = {
        "embedding_width": 512,
        "members": 16,
        "epochs": 100,
        "batch_size": 32,
        "loss": "positive-weighted focal BCE",
        "focal_gamma": 2.0,
        "head_seed_formula": "outer_seed*10000+500+member",
        "fit_labels_present": True,
        "predict_labels_present": False,
    }
    if any(manifest.get(key) != value for key, value in expected.items()):
        raise RuntimeError("Frozen focal MiniMol manifest changed")
    if manifest.get("input_sha256") != _sha256(input_path):
        raise RuntimeError("Frozen focal MiniMol input hash mismatch")
    values = np.load(input_path, allow_pickle=False)
    if set(values.files) != {"x_fit", "y_fit", "x_predict"}:
        raise RuntimeError("Frozen focal MiniMol input fields changed")
    x_fit = values["x_fit"]
    y_fit = values["y_fit"]
    x_predict = values["x_predict"]
    if x_fit.shape != (manifest["fit_rows"], 512):
        raise RuntimeError("Frozen focal MiniMol fit shape mismatch")
    if y_fit.shape != (manifest["fit_rows"],):
        raise RuntimeError("Frozen focal MiniMol label shape mismatch")
    if x_predict.shape != (manifest["predict_rows"], 512):
        raise RuntimeError("Frozen focal MiniMol prediction shape mismatch")
    if not all(np.isfinite(array).all() for array in (x_fit, y_fit, x_predict)):
        raise RuntimeError("Frozen focal MiniMol input contains non-finite values")

    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = LossConfig("focal_2", positive_weight=True, focal_gamma=2.0)
    predictions = []
    seed = int(manifest["seed"])
    for member in range(16):
        predictions.append(
            _train_head(
                x_fit,
                y_fit,
                x_predict,
                config,
                seed * 10000 + 500 + member,
                100,
                32,
                device,
            )
        )
    output = np.mean(predictions, axis=0)
    report = {
        "task": manifest["task"],
        "seed": seed,
        "device": str(device),
        "members": 16,
        "epochs": 100,
        "torch_threads": {"intra_op": 1, "inter_op": 1},
        "fit_labels_present": True,
        "predict_labels_present": False,
        "input_sha256": manifest["input_sha256"],
    }
    return np.asarray(output, dtype=np.float32), report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    predictions, report = run(args.input, args.manifest)
    np.save(args.output, predictions, allow_pickle=False)
    report["predictions_sha256"] = _sha256(args.output)
    args.report.write_text(
        json.dumps(report, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
