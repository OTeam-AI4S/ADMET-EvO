"""Train the preregistered CYP2C9 focal heads in the isolated Torch env."""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import time
from pathlib import Path

import numpy as np
from scipy.stats import rankdata
from sklearn.metrics import average_precision_score


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def percentile_ranks(values: np.ndarray) -> np.ndarray:
    return rankdata(values, method="average") / len(values)


def evaluate(root: Path) -> tuple[dict, dict[str, np.ndarray]]:
    # Torch and the MiniMol head implementation intentionally live only in the
    # isolated MiniMol environment.  Keeping these imports inside execution
    # lets the main ADMET environment inspect and test the protocol helpers.
    import torch

    from scripts.evaluate_minimol_rank_heads import LossConfig, _train_head

    # Tiny 512-d heads are much slower when the host default (72/144 threads)
    # launches a large thread pool for every small matrix multiply.  One thread
    # preserves the frozen optimization protocol and deterministic seeds while
    # removing that scheduling overhead.
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)

    prereg_path = root / "preregistration.json"
    manifest_path = root / "input_manifest.json"
    input_path = root / "confirmation_inputs.npz"
    prereg = json.loads(prereg_path.read_text(encoding="utf-8"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("official_test_loaded") is not False:
        raise RuntimeError("Confirmation input is not explicitly validation-only")
    if manifest.get("test_labels_used") is not False:
        raise RuntimeError("Confirmation manifest reports test-label use")
    if manifest.get("input_sha256") != _sha256(input_path):
        raise RuntimeError("Confirmation input hash mismatch")
    if manifest.get("preregistration_sha256") != prereg.get("preregistration_sha256"):
        raise RuntimeError("Confirmation input/preregistration mismatch")

    config = prereg["candidate"]["focal_minimol"]
    if config != {
        "batch_size": 32,
        "epochs": 100,
        "focal_gamma": 2.0,
        "frozen_embeddings": True,
        "head_seed_formula": "outer_seed*10000+500+member",
        "learning_rate": 0.0003,
        "loss": "positive-weighted focal BCE",
        "members": 16,
        "weight_decay": 0.0001,
    }:
        raise RuntimeError("Frozen focal-head configuration changed")
    if prereg["candidate"]["focal_minimol_weight"] != 0.2:
        raise RuntimeError("Frozen fusion weight changed")

    inputs = np.load(input_path, allow_pickle=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loss_config = LossConfig("focal_2", positive_weight=True, focal_gamma=2.0)
    scores = {"lightgbm": [], "focal_minimol": [], "fixed_rank_fusion": []}
    predictions: dict[str, np.ndarray] = {}
    records = {}
    started = time.monotonic()
    for seed in prereg["fresh_confirmation_seeds"]:
        prefix = f"seed_{seed}"
        x_train = inputs[f"{prefix}_x_train"]
        y_train = inputs[f"{prefix}_y_train"]
        x_valid = inputs[f"{prefix}_x_valid"]
        y_valid = inputs[f"{prefix}_y_valid"]
        baseline = inputs[f"{prefix}_lightgbm"]
        logits = []
        for member in range(config["members"]):
            member_seed = seed * 10000 + 500 + member
            logits.append(
                _train_head(
                    x_train,
                    y_train,
                    x_valid,
                    loss_config,
                    member_seed,
                    config["epochs"],
                    config["batch_size"],
                    device,
                )
            )
        focal = np.mean(logits, axis=0)
        fused = 0.8 * percentile_ranks(baseline) + 0.2 * percentile_ranks(focal)
        seed_scores = {
            "lightgbm": float(average_precision_score(y_valid, baseline)),
            "focal_minimol": float(average_precision_score(y_valid, focal)),
            "fixed_rank_fusion": float(average_precision_score(y_valid, fused)),
        }
        for name, value in seed_scores.items():
            scores[name].append(value)
        predictions[f"{prefix}_lightgbm"] = baseline
        predictions[f"{prefix}_focal_minimol"] = focal
        predictions[f"{prefix}_fixed_rank_fusion"] = fused
        records[str(seed)] = seed_scores
        print(json.dumps({"seed": seed, "scores": seed_scores}, sort_keys=True), flush=True)

    summaries = {
        name: {
            "scores": values,
            "mean": statistics.mean(values),
            "std": statistics.pstdev(values),
        }
        for name, values in scores.items()
    }
    improvements = [
        candidate > baseline
        for candidate, baseline in zip(
            scores["fixed_rank_fusion"], scores["lightgbm"]
        )
    ]
    gate = prereg["promotion_gate"]
    passed = (
        summaries["fixed_rank_fusion"]["mean"] > summaries["lightgbm"]["mean"]
        and sum(improvements) >= gate["minimum_paired_seed_improvements"]
    )
    result = {
        "task": prereg["task"],
        "metric": prereg["metric"],
        "decision": "retain_for_new_freeze" if passed else "reject",
        "summaries": summaries,
        "paired_seed_improvements": sum(improvements),
        "records": records,
        "preregistration_sha256": prereg["preregistration_sha256"],
        "input_sha256": manifest["input_sha256"],
        "device": str(device),
        "torch_threads": {"intra_op": 1, "inter_op": 1},
        "elapsed_seconds": time.monotonic() - started,
        "official_test_loaded": False,
        "test_labels_used": False,
        "confirmation_seeds_used_for_candidate_selection": False,
    }
    return result, predictions


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root", type=Path, default=Path("runs/v25-cyp2c9-focal-confirmation")
    )
    args = parser.parse_args()
    result, predictions = evaluate(args.root)
    result_path = args.root / "result.json"
    result_path.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    prediction_path = args.root / "predictions.npz"
    np.savez_compressed(prediction_path, **predictions)
    result["result_sha256"] = _sha256(result_path)
    result["predictions_sha256"] = _sha256(prediction_path)
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
