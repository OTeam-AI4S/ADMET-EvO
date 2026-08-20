"""Fresh-seed confirmation of one locked MiniMol/chemical kernel recipe."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

import numpy as np

try:
    from scripts.evaluate_minimol_scaffold_kernels import (
        _fit_predict,
        _score,
        build_kernels,
    )
except ModuleNotFoundError:
    from evaluate_minimol_scaffold_kernels import _fit_predict, _score, build_kernels


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--discovery-root",
        type=Path,
        default=Path("runs/v24-minimol-validation"),
    )
    parser.add_argument(
        "--confirmation-root",
        type=Path,
        default=Path("runs/v24-minimol-kernel-confirmation"),
    )
    parser.add_argument("--task", default="Bioavailability_Ma")
    parser.add_argument("--kernel", default="fusion_chem_0.50")
    parser.add_argument("--c", type=float, default=10.0)
    args = parser.parse_args()

    discovery = args.discovery_root / args.task
    confirmation = args.confirmation_root / args.task
    source_smiles = json.loads(
        (discovery / "smiles.json").read_text(encoding="utf-8")
    )
    target_smiles = json.loads(
        (confirmation / "smiles.json").read_text(encoding="utf-8")
    )
    splits = json.loads((confirmation / "splits.json").read_text(encoding="utf-8"))
    if source_smiles.get("official_test_loaded") is not False:
        raise RuntimeError("Discovery SMILES manifest contains test data")
    if target_smiles.get("official_test_loaded") is not False:
        raise RuntimeError("Confirmation SMILES manifest contains test data")
    if splits.get("official_test_loaded") is not False:
        raise RuntimeError("Confirmation split manifest contains test data")
    source_values = np.load(discovery / "embeddings.npy", allow_pickle=False)
    by_smiles = {
        value: source_values[index]
        for index, value in enumerate(source_smiles["smiles"])
    }
    missing = sorted(set(target_smiles["smiles"]).difference(by_smiles))
    if missing:
        raise RuntimeError(f"Confirmation contains {len(missing)} unseen SMILES")
    embeddings = np.asarray(
        [by_smiles[value] for value in target_smiles["smiles"]], dtype=np.float32
    )
    kernels = build_kernels(target_smiles["smiles"], embeddings)
    if args.kernel not in kernels:
        raise ValueError(f"Unknown locked kernel {args.kernel!r}")
    scores = []
    predictions = {}
    for seed in splits["seeds"]:
        split = splits["splits"][str(seed)]
        train_ids = np.asarray(split["train_ids"], dtype=int)
        valid_ids = np.asarray(split["valid_ids"], dtype=int)
        train_y = np.asarray(split["train_y"], dtype=int)
        valid_y = np.asarray(split["valid_y"], dtype=int)
        values = _fit_predict(
            kernels[args.kernel], train_ids, train_y, valid_ids, args.c
        )
        predictions[f"seed_{seed}"] = values
        scores.append(_score(valid_y, values, splits["metric"]))
    result = {
        "task": splits["task"],
        "metric": splits["metric"],
        "locked_recipe": {"kernel": args.kernel, "c": args.c},
        "scores": scores,
        "mean": statistics.mean(scores),
        "std": statistics.pstdev(scores),
        "protocol": {
            "seeds": splits["seeds"],
            "split": splits["split_type"],
            "recipe_selected_on": "outer scaffold seeds 1-5",
            "confirmation_seeds_used_for_selection": False,
            "embeddings_reused_by_exact_smiles_mapping": True,
            "official_test_loaded": False,
            "test_labels_used": False,
        },
    }
    (confirmation / "locked_kernel_confirmation.json").write_text(
        json.dumps(result, indent=2, sort_keys=True), encoding="utf-8"
    )
    np.savez_compressed(
        confirmation / "locked_kernel_confirmation_predictions.npz", **predictions
    )
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
