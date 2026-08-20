"""Validation-only five-model inner-scaffold MiniMol head ensemble."""

from __future__ import annotations

import argparse
import json
import math
import random
import statistics
from pathlib import Path

import numpy as np
import torch
from rdkit.Chem.Scaffolds import MurckoScaffold
from sklearn.metrics import average_precision_score
from sklearn.model_selection import GroupKFold
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

try:
    from scripts.evaluate_minimol_validation import TaskHead
except ModuleNotFoundError:
    from evaluate_minimol_validation import TaskHead


def _scaffolds(smiles: list[str]) -> np.ndarray:
    return np.asarray(
        [
            MurckoScaffold.MurckoScaffoldSmiles(
                smiles=value, includeChirality=True
            )
            or value
            for value in smiles
        ]
    )


def _fit_head(
    x: np.ndarray, y: np.ndarray, seed: int, device: torch.device
) -> TaskHead:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    loader = DataLoader(
        TensorDataset(torch.tensor(x, dtype=torch.float32), torch.tensor(y).float()),
        batch_size=32,
        shuffle=True,
        generator=torch.Generator().manual_seed(seed),
    )
    model = TaskHead().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4, weight_decay=1e-4)
    epochs, warmup = 25, 5
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda epoch: (
            epoch / warmup
            if epoch < warmup
            else (1 + math.cos(math.pi * (epoch - warmup) / (epochs - warmup))) / 2
        ),
    )
    loss_fn = nn.BCEWithLogitsLoss()
    for epoch in range(epochs):
        model.train()
        scheduler.step(epoch)
        for inputs, targets in loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            loss_fn(model(inputs), targets).backward()
            optimizer.step()
    return model.eval()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root", type=Path, default=Path("runs/v24-minimol-validation")
    )
    parser.add_argument("--task", default="CYP2C9_Substrate_CarbonMangels")
    args = parser.parse_args()
    root = args.root / args.task
    payload = json.loads((root / "splits.json").read_text())
    smiles_payload = json.loads((root / "smiles.json").read_text())
    if payload.get("official_test_loaded") is not False:
        raise RuntimeError("Inner ensemble refused a test-bearing split")
    embeddings = np.load(root / "embeddings.npy")
    smiles = smiles_payload["smiles"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    scores, predictions = [], {}
    for outer_seed in payload["seeds"]:
        split = payload["splits"][str(outer_seed)]
        train_ids = np.asarray(split["train_ids"])
        valid_ids = np.asarray(split["valid_ids"])
        y_train = np.asarray(split["train_y"])
        groups = _scaffolds([smiles[index] for index in train_ids])
        splitter = GroupKFold(n_splits=5)
        logits = []
        for fold, (fit_indices, _) in enumerate(
            splitter.split(train_ids, y_train, groups)
        ):
            model = _fit_head(
                embeddings[train_ids[fit_indices]],
                y_train[fit_indices],
                outer_seed * 100 + fold,
                device,
            )
            with torch.no_grad():
                logits.append(
                    model(
                        torch.tensor(
                            embeddings[valid_ids], dtype=torch.float32, device=device
                        )
                    ).cpu().numpy()
                )
        probability = 1 / (1 + np.exp(-np.mean(logits, axis=0)))
        predictions[f"seed_{outer_seed}"] = probability
        scores.append(
            float(average_precision_score(split["valid_y"], probability))
        )
    result = {
        "task": args.task,
        "metric": "auprc",
        "scores": scores,
        "mean": statistics.mean(scores),
        "std": statistics.pstdev(scores),
        "protocol": {
            "outer_seeds": payload["seeds"],
            "inner_folds": 5,
            "inner_split": "scaffold GroupKFold on outer-train only",
            "ensemble": "mean logits from five fixed final-epoch heads",
            "epochs": 25,
            "official_test_loaded": False,
        },
    }
    np.savez_compressed(root / "inner_ensemble_predictions.npz", **predictions)
    (root / "inner_ensemble.json").write_text(
        json.dumps(result, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
