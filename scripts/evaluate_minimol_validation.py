"""Evaluate the fixed MiniMol downstream head on scaffold validation seeds."""

from __future__ import annotations

import argparse
import json
import math
import random
import statistics
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


class TaskHead(nn.Module):
    """Official MiniMol notebook's three-layer residual downstream head."""

    def __init__(self) -> None:
        super().__init__()
        self.dense1 = nn.Linear(512, 512)
        self.dense2 = nn.Linear(512, 512)
        self.final_dense = nn.Linear(1024, 1)
        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(512)
        self.dropout = nn.Dropout(0.10)
        self.relu = nn.ReLU()

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        original = value
        value = self.dropout(self.relu(self.bn1(self.dense1(value))))
        value = self.dropout(self.relu(self.bn2(self.dense2(value))))
        return self.final_dense(torch.cat((value, original), dim=1)).squeeze(-1)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_and_score(
    embeddings: np.ndarray,
    split: dict,
    metric: str,
    seed: int,
    device: torch.device,
) -> tuple[float, np.ndarray]:
    set_seed(seed)
    x_train = torch.tensor(embeddings[split["train_ids"]], dtype=torch.float32)
    y_train = torch.tensor(split["train_y"], dtype=torch.float32)
    x_valid = torch.tensor(
        embeddings[split["valid_ids"]], dtype=torch.float32, device=device
    )
    train_loader = DataLoader(
        TensorDataset(x_train, y_train),
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
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            loss_fn(model(inputs), targets).backward()
            optimizer.step()
    model.eval()
    with torch.no_grad():
        probabilities = torch.sigmoid(model(x_valid)).cpu().numpy()
    labels = np.asarray(split["valid_y"])
    if metric == "auprc":
        value = float(average_precision_score(labels, probabilities))
    else:
        value = float(roc_auc_score(labels, probabilities))
    return value, probabilities


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--splits", type=Path, required=True)
    parser.add_argument("--embeddings", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    payload = json.loads(args.splits.read_text(encoding="utf-8"))
    if payload.get("official_test_loaded") is not False:
        raise RuntimeError("Validation evaluator refused a test-bearing split")
    embeddings = np.load(args.embeddings)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    evaluated = [
        train_and_score(
            embeddings, payload["splits"][str(seed)], payload["metric"], seed, device
        )
        for seed in payload["seeds"]
    ]
    scores = [value for value, _ in evaluated]
    predictions = {
        f"seed_{seed}": prediction
        for seed, (_, prediction) in zip(payload["seeds"], evaluated, strict=True)
    }
    result = {
        "task": payload["task"],
        "metric": payload["metric"],
        "scores": scores,
        "mean": statistics.mean(scores),
        "std": statistics.pstdev(scores),
        "protocol": {
            "source": "MiniMol official downstream_adaptation.ipynb",
            "outer_seeds": payload["seeds"],
            "epochs": 25,
            "batch_size": 32,
            "learning_rate": 3e-4,
            "weight_decay": 1e-4,
            "dropout": 0.1,
            "checkpoint_selection": "fixed final epoch; no validation sweep",
            "official_test_loaded": False,
        },
        "device": str(device),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output.with_suffix(".predictions.npz"), **predictions)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
