"""Validation-only MiniMol head ensembles with metric-aligned rank losses.

MiniMol embeddings are frozen.  Each candidate is preregistered in this file
and evaluated on the five development-only outer scaffold splits.  Candidate
selection quality is reported with leave-one-outer-seed-out replay so that the
same held-out seed is never used to choose its loss configuration.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import statistics
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset

try:
    from scripts.evaluate_minimol_validation import TaskHead
except ModuleNotFoundError:
    from evaluate_minimol_validation import TaskHead


@dataclass(frozen=True)
class LossConfig:
    name: str
    positive_weight: bool = False
    pairwise_weight: float = 0.0
    focal_gamma: float = 0.0


CONFIGS = (
    LossConfig("bce"),
    LossConfig("balanced_bce", positive_weight=True),
    LossConfig("pairwise_025", pairwise_weight=0.25),
    LossConfig("pairwise_100", pairwise_weight=1.0),
    LossConfig("balanced_pairwise_025", positive_weight=True, pairwise_weight=0.25),
    LossConfig("focal_2", positive_weight=True, focal_gamma=2.0),
)


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _metric(labels: np.ndarray, values: np.ndarray, metric: str) -> float:
    if metric == "auprc":
        return float(average_precision_score(labels, values))
    if metric == "auroc":
        return float(roc_auc_score(labels, values))
    raise ValueError(f"Unsupported metric: {metric}")


def _loss(logits: torch.Tensor, targets: torch.Tensor, config: LossConfig) -> torch.Tensor:
    positive_weight = None
    if config.positive_weight:
        positives = targets.sum().clamp_min(1)
        negatives = (targets.numel() - targets.sum()).clamp_min(1)
        positive_weight = negatives / positives
    elementwise = F.binary_cross_entropy_with_logits(
        logits, targets, pos_weight=positive_weight, reduction="none"
    )
    if config.focal_gamma:
        probabilities = torch.sigmoid(logits)
        correct_probability = torch.where(targets > 0.5, probabilities, 1 - probabilities)
        elementwise = elementwise * (1 - correct_probability).pow(config.focal_gamma)
    value = elementwise.mean()
    if config.pairwise_weight:
        positive_logits = logits[targets > 0.5]
        negative_logits = logits[targets <= 0.5]
        if positive_logits.numel() and negative_logits.numel():
            pairwise = F.softplus(
                -(positive_logits[:, None] - negative_logits[None, :])
            ).mean()
            value = value + config.pairwise_weight * pairwise
    return value


def _train_head(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_valid: np.ndarray,
    config: LossConfig,
    seed: int,
    epochs: int,
    batch_size: int,
    device: torch.device,
) -> np.ndarray:
    _set_seed(seed)
    loader = DataLoader(
        TensorDataset(
            torch.tensor(x_train, dtype=torch.float32),
            torch.tensor(y_train, dtype=torch.float32),
        ),
        batch_size=batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(seed),
    )
    model = TaskHead().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    warmup = max(1, epochs // 5)
    for epoch in range(epochs):
        model.train()
        if epoch < warmup:
            multiplier = (epoch + 1) / warmup
        else:
            progress = (epoch - warmup) / max(1, epochs - warmup - 1)
            multiplier = 0.5 * (1 + math.cos(math.pi * progress))
        for group in optimizer.param_groups:
            group["lr"] = 3e-4 * multiplier
        for inputs, targets in loader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            optimizer.zero_grad(set_to_none=True)
            value = _loss(model(inputs), targets, config)
            value.backward()
            optimizer.step()
    model.eval()
    with torch.no_grad():
        return model(
            torch.tensor(x_valid, dtype=torch.float32, device=device)
        ).cpu().numpy()


def evaluate(
    payload: dict,
    embeddings: np.ndarray,
    members: int,
    epochs: int,
    batch_size: int,
    lcb_beta: float,
) -> tuple[dict, dict[str, np.ndarray]]:
    if payload.get("official_test_loaded") is not False:
        raise RuntimeError("Rank-head evaluator refused a test-bearing split")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_float32_matmul_precision("high")
    fixed_scores = {config.name: [] for config in CONFIGS}
    predictions = {}
    per_outer = {}
    started = time.monotonic()
    for outer_seed in payload["seeds"]:
        split = payload["splits"][str(outer_seed)]
        train_ids = np.asarray(split["train_ids"], dtype=int)
        valid_ids = np.asarray(split["valid_ids"], dtype=int)
        train_y = np.asarray(split["train_y"], dtype=np.float32)
        valid_y = np.asarray(split["valid_y"], dtype=np.float32)
        if set(train_ids).intersection(valid_ids):
            raise RuntimeError(f"Outer seed {outer_seed} has overlapping molecule IDs")
        per_outer[str(outer_seed)] = {}
        for config_index, config in enumerate(CONFIGS):
            member_logits = []
            member_scores = []
            for member in range(members):
                head_seed = outer_seed * 10000 + config_index * 100 + member
                logits = _train_head(
                    embeddings[train_ids],
                    train_y,
                    embeddings[valid_ids],
                    config,
                    head_seed,
                    epochs,
                    batch_size,
                    device,
                )
                member_logits.append(logits)
                member_scores.append(_metric(valid_y, logits, payload["metric"]))
            ensemble = np.mean(member_logits, axis=0)
            value = _metric(valid_y, ensemble, payload["metric"])
            fixed_scores[config.name].append(value)
            predictions[f"seed_{outer_seed}_{config.name}"] = ensemble
            per_outer[str(outer_seed)][config.name] = {
                "ensemble_score": value,
                "member_scores": member_scores,
            }
        print(
            json.dumps(
                {
                    "outer_seed": outer_seed,
                    "scores": {
                        name: fixed_scores[name][-1] for name in fixed_scores
                    },
                },
                sort_keys=True,
            ),
            flush=True,
        )
    summary = {
        name: {
            "scores": values,
            "mean": statistics.mean(values),
            "std": statistics.pstdev(values),
        }
        for name, values in fixed_scores.items()
    }
    best_fixed = max(sorted(summary), key=lambda name: summary[name]["mean"])
    loso_selected = {}
    loso_scores = []
    for heldout_index, heldout_seed in enumerate(payload["seeds"]):
        utilities = {}
        for name, values in fixed_scores.items():
            fitting = [
                value for index, value in enumerate(values) if index != heldout_index
            ]
            utilities[name] = (
                statistics.mean(fitting)
                - lcb_beta * statistics.pstdev(fitting)
            )
        selected = max(sorted(utilities), key=utilities.get)
        heldout_score = fixed_scores[selected][heldout_index]
        loso_selected[str(heldout_seed)] = {
            "candidate": selected,
            "utility": utilities[selected],
            "heldout_score": heldout_score,
        }
        loso_scores.append(heldout_score)
    result = {
        "task": payload["task"],
        "metric": payload["metric"],
        "fixed_candidates": summary,
        "best_fixed_candidate": best_fixed,
        "best_fixed_mean": summary[best_fixed]["mean"],
        "loso_selection": {
            "scores": loso_scores,
            "mean": statistics.mean(loso_scores),
            "std": statistics.pstdev(loso_scores),
            "lcb_beta": lcb_beta,
            "selected": loso_selected,
        },
        "outer_records": per_outer,
        "protocol": {
            "outer_seeds": payload["seeds"],
            "outer_split": payload["split_type"],
            "frozen_minimol_embeddings": True,
            "members": members,
            "epochs": epochs,
            "batch_size": batch_size,
            "candidate_losses": [config.__dict__ for config in CONFIGS],
            "official_test_loaded": False,
            "test_labels_used": False,
        },
        "device": str(device),
        "elapsed_seconds": time.monotonic() - started,
    }
    return result, predictions


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root", type=Path, default=Path("runs/v24-minimol-validation")
    )
    parser.add_argument("--task", default="CYP2C9_Substrate_CarbonMangels")
    parser.add_argument("--members", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lcb-beta", type=float, default=0.25)
    args = parser.parse_args()
    task_root = args.root / args.task
    payload = json.loads((task_root / "splits.json").read_text(encoding="utf-8"))
    embeddings = np.load(task_root / "embeddings.npy", allow_pickle=False)
    result, predictions = evaluate(
        payload,
        embeddings,
        args.members,
        args.epochs,
        args.batch_size,
        args.lcb_beta,
    )
    output = task_root / "rank_heads.json"
    output.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    np.savez_compressed(task_root / "rank_head_predictions.npz", **predictions)
    print(
        json.dumps(
            {
                "task": result["task"],
                "metric": result["metric"],
                "best_fixed_candidate": result["best_fixed_candidate"],
                "best_fixed_mean": result["best_fixed_mean"],
                "loso_mean": result["loso_selection"]["mean"],
                "elapsed_seconds": result["elapsed_seconds"],
                "official_test_loaded": False,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
