"""Nested scaffold validation for MiniMol/chemical similarity kernels.

The script is deliberately validation-only.  It consumes the label-free
MiniMol embedding export and the corresponding development split manifest,
and never instantiates a TDC benchmark group or loads an official test frame.
For each outer scaffold split, the kernel family and SVM regularization are
selected using only GroupKFold splits of the outer training partition.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator
from rdkit.Chem.Scaffolds import MurckoScaffold
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.svm import SVC


@dataclass(frozen=True)
class Candidate:
    name: str
    kernel: str
    c: float


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _score(labels: np.ndarray, values: np.ndarray, metric: str) -> float:
    if metric == "auprc":
        return float(average_precision_score(labels, values))
    if metric == "auroc":
        return float(roc_auc_score(labels, values))
    raise ValueError(f"Unsupported metric: {metric}")


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


def _binary_fingerprints(smiles: list[str], radius: int) -> np.ndarray:
    generator = rdFingerprintGenerator.GetMorganGenerator(
        radius=radius, fpSize=2048
    )
    rows = []
    for value in smiles:
        molecule = Chem.MolFromSmiles(value)
        if molecule is None:
            rows.append(np.zeros(2048, dtype=np.float32))
            continue
        fingerprint = generator.GetFingerprint(molecule)
        row = np.zeros(2048, dtype=np.float32)
        DataStructs.ConvertToNumpyArray(fingerprint, row)
        rows.append(row)
    return np.asarray(rows, dtype=np.float32)


def _tanimoto(values: np.ndarray) -> np.ndarray:
    intersection = values @ values.T
    counts = values.sum(axis=1)
    union = counts[:, None] + counts[None, :] - intersection
    return np.divide(
        intersection,
        union,
        out=np.zeros_like(intersection, dtype=np.float32),
        where=union > 0,
    )


def _cosine(values: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    normalized = np.divide(
        values,
        norms,
        out=np.zeros_like(values, dtype=np.float32),
        where=norms > 0,
    )
    return np.asarray(normalized @ normalized.T, dtype=np.float32)


def build_kernels(smiles: list[str], embeddings: np.ndarray) -> dict[str, np.ndarray]:
    morgan_r2 = _tanimoto(_binary_fingerprints(smiles, radius=2))
    morgan_r3 = _tanimoto(_binary_fingerprints(smiles, radius=3))
    chemical = (morgan_r2 + morgan_r3) / 2
    minimol = _cosine(embeddings)
    kernels = {
        "morgan_r2": morgan_r2,
        "morgan_r3": morgan_r3,
        "morgan_multiscale": chemical,
        "minimol": minimol,
    }
    for chemical_weight in (0.25, 0.50, 0.75):
        kernels[f"fusion_chem_{chemical_weight:.2f}"] = (
            chemical_weight * chemical + (1 - chemical_weight) * minimol
        )
    return kernels


def candidates(kernels: dict[str, np.ndarray]) -> list[Candidate]:
    return [
        Candidate(f"{kernel}_c_{c:g}", kernel, c)
        for kernel in sorted(kernels)
        for c in (0.1, 1.0, 10.0, 100.0)
    ]


def _fit_predict(
    kernel: np.ndarray,
    train_ids: np.ndarray,
    train_y: np.ndarray,
    predict_ids: np.ndarray,
    c: float,
) -> np.ndarray:
    model = SVC(
        C=c,
        kernel="precomputed",
        class_weight="balanced",
        cache_size=1024,
        random_state=0,
    )
    model.fit(kernel[np.ix_(train_ids, train_ids)], train_y)
    return model.decision_function(kernel[np.ix_(predict_ids, train_ids)])


def _inner_scores(
    candidate: Candidate,
    kernel: np.ndarray,
    train_ids: np.ndarray,
    train_y: np.ndarray,
    groups: np.ndarray,
    metric: str,
    folds: int,
) -> list[float]:
    splitter = GroupKFold(n_splits=min(folds, len(np.unique(groups))))
    scores = []
    for fit_index, valid_index in splitter.split(train_ids, train_y, groups):
        values = _fit_predict(
            kernel,
            train_ids[fit_index],
            train_y[fit_index],
            train_ids[valid_index],
            candidate.c,
        )
        scores.append(_score(train_y[valid_index], values, metric))
    return scores


def evaluate(
    payload: dict,
    smiles_payload: dict,
    embeddings: np.ndarray,
    folds: int,
    lcb_beta: float,
) -> tuple[dict, dict[str, np.ndarray]]:
    if payload.get("official_test_loaded") is not False:
        raise RuntimeError("Kernel evaluator refused a test-bearing split")
    if smiles_payload.get("official_test_loaded") is not False:
        raise RuntimeError("Kernel evaluator refused a test-bearing SMILES file")
    if smiles_payload.get("contains_labels") is not False:
        raise RuntimeError("SMILES manifest must be label-free")
    smiles = [str(value) for value in smiles_payload["smiles"]]
    if embeddings.shape[0] != len(smiles):
        raise ValueError("Embedding and SMILES row counts differ")
    kernels = build_kernels(smiles, embeddings)
    bank = candidates(kernels)
    metric = payload["metric"]
    outer_records = {}
    outer_predictions: dict[str, np.ndarray] = {}
    fixed_scores = {candidate.name: [] for candidate in bank}

    for seed in payload["seeds"]:
        split = payload["splits"][str(seed)]
        train_ids = np.asarray(split["train_ids"], dtype=int)
        valid_ids = np.asarray(split["valid_ids"], dtype=int)
        train_y = np.asarray(split["train_y"], dtype=int)
        valid_y = np.asarray(split["valid_y"], dtype=int)
        if set(train_ids).intersection(valid_ids):
            raise RuntimeError(f"Outer seed {seed} has overlapping molecule IDs")
        groups = _scaffolds([smiles[index] for index in train_ids])
        utilities = {}
        inner = {}
        for candidate in bank:
            values = _inner_scores(
                candidate,
                kernels[candidate.kernel],
                train_ids,
                train_y,
                groups,
                metric,
                folds,
            )
            mean = statistics.mean(values)
            std = statistics.pstdev(values)
            inner[candidate.name] = {"scores": values, "mean": mean, "std": std}
            utilities[candidate.name] = mean - lcb_beta * std
        selected = max(sorted(utilities), key=utilities.get)
        selected_candidate = next(value for value in bank if value.name == selected)
        for candidate in bank:
            prediction = _fit_predict(
                kernels[candidate.kernel],
                train_ids,
                train_y,
                valid_ids,
                candidate.c,
            )
            fixed_scores[candidate.name].append(_score(valid_y, prediction, metric))
            if candidate == selected_candidate:
                outer_predictions[f"seed_{seed}"] = prediction
        selected_score = fixed_scores[selected][-1]
        outer_records[str(seed)] = {
            "selected": selected,
            "selected_utility": utilities[selected],
            "outer_score": selected_score,
            "n_train": len(train_ids),
            "n_valid": len(valid_ids),
            "inner": inner,
        }

    selected_scores = [
        outer_records[str(seed)]["outer_score"] for seed in payload["seeds"]
    ]
    fixed_summary = {
        name: {
            "scores": values,
            "mean": statistics.mean(values),
            "std": statistics.pstdev(values),
        }
        for name, values in fixed_scores.items()
    }
    best_fixed = max(sorted(fixed_summary), key=lambda name: fixed_summary[name]["mean"])
    result = {
        "task": payload["task"],
        "metric": metric,
        "nested_selection": {
            "scores": selected_scores,
            "mean": statistics.mean(selected_scores),
            "std": statistics.pstdev(selected_scores),
            "lcb_beta": lcb_beta,
            "inner_folds": folds,
            "outer_records": outer_records,
        },
        "fixed_candidates": fixed_summary,
        "best_fixed_candidate": best_fixed,
        "best_fixed_mean": fixed_summary[best_fixed]["mean"],
        "protocol": {
            "outer_seeds": payload["seeds"],
            "outer_split": payload["split_type"],
            "inner_split": "scaffold GroupKFold on outer-train only",
            "selection": "inner mean minus beta times population standard deviation",
            "kernels": sorted(kernels),
            "official_test_loaded": False,
            "test_labels_used": False,
        },
    }
    return result, outer_predictions


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root", type=Path, default=Path("runs/v24-minimol-validation")
    )
    parser.add_argument("--task", default="CYP2C9_Substrate_CarbonMangels")
    parser.add_argument("--inner-folds", type=int, default=5)
    parser.add_argument("--lcb-beta", type=float, default=0.25)
    args = parser.parse_args()
    root = args.root / args.task
    splits_path = root / "splits.json"
    smiles_path = root / "smiles.json"
    embeddings_path = root / "embeddings.npy"
    payload = json.loads(splits_path.read_text(encoding="utf-8"))
    smiles_payload = json.loads(smiles_path.read_text(encoding="utf-8"))
    embeddings = np.load(embeddings_path, allow_pickle=False)
    result, predictions = evaluate(
        payload,
        smiles_payload,
        embeddings,
        args.inner_folds,
        args.lcb_beta,
    )
    result["inputs"] = {
        "splits_sha256": _sha256(splits_path),
        "smiles_sha256": _sha256(smiles_path),
        "embeddings_sha256": _sha256(embeddings_path),
    }
    output = root / "scaffold_kernels.json"
    output.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    np.savez_compressed(root / "scaffold_kernel_predictions.npz", **predictions)
    compact = {
        "task": result["task"],
        "metric": result["metric"],
        "nested_selection": {
            key: result["nested_selection"][key]
            for key in ("scores", "mean", "std")
        },
        "best_fixed_candidate": result["best_fixed_candidate"],
        "best_fixed_mean": result["best_fixed_mean"],
        "official_test_loaded": False,
    }
    print(json.dumps(compact, sort_keys=True))


if __name__ == "__main__":
    main()
