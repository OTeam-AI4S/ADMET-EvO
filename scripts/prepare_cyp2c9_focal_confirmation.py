"""Prepare fresh validation-only inputs for a fixed CYP2C9 rescue candidate.

This runs in the TDC/ADMET environment and deliberately never calls
``group.get`` or otherwise loads the official test split.  The candidate is
written and content-addressed by ``--preregister-only`` before seeds 6--10 are
materialized.  A separate isolated Torch environment trains the frozen focal
heads from the label-bearing NPZ artifact.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
from tdc.benchmark_group import admet_group

from admet_agent.features import FeatureCache
from admet_agent.runner import _columns
from admet_agent.schemas import ExperimentSpec
from admet_agent.selection import build_model, predict
from admet_agent.tasks import resolve_task

TASK = "CYP2C9_Substrate_CarbonMangels"
SEEDS = [6, 7, 8, 9, 10]
RICH = ("morgan", "avalon", "erg", "rdkit2d")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _digest(payload: dict[str, Any]) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def preregistration(repo_root: Path) -> dict[str, Any]:
    root = repo_root.resolve()
    rank_heads = root / "runs/v24-minimol-validation" / TASK / "rank_heads.json"
    rank_predictions = rank_heads.with_name("rank_head_predictions.npz")
    hard_ablation = root / "runs/hard-task-ablation/results.json"
    payload = {
        "version": "cyp2c9-focal-fusion-confirmation-v1",
        "task": TASK,
        "metric": "auprc",
        "fresh_confirmation_seeds": SEEDS,
        "candidate": {
            "aggregation": "percentile_rank",
            "lightgbm_weight": 0.8,
            "focal_minimol_weight": 0.2,
            "lightgbm": {
                "representations": list(RICH),
                "hyperparameters": {},
                "imbalance": "class_weight",
            },
            "focal_minimol": {
                "frozen_embeddings": True,
                "loss": "positive-weighted focal BCE",
                "focal_gamma": 2.0,
                "members": 16,
                "epochs": 100,
                "batch_size": 32,
                "learning_rate": 3e-4,
                "weight_decay": 1e-4,
                "head_seed_formula": "outer_seed*10000+500+member",
            },
        },
        "promotion_gate": {
            "candidate_mean_must_exceed_lightgbm_mean": True,
            "minimum_paired_seed_improvements": 3,
            "no_official_evaluation_without_new_full_freeze": True,
        },
        "discovery_evidence": {
            "lightgbm_mean": 0.48011663267624854,
            "fixed_fusion_mean": 0.489476096522265,
            "fusion_weight_selected_by_loso_on_seeds_1_5": 0.2,
            "artifacts": {
                rank_heads.relative_to(root).as_posix(): _sha256(rank_heads),
                rank_predictions.relative_to(root).as_posix(): _sha256(rank_predictions),
                hard_ablation.relative_to(root).as_posix(): _sha256(hard_ablation),
            },
        },
        "official_test_loaded": False,
        "test_labels_used": False,
        "confirmation_seeds_used_for_candidate_selection": False,
    }
    return {**payload, "preregistration_sha256": _digest(payload)}


def verify_preregistration(path: Path, expected: dict[str, Any]) -> None:
    observed = json.loads(path.read_text(encoding="utf-8"))
    if observed != expected:
        raise RuntimeError("CYP2C9 confirmation preregistration changed")
    payload = {key: value for key, value in observed.items() if key != "preregistration_sha256"}
    if observed["preregistration_sha256"] != _digest(payload):
        raise RuntimeError("CYP2C9 confirmation preregistration hash mismatch")


def build_inputs(
    group: object,
    repo_root: Path,
    output_dir: Path,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    root = repo_root.resolve()
    task = resolve_task(TASK)
    embedding_root = root / "runs/v24-minimol-validation" / TASK
    smiles_payload = json.loads((embedding_root / "smiles.json").read_text(encoding="utf-8"))
    if smiles_payload.get("official_test_loaded") is not False:
        raise RuntimeError("Frozen MiniMol embedding source is not validation-only")
    embeddings = np.load(embedding_root / "embeddings.npy", allow_pickle=False)
    smiles_to_embedding = {
        str(smiles): embeddings[index]
        for index, smiles in enumerate(smiles_payload["smiles"])
    }
    cache = FeatureCache(output_dir / "feature_cache")
    arrays: dict[str, np.ndarray] = {}
    records = {}
    for seed in SEEDS:
        train, valid = group.get_train_valid_split(
            benchmark=task.name, split_type="default", seed=seed
        )
        smiles_col, y_col = _columns(train)
        train_smiles = train[smiles_col].astype(str).tolist()
        valid_smiles = valid[smiles_col].astype(str).tolist()
        missing = sorted(
            set(train_smiles + valid_smiles).difference(smiles_to_embedding)
        )
        if missing:
            raise RuntimeError(
                f"Fresh split contains {len(missing)} molecules absent from the frozen "
                "validation-only MiniMol embedding bank"
            )
        x_train_mini = np.asarray([smiles_to_embedding[value] for value in train_smiles])
        x_valid_mini = np.asarray([smiles_to_embedding[value] for value in valid_smiles])
        y_train = train[y_col].to_numpy(dtype=np.float32)
        y_valid = valid[y_col].to_numpy(dtype=np.float32)

        rich = cache.get_or_compute(train_smiles + valid_smiles, RICH)
        spec = ExperimentSpec(
            task=task.name,
            seed=seed,
            representations=RICH,
            model="lightgbm",
            imbalance="class_weight",
            device="cpu",
            n_jobs=8,
            hyperparameters={},
            hypothesis="preregistered CYP2C9 focal-fusion confirmation baseline",
        )
        baseline = build_model(spec, task, y_train)
        baseline.fit(rich[: len(train)], y_train)
        baseline_predictions = predict(baseline, rich[len(train) :], task)

        prefix = f"seed_{seed}"
        arrays[f"{prefix}_x_train"] = x_train_mini.astype(np.float32)
        arrays[f"{prefix}_y_train"] = y_train
        arrays[f"{prefix}_x_valid"] = x_valid_mini.astype(np.float32)
        arrays[f"{prefix}_y_valid"] = y_valid
        arrays[f"{prefix}_lightgbm"] = np.asarray(
            baseline_predictions, dtype=np.float64
        )
        records[str(seed)] = {
            "n_train": len(train),
            "n_valid": len(valid),
            "positive_train": int(y_train.sum()),
            "positive_valid": int(y_valid.sum()),
        }
    manifest = {
        "task": TASK,
        "metric": task.metric,
        "seeds": SEEDS,
        "split": "TDC default train/validation scaffold split",
        "records": records,
        "embedding_bank": {
            "smiles_sha256": _sha256(embedding_root / "smiles.json"),
            "embeddings_sha256": _sha256(embedding_root / "embeddings.npy"),
        },
        "official_test_loaded": False,
        "test_labels_used": False,
    }
    return manifest, arrays


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument(
        "--output-dir", type=Path, default=Path("runs/v25-cyp2c9-focal-confirmation")
    )
    parser.add_argument("--preregister-only", action="store_true")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    prereg_path = args.output_dir / "preregistration.json"
    expected = preregistration(args.repo_root)
    if prereg_path.exists():
        verify_preregistration(prereg_path, expected)
    else:
        prereg_path.write_text(
            json.dumps(expected, indent=2, sort_keys=True), encoding="utf-8"
        )
    if args.preregister_only:
        print(json.dumps({"status": "preregistered", "sha256": expected["preregistration_sha256"]}))
        return

    verify_preregistration(prereg_path, expected)
    group = admet_group(path=str(args.data_dir))
    manifest, arrays = build_inputs(group, args.repo_root, args.output_dir)
    input_path = args.output_dir / "confirmation_inputs.npz"
    np.savez_compressed(input_path, **arrays)
    manifest["input_sha256"] = _sha256(input_path)
    manifest["preregistration_sha256"] = expected["preregistration_sha256"]
    (args.output_dir / "input_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps({"status": "prepared", **manifest}, sort_keys=True))


if __name__ == "__main__":
    main()
