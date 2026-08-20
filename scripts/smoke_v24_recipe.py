"""Validation-only runtime smoke test for a frozen v2.4 recipe.

This program never calls ``admet_group.get`` and passes only validation SMILES,
not validation labels, into the frozen executor. Labels are used once afterwards
to report the already-frozen recipe's validation score.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import time
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import numpy as np
from tdc.benchmark_group import admet_group

from admet_agent.features import (
    MINIMOL_PINNED_VERSION,
    FeatureCache,
)
from admet_agent.frozen_executor import FrozenRecipeExecutor
from admet_agent.frozen_recipe import parse_recipe
from admet_agent.runner import ADMETAgent, _columns
from admet_agent.selection import score
from admet_agent.tasks import resolve_task


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_frozen_config(mapping_path: Path, task_name: str) -> dict:
    mapping = json.loads(mapping_path.read_text(encoding="utf-8"))
    task = resolve_task(task_name)
    if task.name not in mapping:
        raise ValueError(f"No frozen recipe for {task.name} in {mapping_path}")
    return ADMETAgent._normalize_frozen(task, mapping[task.name])


def _package_version(name: str) -> str | None:
    try:
        return version(name)
    except PackageNotFoundError:
        return None


def read_training_seeds(path: Path) -> dict[str, int]:
    """Read the two reproducibility seeds from unimol-tools' saved YAML."""
    values: dict[str, int] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        key, separator, raw = line.partition(":")
        if separator and key.strip() in {"seed", "split_seed"}:
            values[key.strip()] = int(raw.strip())
    if set(values) != {"seed", "split_seed"}:
        raise RuntimeError(f"Missing Uni-Mol training seeds in {path}")
    return values


def run_smoke(args: argparse.Namespace) -> dict:
    task = resolve_task(args.task)
    frozen = load_frozen_config(args.mapping, task.name)
    if frozen["kind"] != "recipe":
        raise ValueError("v2.4 smoke runner requires a recipe finalist")
    recipe = parse_recipe(frozen["recipe"])
    requires_unimol_finetune = bool(
        {"finetuned_unimol2", "fixed_rank_top3"}.intersection(
            recipe.expert_names
        )
    )
    try:
        import torch
    except ImportError:
        torch = None
    if requires_unimol_finetune and (
        torch is None or not torch.cuda.is_available()
    ):
        raise RuntimeError("Frozen recipe smoke test requires an allocated CUDA GPU")
    device = "cuda" if requires_unimol_finetune else "cpu"
    group = admet_group(path=str(args.data_dir))
    train, valid = group.get_train_valid_split(
        benchmark=task.name, split_type="default", seed=args.seed
    )
    smiles_col, y_col = _columns(train)
    output = args.output_dir / task.name / f"seed_{args.seed}"
    cache = FeatureCache(
        args.output_dir / "feature_cache", device=device, unimol_batch_size=16
    )
    executor = FrozenRecipeExecutor(
        cache=cache,
        artifact_dir=output / "artifacts",
        device=device,
        n_jobs=args.workers,
    )
    started = time.monotonic()
    predictions = executor.predict(
        task=task,
        recipe_value=frozen["recipe"],
        seed=args.seed,
        fit_frame=train,
        predict_smiles=valid[smiles_col].astype(str).tolist(),
        smiles_col=smiles_col,
        y_col=y_col,
        protocol=frozen["protocol"],
    )
    expected_seed = int(args.seed)
    if requires_unimol_finetune:
        training_config_path = (
            output / "artifacts" / "unimol2_finetune" / "config.yaml"
        )
        training_seeds = read_training_seeds(training_config_path)
        if training_seeds != {"seed": expected_seed, "split_seed": expected_seed}:
            raise RuntimeError(
                "Uni-Mol runtime seed mismatch: "
                f"expected {expected_seed}, observed {training_seeds}"
            )
        reproducibility = {
            "requested_seed": expected_seed,
            "training_seed": training_seeds["seed"],
            "split_seed": training_seeds["split_seed"],
            "training_config": training_config_path.as_posix(),
        }
    elif "focal_minimol_16" in recipe.expert_names:
        reproducibility = {
            "requested_seed": expected_seed,
            "focal_head_seeds": [
                expected_seed * 10000 + 500 + member for member in range(16)
            ],
            "torch_threads": {"intra_op": 1, "inter_op": 1},
        }
    elif "maplight_gnn" in recipe.expert_names:
        runtime_report = json.loads(
            (
                output
                / "artifacts"
                / "maplight_gnn"
                / "runtime_report.json"
            ).read_text(encoding="utf-8")
        )
        reproducibility = {
            "requested_seed": expected_seed,
            "upstream_sha256": runtime_report["upstream_sha256"],
            "checkpoint_sha256": runtime_report["checkpoint_sha256"],
            "feature_manifest_sha256": runtime_report["feature_manifest_sha256"],
            "prediction_sha256": runtime_report["predictions_sha256"],
            "predict_labels_present": runtime_report["predict_labels_present"],
        }
    else:
        reproducibility = {
            "requested_seed": expected_seed,
            "kernel_random_state": 0,
        }
    minimol_version = _package_version("minimol")
    if minimol_version is None and {
        "minimol_morgan_kernel", "focal_minimol_16"
    }.intersection(recipe.expert_names):
        minimol_version = MINIMOL_PINNED_VERSION
    validation_score = score(valid[y_col].to_numpy(), predictions, task.metric)
    output.mkdir(parents=True, exist_ok=True)
    np.save(output / "validation_predictions.npy", predictions)
    result = {
        "status": "passed",
        "task": task.name,
        "seed": args.seed,
        "metric": task.metric,
        "score": validation_score,
        "n_train": len(train),
        "n_valid": len(valid),
        "elapsed_seconds": time.monotonic() - started,
        "frozen_config": frozen,
        "mapping_path": args.mapping.as_posix(),
        "mapping_sha256": sha256_file(args.mapping),
        "official_test_loaded": False,
        "test_labels_passed_to_executor": False,
        "reproducibility": reproducibility,
        "environment": {
            "python": platform.python_version(),
            "torch": None if torch is None else torch.__version__,
            "cuda": None if torch is None else torch.version.cuda,
            "gpu": (
                torch.cuda.get_device_name(0)
                if torch is not None and torch.cuda.is_available()
                else None
            ),
            "unimol_tools": _package_version("unimol-tools"),
            "minimol": minimol_version,
        },
    }
    (output / "result.json").write_text(
        json.dumps(result, indent=2, sort_keys=True), encoding="utf-8"
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", required=True)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument(
        "--mapping",
        type=Path,
        default=Path("runs/v24-freeze/finalists.mapping.json"),
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("runs/v24-recipe-smoke")
    )
    args = parser.parse_args()
    result = run_smoke(args)
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
