"""Run preregistered Caco2 rich-2D feature and learner ablations.

This script reads only TDC ``train_val`` through ``get_train_valid_split``.
It never requests or evaluates the official test split.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
import time
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

import numpy as np
from rdkit import Chem, rdBase
from rdkit.Chem.Scaffolds import MurckoScaffold
from tdc.benchmark_group import admet_group

from admet_agent.features import FEATURE_SCHEMA_VERSION, FeatureCache
from admet_agent.guard import validate_experiment
from admet_agent.runner import _columns
from admet_agent.schemas import ExperimentSpec
from admet_agent.selection import build_model, predict, score
from admet_agent.tasks import resolve_task

TASK = "Caco2_Wang"
METRIC = "mae"
MINIMUM_USEFUL_MAE_REDUCTION = 0.01

FEATURE_LADDER = (
    ("catboost_rdkit", ("rdkit2d",)),
    ("catboost_rdkit_erg", ("rdkit2d", "erg")),
    (
        "catboost_rdkit_erg_morgan_count",
        ("rdkit2d", "erg", "morgan_count"),
    ),
    (
        "catboost_rdkit_erg_morgan_count_avalon",
        ("rdkit2d", "erg", "morgan_count", "avalon"),
    ),
)

FULL_COUNT_VIEW = ("rdkit2d", "erg", "morgan_count", "avalon")
CURRENT_BINARY_VIEW = ("morgan", "avalon", "erg", "rdkit2d")


def _recipes(suite: str = "discovery") -> list[dict[str, Any]]:
    recipes = [
        {
            "name": name,
            "model": "catboost",
            "representations": representations,
            "hypothesis": "MapLight-style forward feature addition",
        }
        for name, representations in FEATURE_LADDER
    ]
    recipes.extend(
        (
            {
                "name": "catboost_current_binary_rich",
                "model": "catboost",
                "representations": CURRENT_BINARY_VIEW,
                "hypothesis": "current binary rich-view CatBoost control",
            },
            {
                "name": "lightgbm_full_count",
                "model": "lightgbm",
                "representations": FULL_COUNT_VIEW,
                "hypothesis": "matched rich-view learner control",
            },
            {
                "name": "extra_trees_full_count",
                "model": "extra_trees",
                "representations": FULL_COUNT_VIEW,
                "hypothesis": "matched rich-view learner control",
            },
            {
                "name": "random_forest_full_count",
                "model": "random_forest",
                "representations": FULL_COUNT_VIEW,
                "hypothesis": "matched rich-view learner control",
            },
            {
                "name": "lightgbm_current_binary_rich",
                "model": "lightgbm",
                "representations": CURRENT_BINARY_VIEW,
                "hypothesis": "existing rich-2D validation baseline",
            },
        )
    )
    if suite == "confirmation":
        keep = {
            "catboost_rdkit",
            "catboost_rdkit_erg_morgan_count",
            "lightgbm_full_count",
            "extra_trees_full_count",
            "lightgbm_current_binary_rich",
        }
        return [recipe for recipe in recipes if recipe["name"] in keep]
    return recipes


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_state() -> dict[str, Any]:
    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    status = subprocess.run(
        ["git", "status", "--short"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.splitlines()
    return {"revision": revision, "dirty": bool(status), "status": status}


def _packages() -> dict[str, str | None]:
    names = (
        "PyTDC",
        "numpy",
        "scipy",
        "scikit-learn",
        "rdkit",
        "catboost",
        "lightgbm",
    )
    result: dict[str, str | None] = {}
    for name in names:
        try:
            result[name] = version(name)
        except PackageNotFoundError:
            result[name] = None
    return result


def _scaffolds(smiles: list[str]) -> set[str]:
    values = set()
    for value in smiles:
        molecule = Chem.MolFromSmiles(value)
        if molecule is None:
            continue
        values.add(
            MurckoScaffold.MurckoScaffoldSmiles(
                mol=molecule,
                includeChirality=False,
            )
        )
    return values


def _summaries(
    recipes: list[dict[str, Any]],
    records: dict[str, dict[int, dict[str, Any]]],
) -> dict[str, dict[str, Any]]:
    result = {}
    for recipe in recipes:
        name = recipe["name"]
        seed_records = records[name]
        scores = [seed_records[seed]["score"] for seed in sorted(seed_records)]
        elapsed = [seed_records[seed]["elapsed_seconds"] for seed in sorted(seed_records)]
        result[name] = {
            **recipe,
            "representations": list(recipe["representations"]),
            "scores": scores,
            "mean": float(np.mean(scores)),
            "sample_std": float(np.std(scores, ddof=1)),
            "total_wall_seconds": float(np.sum(elapsed)),
            "mean_wall_seconds": float(np.mean(elapsed)),
        }
    return result


def _comparison_payloads(
    seeds: list[int],
    summaries: dict[str, dict[str, Any]],
    phase: str = "discovery",
) -> dict[str, dict[str, Any]]:
    protocol = {
        "endpoint": TASK,
        "split": "TDC default scaffold validation",
        "phase": phase,
        "selection": "fixed recipes; no target-dependent selection",
        "test_labels_used": False,
        "official_test_loaded": False,
        "minimum_useful_mae_reduction": MINIMUM_USEFUL_MAE_REDUCTION,
    }

    def method(name: str) -> dict[str, Any]:
        value = summaries[name]
        return {
            "name": name,
            "scores": value["scores"],
            "cost": {"wall_seconds": value["total_wall_seconds"]},
        }

    if phase == "confirmation":
        return {
            "feature_ladder_confirmation": {
                "metric": METRIC,
                "higher_is_better": False,
                "seeds": seeds,
                "baseline": method("catboost_rdkit"),
                "candidates": [
                    method("catboost_rdkit_erg_morgan_count")
                ],
                "protocol": protocol,
            },
            "current_baseline_confirmation": {
                "metric": METRIC,
                "higher_is_better": False,
                "seeds": seeds,
                "baseline": method("lightgbm_current_binary_rich"),
                "candidates": [
                    method("catboost_rdkit_erg_morgan_count"),
                    method("lightgbm_full_count"),
                    method("extra_trees_full_count"),
                ],
                "protocol": protocol,
            },
        }

    ladder_names = [name for name, _ in FEATURE_LADDER]
    learner_names = (
        "catboost_rdkit_erg_morgan_count_avalon",
        "extra_trees_full_count",
        "random_forest_full_count",
    )
    current_names = (
        "catboost_current_binary_rich",
        "catboost_rdkit_erg_morgan_count_avalon",
        "lightgbm_full_count",
        "extra_trees_full_count",
        "random_forest_full_count",
    )
    return {
        "feature_ladder": {
            "metric": METRIC,
            "higher_is_better": False,
            "seeds": seeds,
            "baseline": method(ladder_names[0]),
            "candidates": [method(name) for name in ladder_names[1:]],
            "protocol": protocol,
        },
        "matched_learners": {
            "metric": METRIC,
            "higher_is_better": False,
            "seeds": seeds,
            "baseline": method("lightgbm_full_count"),
            "candidates": [method(name) for name in learner_names],
            "protocol": protocol,
        },
        "current_baseline": {
            "metric": METRIC,
            "higher_is_better": False,
            "seeds": seeds,
            "baseline": method("lightgbm_current_binary_rich"),
            "candidates": [method(name) for name in current_names],
            "protocol": protocol,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument(
        "--run-dir", type=Path, default=Path("runs/caco2-feature-ablation-v24")
    )
    parser.add_argument("--seeds", default="1,2,3,4,5")
    parser.add_argument(
        "--phase", choices=("discovery", "confirmation"), default="discovery"
    )
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    seeds = [int(value) for value in args.seeds.split(",") if value.strip()]
    if len(seeds) < 2 or len(set(seeds)) != len(seeds):
        parser.error("--seeds must contain at least two unique integers")
    if not 1 <= args.workers <= 256:
        parser.error("--workers must be between 1 and 256")

    run_dir = args.run_dir
    if (run_dir / "results.json").exists():
        raise FileExistsError(f"Refusing to overwrite completed run: {run_dir}")
    run_dir.mkdir(parents=True, exist_ok=True)
    recipes = _recipes(args.phase)
    preregistration = {
        "task": TASK,
        "metric": METRIC,
        "higher_is_better": False,
        "phase": args.phase,
        "seeds": seeds,
        "primary_hypothesis": (
            "Fresh-seed confirmation of the discovery-selected rich-2D "
            "baseline and CatBoost RDKit+ErG+Morgan-count candidate"
            if args.phase == "confirmation"
            else (
                "Forward addition RDKit -> ErG -> Morgan-count -> Avalon "
                "reduces paired scaffold-validation MAE for CatBoost"
            )
        ),
        "minimum_useful_mae_reduction": MINIMUM_USEFUL_MAE_REDUCTION,
        "recipes": [
            {**recipe, "representations": list(recipe["representations"])}
            for recipe in recipes
        ],
        "test_labels_used": False,
        "official_test_loaded": False,
        "promotion_rule": (
            "Confirmation must clear the preregistered practical threshold "
            "and isolation audit before freeze"
            if args.phase == "confirmation"
            else "Discovery winner requires fresh-seed confirmation before freeze"
        ),
    }
    (run_dir / "preregistration.json").write_text(
        json.dumps(preregistration, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    train_val_path = args.data_dir / "admet_group" / "caco2_wang" / "train_val.csv"
    environment = {
        "python": sys.version,
        "platform": platform.platform(),
        "packages": _packages(),
        "rdkit_runtime": rdBase.rdkitVersion,
        "feature_schema": FEATURE_SCHEMA_VERSION,
        "data_train_val_sha256": _sha256(train_val_path),
        "git": _git_state(),
        "workers": args.workers,
        "device": "cpu",
    }
    (run_dir / "environment.json").write_text(
        json.dumps(environment, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    task = resolve_task(TASK)
    group = admet_group(path=str(args.data_dir))
    cache = FeatureCache(run_dir / "feature_cache", device="cpu")
    records: dict[str, dict[int, dict[str, Any]]] = {
        recipe["name"]: {} for recipe in recipes
    }
    audits = []
    for seed in seeds:
        train, valid = group.get_train_valid_split(
            benchmark=TASK,
            split_type="default",
            seed=seed,
        )
        smiles_col, y_col = _columns(train)
        train_smiles = train[smiles_col].astype(str).tolist()
        valid_smiles = valid[smiles_col].astype(str).tolist()
        train_scaffolds = _scaffolds(train_smiles)
        valid_scaffolds = _scaffolds(valid_smiles)
        audit = {
            "seed": seed,
            "n_train": len(train),
            "n_valid": len(valid),
            "smiles_overlap": len(set(train_smiles) & set(valid_smiles)),
            "bemis_murcko_scaffold_overlap": len(
                train_scaffolds & valid_scaffolds
            ),
            "invalid_smiles_train": sum(
                Chem.MolFromSmiles(value) is None for value in train_smiles
            ),
            "invalid_smiles_valid": sum(
                Chem.MolFromSmiles(value) is None for value in valid_smiles
            ),
        }
        audits.append(audit)
        if audit["smiles_overlap"] or audit["bemis_murcko_scaffold_overlap"]:
            raise RuntimeError(f"Data-isolation audit failed for seed {seed}: {audit}")

        all_smiles = train_smiles + valid_smiles
        matrices: dict[tuple[str, ...], np.ndarray] = {}
        for recipe in recipes:
            name = recipe["name"]
            representations = tuple(recipe["representations"])
            if representations not in matrices:
                matrices[representations] = cache.get_or_compute(
                    all_smiles,
                    representations,
                )
            matrix = matrices[representations]
            experiment = ExperimentSpec(
                task=TASK,
                seed=seed,
                representations=representations,
                model=recipe["model"],
                device="cpu",
                n_jobs=args.workers,
                hypothesis=recipe["hypothesis"],
            )
            validate_experiment(experiment)
            started = time.perf_counter()
            model = build_model(experiment, task, train[y_col].to_numpy())
            model.fit(matrix[: len(train)], train[y_col].to_numpy())
            predictions = predict(model, matrix[len(train) :], task)
            elapsed = time.perf_counter() - started
            value = score(valid[y_col].to_numpy(), predictions, task.metric)
            records[name][seed] = {
                "score": value,
                "elapsed_seconds": elapsed,
                "n_features": int(matrix.shape[1]),
            }
            artifact_dir = run_dir / "artifacts" / f"seed_{seed}"
            artifact_dir.mkdir(parents=True, exist_ok=True)
            drug_id_col = "Drug_ID" if "Drug_ID" in valid else smiles_col
            np.savez_compressed(
                artifact_dir / f"{name}.npz",
                drug_ids=np.asarray(
                    valid[drug_id_col].astype(str).tolist(), dtype=np.str_
                ),
                smiles=np.asarray(
                    valid[smiles_col].astype(str).tolist(), dtype=np.str_
                ),
                targets=valid[y_col].to_numpy(dtype=float),
                predictions=np.asarray(predictions, dtype=float),
            )
            print(
                f"seed={seed} recipe={name} mae={value:.6f} "
                f"features={matrix.shape[1]} seconds={elapsed:.2f}",
                flush=True,
            )
        partial = {
            "preregistration": preregistration,
            "environment": environment,
            "data_isolation": audits,
            "records": records,
            "test_labels_used": False,
            "official_test_loaded": False,
        }
        (run_dir / "results.partial.json").write_text(
            json.dumps(partial, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    summaries = _summaries(recipes, records)
    comparisons = _comparison_payloads(seeds, summaries, args.phase)
    comparison_dir = run_dir / "comparisons"
    comparison_dir.mkdir(parents=True, exist_ok=True)
    for name, payload in comparisons.items():
        (comparison_dir / f"{name}.input.json").write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    result = {
        "preregistration": preregistration,
        "environment": environment,
        "data_isolation": audits,
        "summaries": summaries,
        "test_labels_used": False,
        "official_test_loaded": False,
        "decision": {
            "status": "confirm",
            "reason": (
                "Discovery results require paired comparison and fresh-seed "
                "confirmation before any freeze decision"
            ),
        },
    }
    (run_dir / "results.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
