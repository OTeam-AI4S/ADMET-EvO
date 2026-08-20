"""Run outer-train-only nested RDKit descriptor selection for Caco2.

The feature count is selected with inner scaffold folds. Within each inner
development split, permutation ranking uses a second scaffold holdout. The
official TDC test split is never requested.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.Chem.Scaffolds import MurckoScaffold
from tdc.benchmark_group import admet_group

from admet_agent.feature_selection import nested_permutation_select
from admet_agent.features import FeatureCache
from admet_agent.guard import validate_experiment
from admet_agent.runner import _columns
from admet_agent.schemas import ExperimentSpec
from admet_agent.selection import build_model, predict, score
from admet_agent.tasks import resolve_task

TASK = "Caco2_Wang"
REPRESENTATIONS = ("rdkit2d",)
CANDIDATE_COUNTS = (32, 64, 96, 128)
MINIMUM_USEFUL_MAE_REDUCTION = 0.01


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument(
        "--run-dir", type=Path, default=Path("runs/caco2-nested-selection-v24")
    )
    parser.add_argument("--model", choices=("catboost", "lightgbm"), required=True)
    parser.add_argument("--seeds", default="1,2,3,4,5")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--inner-splits", type=int, default=5)
    parser.add_argument("--permutation-repeats", type=int, default=3)
    args = parser.parse_args()
    seeds = [int(value) for value in args.seeds.split(",") if value.strip()]
    if len(seeds) < 2 or len(set(seeds)) != len(seeds):
        parser.error("--seeds must contain at least two unique integers")
    if (args.run_dir / "results.json").exists():
        raise FileExistsError(f"Refusing to overwrite completed run: {args.run_dir}")
    args.run_dir.mkdir(parents=True, exist_ok=True)

    preregistration = {
        "task": TASK,
        "metric": "mae",
        "phase": "discovery",
        "model": args.model,
        "representations": list(REPRESENTATIONS),
        "seeds": seeds,
        "inner_splits": args.inner_splits,
        "selector_holdout_fraction": 0.2,
        "permutation_repeats": args.permutation_repeats,
        "candidate_feature_counts": [*CANDIDATE_COUNTS, len(Descriptors._descList)],
        "minimum_useful_mae_reduction": MINIMUM_USEFUL_MAE_REDUCTION,
        "primary_hypothesis": (
            "outer-train-only nested permutation selection reduces paired "
            "Caco2 scaffold-validation MAE versus all RDKit descriptors"
        ),
        "test_labels_used": False,
        "official_test_loaded": False,
        "promotion_rule": "fresh-seed confirmation required before freeze",
    }
    (args.run_dir / "preregistration.json").write_text(
        json.dumps(preregistration, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    data_path = args.data_dir / "admet_group" / "caco2_wang" / "train_val.csv"
    environment = {
        "data_train_val_sha256": _sha256(data_path),
        "workers": args.workers,
        "device": "cpu",
        "descriptor_names": [name for name, _ in Descriptors._descList],
    }
    (args.run_dir / "environment.json").write_text(
        json.dumps(environment, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    group = admet_group(path=str(args.data_dir))
    task = resolve_task(TASK)
    cache = FeatureCache(args.run_dir / "feature_cache", device="cpu")
    records: dict[str, dict[int, dict[str, Any]]] = {
        "all_rdkit2d": {},
        "nested_selected_rdkit2d": {},
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
        audit = {
            "seed": seed,
            "n_train": len(train),
            "n_valid": len(valid),
            "smiles_overlap": len(set(train_smiles) & set(valid_smiles)),
            "bemis_murcko_scaffold_overlap": len(
                _scaffolds(train_smiles) & _scaffolds(valid_smiles)
            ),
        }
        audits.append(audit)
        if audit["smiles_overlap"] or audit["bemis_murcko_scaffold_overlap"]:
            raise RuntimeError(f"Data-isolation audit failed: {audit}")

        all_smiles = train_smiles + valid_smiles
        features = cache.get_or_compute(all_smiles, REPRESENTATIONS)
        x_train = features[: len(train)]
        x_valid = features[len(train) :]
        y_train = train[y_col].to_numpy(dtype=float)
        y_valid = valid[y_col].to_numpy(dtype=float)
        feature_names = [name for name, _ in Descriptors._descList]
        if len(feature_names) != x_train.shape[1]:
            raise RuntimeError("RDKit descriptor names do not match feature width")

        def estimator_factory(
            model_seed: int,
            training_targets: np.ndarray = y_train,
        ):
            experiment = ExperimentSpec(
                task=TASK,
                seed=model_seed,
                representations=REPRESENTATIONS,
                model=args.model,
                device="cpu",
                n_jobs=args.workers,
                hypothesis="nested descriptor selection estimator",
            )
            validate_experiment(experiment)
            return build_model(experiment, task, training_targets)

        baseline_started = time.perf_counter()
        baseline_model = estimator_factory(seed)
        baseline_model.fit(x_train, y_train)
        baseline_predictions = predict(baseline_model, x_valid, task)
        baseline_elapsed = time.perf_counter() - baseline_started
        baseline_score = score(y_valid, baseline_predictions, task.metric)

        selector_started = time.perf_counter()
        selection = nested_permutation_select(
            x_train,
            y_train,
            train_smiles,
            estimator_factory,
            candidate_feature_counts=CANDIDATE_COUNTS,
            feature_names=feature_names,
            inner_splits=args.inner_splits,
            selector_holdout_fraction=0.2,
            permutation_repeats=args.permutation_repeats,
            random_state=seed,
        )
        selected = np.asarray(selection.selected_indices, dtype=int)
        selected_model = estimator_factory(seed)
        selected_model.fit(x_train[:, selected], y_train)
        selected_predictions = predict(selected_model, x_valid[:, selected], task)
        selected_elapsed = time.perf_counter() - selector_started
        selected_score = score(y_valid, selected_predictions, task.metric)
        records["all_rdkit2d"][seed] = {
            "score": baseline_score,
            "elapsed_seconds": baseline_elapsed,
            "n_features": int(x_train.shape[1]),
        }
        records["nested_selected_rdkit2d"][seed] = {
            "score": selected_score,
            "elapsed_seconds": selected_elapsed,
            "n_features": selection.chosen_feature_count,
            "mean_selected_set_jaccard": selection.mean_selected_set_jaccard,
        }
        artifact_dir = args.run_dir / "artifacts" / f"seed_{seed}"
        artifact_dir.mkdir(parents=True, exist_ok=True)
        (artifact_dir / "selection.json").write_text(
            json.dumps(selection.to_dict(), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        drug_id_col = "Drug_ID" if "Drug_ID" in valid else smiles_col
        np.savez_compressed(
            artifact_dir / "validation_predictions.npz",
            drug_ids=np.asarray(
                valid[drug_id_col].astype(str).tolist(), dtype=np.str_
            ),
            smiles=np.asarray(valid_smiles, dtype=np.str_),
            targets=y_valid,
            all_rdkit2d=np.asarray(baseline_predictions, dtype=float),
            nested_selected_rdkit2d=np.asarray(selected_predictions, dtype=float),
        )
        print(
            f"seed={seed} all_mae={baseline_score:.6f} "
            f"selected_mae={selected_score:.6f} "
            f"features={selection.chosen_feature_count}/{x_train.shape[1]} "
            f"selection_seconds={selected_elapsed:.2f}",
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
        (args.run_dir / "results.partial.json").write_text(
            json.dumps(partial, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    def summary(name: str) -> dict[str, Any]:
        seed_records = records[name]
        values = [seed_records[seed]["score"] for seed in seeds]
        elapsed = [seed_records[seed]["elapsed_seconds"] for seed in seeds]
        return {
            "name": name,
            "scores": values,
            "mean": float(np.mean(values)),
            "sample_std": float(np.std(values, ddof=1)),
            "total_wall_seconds": float(np.sum(elapsed)),
        }

    baseline_summary = summary("all_rdkit2d")
    selected_summary = summary("nested_selected_rdkit2d")
    comparison = {
        "metric": "mae",
        "higher_is_better": False,
        "seeds": seeds,
        "baseline": {
            "name": baseline_summary["name"],
            "scores": baseline_summary["scores"],
            "cost": {"wall_seconds": baseline_summary["total_wall_seconds"]},
        },
        "candidates": [
            {
                "name": selected_summary["name"],
                "scores": selected_summary["scores"],
                "cost": {"wall_seconds": selected_summary["total_wall_seconds"]},
            }
        ],
        "protocol": {
            "endpoint": TASK,
            "split": "TDC default scaffold validation",
            "phase": "discovery",
            "selection": (
                "outer-train-only nested scaffold CV with permutation ranking "
                "on a second scaffold holdout"
            ),
            "test_labels_used": False,
            "official_test_loaded": False,
            "minimum_useful_mae_reduction": MINIMUM_USEFUL_MAE_REDUCTION,
        },
    }
    comparison_dir = args.run_dir / "comparisons"
    comparison_dir.mkdir(parents=True, exist_ok=True)
    (comparison_dir / "nested_selection.input.json").write_text(
        json.dumps(comparison, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    result = {
        "preregistration": preregistration,
        "environment": environment,
        "data_isolation": audits,
        "summaries": {
            "all_rdkit2d": baseline_summary,
            "nested_selected_rdkit2d": selected_summary,
        },
        "test_labels_used": False,
        "official_test_loaded": False,
        "decision": {
            "status": "confirm",
            "reason": "paired comparison and fresh-seed confirmation required",
        },
    }
    (args.run_dir / "results.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
