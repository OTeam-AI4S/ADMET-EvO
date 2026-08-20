"""Validation-only reproduction and MiniMol extension of MapLight.

The MapLight feature implementation is imported from a pinned vendored checkout
of the authors' MIT-licensed repository.  The script consumes only the exported
development splits used by the MiniMol validation study and never loads TDC's
official test frame.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import statistics
import time
from importlib.metadata import version
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from rdkit import DataStructs
from sklearn.metrics import average_precision_score, roc_auc_score


def _load_maplight(repository: Path):
    path = repository / "maplight.py"
    spec = importlib.util.spec_from_file_location("pinned_maplight", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import MapLight from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    # The authors' code allocated count fingerprints as int8.  Newer NumPy
    # raises instead of silently wrapping counts above 127.  Use a wider integer
    # without altering the vendored source or the fingerprint values.
    def count_to_array(fingerprint):
        values = np.zeros((0,), dtype=np.int32)
        DataStructs.ConvertToNumpyArray(fingerprint, values)
        return values

    module.count_to_array = count_to_array
    return module


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _score(labels: np.ndarray, predictions: np.ndarray, metric: str) -> float:
    if metric == "auprc":
        return float(average_precision_score(labels, predictions))
    if metric == "auroc":
        return float(roc_auc_score(labels, predictions))
    raise ValueError(f"Unsupported metric: {metric}")


def _fit_member(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_valid: np.ndarray,
    seed: int,
    workers: int,
    balanced: bool,
) -> tuple[np.ndarray, dict]:
    params = {
        "random_strength": 2,
        "random_seed": seed,
        "verbose": 0,
        "loss_function": "Logloss",
        "thread_count": workers,
        "allow_writing_files": False,
    }
    if balanced:
        params["auto_class_weights"] = "Balanced"
    model = CatBoostClassifier(**params)
    model.fit(x_train, y_train)
    return model.predict_proba(x_valid)[:, 1], model.get_all_params()


def evaluate(
    payload: dict,
    smiles_payload: dict,
    maplight_features: np.ndarray,
    minimol_features: np.ndarray,
    members: int,
    workers: int,
    lcb_beta: float,
) -> tuple[dict, dict[str, np.ndarray]]:
    if payload.get("official_test_loaded") is not False:
        raise RuntimeError("MapLight evaluator refused a test-bearing split")
    if smiles_payload.get("official_test_loaded") is not False:
        raise RuntimeError("MapLight evaluator refused a test-bearing SMILES file")
    views = {
        "maplight": maplight_features,
        "maplight_minimol": np.concatenate(
            [maplight_features, minimol_features], axis=1
        ),
    }
    candidates = {
        "maplight_official": ("maplight", False, 1),
        "maplight_bagged": ("maplight", False, members),
        "maplight_minimol_bagged": ("maplight_minimol", False, members),
        "maplight_minimol_balanced_bagged": (
            "maplight_minimol",
            True,
            members,
        ),
    }
    fixed_scores = {name: [] for name in candidates}
    predictions = {}
    records = {}
    resolved_parameters = {}
    started = time.monotonic()
    for outer_seed in payload["seeds"]:
        split = payload["splits"][str(outer_seed)]
        train_ids = np.asarray(split["train_ids"], dtype=int)
        valid_ids = np.asarray(split["valid_ids"], dtype=int)
        train_y = np.asarray(split["train_y"], dtype=int)
        valid_y = np.asarray(split["valid_y"], dtype=int)
        if set(train_ids).intersection(valid_ids):
            raise RuntimeError(f"Outer seed {outer_seed} has overlapping molecule IDs")
        records[str(outer_seed)] = {}
        print(
            json.dumps({"outer_seed": outer_seed, "status": "started"}),
            flush=True,
        )
        for candidate_index, (name, (view, balanced, n_members)) in enumerate(
            candidates.items()
        ):
            print(
                json.dumps(
                    {
                        "candidate": name,
                        "members": n_members,
                        "outer_seed": outer_seed,
                        "status": "started",
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
            members_predictions = []
            member_scores = []
            for member in range(n_members):
                seed = outer_seed * 10000 + candidate_index * 100 + member
                values, parameters = _fit_member(
                    views[view][train_ids],
                    train_y,
                    views[view][valid_ids],
                    seed,
                    workers,
                    balanced,
                )
                resolved_parameters.setdefault(name, parameters)
                members_predictions.append(values)
                member_scores.append(_score(valid_y, values, payload["metric"]))
            ensemble = np.mean(members_predictions, axis=0)
            score = _score(valid_y, ensemble, payload["metric"])
            fixed_scores[name].append(score)
            predictions[f"seed_{outer_seed}_{name}"] = ensemble
            records[str(outer_seed)][name] = {
                "score": score,
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
    loso_scores = []
    loso_selected = {}
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
        score = fixed_scores[selected][heldout_index]
        loso_scores.append(score)
        loso_selected[str(heldout_seed)] = {
            "candidate": selected,
            "utility": utilities[selected],
            "heldout_score": score,
        }
    return {
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
        "outer_records": records,
        "resolved_catboost_parameters": resolved_parameters,
        "protocol": {
            "outer_seeds": payload["seeds"],
            "outer_split": payload["split_type"],
            "members": members,
            "official_maplight_candidate": "authors' features and default classifier parameters",
            "official_test_loaded": False,
            "test_labels_used": False,
        },
        "elapsed_seconds": time.monotonic() - started,
    }, predictions


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root", type=Path, default=Path("runs/v24-minimol-validation")
    )
    parser.add_argument("--task", default="CYP2C9_Substrate_CarbonMangels")
    parser.add_argument(
        "--maplight-repository",
        type=Path,
        default=Path("vendor/MapLight-TDC"),
    )
    parser.add_argument("--members", type=int, default=8)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--lcb-beta", type=float, default=0.25)
    args = parser.parse_args()
    task_root = args.root / args.task
    splits_path = task_root / "splits.json"
    smiles_path = task_root / "smiles.json"
    embeddings_path = task_root / "embeddings.npy"
    payload = json.loads(splits_path.read_text(encoding="utf-8"))
    smiles_payload = json.loads(smiles_path.read_text(encoding="utf-8"))
    if smiles_payload.get("contains_labels") is not False:
        raise RuntimeError("MapLight featurization accepts only a label-free manifest")
    maplight = _load_maplight(args.maplight_repository)
    maplight_path = args.maplight_repository / "maplight.py"
    maplight_features_path = task_root / "maplight_features.npy"
    maplight_manifest_path = task_root / "maplight_features.manifest.json"
    expected_feature_manifest = {
        "compatibility_patch": "count_to_array int8 -> int32 to prevent overflow",
        "maplight_py_sha256": _sha256(maplight_path),
        "smiles_sha256": _sha256(smiles_path),
    }
    use_cache = False
    if maplight_features_path.exists() and maplight_manifest_path.exists():
        cached_manifest = json.loads(
            maplight_manifest_path.read_text(encoding="utf-8")
        )
        use_cache = cached_manifest == expected_feature_manifest
    if use_cache:
        maplight_features = np.load(maplight_features_path, allow_pickle=False)
        print(
            json.dumps(
                {
                    "feature_cache": str(maplight_features_path),
                    "shape": list(maplight_features.shape),
                    "status": "loaded",
                },
                sort_keys=True,
            ),
            flush=True,
        )
    else:
        print(
            json.dumps({"feature_cache": "miss", "status": "computing"}),
            flush=True,
        )
        maplight_features = maplight.get_fingerprints(
            pd.Series(smiles_payload["smiles"], dtype=str)
        )
        np.save(maplight_features_path, maplight_features, allow_pickle=False)
        maplight_manifest_path.write_text(
            json.dumps(expected_feature_manifest, indent=2, sort_keys=True),
            encoding="utf-8",
        )
    minimol_features = np.load(embeddings_path, allow_pickle=False)
    if maplight_features.shape[0] != minimol_features.shape[0]:
        raise ValueError("MapLight and MiniMol row counts differ")
    result, predictions = evaluate(
        payload,
        smiles_payload,
        maplight_features,
        minimol_features,
        args.members,
        args.workers,
        args.lcb_beta,
    )
    result["provenance"] = {
        "maplight_git_commit": "c249378c63232354d17083c83fe94fe728960a27",
        "maplight_py_sha256": _sha256(maplight_path),
        "splits_sha256": _sha256(splits_path),
        "smiles_sha256": _sha256(smiles_path),
        "embeddings_sha256": _sha256(embeddings_path),
        "catboost": version("catboost"),
        "license": "MIT",
        "compatibility_patch": "count_to_array int8 -> int32 to prevent overflow",
    }
    output = task_root / "maplight_minimol.json"
    output.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    np.savez_compressed(task_root / "maplight_minimol_predictions.npz", **predictions)
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
