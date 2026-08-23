"""Fresh, validation-only confirmation of the externally fixed MapLight+GNN.

The candidate is the exact MapLight+GNN classifier independently published and
reproduced by receptor-ai/tdc-admet-bench.  ``--preregister-only`` writes an
immutable protocol before any fresh split labels are read.  The evaluation path
uses only ``get_train_valid_split`` and never calls ``admet_group.get`` (which
would materialize TDC's official test frame).
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import re
import statistics
import subprocess
import time
from importlib.metadata import version
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from rdkit import Chem, DataStructs
from sklearn.metrics import average_precision_score, roc_auc_score
from tdc.benchmark_group import admet_group


TASKS = {
    "CYP2C9_Substrate_CarbonMangels": {
        "metric": "auprc",
        "external_official_score": 0.437,
        "current_top10_threshold": 0.392,
    },
    "Bioavailability_Ma": {
        "metric": "auroc",
        "external_official_score": 0.938,
        "current_top10_threshold": 0.922,
    },
}
SEEDS = [6, 7, 8, 9, 10]
UPSTREAM_COMMIT = "c249378c63232354d17083c83fe94fe728960a27"
AUDIT_COMMIT = "3e5472ce159a29b872219ffaaa4a11b4a2660f77"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _digest(payload: dict[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _git_commit(repository: Path) -> str:
    marker = repository / ".git-commit"
    metadata = repository / ".git"
    if marker.is_file() and not metadata.exists():
        commit = marker.read_text(encoding="utf-8").strip()
    else:
        try:
            commit = subprocess.check_output(
                ["git", "-C", str(repository), "rev-parse", "HEAD"],
                text=True,
                stderr=subprocess.DEVNULL,
            ).strip()
        except (subprocess.CalledProcessError, FileNotFoundError):
            if not marker.is_file():
                raise RuntimeError(
                    f"Neither Git metadata nor an archive commit marker exists: {repository}"
                )
            commit = marker.read_text(encoding="utf-8").strip()
    if not re.fullmatch(r"[0-9a-f]{40}", commit):
        raise RuntimeError(f"Invalid pinned repository commit: {commit!r}")
    return commit


def _load_upstream(path: Path):
    spec = importlib.util.spec_from_file_location("pinned_maplight_gnn", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import pinned MapLight+GNN source: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    # The upstream code uses int8 for count fingerprints.  Recent NumPy/RDKit
    # raises when a count exceeds 127, whereas older releases silently wrapped.
    # int32 preserves the actual count and is the sole compatibility patch.
    def count_to_array(fingerprint):
        values = np.zeros((0,), dtype=np.int32)
        DataStructs.ConvertToNumpyArray(fingerprint, values)
        return values

    module.count_to_array = count_to_array
    return module


def _canonical_identity(smiles: str) -> str:
    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        raise ValueError(f"Invalid SMILES in TDC development split: {smiles}")
    return Chem.MolToSmiles(molecule, canonical=True, isomericSmiles=True)


def _score(y_true: np.ndarray, y_pred: np.ndarray, metric: str) -> float:
    if metric == "auprc":
        return float(average_precision_score(y_true, y_pred))
    if metric == "auroc":
        return float(roc_auc_score(y_true, y_pred))
    raise ValueError(f"Unsupported metric: {metric}")


def preregistration(repo_root: Path, upstream_path: Path) -> dict[str, Any]:
    audit_repository = repo_root / "vendor/tdc-admet-bench-audit"
    source = upstream_path.resolve()
    payload = {
        "version": "maplight-gnn-fresh-confirmation-v1",
        "tasks": TASKS,
        "fresh_confirmation_seeds": SEEDS,
        "candidate": {
            "name": "MapLight+GNN",
            "status": "fixed independently before local confirmation",
            "features": [
                "Morgan count fingerprint (radius=2, 1024)",
                "Avalon count fingerprint (1024)",
                "ErG fingerprint",
                "fixed upstream RDKit descriptor list",
                "GIN supervised-masking pretrained embedding (300)",
            ],
            "classifier": {
                "type": "CatBoostClassifier",
                "parameters": {
                    "random_strength": 2,
                    "random_seed": "outer validation seed",
                    "verbose": 0,
                    "loss_function": "Logloss",
                },
                "all_other_parameters": "CatBoost 1.2.8 defaults",
            },
            "upstream_commit": UPSTREAM_COMMIT,
            "upstream_source_sha256": _sha256(source),
            "compatibility_patch": (
                "count_to_array allocation int8 -> int32; no feature/model tuning"
            ),
        },
        "independent_prior": {
            "audit_repository_commit": _git_commit(audit_repository),
            "expected_audit_commit": AUDIT_COMMIT,
            "paper_doi": "10.1021/acs.jcim.6c00819",
            "candidate_selected_from_fresh_labels": False,
        },
        "promotion_gate": {
            "per_task": {
                "mean_maplight_gnn_must_exceed_mean_maplight": True,
                "minimum_paired_seed_improvements": 3,
            },
            "all_tasks_must_pass": True,
            "no_official_evaluation_without_new_full_freeze": True,
        },
        "split_protocol": {
            "api": "admet_group.get_train_valid_split only",
            "split_type": "default",
            "identity_overlap_guard": "canonical isomeric SMILES",
            "official_test_loaded": False,
            "test_labels_used": False,
        },
    }
    if payload["independent_prior"]["audit_repository_commit"] != AUDIT_COMMIT:
        raise RuntimeError("Vendored independent-audit checkout moved")
    return {**payload, "preregistration_sha256": _digest(payload)}


def verify_preregistration(path: Path, expected: dict[str, Any]) -> None:
    observed = json.loads(path.read_text(encoding="utf-8"))
    if observed != expected:
        raise RuntimeError("MapLight+GNN preregistration changed")
    payload = {
        key: value
        for key, value in observed.items()
        if key != "preregistration_sha256"
    }
    if observed["preregistration_sha256"] != _digest(payload):
        raise RuntimeError("MapLight+GNN preregistration hash mismatch")


def _fit_predict(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_valid: np.ndarray,
    seed: int,
    workers: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    model = CatBoostClassifier(
        random_strength=2,
        random_seed=seed,
        verbose=0,
        loss_function="Logloss",
        thread_count=workers,
        allow_writing_files=False,
    )
    model.fit(x_train, y_train)
    return model.predict_proba(x_valid)[:, 1], model.get_all_params()


def _checkpoint_files() -> list[dict[str, Any]]:
    roots = [
        Path.home() / ".cache" / "molfeat",
        Path.home() / ".cache" / "torch" / "hub" / "checkpoints",
        Path.home() / ".dgl",
        Path.home() / ".cache" / "dgl",
    ]
    records = []
    for root in roots:
        if not root.exists():
            continue
        for path in sorted(item for item in root.rglob("*") if item.is_file()):
            records.append(
                {
                    "path": str(path),
                    "size": path.stat().st_size,
                    "sha256": _sha256(path),
                }
            )
    return records


def evaluate(
    repo_root: Path,
    data_dir: Path,
    upstream_path: Path,
    output_dir: Path,
    workers: int,
) -> dict[str, Any]:
    group = admet_group(path=str(data_dir))
    splits: dict[str, dict[int, dict[str, Any]]] = {}
    ordered_smiles: list[str] = []
    seen_smiles: set[str] = set()
    identity_audit: dict[str, dict[str, Any]] = {}

    for task in TASKS:
        splits[task] = {}
        identity_audit[task] = {}
        for seed in SEEDS:
            train, valid = group.get_train_valid_split(
                benchmark=task, split_type="default", seed=seed
            )
            train_smiles = train["Drug"].astype(str).tolist()
            valid_smiles = valid["Drug"].astype(str).tolist()
            train_ids = {_canonical_identity(value) for value in train_smiles}
            valid_ids = {_canonical_identity(value) for value in valid_smiles}
            overlap = sorted(train_ids.intersection(valid_ids))
            if overlap:
                raise RuntimeError(
                    f"{task} seed {seed}: {len(overlap)} canonical molecule "
                    "identities cross train/validation"
                )
            identity_audit[task][str(seed)] = {
                "train_rows": len(train),
                "valid_rows": len(valid),
                "train_unique_identities": len(train_ids),
                "valid_unique_identities": len(valid_ids),
                "cross_split_identity_overlap": 0,
            }
            splits[task][seed] = {
                "train_smiles": train_smiles,
                "valid_smiles": valid_smiles,
                "y_train": train["Y"].to_numpy(dtype=np.int64),
                "y_valid": valid["Y"].to_numpy(dtype=np.int64),
            }
            for value in train_smiles + valid_smiles:
                if value not in seen_smiles:
                    seen_smiles.add(value)
                    ordered_smiles.append(value)

    upstream = _load_upstream(upstream_path)
    feature_path = output_dir / "all_development_features.npy"
    feature_manifest_path = output_dir / "all_development_features.manifest.json"
    feature_manifest_core = {
        "ordered_smiles_sha256": hashlib.sha256(
            "\n".join(ordered_smiles).encode()
        ).hexdigest(),
        "rows": len(ordered_smiles),
        "upstream_source_sha256": _sha256(upstream_path),
        "compatibility_patch": "count_to_array int8 -> int32",
        "official_test_loaded": False,
    }
    use_cache = False
    if feature_path.exists() and feature_manifest_path.exists():
        cached_manifest = json.loads(feature_manifest_path.read_text(encoding="utf-8"))
        use_cache = all(
            cached_manifest.get(key) == value
            for key, value in feature_manifest_core.items()
        )
    if use_cache:
        features = np.load(feature_path, allow_pickle=False)
    else:
        features = np.asarray(
            upstream.get_fingerprints(pd.Series(ordered_smiles, dtype=str))
        )
        if features.shape[0] != len(ordered_smiles):
            raise RuntimeError("MapLight+GNN feature row count mismatch")
        np.save(feature_path, features, allow_pickle=False)
        manifest = {
            **feature_manifest_core,
            "shape": list(features.shape),
            "dtype": str(features.dtype),
            "feature_sha256": _sha256(feature_path),
            "checkpoint_files": _checkpoint_files(),
        }
        feature_manifest_path.write_text(
            json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
        )
    manifest = json.loads(feature_manifest_path.read_text(encoding="utf-8"))
    if list(features.shape) != manifest["shape"]:
        raise RuntimeError("Cached MapLight+GNN feature shape mismatch")
    if _sha256(feature_path) != manifest["feature_sha256"]:
        raise RuntimeError("Cached MapLight+GNN feature hash mismatch")

    row_for_smiles = {value: index for index, value in enumerate(ordered_smiles)}
    # The GIN embedding is appended last and is exactly 300 dimensions upstream.
    if features.shape[1] <= 300:
        raise RuntimeError("MapLight+GNN feature matrix is missing the GIN view")
    maplight_features = features[:, :-300]
    records: dict[str, Any] = {}
    predictions: dict[str, np.ndarray] = {}
    resolved_parameters: dict[str, Any] = {}
    started = time.monotonic()
    for task, task_config in TASKS.items():
        baseline_scores = []
        candidate_scores = []
        records[task] = {}
        for seed in SEEDS:
            split = splits[task][seed]
            train_rows = np.asarray(
                [row_for_smiles[value] for value in split["train_smiles"]], dtype=int
            )
            valid_rows = np.asarray(
                [row_for_smiles[value] for value in split["valid_smiles"]], dtype=int
            )
            y_train = split["y_train"]
            y_valid = split["y_valid"]
            baseline_pred, baseline_params = _fit_predict(
                maplight_features[train_rows],
                y_train,
                maplight_features[valid_rows],
                seed,
                workers,
            )
            candidate_pred, candidate_params = _fit_predict(
                features[train_rows],
                y_train,
                features[valid_rows],
                seed,
                workers,
            )
            baseline_score = _score(y_valid, baseline_pred, task_config["metric"])
            candidate_score = _score(y_valid, candidate_pred, task_config["metric"])
            baseline_scores.append(baseline_score)
            candidate_scores.append(candidate_score)
            predictions[f"{task}_seed_{seed}_y_valid"] = y_valid
            predictions[f"{task}_seed_{seed}_maplight"] = baseline_pred
            predictions[f"{task}_seed_{seed}_maplight_gnn"] = candidate_pred
            records[task][str(seed)] = {
                "maplight": baseline_score,
                "maplight_gnn": candidate_score,
                "paired_gain": candidate_score - baseline_score,
            }
            resolved_parameters.setdefault(
                "maplight", baseline_params
            )
            resolved_parameters.setdefault(
                "maplight_gnn", candidate_params
            )
            print(
                json.dumps(
                    {
                        "task": task,
                        "seed": seed,
                        "maplight": baseline_score,
                        "maplight_gnn": candidate_score,
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
        improvements = sum(
            candidate > baseline
            for candidate, baseline in zip(candidate_scores, baseline_scores)
        )
        records[task]["summary"] = {
            "metric": task_config["metric"],
            "maplight_scores": baseline_scores,
            "maplight_mean": statistics.mean(baseline_scores),
            "maplight_std": statistics.pstdev(baseline_scores),
            "maplight_gnn_scores": candidate_scores,
            "maplight_gnn_mean": statistics.mean(candidate_scores),
            "maplight_gnn_std": statistics.pstdev(candidate_scores),
            "mean_paired_gain": statistics.mean(candidate_scores)
            - statistics.mean(baseline_scores),
            "paired_seed_improvements": improvements,
            "gate_passed": (
                statistics.mean(candidate_scores) > statistics.mean(baseline_scores)
                and improvements >= 3
            ),
        }
    gate_passed = all(records[task]["summary"]["gate_passed"] for task in TASKS)
    predictions_path = output_dir / "fresh_predictions.npz"
    np.savez_compressed(predictions_path, **predictions)
    result = {
        "version": "maplight-gnn-fresh-confirmation-v1",
        "tasks": records,
        "identity_audit": identity_audit,
        "promotion_gate_passed": gate_passed,
        "resolved_catboost_parameters": resolved_parameters,
        "provenance": {
            "audit_repository_commit": _git_commit(
                repo_root / "vendor/tdc-admet-bench-audit"
            ),
            "upstream_commit": UPSTREAM_COMMIT,
            "upstream_source_sha256": _sha256(upstream_path),
            "feature_manifest_sha256": _sha256(feature_manifest_path),
            "predictions_sha256": _sha256(predictions_path),
            "catboost": version("catboost"),
            "molfeat": version("molfeat"),
            "dgl": version("dgl"),
            "dgllife": version("dgllife"),
            "torch": version("torch"),
            "runtime_note": (
                "torch 2.2.1 CPU wheel replaces the same-version default wheel; "
                "feature/model code and pinned DGL stack are unchanged"
            ),
            "compatibility_patch": "count_to_array int8 -> int32",
        },
        "protocol": {
            "fresh_seeds": SEEDS,
            "split_type": "TDC default train/validation scaffold split",
            "official_test_loaded": False,
            "test_labels_used": False,
            "fresh_labels_used_for_candidate_selection": False,
        },
        "elapsed_seconds": time.monotonic() - started,
    }
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument(
        "--upstream-path",
        type=Path,
        default=Path(
            "vendor/tdc-admet-bench-audit/models/maplight_gnn/"
            "upstream/maplight_gnn.py"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("runs/v27-maplight-gnn-confirmation"),
    )
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--preregister-only", action="store_true")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    expected = preregistration(args.repo_root.resolve(), args.upstream_path.resolve())
    prereg_path = args.output_dir / "preregistration.json"
    if prereg_path.exists():
        verify_preregistration(prereg_path, expected)
    else:
        prereg_path.write_text(
            json.dumps(expected, indent=2, sort_keys=True), encoding="utf-8"
        )
    if args.preregister_only:
        print(
            json.dumps(
                {
                    "status": "preregistered",
                    "sha256": expected["preregistration_sha256"],
                },
                sort_keys=True,
            )
        )
        return
    verify_preregistration(prereg_path, expected)
    result = evaluate(
        args.repo_root.resolve(),
        args.data_dir.resolve(),
        args.upstream_path.resolve(),
        args.output_dir.resolve(),
        args.workers,
    )
    result["preregistration_sha256"] = expected["preregistration_sha256"]
    result_path = args.output_dir / "result.json"
    result_path.write_text(
        json.dumps(result, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "status": "complete",
                "promotion_gate_passed": result["promotion_gate_passed"],
                "result": str(result_path),
                "official_test_loaded": False,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
