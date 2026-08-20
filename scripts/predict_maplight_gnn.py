"""Offline, leakage-safe predictor for the frozen MapLight+GNN expert."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from catboost import CatBoostClassifier
from molfeat.store import ModelStore
from molfeat.trans.pretrained import PretrainedDGLTransformer
from molfeat.trans.pretrained.dgl_pretrained import DGLModel
from rdkit import DataStructs


UPSTREAM_SHA256 = "74fbd1c98d9afa7fa4bda1add21efd429e20dee0a4b0fb8fa7e9b3825c21fe13"
CHECKPOINT_SHA256 = "6d0f8febad73e437772ebffc2ac32253d79f86ee138cfc233590ae50fb1cfeb9"
METADATA_SHA256 = "75ea305d643d800b8b272819a78b842d5aca1c4ad55d47207321ab9b81d44d02"
EXPECTED_WIDTH = 2863


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _digest(payload: Any) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _load_upstream(path: Path, checkpoint_root: Path):
    if _sha256(path) != UPSTREAM_SHA256:
        raise RuntimeError("Pinned MapLight+GNN source hash mismatch")
    model_path = checkpoint_root / "gin_supervised_masking/model.save"
    metadata_path = checkpoint_root / "gin_supervised_masking/metadata.json"
    if _sha256(model_path) != CHECKPOINT_SHA256:
        raise RuntimeError("Pinned GIN checkpoint hash mismatch")
    if _sha256(metadata_path) != METADATA_SHA256:
        raise RuntimeError("Pinned GIN metadata hash mismatch")
    spec = importlib.util.spec_from_file_location("frozen_maplight_gnn", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import pinned MapLight+GNN source: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    def count_to_array(fingerprint):
        values = np.zeros((0,), dtype=np.int32)
        DataStructs.ConvertToNumpyArray(fingerprint, values)
        return values

    def local_gin_supervised_masking(molecules):
        # ModelStore's default constructor scans its remote bucket even when the
        # model is cached.  A local store removes that nondeterministic network
        # dependency while loading the exact same content-addressed checkpoint.
        store = ModelStore(model_store_root=str(checkpoint_root))
        frozen = DGLModel(
            name="gin_supervised_masking",
            cache_path=str(checkpoint_root / "gin_supervised_masking"),
            store=store,
        )
        transformer = PretrainedDGLTransformer(kind=frozen, dtype=float)
        return transformer(molecules)

    module.count_to_array = count_to_array
    module.get_gin_supervised_masking = local_gin_supervised_masking
    return module


def _features(
    smiles: list[str],
    upstream_path: Path,
    checkpoint_root: Path,
    cache_root: Path,
) -> tuple[np.ndarray, dict[str, Any], Path]:
    cache_key = _digest(
        {
            "smiles": smiles,
            "upstream_sha256": UPSTREAM_SHA256,
            "checkpoint_sha256": CHECKPOINT_SHA256,
            "metadata_sha256": METADATA_SHA256,
            "compatibility_patch": "count_to_array int8 -> int32",
            "offline_loader": "local content-addressed ModelStore",
        }
    )
    cache_root.mkdir(parents=True, exist_ok=True)
    array_path = cache_root / f"{cache_key}.npy"
    manifest_path = cache_root / f"{cache_key}.manifest.json"
    expected = {
        "cache_key": cache_key,
        "rows": len(smiles),
        "upstream_sha256": UPSTREAM_SHA256,
        "checkpoint_sha256": CHECKPOINT_SHA256,
        "metadata_sha256": METADATA_SHA256,
        "compatibility_patch": "count_to_array int8 -> int32",
        "offline_loader": "local content-addressed ModelStore",
    }
    cache_hit = False
    if array_path.exists() and manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        cache_hit = all(manifest.get(key) == value for key, value in expected.items())
        cache_hit = cache_hit and manifest.get("feature_sha256") == _sha256(array_path)
    if cache_hit:
        values = np.load(array_path, allow_pickle=False)
    else:
        upstream = _load_upstream(upstream_path, checkpoint_root)
        values = np.asarray(upstream.get_fingerprints(pd.Series(smiles, dtype=str)))
        if values.shape != (len(smiles), EXPECTED_WIDTH):
            raise RuntimeError(
                f"Unexpected MapLight+GNN feature shape: {values.shape}"
            )
        np.save(array_path, values, allow_pickle=False)
        manifest = {
            **expected,
            "shape": list(values.shape),
            "dtype": str(values.dtype),
            "feature_sha256": _sha256(array_path),
        }
        manifest_path.write_text(
            json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
        )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if values.shape != (len(smiles), EXPECTED_WIDTH):
        raise RuntimeError("Cached MapLight+GNN feature shape mismatch")
    return values, {**manifest, "cache_hit": cache_hit}, manifest_path


def predict(args: argparse.Namespace) -> dict[str, Any]:
    payload = json.loads(args.input.read_text(encoding="utf-8"))
    required = {
        "task",
        "seed",
        "workers",
        "fit_smiles",
        "fit_labels",
        "predict_smiles",
        "predict_labels_present",
    }
    if set(payload) != required:
        raise RuntimeError("Frozen MapLight+GNN input schema changed")
    if payload["predict_labels_present"] is not False:
        raise RuntimeError("Frozen MapLight+GNN refused prediction labels")
    fit_smiles = [str(value) for value in payload["fit_smiles"]]
    predict_smiles = [str(value) for value in payload["predict_smiles"]]
    y_fit = np.asarray(payload["fit_labels"], dtype=np.int64)
    if len(fit_smiles) != len(y_fit) or not fit_smiles or not predict_smiles:
        raise RuntimeError("Frozen MapLight+GNN input lengths are invalid")
    if set(np.unique(y_fit)) - {0, 1}:
        raise RuntimeError("Frozen MapLight+GNN supports binary labels only")
    workers = max(1, min(int(payload["workers"]), 32))
    torch.set_num_threads(workers)
    all_smiles = fit_smiles + predict_smiles
    features, feature_manifest, feature_manifest_path = _features(
        all_smiles,
        args.upstream,
        args.checkpoint_root,
        args.cache_root,
    )
    model = CatBoostClassifier(
        random_strength=2,
        random_seed=int(payload["seed"]),
        verbose=0,
        loss_function="Logloss",
        thread_count=workers,
        allow_writing_files=False,
    )
    model.fit(features[: len(fit_smiles)], y_fit)
    predictions = model.predict_proba(features[len(fit_smiles) :])[:, 1]
    if predictions.shape != (len(predict_smiles),) or not np.isfinite(predictions).all():
        raise RuntimeError("Frozen MapLight+GNN produced invalid predictions")
    np.save(args.output, predictions, allow_pickle=False)
    report = {
        "task": payload["task"],
        "seed": int(payload["seed"]),
        "fit_rows": len(fit_smiles),
        "predict_rows": len(predict_smiles),
        "predict_labels_present": False,
        "input_sha256": _sha256(args.input),
        "predictions_sha256": _sha256(args.output),
        "feature_manifest": str(feature_manifest_path),
        "feature_manifest_sha256": _sha256(feature_manifest_path),
        "feature_cache_hit": feature_manifest["cache_hit"],
        "feature_shape": feature_manifest["shape"],
        "upstream_sha256": UPSTREAM_SHA256,
        "checkpoint_sha256": CHECKPOINT_SHA256,
        "metadata_sha256": METADATA_SHA256,
        "catboost_parameters": model.get_all_params(),
        "runtime": {
            "torch": torch.__version__,
            "torch_threads": workers,
            "offline_checkpoint_loader": True,
        },
    }
    args.report.write_text(
        json.dumps(report, indent=2, sort_keys=True), encoding="utf-8"
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument(
        "--upstream",
        type=Path,
        default=Path(
            "vendor/tdc-admet-bench-audit/models/maplight_gnn/"
            "upstream/maplight_gnn.py"
        ),
    )
    parser.add_argument(
        "--checkpoint-root",
        type=Path,
        default=Path.home() / ".cache/molfeat",
    )
    parser.add_argument(
        "--cache-root",
        type=Path,
        default=Path("runs/_maplight_gnn_feature_cache"),
    )
    args = parser.parse_args()
    report = predict(args)
    print(json.dumps(report, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
