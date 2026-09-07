"""Freeze OOF-selected thresholds and final admissible recipes without holdout access."""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import matthews_corrcoef, mean_absolute_error

from scripts.run_admet_fusion_development_endpoint import FUSION_WEIGHTS, SINGLE_ACTIONS


THRESHOLDS = tuple(round(value, 2) for value in np.arange(0.05, 1.0, 0.05))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _fold_scores(y: np.ndarray, prediction: np.ndarray, folds: np.ndarray, task_type: str, threshold: float | None) -> list[float]:
    scores: list[float] = []
    for fold in (0, 1, 2):
        mask = folds == fold
        if task_type == "classification":
            score = matthews_corrcoef(y[mask].astype(int), (prediction[mask] >= float(threshold)).astype(int))
        else:
            score = -mean_absolute_error(y[mask].astype(float), prediction[mask].astype(float))
        scores.append(float(score))
    return scores


def _best_threshold(y: np.ndarray, prediction: np.ndarray, folds: np.ndarray) -> tuple[float, list[float]]:
    candidates = []
    for threshold in THRESHOLDS:
        scores = _fold_scores(y, prediction, folds, "classification", threshold)
        candidates.append((float(np.mean(scores)), -abs(threshold - 0.5), -threshold, threshold, scores))
    return max(candidates, key=lambda row: row[:3])[3:]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-endpoints", type=int, default=372)
    parser.add_argument("--amendment", type=Path, required=True)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1)
    args = parser.parse_args()
    source_paths = sorted(args.input_dir.glob("*.manifest.json"))
    if len(source_paths) != args.expected_endpoints:
        raise RuntimeError(f"Expected {args.expected_endpoints} source manifests, found {len(source_paths)}")
    if args.shard_count < 1 or not 0 <= args.shard_index < args.shard_count:
        raise ValueError("Invalid shard index/count")
    source_paths = source_paths[args.shard_index::args.shard_count]
    args.output_dir.mkdir(parents=True, exist_ok=True)

    seen: set[str] = set()
    fusion_selected = 0
    for source_path in source_paths:
        source = json.loads(source_path.read_text())
        endpoint_id = str(source["endpoint_id"])
        if endpoint_id in seen:
            raise RuntimeError(f"Duplicate endpoint: {endpoint_id}")
        seen.add(endpoint_id)
        if source.get("holdout_labels_loaded") is not False or source.get("integrity_passed") is not True:
            raise RuntimeError(f"Invalid source development manifest: {endpoint_id}")
        original = pd.read_csv(source["result_path"])
        task_type = str(original["task_type"].iloc[0])
        with np.load(source["prediction_path"]) as archive:
            y = archive["target"]
            folds = archive["outer_fold"].astype(int)
            predictions = {action: archive[action] for action in SINGLE_ACTIONS}
        if sorted(np.unique(folds).tolist()) != [0, 1, 2]:
            raise RuntimeError(f"Unexpected OOF folds: {endpoint_id}")

        rows: list[dict] = []
        single_scores: dict[str, list[float]] = {}
        single_thresholds: dict[str, float | None] = {}
        for action in SINGLE_ACTIONS:
            if task_type == "classification":
                threshold, scores = _best_threshold(y, predictions[action], folds)
            else:
                threshold = None
                scores = _fold_scores(y, predictions[action], folds, task_type, None)
            single_scores[action] = scores
            single_thresholds[action] = threshold
            rows.append({
                "endpoint_id": endpoint_id, "task_type": task_type,
                "action": action, "kind": "single", "left": action, "right": None,
                "left_weight": 1.0, "threshold": threshold,
                "fold0": scores[0], "fold1": scores[1], "fold2": scores[2],
                "mean_oriented_score": float(np.mean(scores)),
                "gain_over_stronger_member": 0.0, "positive_fold_rate": 1.0,
                "fusion_supported": False,
            })
        for left, right in itertools.combinations(SINGLE_ACTIONS, 2):
            for left_weight in FUSION_WEIGHTS:
                prediction = left_weight * predictions[left] + (1.0 - left_weight) * predictions[right]
                if task_type == "classification":
                    threshold, scores = _best_threshold(y, prediction, folds)
                else:
                    threshold = None
                    scores = _fold_scores(y, prediction, folds, task_type, None)
                stronger = left if np.mean(single_scores[left]) >= np.mean(single_scores[right]) else right
                deltas = np.asarray(scores) - np.asarray(single_scores[stronger])
                gain = float(np.mean(deltas))
                positive_rate = float(np.mean(deltas > 0))
                supported = gain >= 0.005 and positive_rate >= (2.0 / 3.0)
                rows.append({
                    "endpoint_id": endpoint_id, "task_type": task_type,
                    "action": f"blend({left}@{left_weight:.2f}+{right}@{1-left_weight:.2f})",
                    "kind": "fusion", "left": left, "right": right,
                    "left_weight": left_weight, "threshold": threshold,
                    "fold0": scores[0], "fold1": scores[1], "fold2": scores[2],
                    "mean_oriented_score": float(np.mean(scores)),
                    "gain_over_stronger_member": gain, "positive_fold_rate": positive_rate,
                    "fusion_supported": supported,
                })
        table = pd.DataFrame(rows)
        admissible = table.loc[table["kind"].eq("single") | table["fusion_supported"]].copy()
        admissible["_kind_priority"] = admissible["kind"].map({"single": 0, "fusion": 1})
        selected = admissible.sort_values(
            ["mean_oriented_score", "_kind_priority", "action"],
            ascending=[False, True, True], kind="stable",
        ).iloc[0]
        table["selected"] = table["action"].eq(selected["action"])
        baseline = float(table.loc[table["action"].eq("extra_trees_morgan"), "mean_oriented_score"].iloc[0])
        if str(selected["kind"]) == "fusion":
            fusion_selected += 1
        result_path = args.output_dir / source["result_path"].split("/")[-1]
        manifest_path = args.output_dir / source_path.name
        table.to_csv(result_path, index=False)
        frozen = {
            "kind": str(selected["kind"]), "action": str(selected["action"]),
            "left": str(selected["left"]),
            "right": None if pd.isna(selected["right"]) else str(selected["right"]),
            "left_weight": float(selected["left_weight"]),
            "threshold": None if pd.isna(selected["threshold"]) else float(selected["threshold"]),
        }
        manifest = {
            "version": "admet-toxicology-oof-fusion-calibrated-development-v2",
            "endpoint_id": endpoint_id, "task_type": task_type,
            "source_manifest": str(source_path.resolve()), "source_manifest_sha256": _sha256(source_path),
            "prediction_path": source["prediction_path"],
            "prediction_sha256": source["prediction_sha256"],
            "threshold_amendment": str(args.amendment.resolve()),
            "threshold_amendment_sha256": _sha256(args.amendment),
            "threshold_grid": list(THRESHOLDS) if task_type == "classification" else [],
            "threshold_tie_break": "closest_to_0.5_then_lower",
            "single_thresholds": single_thresholds,
            "selected_recipe": frozen,
            "development_score": float(selected["mean_oriented_score"]),
            "development_baseline": baseline,
            "development_gain": float(selected["mean_oriented_score"] - baseline),
            "result_path": str(result_path.resolve()), "result_sha256": _sha256(result_path),
            "holdout_rows_loaded": False, "holdout_labels_loaded": False,
            "integrity_passed": len(table) == 51 and int(table["selected"].sum()) == 1,
        }
        manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "shard_index": args.shard_index, "shard_count": args.shard_count,
        "endpoints": len(seen), "fusion_selected": fusion_selected,
    }, sort_keys=True))


if __name__ == "__main__":
    main()
