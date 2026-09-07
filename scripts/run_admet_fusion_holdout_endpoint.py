"""Evaluate one development-frozen Agent recipe on untouched folds 3 and 4."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import matthews_corrcoef

from admet_agent.features import MolecularFeaturizer
from scripts.run_admet_fusion_development_endpoint import (
    RICH_REPRESENTATIONS,
    SINGLE_ACTIONS,
    _model,
    _predict,
    _score,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _safe(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "__", value).strip("_")


def _score_with_threshold(
    y: np.ndarray, prediction: np.ndarray, task_type: str, threshold: float | None,
) -> float:
    if task_type == "classification":
        if threshold is None:
            raise RuntimeError("Classification recipe lacks a frozen threshold")
        return float(matthews_corrcoef(y.astype(int), (prediction >= threshold).astype(int)))
    return _score(y, prediction, task_type)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cohort", type=Path, required=True)
    parser.add_argument("--recipe-dir", type=Path, required=True)
    parser.add_argument("--endpoint")
    parser.add_argument("--array-index", type=int)
    parser.add_argument("--development-seed", type=int, default=17)
    parser.add_argument("--confirmation-seeds", default="29,43")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    cohort_path = args.cohort.resolve()
    cohort = pd.read_csv(cohort_path)
    endpoints = sorted(cohort["endpoint_id"].astype(str).unique())
    index = args.array_index
    if args.endpoint is None:
        if index is None:
            index = int(os.environ["SLURM_ARRAY_TASK_ID"])
        if not 0 <= index < len(endpoints):
            raise ValueError("Array index out of bounds")
        endpoint_id = endpoints[index]
    else:
        endpoint_id = args.endpoint
    endpoint = cohort.loc[cohort["endpoint_id"].astype(str).eq(endpoint_id)].copy()
    if endpoint.empty:
        raise ValueError("Unknown endpoint")
    folds = [int(value) for value in sorted(endpoint["outer_fold"].astype(int).unique())]
    if folds != [0, 1, 2, 3, 4]:
        raise RuntimeError(f"Confirmation requires frozen folds 0-4; got {folds}")
    if int((endpoint.groupby("chemistry_group")["outer_fold"].nunique() > 1).sum()):
        raise RuntimeError("Chemistry-group leakage")

    recipe_path = args.recipe_dir / f"{_safe(endpoint_id)}__seed{args.development_seed}.manifest.json"
    if not recipe_path.exists():
        raise FileNotFoundError(f"Missing frozen development recipe: {recipe_path}")
    recipe_manifest = json.loads(recipe_path.read_text())
    if recipe_manifest["endpoint_id"] != endpoint_id:
        raise RuntimeError("Recipe endpoint mismatch")
    if recipe_manifest.get("holdout_labels_loaded") is not False:
        raise RuntimeError("Development recipe does not certify holdout-label firewall")
    frozen = recipe_manifest["selected_recipe"]
    single_thresholds = recipe_manifest.get(
        "single_thresholds", {action: 0.5 if str(endpoint["task_type"].iloc[0]) == "classification" else None for action in SINGLE_ACTIONS}
    )
    required_actions = {frozen["left"]}
    if frozen["right"] is not None:
        required_actions.add(frozen["right"])
    if not required_actions.issubset(set(SINGLE_ACTIONS)):
        raise RuntimeError("Frozen recipe references an unknown action")

    task_type = str(endpoint["task_type"].iloc[0])
    metric = "mcc" if task_type == "classification" else "mae"
    train_mask = endpoint["outer_fold"].astype(int).isin([0, 1, 2]).to_numpy()
    holdout_mask = endpoint["outer_fold"].astype(int).isin([3, 4]).to_numpy()
    train = endpoint.loc[train_mask]
    holdout = endpoint.loc[holdout_mask]
    identities = endpoint[["inchikey", "standard_smiles"]].drop_duplicates("inchikey")
    identities = identities.sort_values("inchikey", kind="stable").reset_index(drop=True)
    position = {value: position for position, value in enumerate(identities["inchikey"])}
    smiles = identities["standard_smiles"].astype(str).tolist()
    features = {
        "morgan": MolecularFeaturizer(("morgan",)).transform(smiles),
        "rich": MolecularFeaturizer(RICH_REPRESENTATIONS).transform(smiles),
    }
    train_index = np.asarray([position[value] for value in train["inchikey"]], dtype=int)
    holdout_index = np.asarray([position[value] for value in holdout["inchikey"]], dtype=int)
    y_train = train["target"].to_numpy()
    y_holdout = holdout["target"].to_numpy()
    seeds = [int(value) for value in args.confirmation_seeds.split(",") if value.strip()]
    if not seeds or set(seeds) & {args.development_seed}:
        raise ValueError("Confirmation seeds must be nonempty and disjoint from development seed")

    rows: list[dict] = []
    prediction_payload: dict[str, np.ndarray] = {
        "inchikey": holdout["inchikey"].astype(str).to_numpy(),
        "outer_fold": holdout["outer_fold"].astype(int).to_numpy(),
        "target": y_holdout,
    }
    for seed in seeds:
        predictions: dict[str, np.ndarray] = {}
        for action in SINGLE_ACTIONS:
            representation = "morgan" if action == "extra_trees_morgan" else "rich"
            started = time.monotonic()
            model = _model(action, task_type, seed, args.workers, y_train)
            model.fit(features[representation][train_index], y_train)
            prediction = _predict(model, features[representation][holdout_index], task_type)
            elapsed = time.monotonic() - started
            predictions[action] = prediction
            prediction_payload[f"seed{seed}__{action}"] = prediction
            threshold = single_thresholds[action]
            rows.append({
                "endpoint_id": endpoint_id, "task_type": task_type, "metric": metric,
                "seed": seed, "model": action, "kind": "single",
                "threshold": threshold,
                "score": _score_with_threshold(y_holdout, prediction, task_type, threshold),
                "fold3_score": _score_with_threshold(
                    y_holdout[holdout["outer_fold"].astype(int).eq(3).to_numpy()],
                    prediction[holdout["outer_fold"].astype(int).eq(3).to_numpy()], task_type, threshold,
                ),
                "fold4_score": _score_with_threshold(
                    y_holdout[holdout["outer_fold"].astype(int).eq(4).to_numpy()],
                    prediction[holdout["outer_fold"].astype(int).eq(4).to_numpy()], task_type, threshold,
                ),
                "fit_seconds": elapsed,
            })
        agent_prediction = predictions[frozen["left"]].copy()
        if frozen["right"] is not None:
            weight = float(frozen["left_weight"])
            agent_prediction = weight * agent_prediction + (1.0 - weight) * predictions[frozen["right"]]
        prediction_payload[f"seed{seed}__agent_frozen"] = agent_prediction
        agent_threshold = frozen["threshold"]
        rows.append({
            "endpoint_id": endpoint_id, "task_type": task_type, "metric": metric,
            "seed": seed, "model": "agent_frozen", "kind": str(frozen["kind"]),
            "threshold": agent_threshold,
            "score": _score_with_threshold(y_holdout, agent_prediction, task_type, agent_threshold),
            "fold3_score": _score_with_threshold(
                y_holdout[holdout["outer_fold"].astype(int).eq(3).to_numpy()],
                agent_prediction[holdout["outer_fold"].astype(int).eq(3).to_numpy()], task_type, agent_threshold,
            ),
            "fold4_score": _score_with_threshold(
                y_holdout[holdout["outer_fold"].astype(int).eq(4).to_numpy()],
                agent_prediction[holdout["outer_fold"].astype(int).eq(4).to_numpy()], task_type, agent_threshold,
            ),
            "fit_seconds": 0.0,
        })

    table = pd.DataFrame(rows)
    agent = table.loc[table["model"].eq("agent_frozen")]
    baseline = table.loc[table["model"].eq("extra_trees_morgan")]
    summary = {
        "agent_mean_oriented_score": float(agent["score"].mean()),
        "baseline_mean_oriented_score": float(baseline["score"].mean()),
        "agent_gain": float(agent["score"].mean() - baseline["score"].mean()),
        "agent_wins_by_seed": int(sum(
            agent.set_index("seed")["score"] > baseline.set_index("seed")["score"]
        )),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{_safe(endpoint_id)}__confirm"
    result_path = args.output_dir / f"{stem}.csv"
    prediction_path = args.output_dir / f"{stem}.predictions.npz"
    manifest_path = args.output_dir / f"{stem}.manifest.json"
    table.to_csv(result_path, index=False)
    np.savez_compressed(prediction_path, **prediction_payload)
    manifest = {
        "version": "admet322-frozen-fusion-holdout-v1",
        "cohort_path": str(cohort_path), "cohort_sha256": _sha256(cohort_path),
        "endpoint_id": endpoint_id,
        "development_recipe_path": str(recipe_path.resolve()),
        "development_recipe_sha256": _sha256(recipe_path),
        "frozen_recipe": frozen,
        "development_seed": args.development_seed,
        "confirmation_seeds": seeds,
        "train_folds": [0, 1, 2], "holdout_folds": [3, 4],
        "train_rows": len(train), "holdout_rows": len(holdout),
        "result_path": str(result_path.resolve()), "result_sha256": _sha256(result_path),
        "prediction_path": str(prediction_path.resolve()), "prediction_sha256": _sha256(prediction_path),
        "confirmation_labels_modified_recipe": False,
        "summary": summary,
        "integrity_passed": len(table) == len(seeds) * (len(SINGLE_ACTIONS) + 1),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"endpoint_id": endpoint_id, **summary}, sort_keys=True))


if __name__ == "__main__":
    main()
