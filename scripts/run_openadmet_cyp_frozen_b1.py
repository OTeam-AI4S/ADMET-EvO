"""One-time certification and label-blind inference for the frozen CYP B1 recipe."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier, LGBMRegressor
from sklearn.metrics import matthews_corrcoef

from admet_agent.external_structure_firewall import _standard_molecule
from admet_agent.features import FeatureCache
from scripts.openadmet_cyp_metrics import official_st_rae
from scripts.run_openadmet_cyp_tree_baselines import RICH_2D


SEEDS = (1, 2, 3)
ENDPOINT_COLUMNS = {
    "CYP1A2|direct_inhibition|pIC50|primary_direct": "CYP1A2_pIC50_direct_inhibition",
    "CYP2C9|direct_inhibition|pIC50|primary_direct": "CYP2C9_pIC50_direct_inhibition",
    "CYP2D6|direct_inhibition|pIC50|primary_direct": "CYP2D6_pIC50_direct_inhibition",
    "CYP3A4|direct_inhibition|pIC50|primary_direct": "CYP3A4_pIC50_direct_inhibition",
    "CYP2D6|TDI_shift|is_TDI|primary_classification": "CYP2D6_is_TDI",
    "CYP3A4|TDI_shift|is_TDI|primary_classification": "CYP3A4_is_TDI",
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _recipe(path: Path, root: Path) -> dict:
    payload = json.loads(path.read_text())
    core = {key: value for key, value in payload.items() if key != "recipe_sha256"}
    digest = hashlib.sha256(json.dumps(core, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    if payload.get("recipe_sha256") != digest:
        raise RuntimeError("Frozen recipe digest mismatch")
    if payload.get("components", {}).keys() != {"B1"} or payload.get("recipe_mode") != "single_component_fallback":
        raise RuntimeError("This executor accepts only the frozen single-component B1 recipe")
    for relative, expected in payload["execution_source_sha256"].items():
        if _sha256(root / relative) != expected:
            raise RuntimeError(f"Frozen execution source changed: {relative}")
    return payload


def _model(task_type: str, seed: int, threads: int):
    common = dict(n_estimators=600, num_leaves=31, learning_rate=0.04, subsample=0.85, subsample_freq=1, colsample_bytree=0.7, min_child_samples=15, reg_lambda=2.0, n_jobs=threads, random_state=seed, verbosity=-1)
    if task_type == "classification":
        return LGBMClassifier(class_weight="balanced", **common)
    return LGBMRegressor(objective="regression_l1", **common)


def _primary(longform: pd.DataFrame) -> pd.DataFrame:
    return longform.loc[longform["fidelity"].isin({"primary_direct", "primary_classification"})].copy()


def certify(args: argparse.Namespace) -> dict:
    recipe = _recipe(args.recipe, args.root.resolve())
    protocol = json.loads(args.certification_protocol.read_text())
    if protocol["recipe_sha256"] != recipe["recipe_sha256"]:
        raise RuntimeError("Certification protocol does not bind the recipe")
    longform_path = Path(recipe["longform"]["path"])
    if _sha256(longform_path) != recipe["longform"]["sha256"]:
        raise RuntimeError("Frozen long-form hash mismatch")
    data = _primary(pd.read_csv(longform_path))
    identities = data[["inchikey", "standard_smiles"]].drop_duplicates("inchikey").sort_values("inchikey", kind="stable").reset_index(drop=True)
    identity_index = {value: index for index, value in enumerate(identities["inchikey"])}
    features = FeatureCache(args.feature_cache.resolve()).get_or_compute(identities["standard_smiles"].astype(str).tolist(), RICH_2D)
    predictions = []
    metrics = {}
    for endpoint, frame in data.groupby("endpoint_key", sort=True):
        train = frame.loc[frame["partition"] == "development"].copy()
        holdout = frame.loc[frame["partition"] == "certification"].copy()
        train_index = np.asarray([identity_index[value] for value in train["inchikey"]])
        holdout_index = np.asarray([identity_index[value] for value in holdout["inchikey"]])
        seed_predictions = []
        for seed in SEEDS:
            model = _model(str(train["task_type"].iloc[0]), seed, args.threads)
            model.fit(features[train_index], train["target"].astype(int) if train["task_type"].iloc[0] == "classification" else train["target"])
            seed_predictions.append(model.predict_proba(features[holdout_index])[:, 1] if train["task_type"].iloc[0] == "classification" else model.predict(features[holdout_index]))
        prediction = np.mean(seed_predictions, axis=0)
        scored = holdout.copy()
        scored["prediction"] = prediction
        scored["endpoint_column"] = ENDPOINT_COLUMNS[endpoint]
        predictions.append(scored)
        if train["task_type"].iloc[0] == "classification":
            metrics[endpoint] = {"mcc": float(matthews_corrcoef(holdout["target"].astype(int), (prediction >= 0.5).astype(int))), "rows": len(holdout)}
        else:
            metrics[endpoint] = {"st_rae": official_st_rae(target=holdout["target"].to_numpy(), prediction=prediction, lower=holdout["credible_low"].to_numpy(), upper=holdout["credible_high"].to_numpy()), "mae": float(np.mean(np.abs(prediction - holdout["target"].to_numpy()))), "rows": len(holdout)}
    regression = [value["st_rae"] for value in metrics.values() if "st_rae" in value]
    classification = [value["mcc"] for value in metrics.values() if "mcc" in value]
    macro_st_rae, macro_mcc = float(np.mean(regression)), float(np.mean(classification))
    checks = [
        {"name": "macro_st_rae", "observed": macro_st_rae, "requirement": f"<={protocol['gates']['macro_st_rae_max']}", "passed": macro_st_rae <= protocol["gates"]["macro_st_rae_max"]},
        {"name": "macro_mcc", "observed": macro_mcc, "requirement": f">={protocol['gates']['macro_mcc_min']}", "passed": macro_mcc >= protocol["gates"]["macro_mcc_min"]},
        {"name": "worst_regression_st_rae", "observed": max(regression), "requirement": f"<={protocol['gates']['worst_endpoint_st_rae_max']}", "passed": max(regression) <= protocol["gates"]["worst_endpoint_st_rae_max"]},
        {"name": "worst_classification_mcc", "observed": min(classification), "requirement": f">={protocol['gates']['worst_endpoint_mcc_min']}", "passed": min(classification) >= protocol["gates"]["worst_endpoint_mcc_min"]},
    ]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    prediction_path = args.output_dir / "certification_predictions.csv"
    pd.concat(predictions, ignore_index=True).to_csv(prediction_path, index=False)
    result = {"version": "openadmet-cyp-b1-certification-v1", "recipe_sha256": recipe["recipe_sha256"], "certification_protocol_sha256": _sha256(args.certification_protocol), "metrics": metrics, "macro_st_rae": macro_st_rae, "macro_mcc": macro_mcc, "checks": checks, "certification_passed": all(item["passed"] for item in checks), "prediction_sha256": _sha256(prediction_path), "blind_test_labels_loaded": False, "leaderboard_feedback_used": False, "selection_reopened": False}
    result_path = args.output_dir / "certification_result.json"
    result_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return result


def blind_inference(args: argparse.Namespace) -> dict:
    recipe = _recipe(args.recipe, args.root.resolve())
    certification = json.loads(args.certification_result.read_text())
    if certification.get("certification_passed") is not True or certification.get("recipe_sha256") != recipe["recipe_sha256"]:
        raise RuntimeError("Blind inference requires a passing certification bound to this recipe")
    schema = json.loads(Path(recipe["submission_schema"]["path"]).read_text())
    blind_path = args.blind_table.resolve()
    if _sha256(blind_path) != schema["blind_table_sha256"]:
        raise RuntimeError("Blind structure table hash mismatch")
    blind = pd.read_csv(blind_path)
    if list(blind.columns) != schema["blind_table_columns"] or len(blind) != schema["activity_dataset_size"]:
        raise RuntimeError("Blind table schema mismatch")
    standardized = [_standard_molecule(value) for value in blind["SMILES"].astype(str)]
    if any(value is None for value in standardized):
        raise RuntimeError("Every blind structure must standardize before frozen inference")
    longform = _primary(pd.read_csv(Path(recipe["longform"]["path"])))
    identities = longform[["inchikey", "standard_smiles"]].drop_duplicates("inchikey").sort_values("inchikey", kind="stable").reset_index(drop=True)
    train_count = len(identities)
    combined_smiles = identities["standard_smiles"].astype(str).tolist() + [value.smiles for value in standardized]
    features = FeatureCache(args.feature_cache.resolve()).get_or_compute(combined_smiles, RICH_2D)
    identity_index = {value: index for index, value in enumerate(identities["inchikey"])}
    regression = blind[["SMILES", "Molecule_Name"]].copy()
    classification = blind[["SMILES", "Molecule_Name"]].copy()
    for endpoint, frame in longform.groupby("endpoint_key", sort=True):
        index = np.asarray([identity_index[value] for value in frame["inchikey"]])
        seed_predictions = []
        for seed in SEEDS:
            model = _model(str(frame["task_type"].iloc[0]), seed, args.threads)
            model.fit(features[index], frame["target"].astype(int) if frame["task_type"].iloc[0] == "classification" else frame["target"])
            seed_predictions.append(model.predict_proba(features[train_count:])[:, 1] if frame["task_type"].iloc[0] == "classification" else model.predict(features[train_count:]))
        prediction = np.mean(seed_predictions, axis=0)
        column = ENDPOINT_COLUMNS[endpoint]
        if frame["task_type"].iloc[0] == "classification":
            classification[column] = (prediction >= 0.5).astype(int)
        else:
            regression[column] = prediction
    if list(regression.columns) != schema["regression_columns"] or list(classification.columns) != schema["classification_columns"]:
        raise RuntimeError("Generated submission columns do not match frozen official schema")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    regression_path = args.output_dir / "openadmet_cyp_regression_submission.csv"
    classification_path = args.output_dir / "openadmet_cyp_classification_submission.csv"
    regression.to_csv(regression_path, index=False)
    classification.to_csv(classification_path, index=False)
    manifest = {"version": "openadmet-cyp-b1-blind-inference-v1", "recipe_sha256": recipe["recipe_sha256"], "certification_result_sha256": _sha256(args.certification_result), "challenge_space_commit": schema["challenge_space_commit"], "blind_input_sha256": _sha256(blind_path), "regression_submission_sha256": _sha256(regression_path), "classification_submission_sha256": _sha256(classification_path), "rows": len(blind), "blind_test_labels_loaded": False, "leaderboard_feedback_used": False, "selection_reopened": False}
    (args.output_dir / "blind_inference_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("certify", "blind"))
    parser.add_argument("--recipe", type=Path, required=True)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--feature-cache", type=Path, default=Path("cache/openadmet-cyp-frozen-b1"))
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--certification-protocol", type=Path)
    parser.add_argument("--certification-result", type=Path)
    parser.add_argument("--blind-table", type=Path)
    args = parser.parse_args()
    if args.mode == "certify" and args.certification_protocol is None:
        parser.error("certify requires --certification-protocol")
    if args.mode == "blind" and (args.certification_result is None or args.blind_table is None):
        parser.error("blind requires --certification-result and --blind-table")
    return args


if __name__ == "__main__":
    arguments = parse_args()
    print(json.dumps(certify(arguments) if arguments.mode == "certify" else blind_inference(arguments), indent=2, sort_keys=True))
