"""Paired classic-ML confirmation of task-conditioned external data.

The experiment changes only the training rows.  Molecular representation,
learner hyperparameters, TDC scaffold split, validation rows, and random seed
are identical between the TDC-only and augmented arms.  TDC test files are
never opened.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier, LGBMRegressor
from rdkit import Chem, DataStructs, RDLogger
from run_scaling_parallel_adapters import _curate_binary, _curate_regression
from scipy import sparse
from scipy.stats import spearmanr
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import average_precision_score, mean_absolute_error, roc_auc_score
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.svm import LinearSVC, LinearSVR
from tdc.benchmark_group import admet_group

from admet_agent.external_data_admission import require_bound_task_admission
from admet_agent.external_structure_firewall import _standard_molecule
from admet_agent.herg_data_scaling import (
    fingerprints_to_csr,
    morgan_fingerprints,
    near_neighbor_keep_mask,
)
from admet_agent.heldout_structure_firewall import (
    load_structure_only_inchikey_blocklist,
)

RDLogger.DisableLog("rdApp.warning")

PROTOCOL = "classic-ml-task-conditioned-external-data-all22-confirmation-v3"
PRACTICAL_DELTA = 0.005
DEFAULT_SEEDS = (2, 3, 4, 5)
DEFAULT_MODELS = ("random_forest", "linear_svm", "morgan_mlp")
ALL_MODELS = (*DEFAULT_MODELS, "lightgbm")

# All 22 TDC ADMET Group endpoints, with the official task metric. Seed 1 was
# used in the earlier LightGBM development screen; this confirmation is frozen
# to seeds 2-5 and never opens the benchmark test split.
TASKS = {
    "Caco2_Wang": ("caco2_wang", "regression", "mae", "development_positive"),
    "HIA_Hou": ("hia_hou", "classification", "auroc", "development_positive"),
    "Pgp_Broccatelli": ("pgp_broccatelli", "classification", "auroc", "development_positive"),
    "Bioavailability_Ma": ("bioavailability_ma", "classification", "auroc", "development_positive"),
    "Lipophilicity_AstraZeneca": ("lipophilicity_astrazeneca", "regression", "mae", "all22_extension"),
    "AMES": ("ames", "classification", "auroc", "development_inconclusive_control"),
    "Solubility_AqSolDB": ("solubility_aqsoldb", "regression", "mae", "development_negative_control"),
    "BBB_Martins": ("bbb_martins", "classification", "auroc", "all22_extension"),
    "PPBR_AZ": ("ppbr_az", "regression", "mae", "all22_extension"),
    "VDss_Lombardo": ("vdss_lombardo", "regression", "spearman", "development_positive"),
    "CYP2C9_Veith": ("cyp2c9_veith", "classification", "auprc", "all22_extension"),
    "CYP2D6_Veith": ("cyp2d6_veith", "classification", "auprc", "development_positive_official_metric"),
    "CYP3A4_Veith": ("cyp3a4_veith", "classification", "auprc", "all22_extension"),
    "CYP2C9_Substrate_CarbonMangels": ("cyp2c9_substrate_carbonmangels", "classification", "auprc", "all22_extension"),
    "CYP2D6_Substrate_CarbonMangels": ("cyp2d6_substrate_carbonmangels", "classification", "auprc", "all22_extension"),
    "CYP3A4_Substrate_CarbonMangels": ("cyp3a4_substrate_carbonmangels", "classification", "auroc", "all22_extension"),
    "Half_Life_Obach": ("half_life_obach", "regression", "spearman", "all22_extension"),
    "Clearance_Hepatocyte_AZ": ("clearance_hepatocyte_az", "regression", "spearman", "all22_extension"),
    "Clearance_Microsome_AZ": ("clearance_microsome_az", "regression", "spearman", "all22_extension"),
    "LD50_Zhu": ("ld50_zhu", "regression", "mae", "all22_extension"),
    "hERG": ("herg", "classification", "auroc", "all22_extension_bridge_source"),
    "DILI": ("dili", "classification", "auroc", "all22_extension_bridge_source"),
}

SPECIAL_SOURCES = {
    "BBB_Martins": {
        "path": Path("data/v5_public_cohort_raw/b3db_classification.tab"),
        "name": "B3DB_classification",
        "smiles_column": "Drug",
        "label_column": "Y",
        "id_column": "Drug_ID",
        "lineage_status": "bridge_source_pending_document_lineage_clearance",
        "semantic_role": "BBB permeability classification bridge source",
    },
    "hERG": {
        "path": Path("external/ecvoi_v6_tdc_cache/herg_central.tab"),
        "name": "hERGCentral_hERG_inhib",
        "smiles_column": "X",
        "label_column": "hERG_inhib",
        "id_column": "ID",
        "lineage_status": "bridge_not_endpoint_equivalent_direct_merge",
        "semantic_role": "fixed-concentration electrophysiology screen bridge source",
    },
    "DILI": {
        "path": Path("data/external/fda/dilirank2/dilirank2_structures.tab"),
        "name": "FDA_DILIrank2_PubChem",
        "smiles_column": "X",
        "label_column": "Y",
        "id_column": "source_record_id",
        "lineage_status": "bridge_binary_mapping_sensitivity_only",
        "semantic_role": "clinical risk-category bridge source",
    },
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _standardized_tdc(frame: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Standardize without dropping TDC rows; use raw-RDKit fallback if needed."""

    rows = []
    fallback_count = 0
    for raw_smiles, target in frame[["Drug", "Y"]].itertuples(index=False, name=None):
        raw_smiles = str(raw_smiles)
        molecule = _standard_molecule(raw_smiles)
        if molecule is not None and Chem.MolFromSmiles(molecule.smiles) is not None:
            feature_smiles = molecule.smiles
            identity = molecule.inchikey
        else:
            raw_molecule = Chem.MolFromSmiles(raw_smiles)
            if raw_molecule is None:
                raise ValueError("A TDC structure is not fingerprintable")
            feature_smiles = Chem.MolToSmiles(
                raw_molecule, canonical=True, isomericSmiles=True
            )
            identity = Chem.MolToInchiKey(raw_molecule)
            fallback_count += 1
        rows.append(
            {
                "standard_smiles": feature_smiles,
                "standard_inchikey": identity,
                "Y": target,
            }
        )
    return pd.DataFrame(rows), fallback_count


def _curate_external(
    source: Path,
    *,
    task_type: str,
    smiles_column: str,
    label_column: str,
    id_column: str,
    n_jobs: int,
    majority_cap: int,
) -> tuple[pd.DataFrame, dict, pd.DataFrame]:
    raw = pd.read_csv(source, sep="\t")
    if task_type == "classification":
        external, audit = _curate_binary(
            raw,
            smiles_column=smiles_column,
            label_column=label_column,
            id_column=id_column,
            majority_cap=majority_cap,
        )
        audit_dict = asdict(audit)
    else:
        args = SimpleNamespace(
            smiles_column=smiles_column,
            label_column=label_column,
            id_column=id_column,
            n_jobs=n_jobs,
        )
        external, audit_dict = _curate_regression(raw, args)
    return external, {"raw_rows": len(raw), **audit_dict}, raw


def _source_spec(args: argparse.Namespace, task: str) -> dict:
    if task in SPECIAL_SOURCES:
        spec = dict(SPECIAL_SOURCES[task])
        spec["path"] = spec["path"].resolve()
        return spec
    return {
        "path": (args.cohort_dir / f"{task}.tab").resolve(),
        "name": f"ChEMBL37_{task}",
        "smiles_column": "X",
        "label_column": "Y",
        "id_column": "ID",
        "lineage_status": "pending_document_lineage_clearance",
        "semantic_role": "endpoint-conditioned ChEMBL 37 cohort",
    }


def _raw_quality_diagnostics(
    raw: pd.DataFrame,
    *,
    smiles_column: str,
    label_column: str,
) -> dict:
    rows = max(1, len(raw))
    result = {
        "raw_missing_smiles_fraction": float(raw[smiles_column].isna().sum() / rows),
        "raw_missing_label_fraction": float(raw[label_column].isna().sum() / rows),
        "raw_duplicate_smiles_fraction": float(
            raw.duplicated(smiles_column, keep=False).sum() / rows
        ),
    }
    for column in ("standard_type", "standard_units", "assay_organism"):
        if column in raw:
            values = raw[column].fillna("").astype(str).str.strip()
            nonempty = values[values != ""]
            result[f"unique_{column}"] = int(nonempty.nunique())
            if column == "assay_organism":
                result["human_organism_fraction"] = float(
                    (values == "Homo sapiens").sum() / rows
                )
    return result


def _overlap_and_target_diagnostics(
    external: pd.DataFrame,
    discovery: pd.DataFrame,
    *,
    task_type: str,
) -> dict:
    if task_type == "classification":
        tdc = discovery.groupby("standard_inchikey", as_index=False).agg(
            tdc_y=("Y", "mean")
        )
        tdc["tdc_y"] = (tdc["tdc_y"] >= 0.5).astype(int)
        overlap = external.merge(tdc, on="standard_inchikey", how="inner")
        return {
            "tdc_positive_rate": float(discovery["Y"].mean()),
            "external_positive_rate": float(external["Y"].mean()),
            "absolute_positive_rate_shift": float(
                abs(external["Y"].mean() - discovery["Y"].mean())
            ),
            "exact_overlap_identities": len(overlap),
            "exact_overlap_label_disagreement_fraction": (
                float((overlap["Y"] != overlap["tdc_y"]).mean())
                if len(overlap)
                else None
            ),
        }
    tdc = discovery.groupby("standard_inchikey", as_index=False).agg(tdc_y=("Y", "median"))
    overlap = external.merge(tdc, on="standard_inchikey", how="inner")
    tdc_iqr = float(discovery["Y"].quantile(0.75) - discovery["Y"].quantile(0.25))
    ext_iqr = float(external["Y"].quantile(0.75) - external["Y"].quantile(0.25))
    scale = tdc_iqr if math.isfinite(tdc_iqr) and tdc_iqr > 0 else 1.0
    return {
        "tdc_target_median": float(discovery["Y"].median()),
        "external_target_median": float(external["Y"].median()),
        "tdc_target_iqr": tdc_iqr,
        "external_target_iqr": ext_iqr,
        "normalized_median_shift": float(
            abs(external["Y"].median() - discovery["Y"].median()) / scale
        ),
        "external_to_tdc_iqr_ratio": float(ext_iqr / scale),
        "exact_overlap_identities": len(overlap),
        "exact_overlap_median_absolute_target_difference": (
            float(np.median(np.abs(overlap["Y"] - overlap["tdc_y"])))
            if len(overlap)
            else None
        ),
        "exact_overlap_normalized_median_absolute_target_difference": (
            float(np.median(np.abs(overlap["Y"] - overlap["tdc_y"])) / scale)
            if len(overlap)
            else None
        ),
    }


def _sampled_similarity_diagnostics(
    external_fps: list,
    discovery_fps: list,
    *,
    sample_size: int = 2_000,
) -> dict:
    rng = np.random.default_rng(1729)
    ext_index = np.sort(
        rng.choice(len(external_fps), min(sample_size, len(external_fps)), replace=False)
    )
    tdc_index = np.sort(
        rng.choice(len(discovery_fps), min(sample_size, len(discovery_fps)), replace=False)
    )
    reference = [discovery_fps[index] for index in tdc_index]
    maxima = np.asarray(
        [
            max(DataStructs.BulkTanimotoSimilarity(external_fps[index], reference))
            for index in ext_index
        ],
        dtype=float,
    )
    return {
        "method": "deterministic 2000x2000 sampled Morgan-r2 nearest Tanimoto",
        "external_sample_rows": len(ext_index),
        "tdc_discovery_sample_rows": len(tdc_index),
        "nearest_tanimoto_mean": float(maxima.mean()),
        "nearest_tanimoto_median": float(np.median(maxima)),
        "nearest_tanimoto_q25": float(np.quantile(maxima, 0.25)),
        "nearest_tanimoto_q75": float(np.quantile(maxima, 0.75)),
        "fraction_nearest_at_least_0_4": float((maxima >= 0.4).mean()),
        "fraction_nearest_at_least_0_6": float((maxima >= 0.6).mean()),
    }


def _classification_model(name: str, seed: int, n_jobs: int):
    if name == "lightgbm":
        return LGBMClassifier(
            n_estimators=400,
            learning_rate=0.03,
            num_leaves=31,
            subsample=0.9,
            colsample_bytree=0.9,
            reg_lambda=1.0,
            class_weight="balanced",
            random_state=seed,
            n_jobs=n_jobs,
            verbosity=-1,
        )
    if name == "random_forest":
        return RandomForestClassifier(
            n_estimators=200,
            max_features="sqrt",
            min_samples_leaf=2,
            class_weight="balanced_subsample",
            random_state=seed,
            n_jobs=n_jobs,
        )
    if name == "linear_svm":
        return LinearSVC(
            C=1.0,
            class_weight="balanced",
            dual="auto",
            max_iter=10_000,
            random_state=seed,
        )
    if name == "morgan_mlp":
        return MLPClassifier(
            hidden_layer_sizes=(64,),
            activation="relu",
            solver="adam",
            alpha=1e-4,
            batch_size=256,
            learning_rate_init=1e-3,
            max_iter=60,
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=6,
            random_state=seed,
        )
    raise ValueError(f"Unknown classification model {name!r}")


def _regression_model(name: str, seed: int, n_jobs: int):
    if name == "lightgbm":
        return LGBMRegressor(
            n_estimators=400,
            learning_rate=0.03,
            num_leaves=31,
            subsample=0.9,
            colsample_bytree=0.9,
            reg_lambda=1.0,
            random_state=seed,
            n_jobs=n_jobs,
            verbosity=-1,
        )
    if name == "random_forest":
        return RandomForestRegressor(
            n_estimators=200,
            max_features="sqrt",
            min_samples_leaf=2,
            random_state=seed,
            n_jobs=n_jobs,
        )
    if name == "linear_svm":
        return LinearSVR(
            C=1.0,
            epsilon=0.1,
            dual="auto",
            max_iter=10_000,
            random_state=seed,
        )
    if name == "morgan_mlp":
        return MLPRegressor(
            hidden_layer_sizes=(64,),
            activation="relu",
            solver="adam",
            alpha=1e-4,
            batch_size=256,
            learning_rate_init=1e-3,
            max_iter=60,
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=6,
            random_state=seed,
        )
    raise ValueError(f"Unknown regression model {name!r}")


def _classification_prediction(model, x_train, y_train, x_valid) -> np.ndarray:
    fitted = model.fit(x_train, y_train)
    if hasattr(fitted, "predict_proba"):
        return np.asarray(fitted.predict_proba(x_valid)[:, 1], dtype=float)
    return np.asarray(fitted.decision_function(x_valid), dtype=float)


def _regression_prediction(model, x_train, y_train, x_valid) -> np.ndarray:
    # Target normalization makes LinearSVR and MLP comparable across endpoints.
    mean = float(np.mean(y_train))
    scale = float(np.std(y_train))
    if not math.isfinite(scale) or scale <= 0:
        scale = 1.0
    fitted = model.fit(x_train, (y_train - mean) / scale)
    return np.asarray(fitted.predict(x_valid), dtype=float) * scale + mean


def _score(metric: str, y_true: np.ndarray, prediction: np.ndarray) -> float:
    if metric == "auroc":
        return float(roc_auc_score(y_true, prediction))
    if metric == "auprc":
        return float(average_precision_score(y_true, prediction))
    if metric == "mae":
        return float(mean_absolute_error(y_true, prediction))
    if metric == "spearman":
        return float(spearmanr(y_true, prediction).statistic)
    raise ValueError(f"Unknown metric {metric!r}")


def _oriented_delta(metric: str, baseline: float, augmented: float) -> float:
    return baseline - augmented if metric == "mae" else augmented - baseline


def _sign_flip_pvalue(values: np.ndarray) -> float:
    """Exact two-sided paired sign-flip p-value for the mean delta."""

    observed = abs(float(values.mean()))
    extreme = 0
    total = 1 << len(values)
    for mask in range(total):
        signs = np.asarray(
            [1.0 if mask & (1 << index) else -1.0 for index in range(len(values))]
        )
        if abs(float((values * signs).mean())) >= observed - 1e-15:
            extreme += 1
    return extreme / total


def _summarize(rows: list[dict], model_name: str) -> dict:
    selected = [row for row in rows if row["model"] == model_name]
    deltas = np.asarray([row["oriented_delta"] for row in selected], dtype=float)
    baseline = np.asarray([row["baseline_score"] for row in selected], dtype=float)
    augmented = np.asarray([row["augmented_score"] for row in selected], dtype=float)
    return {
        "baseline_mean": float(baseline.mean()),
        "baseline_std": float(baseline.std()),
        "augmented_mean": float(augmented.mean()),
        "augmented_std": float(augmented.std()),
        "oriented_delta_mean": float(deltas.mean()),
        "oriented_delta_std": float(deltas.std()),
        "positive_seed_fraction": float((deltas > 0).mean()),
        "exact_two_sided_sign_flip_p": _sign_flip_pvalue(deltas),
        "promotion_passed": bool(
            deltas.mean() >= PRACTICAL_DELTA and (deltas > 0).mean() >= 0.75
        ),
    }


def run_task(args: argparse.Namespace, task: str) -> dict:
    benchmark, task_type, metric, selection_role = TASKS[task]
    source_spec = _source_spec(args, task)
    if not args.audit_only:
        require_bound_task_admission(
            args.admission_report,
            task,
            source_path=source_spec["path"],
            source_name=source_spec["name"],
            augmentation_mode=args.augmentation_mode,
        )
    source = source_spec["path"]
    started = time.perf_counter()
    external, audit, raw = _curate_external(
        source,
        task_type=task_type,
        smiles_column=source_spec["smiles_column"],
        label_column=source_spec["label_column"],
        id_column=source_spec["id_column"],
        n_jobs=args.n_jobs,
        majority_cap=args.majority_cap,
    )

    discovery_path = args.data_dir / "admet_group" / benchmark / "train_val.csv"
    discovery = pd.read_csv(discovery_path, usecols=["Drug", "Y"])
    discovery_standard, discovery_fallbacks = _standardized_tdc(discovery)
    test_ids, test_firewall_audit = load_structure_only_inchikey_blocklist(
        args.data_dir / "admet_group" / benchmark / "test.csv"
    )
    raw_quality = _raw_quality_diagnostics(
        raw,
        smiles_column=source_spec["smiles_column"],
        label_column=source_spec["label_column"],
    )
    target_diagnostics = _overlap_and_target_diagnostics(
        external, discovery_standard, task_type=task_type
    )
    discovery_ids = set(discovery_standard["standard_inchikey"])
    discovery_exact_mask = external["standard_inchikey"].isin(discovery_ids)
    test_exact_mask = external["standard_inchikey"].isin(test_ids)
    discovery_exact_exclusions = int(discovery_exact_mask.sum())
    test_exact_exclusions = int(test_exact_mask.sum())
    exact_mask = discovery_exact_mask | test_exact_mask
    external = external.loc[~exact_mask].reset_index(drop=True)
    fingerprintable = external["standard_smiles"].map(
        lambda value: Chem.MolFromSmiles(str(value)) is not None
    )
    parse_exclusions = int((~fingerprintable).sum())
    external = external.loc[fingerprintable].reset_index(drop=True)
    if external.empty:
        raise RuntimeError("No external identities survived the exact-overlap firewall")

    external_fps = morgan_fingerprints(external["standard_smiles"].tolist())
    discovery_fps = morgan_fingerprints(
        discovery_standard["standard_smiles"].tolist()
    )
    similarity_diagnostics = _sampled_similarity_diagnostics(
        external_fps, discovery_fps
    )

    audit_result = {
        "protocol": PROTOCOL,
        "run_mode": "data_audit_only" if args.audit_only else "model_confirmation",
        "task": task,
        "benchmark": benchmark,
        "task_type": task_type,
        "metric": metric,
        "selection_role": selection_role,
        "source": {
            "name": source_spec["name"],
            "path": str(source),
            "sha256": _sha256(source),
            "lineage_status": source_spec["lineage_status"],
            "semantic_role": source_spec["semantic_role"],
        },
        "curation": {
            **audit,
            **raw_quality,
            "exact_tdc_discovery_overlap_exclusions": discovery_exact_exclusions,
            "exact_tdc_test_structure_overlap_exclusions": test_exact_exclusions,
            **test_firewall_audit,
            "post_standardization_fingerprint_parse_exclusions": parse_exclusions,
            "post_exact_firewall_identities": len(external),
            "near_neighbor_rule": f"max Morgan Tanimoto < {args.near_threshold:.2f}",
            "tdc_discovery_standardization_fallback_count": discovery_fallbacks,
        },
        "target_alignment": target_diagnostics,
        "chemical_space_alignment": similarity_diagnostics,
        "safety": {
            "tdc_test_loaded": True,
            "tdc_test_structures_loaded_for_firewall": True,
            "tdc_test_columns_loaded": ["Drug"],
            "tdc_test_labels_loaded": False,
            "test_labels_accessible": False,
        },
    }
    if args.audit_only:
        audit_result["wall_seconds"] = time.perf_counter() - started
        audit_result["claim_boundary"] = (
            "Data audit only. No model was fit; only the TDC test Drug column "
            "was loaded for the structure firewall and no test labels were loaded. "
            "Sampled chemical-space similarity is diagnostic, not a causal test."
        )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(audit_result, indent=2, sort_keys=True) + "\n"
        )
        return audit_result

    external_x = fingerprints_to_csr(external_fps)
    external_y = external["Y"].to_numpy(
        dtype=int if task_type == "classification" else float
    )

    group = admet_group(path=str(args.data_dir))
    rows: list[dict] = []
    fallback_count = 0
    for seed in args.seeds:
        train_raw, valid_raw = group.get_train_valid_split(
            benchmark=benchmark, split_type="default", seed=seed
        )
        train, train_fallbacks = _standardized_tdc(train_raw)
        valid, valid_fallbacks = _standardized_tdc(valid_raw)
        fallback_count += train_fallbacks + valid_fallbacks
        train_fps = morgan_fingerprints(train["standard_smiles"].tolist())
        valid_fps = morgan_fingerprints(valid["standard_smiles"].tolist())
        train_x = fingerprints_to_csr(train_fps)
        valid_x = fingerprints_to_csr(valid_fps)
        train_y = train["Y"].to_numpy(
            dtype=int if task_type == "classification" else float
        )
        valid_y = valid["Y"].to_numpy(
            dtype=int if task_type == "classification" else float
        )
        keep = near_neighbor_keep_mask(
            external_fps, valid_fps, threshold=args.near_threshold
        )
        augmented_x = sparse.vstack([train_x, external_x[keep]], format="csr")
        augmented_y = np.concatenate([train_y, external_y[keep]])

        for model_name in args.models:
            fit_started = time.perf_counter()
            if task_type == "classification":
                baseline_prediction = _classification_prediction(
                    _classification_model(model_name, seed, args.n_jobs),
                    train_x,
                    train_y,
                    valid_x,
                )
                augmented_prediction = _classification_prediction(
                    _classification_model(model_name, seed, args.n_jobs),
                    augmented_x,
                    augmented_y,
                    valid_x,
                )
            else:
                baseline_prediction = _regression_prediction(
                    _regression_model(model_name, seed, args.n_jobs),
                    train_x,
                    train_y,
                    valid_x,
                )
                augmented_prediction = _regression_prediction(
                    _regression_model(model_name, seed, args.n_jobs),
                    augmented_x,
                    augmented_y,
                    valid_x,
                )
            baseline = _score(metric, valid_y, baseline_prediction)
            augmented = _score(metric, valid_y, augmented_prediction)
            row = {
                "seed": seed,
                "model": model_name,
                "tdc_train_rows": len(train),
                "tdc_validation_rows": len(valid),
                "external_firewalled_rows": int(keep.sum()),
                "near_neighbor_exclusions": int((~keep).sum()),
                "baseline_score": baseline,
                "augmented_score": augmented,
                "oriented_delta": _oriented_delta(metric, baseline, augmented),
                "fit_wall_seconds": time.perf_counter() - fit_started,
            }
            rows.append(row)
            print(json.dumps({"task": task, **row}), flush=True)

    summaries = {name: _summarize(rows, name) for name in args.models}
    promoted_models = [
        name for name, summary in summaries.items() if summary["promotion_passed"]
    ]
    result = {
        **audit_result,
        "curation": {
            **audit_result["curation"],
            "tdc_standardization_fallback_count": fallback_count,
        },
        "seeds": list(args.seeds),
        "models": list(args.models),
        "per_seed_model": rows,
        "model_summaries": summaries,
        "task_robustness": {
            "promoted_models": promoted_models,
            "promoted_model_count": len(promoted_models),
            "robust_across_models": len(promoted_models) >= 2,
        },
        "wall_seconds": time.perf_counter() - started,
        "safety": {
            **audit_result["safety"],
            "leaderboard_used_only_to_choose_average_model_families": True,
            "seed_1_used_only_for_development_task_selection": True,
            "confirmation_seeds_disjoint_from_selection_seed": True,
        },
        "claim_boundary": (
            "Development confirmation on TDC train/validation splits only. "
            "Source-document lineage remains pending; this cannot establish an "
            "independent external-data or official leaderboard gain."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=tuple(TASKS), required=True)
    parser.add_argument(
        "--model", choices=ALL_MODELS, action="append", dest="models"
    )
    parser.add_argument("--seed", type=int, action="append", dest="seeds")
    parser.add_argument(
        "--cohort-dir",
        type=Path,
        default=Path("data/external/chembl/chembl_37/tdc22_cohorts"),
    )
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--majority-cap", type=int, default=15_000)
    parser.add_argument("--near-threshold", type=float, default=0.80)
    parser.add_argument("--n-jobs", type=int, default=8)
    parser.add_argument(
        "--admission-report",
        type=Path,
        default=Path(
            "reports/classic_ml_external_data_all22_audit_v2/admission_report_v2.json"
        ),
        help="Frozen pre-model admission decisions; non-admitted tasks fail closed.",
    )
    parser.add_argument(
        "--augmentation-mode",
        choices=("direct", "controlled_bridge"),
        default="direct",
        help="Controlled bridge runs are developmental and cannot support direct-data claims.",
    )
    parser.add_argument(
        "--audit-only",
        action="store_true",
        help="Run curation/alignment diagnostics only; do not fit any model.",
    )
    args = parser.parse_args()
    args.models = tuple(dict.fromkeys(args.models or DEFAULT_MODELS))
    args.seeds = tuple(dict.fromkeys(args.seeds or DEFAULT_SEEDS))
    if args.near_threshold != 0.80:
        raise ValueError("The frozen confirmation threshold is Tanimoto 0.80")
    if 1 in args.seeds:
        raise ValueError("Seed 1 was used for task selection and is excluded from confirmation")
    if (
        args.augmentation_mode == "controlled_bridge"
        and args.task == "hERG"
        and set(args.seeds) & {1, 2, 3}
    ):
        raise ValueError(
            "hERG controlled-bridge confirmation must be disjoint from pilot seeds 1-3"
        )
    return args


if __name__ == "__main__":
    parsed = parse_args()
    payload = run_task(parsed, parsed.task)
    if parsed.audit_only:
        print(json.dumps(payload, indent=2))
    else:
        print(
            json.dumps(
                {
                    "task": payload["task"],
                    "model_summaries": payload["model_summaries"],
                    "task_robustness": payload["task_robustness"],
                },
                indent=2,
            )
        )
