#!/usr/bin/env python3
"""Run frozen model/feature/data ablations on all 22 TDC ADMET endpoints.

Discovery and confirmation read the hidden-test structure column only.  The
certification phase is a separate entry point that requires complete frozen
discovery and confirmation artifacts before it may read hidden-test labels.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
import sys
import time
import traceback
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs, RDLogger
from tdc.benchmark_group import admet_group

from admet_agent.adaptive_external_data import (
    external_mass_capped_weights,
    molecular_scaffold,
    select_relevance_diverse_binary_coreset,
    select_relevance_diverse_regression_coreset,
)
from admet_agent.brain import RuleBasedBrain
from admet_agent.external_data_admission import require_bound_task_admission
from admet_agent.features import FeatureCache
from admet_agent.herg_data_scaling import (
    morgan_fingerprints,
    near_neighbor_keep_mask,
)
from admet_agent.external_structure_firewall import _standard_molecule
from admet_agent.schemas import ExperimentSpec
from admet_agent.selection import build_model, predict, score
from admet_agent.tasks import resolve_task

from run_all22_optimal_recipe_adaptive_data import _fit_with_sample_weight
from run_classic_ml_external_data_confirmation import TASKS, _standardized_tdc


RDLogger.DisableLog("rdApp.*")
PROTOCOL_NAME = "all22-three-axis-agent-ablation-v1"
DEFAULT_PROTOCOL = ROOT / "reports/all22_three_axis_ablation_v1/protocol.json"
DEFAULT_OUTPUT = ROOT / "reports/all22_three_axis_ablation_v1"
DEFAULT_EXTERNAL = ROOT / "data/ADMET_v3_clean_external_candidate_pool.csv"
BASELINE_MODEL = {
    "kind": "single",
    "model": "lightgbm",
    "hyperparameters": {},
    "imbalance": "none",
}
BASELINE_FEATURE = ["morgan"]
NO_DATA = {"enabled": False, "method": "official", "ratio": 0.0}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _signature(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode()).hexdigest()


def _normalized_improvement(metric: str, baseline: float, candidate: float) -> float:
    denominator = max(abs(float(baseline)), 1e-12)
    if metric == "mae":
        return float((baseline - candidate) / denominator)
    return float((candidate - baseline) / denominator)


def _raw_oriented_delta(metric: str, baseline: float, candidate: float) -> float:
    return float(baseline - candidate if metric == "mae" else candidate - baseline)


def _model_single(
    model: str,
    hyperparameters: dict[str, Any] | None = None,
    imbalance: str = "none",
) -> dict[str, Any]:
    return {
        "kind": "single",
        "model": model,
        "hyperparameters": dict(hyperparameters or {}),
        "imbalance": imbalance,
    }


def _members(model_config: dict[str, Any]) -> list[tuple[dict[str, Any], float]]:
    if model_config["kind"] == "single":
        return [(model_config, 1.0)]
    weights = [float(value) for value in model_config["weights"]]
    total = sum(weights)
    return [
        (member, weight / total)
        for member, weight in zip(model_config["members"], weights, strict=True)
    ]


def _standardize_structure_frame(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    fallback_count = 0
    unparseable_count = 0
    for input_index, raw_value in enumerate(frame["Drug"].astype(str)):
        molecule = _standard_molecule(raw_value)
        used_fallback = False
        if molecule is not None and Chem.MolFromSmiles(molecule.smiles) is not None:
            smiles = molecule.smiles
            inchikey = molecule.inchikey
        else:
            raw_molecule = Chem.MolFromSmiles(raw_value)
            if raw_molecule is None:
                # Preserve evaluator row order. MolecularFeaturizer already
                # maps an invalid structure to an all-zero feature row; the
                # opaque raw hash keeps it in the exact structure firewall.
                smiles = raw_value
                inchikey = "RAW-" + hashlib.sha256(raw_value.encode()).hexdigest()
                unparseable_count += 1
            else:
                smiles = Chem.MolToSmiles(
                    raw_molecule, canonical=True, isomericSmiles=True
                )
                inchikey = Chem.MolToInchiKey(raw_molecule)
            fallback_count += 1
            used_fallback = True
        rows.append(
            {
                "input_index": input_index,
                "standard_smiles": smiles,
                "standard_inchikey": inchikey,
                "standardization_fallback": used_fallback,
                "fingerprintable": Chem.MolFromSmiles(smiles) is not None,
            }
        )
    output = pd.DataFrame(rows)
    output.attrs["standardization_fallback_count"] = fallback_count
    output.attrs["unparseable_structure_count"] = unparseable_count
    return output


def _load_protocol(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("protocol") != PROTOCOL_NAME or payload.get("test_labels_loaded") is not False:
        raise ValueError("Invalid three-axis ablation protocol")
    stored = payload.get("sha256")
    without_hash = dict(payload)
    without_hash.pop("sha256", None)
    encoded = json.dumps(without_hash, indent=2, sort_keys=True) + "\n"
    if hashlib.sha256(encoded.encode()).hexdigest() != stored:
        raise ValueError("Protocol self hash mismatch")
    return payload


def _complete_artifact(
    path: Path,
    *,
    phase: str,
    task: str,
    protocol_sha: str,
) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    value = json.loads(path.read_text(encoding="utf-8"))
    checks = {
        "protocol": value.get("protocol") == PROTOCOL_NAME,
        "protocol_sha": value.get("protocol_sha256") == protocol_sha,
        "phase": value.get("phase") == phase,
        "task": value.get("task") == task,
        "status": value.get("status") == "completed",
    }
    failed = sorted(name for name, passed in checks.items() if not passed)
    if failed:
        raise ValueError(f"Invalid {phase} artifact for {task}: {failed}")
    return value


def _load_external(
    *,
    task: str,
    path: Path,
    protocol: dict[str, Any],
    test_structures: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    data_meta = protocol["external_data"]
    if _sha256(path) != data_meta["sha256"]:
        raise ValueError("External candidate pool hash differs from frozen protocol")
    audit_path = Path(data_meta["audit_path"])
    if _sha256(audit_path) != data_meta["audit_sha256"]:
        raise ValueError("External candidate audit hash differs from frozen protocol")
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    task_audit = audit["tasks"][task]
    minimum = 100 if TASKS[task][1] == "classification" else 30
    test_unique = max(1, int(task_audit["test_structure_unique_inchikeys"]))
    raw_overlap = int(task_audit["exact_tdc_test_structure_overlap_exclusions"])
    overlap_fraction = raw_overlap / test_unique
    threshold = float(
        protocol["data_axis"]["same_source_test_overlap_rejection_threshold"]
    )
    source_rejected = overlap_fraction > threshold
    base_audit = {
        "source_rejected_same_source": source_rejected,
        "raw_exact_test_overlap": raw_overlap,
        "test_unique_identities": test_unique,
        "raw_exact_test_overlap_fraction": overlap_fraction,
        "same_source_rejection_threshold": threshold,
        "minimum_required_rows": minimum,
        "test_structure_columns_loaded": ["Drug"],
        "test_labels_loaded": False,
    }
    admission_path_value = data_meta.get("admission_report_path")
    admission_sha256 = data_meta.get("admission_report_sha256")
    if not admission_path_value or not admission_sha256:
        return pd.DataFrame(), {
            **base_audit,
            "admitted": False,
            "admission_status": "fail_closed",
            "admission_error": "protocol_missing_source_admission_binding",
            "weight_policy": None,
        }
    admission_path = Path(admission_path_value)
    if not admission_path.is_file() or _sha256(admission_path) != admission_sha256:
        raise ValueError("Admission report hash differs from frozen protocol")
    metadata = pd.read_csv(
        path,
        usecols=[
            "endpoint",
            "external_source_name",
            "external_source_path",
            "external_source_sha256",
        ],
        low_memory=False,
    )
    metadata = metadata.loc[metadata["endpoint"] == task].drop_duplicates()
    if len(metadata) != 1:
        return pd.DataFrame(), {
            **base_audit,
            "admitted": False,
            "admission_status": "fail_closed",
            "admission_error": (
                f"expected one source identity in candidate pool, found {len(metadata)}"
            ),
            "weight_policy": None,
        }
    source_meta = metadata.iloc[0]
    try:
        admission = require_bound_task_admission(
            admission_path,
            task,
            source_path=Path(str(source_meta["external_source_path"])),
            source_name=str(source_meta["external_source_name"]),
            augmentation_mode="auto",
        )
        if source_meta["external_source_sha256"] != admission["audited_source_sha256"]:
            raise RuntimeError("Candidate-pool source SHA-256 differs from admission")
    except RuntimeError as exc:
        return pd.DataFrame(), {
            **base_audit,
            "admitted": False,
            "admission_status": "fail_closed",
            "admission_error": str(exc),
            "source_name": str(source_meta["external_source_name"]),
            "source_path": str(source_meta["external_source_path"]),
            "weight_policy": None,
        }
    usecols = [
        "endpoint",
        "standard_smiles",
        "standard_inchikey",
        "external_label",
        "source_record_id",
        "scaffold",
        "external_source_name",
        "external_source_path",
        "external_source_sha256",
    ]
    external = pd.read_csv(path, usecols=usecols, low_memory=False)
    external = external.loc[external["endpoint"] == task].copy().reset_index(drop=True)
    external = external.rename(columns={"external_label": "Y"})
    if external["standard_inchikey"].duplicated().any():
        raise ValueError(f"{task} clean pool contains duplicate identities")
    test_ids = set(test_structures["standard_inchikey"])
    exact_overlap_after = int(external["standard_inchikey"].isin(test_ids).sum())
    if exact_overlap_after:
        raise ValueError(f"{task} clean pool still overlaps hidden-test identities")
    before_near = len(external)
    if len(external):
        external_fps = morgan_fingerprints(external["standard_smiles"].astype(str).tolist())
        test_fps = morgan_fingerprints(
            test_structures.loc[test_structures["fingerprintable"], "standard_smiles"]
            .astype(str)
            .tolist()
        )
        keep = near_neighbor_keep_mask(
            external_fps,
            test_fps,
            threshold=float(protocol["data_axis"]["hidden_test_near_analog_threshold"]),
        )
        external = external.loc[keep].reset_index(drop=True)
    if "scaffold" not in external or external["scaffold"].isna().any():
        external["scaffold"] = external["standard_smiles"].map(molecular_scaffold)
    task_type = TASKS[task][1]
    external["Y"] = pd.to_numeric(external["Y"], errors="raise")
    if task_type == "classification":
        external["Y"] = external["Y"].astype(int)
    admitted = (not source_rejected) and len(external) >= minimum
    return external, {
        **base_audit,
        "test_structure_standardization_fallback_count": int(
            test_structures.attrs.get("standardization_fallback_count", 0)
        ),
        "test_structure_unparseable_count": int(
            test_structures.attrs.get("unparseable_structure_count", 0)
        ),
        "clean_pool_rows_before_near_analog": before_near,
        "hidden_test_near_analog_exclusions": before_near - len(external),
        "post_firewall_rows": len(external),
        "admitted": admitted,
        "admission_status": (
            admission["resolved_augmentation_mode"] if admitted else "fail_closed"
        ),
        "admission": admission,
        "weight_policy": admission["weight_policy"] if admitted else None,
    }


class Evaluator:
    def __init__(
        self,
        *,
        task: str,
        seed: int,
        train: pd.DataFrame,
        predict_frame: pd.DataFrame,
        external: pd.DataFrame,
        external_admitted: bool,
        external_weight_policy: dict[str, Any] | None,
        protocol: dict[str, Any],
        feature_cache: FeatureCache,
        n_jobs: int,
        certification: bool = False,
    ):
        self.task_name = task
        self.task = resolve_task(task)
        self.seed = int(seed)
        self.train = train.reset_index(drop=True)
        self.predict_frame = predict_frame.reset_index(drop=True)
        self.external = external.reset_index(drop=True)
        self.external_admitted = bool(external_admitted)
        self.external_weight_policy = (
            dict(external_weight_policy) if external_weight_policy else None
        )
        if self.external_admitted and self.external_weight_policy is None:
            raise ValueError("Admitted external data require a verified weight policy")
        self.protocol = protocol
        self.feature_cache = feature_cache
        self.n_jobs = int(n_jobs)
        self.certification = certification
        self._feature_values: dict[
            tuple[tuple[str, ...], bool], tuple[np.ndarray, np.ndarray, np.ndarray]
        ] = {}
        self._selected_external: dict[str, tuple[np.ndarray, dict[str, Any]]] = {}
        self._prediction_cache: dict[str, np.ndarray] = {}
        self._evaluation_cache: dict[str, dict[str, Any]] = {}
        self.external_fps = (
            morgan_fingerprints(self.external["standard_smiles"].astype(str).tolist())
            if len(self.external)
            else []
        )
        self.train_fps = morgan_fingerprints(self.train["standard_smiles"].astype(str).tolist())
        self.predict_fps = (
            []
            if certification
            else morgan_fingerprints(
                self.predict_frame["standard_smiles"].astype(str).tolist()
            )
        )

    def _features(
        self, representations: list[str], *, include_external: bool
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        key = (tuple(representations), bool(include_external))
        if key not in self._feature_values:
            smiles = (
                self.train["standard_smiles"].astype(str).tolist()
                + self.predict_frame["standard_smiles"].astype(str).tolist()
                + (
                    self.external["standard_smiles"].astype(str).tolist()
                    if include_external
                    else []
                )
            )
            values = self.feature_cache.get_or_compute(smiles, representations)
            n_train = len(self.train)
            n_predict = len(self.predict_frame)
            self._feature_values[key] = (
                values[:n_train],
                values[n_train : n_train + n_predict],
                values[n_train + n_predict :]
                if include_external
                else np.empty((0, values.shape[1]), dtype=values.dtype),
            )
        return self._feature_values[key]

    def _external_indices(self, config: dict[str, Any]) -> tuple[np.ndarray, dict[str, Any]]:
        key = _signature(config)
        if key in self._selected_external:
            return self._selected_external[key]
        if not config.get("enabled") or not self.external_admitted:
            value = (
                np.asarray([], dtype=int),
                {
                    "status": "official_only" if not config.get("enabled") else "abstained_not_admitted",
                    "selected_rows": 0,
                },
            )
            self._selected_external[key] = value
            return value
        keep = np.ones(len(self.external), dtype=bool)
        if not self.certification:
            keep = near_neighbor_keep_mask(
                self.external_fps,
                self.predict_fps,
                threshold=float(
                    self.protocol["data_axis"]["validation_near_neighbor_threshold"]
                ),
            )
        relevance = np.asarray(
            [
                max(DataStructs.BulkTanimotoSimilarity(fp, self.train_fps))
                for fp in self.external_fps
            ],
            dtype=float,
        )
        relevance[~keep] = -1.0
        requested = (
            max(1, int(config["rows"]))
            if config.get("rows") is not None
            else max(1, int(round(float(config["ratio"]) * len(self.train))))
        )
        if config["method"] == "random":
            eligible = np.flatnonzero(keep)
            ranked = sorted(
                eligible.tolist(),
                key=lambda index: hashlib.sha256(
                    f"three-axis|{self.task_name}|{self.seed}|{self.external.iloc[index]['standard_inchikey']}".encode()
                ).hexdigest(),
            )
            selected = np.asarray(sorted(ranked[:requested]), dtype=int)
            selection_audit = {
                "method": "deterministic_hash_random",
                "requested_rows": requested,
                "eligible_rows": len(eligible),
                "selected_rows": len(selected),
            }
        else:
            minimum = float(self.protocol["data_axis"]["minimum_relevance"])
            if self.task.task_type == "classification":
                selected, selection_audit = select_relevance_diverse_binary_coreset(
                    self.external,
                    relevance,
                    total_rows=requested,
                    target_positive_rate=float(self.train["Y"].mean()),
                    minimum_relevance=minimum,
                )
            else:
                selected, selection_audit = select_relevance_diverse_regression_coreset(
                    self.external,
                    relevance,
                    total_rows=requested,
                    reference_targets=self.train["Y"].to_numpy(dtype=float),
                    minimum_relevance=minimum,
                )
            selection_audit = {"method": "relevance_diverse_coreset", **selection_audit}
        selection_audit.update(
            {
                "validation_near_neighbor_exclusions": int((~keep).sum()),
                "selected_identity_sha256": hashlib.sha256(
                    "\n".join(
                        self.external.iloc[selected]["standard_inchikey"].astype(str).tolist()
                    ).encode()
                ).hexdigest(),
            }
        )
        value = selected, selection_audit
        self._selected_external[key] = value
        return value

    def _single_prediction(
        self,
        member: dict[str, Any],
        representations: list[str],
        data_config: dict[str, Any],
    ) -> tuple[np.ndarray, dict[str, Any]]:
        cache_key = _signature(
            {"member": member, "features": representations, "data": data_config}
        )
        indices, selection_audit = self._external_indices(data_config)
        if cache_key in self._prediction_cache:
            return self._prediction_cache[cache_key], selection_audit
        x_train, x_predict, x_external = self._features(
            representations, include_external=bool(len(indices))
        )
        y_train = self.train["Y"].to_numpy(
            dtype=int if self.task.task_type == "classification" else float
        )
        sample_weight = None
        if len(indices):
            x_fit = np.concatenate([x_train, x_external[indices]], axis=0)
            y_external = self.external.iloc[indices]["Y"].to_numpy(dtype=y_train.dtype)
            y_fit = np.concatenate([y_train, y_external])
            sample_weight = external_mass_capped_weights(
                len(y_train),
                len(indices),
                external_mass_ratio=float(
                    self.external_weight_policy["external_mass_ratio"]
                ),
                external_row_weight_cap=float(
                    self.external_weight_policy["external_row_weight_cap"]
                ),
                external_to_min_native_weight_ratio_cap=float(
                    self.external_weight_policy[
                        "external_to_min_native_weight_ratio_cap"
                    ]
                ),
            )
        else:
            x_fit = x_train
            y_fit = y_train
        target_mean = 0.0
        target_scale = 1.0
        model_y = y_fit
        if self.task.task_type == "regression":
            target_mean = float(np.mean(y_fit))
            target_scale = float(np.std(y_fit))
            if not math.isfinite(target_scale) or target_scale <= 0:
                target_scale = 1.0
            model_y = (y_fit - target_mean) / target_scale
        experiment = ExperimentSpec.from_dict(
            {
                "task": self.task_name,
                "seed": self.seed,
                "representations": representations,
                "model": member["model"],
                "imbalance": member.get("imbalance", "none"),
                "data_policy": "official",
                "device": "cpu",
                # CalibratedClassifierCV otherwise creates a nested loky pool
                # for every SLURM array task and relies on short-lived /dev/shm
                # memmaps.  The scientific configuration is unchanged; only
                # the execution parallelism is serialized for the SVM family.
                "n_jobs": 1 if member["model"] == "svm_rbf" else self.n_jobs,
                "hyperparameters": member.get("hyperparameters", {}),
            }
        )
        model = build_model(experiment, self.task, np.asarray(model_y))
        fitted = _fit_with_sample_weight(model, x_fit, np.asarray(model_y), sample_weight)
        values = predict(fitted, x_predict, self.task)
        if self.task.task_type == "regression":
            values = values * target_scale + target_mean
        values = np.asarray(values, dtype=float)
        self._prediction_cache[cache_key] = values
        return values, selection_audit

    def evaluate(
        self,
        model_config: dict[str, Any],
        feature_config: list[str],
        data_config: dict[str, Any],
    ) -> dict[str, Any]:
        key = _signature(
            {"model": model_config, "feature": feature_config, "data": data_config}
        )
        if key in self._evaluation_cache:
            return dict(self._evaluation_cache[key])
        started = time.perf_counter()
        combined = None
        selection_audit: dict[str, Any] = {}
        for member, weight in _members(model_config):
            values, selection_audit = self._single_prediction(
                member, feature_config, data_config
            )
            combined = weight * values if combined is None else combined + weight * values
        if combined is None:
            raise RuntimeError("Model configuration produced no predictions")
        metric_value = score(
            self.predict_frame["Y"].to_numpy(), np.asarray(combined), self.task.metric
        )
        result = {
            "score": float(metric_value),
            "fit_wall_seconds": time.perf_counter() - started,
            "external_selection": selection_audit,
            "prediction_sha256": hashlib.sha256(
                np.asarray(combined, dtype=np.float64).tobytes()
            ).hexdigest(),
        }
        self._evaluation_cache[key] = dict(result)
        return result


def _row(
    *,
    seed: int,
    arm: str,
    model: dict[str, Any],
    feature: list[str],
    data: dict[str, Any],
    baseline: float,
    result: dict[str, Any],
    metric: str,
) -> dict[str, Any]:
    value = {
        "seed": int(seed),
        "arm": arm,
        "model_config": model,
        "feature_config": feature,
        "data_config": data,
        **result,
    }
    value["raw_oriented_delta"] = _raw_oriented_delta(metric, baseline, result["score"])
    value["normalized_improvement"] = _normalized_improvement(
        metric, baseline, result["score"]
    )
    return value


def _summaries(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for arm in sorted({row["arm"] for row in rows}):
        selected = [row for row in rows if row["arm"] == arm]
        improvements = [float(row["normalized_improvement"]) for row in selected]
        scores = [float(row["score"]) for row in selected]
        result[arm] = {
            "mean_score": float(np.mean(scores)),
            "sample_score_sd": float(statistics.stdev(scores)) if len(scores) > 1 else 0.0,
            "mean_normalized_improvement": float(np.mean(improvements)),
            "sample_normalized_improvement_sd": (
                float(statistics.stdev(improvements)) if len(improvements) > 1 else 0.0
            ),
            "positive_seed_fraction": float(np.mean(np.asarray(improvements) > 0)),
            "seeds": [int(row["seed"]) for row in selected],
        }
    return result


def _best_arm(
    rows: list[dict[str, Any]],
    configs: dict[str, Any],
    candidates: list[str],
) -> tuple[str, Any, dict[str, Any]]:
    summaries = _summaries([row for row in rows if row["arm"] in candidates])
    best = max(
        candidates,
        key=lambda arm: (
            summaries[arm]["mean_normalized_improvement"],
            summaries[arm]["positive_seed_fraction"],
            -len(json.dumps(configs[arm], sort_keys=True)),
            arm,
        ),
    )
    return best, configs[best], summaries[best]


def _evaluate_across(
    evaluators: dict[int, Evaluator],
    *,
    arm: str,
    model: dict[str, Any],
    feature: list[str],
    data: dict[str, Any],
    baselines: dict[int, float],
    metric: str,
) -> list[dict[str, Any]]:
    rows = []
    for seed, evaluator in evaluators.items():
        result = evaluator.evaluate(model, feature, data)
        rows.append(
            _row(
                seed=seed,
                arm=arm,
                model=model,
                feature=feature,
                data=data,
                baseline=baselines[seed],
                result=result,
                metric=metric,
            )
        )
    return rows


def _model_discovery(
    evaluators: dict[int, Evaluator], baselines: dict[int, float], metric: str
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    configs: dict[str, Any] = {"M0": BASELINE_MODEL}
    n_train = len(next(iter(evaluators.values())).train)
    families = ["lightgbm", "xgboost", "extra_trees", "random_forest", "catboost"]
    if n_train <= 5000:
        families.append("svm_rbf")
    default_arms = []
    for model in families:
        arm = f"M1_default_{model}"
        configs[arm] = _model_single(model)
        default_arms.append(arm)
        rows.extend(
            _evaluate_across(
                evaluators,
                arm=arm,
                model=configs[arm],
                feature=BASELINE_FEATURE,
                data=NO_DATA,
                baselines=baselines,
                metric=metric,
            )
        )
    m1_arm, m1, m1_summary = _best_arm(rows, configs, default_arms)
    default_summaries = _summaries(rows)
    top_families = [
        arm.removeprefix("M1_default_")
        for arm in sorted(
            default_arms,
            key=lambda arm: default_summaries[arm]["mean_normalized_improvement"],
            reverse=True,
        )[:3]
    ]
    tuned_arms = [m1_arm]
    for model in top_families:
        for variant in range(3):
            hyper = RuleBasedBrain._tuned(model, variant, resolve_task(next(iter(evaluators.values())).task_name).task_type)
            imbalance_options = ["none"]
            if resolve_task(next(iter(evaluators.values())).task_name).task_type == "classification":
                imbalance_options.append("class_weight")
            for imbalance in imbalance_options:
                arm = f"M2_{model}_v{variant}_{imbalance}"
                configs[arm] = _model_single(model, hyper, imbalance)
                tuned_arms.append(arm)
                rows.extend(
                    _evaluate_across(
                        evaluators,
                        arm=arm,
                        model=configs[arm],
                        feature=BASELINE_FEATURE,
                        data=NO_DATA,
                        baselines=baselines,
                        metric=metric,
                    )
                )
    m2_arm, m2, m2_summary = _best_arm(rows, configs, tuned_arms)
    all_summaries = _summaries(rows)
    best_by_family: dict[str, str] = {}
    for arm in tuned_arms:
        family = configs[arm]["model"]
        if family not in best_by_family or (
            all_summaries[arm]["mean_normalized_improvement"]
            > all_summaries[best_by_family[family]]["mean_normalized_improvement"]
        ):
            best_by_family[family] = arm
    ranked = sorted(
        best_by_family.values(),
        key=lambda arm: all_summaries[arm]["mean_normalized_improvement"],
        reverse=True,
    )[:3]
    fusion_arms = [m2_arm]
    for left_index in range(len(ranked)):
        for right_index in range(left_index + 1, len(ranked)):
            left, right = ranked[left_index], ranked[right_index]
            for weight in (0.25, 0.5, 0.75):
                arm = f"M3_{configs[left]['model']}_{configs[right]['model']}_w{weight:.2f}"
                configs[arm] = {
                    "kind": "ensemble",
                    "members": [configs[left], configs[right]],
                    "weights": [weight, 1.0 - weight],
                    "aggregation": "probability_or_raw",
                }
                fusion_arms.append(arm)
                rows.extend(
                    _evaluate_across(
                        evaluators,
                        arm=arm,
                        model=configs[arm],
                        feature=BASELINE_FEATURE,
                        data=NO_DATA,
                        baselines=baselines,
                        metric=metric,
                    )
                )
    m3_arm, m3, m3_summary = _best_arm(rows, configs, fusion_arms)
    selected = {
        "M1": {"source_arm": m1_arm, "config": m1, "development": m1_summary},
        "M2": {"source_arm": m2_arm, "config": m2, "development": m2_summary},
        "M3": {"source_arm": m3_arm, "config": m3, "development": m3_summary},
    }
    return rows, selected, {"top_default_families_tuned": top_families}


def _feature_discovery(
    evaluators: dict[int, Evaluator],
    baselines: dict[int, float],
    metric: str,
    protocol: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    configs: dict[str, list[str]] = {"F0": BASELINE_FEATURE}
    single_arms = []
    for index, view in enumerate(protocol["feature_axis"]["single_views"]):
        arm = f"F1_single_{index}"
        configs[arm] = list(view)
        single_arms.append(arm)
        rows.extend(
            _evaluate_across(
                evaluators,
                arm=arm,
                model=BASELINE_MODEL,
                feature=configs[arm],
                data=NO_DATA,
                baselines=baselines,
                metric=metric,
            )
        )
    f1_arm, f1, f1_summary = _best_arm(rows, configs, single_arms)
    composed_arms = [f1_arm]
    for index, view in enumerate(protocol["feature_axis"]["composed_views"]):
        arm = f"F2_composed_{index}"
        configs[arm] = list(view)
        composed_arms.append(arm)
        rows.extend(
            _evaluate_across(
                evaluators,
                arm=arm,
                model=BASELINE_MODEL,
                feature=configs[arm],
                data=NO_DATA,
                baselines=baselines,
                metric=metric,
            )
        )
    f2_arm, f2, f2_summary = _best_arm(rows, configs, composed_arms)
    summaries = _summaries(rows)
    ranked_composed = sorted(
        [arm for arm in composed_arms if arm.startswith("F2_")],
        key=lambda arm: summaries[arm]["mean_normalized_improvement"],
        reverse=True,
    )
    fusion_arms = [f2_arm]
    if len(ranked_composed) >= 2:
        for count in (2, min(3, len(ranked_composed))):
            selected_views = [configs[arm] for arm in ranked_composed[:count]] + [f1]
            union = []
            for view in selected_views:
                for name in view:
                    if name not in union:
                        union.append(name)
            arm = f"F3_union_top{count}"
            configs[arm] = union
            fusion_arms.append(arm)
            rows.extend(
                _evaluate_across(
                    evaluators,
                    arm=arm,
                    model=BASELINE_MODEL,
                    feature=union,
                    data=NO_DATA,
                    baselines=baselines,
                    metric=metric,
                )
            )
    f3_arm, f3, f3_summary = _best_arm(rows, configs, fusion_arms)
    return rows, {
        "F1": {"source_arm": f1_arm, "config": f1, "development": f1_summary},
        "F2": {"source_arm": f2_arm, "config": f2, "development": f2_summary},
        "F3": {"source_arm": f3_arm, "config": f3, "development": f3_summary},
    }


def _data_discovery(
    evaluators: dict[int, Evaluator],
    baselines: dict[int, float],
    metric: str,
    protocol: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    configs: dict[str, dict[str, Any]] = {"D0": NO_DATA}
    configs["D1"] = {"enabled": True, "method": "random", "ratio": 1.0}
    configs["D2"] = {"enabled": True, "method": "coreset", "ratio": 1.0}
    for ratio in protocol["data_axis"]["ratios"]:
        configs[f"D3_ratio_{ratio:g}"] = {
            "enabled": True,
            "method": "coreset",
            "ratio": float(ratio),
        }
    rows: list[dict[str, Any]] = []
    for arm, config in configs.items():
        if arm == "D0":
            continue
        rows.extend(
            _evaluate_across(
                evaluators,
                arm=arm,
                model=BASELINE_MODEL,
                feature=BASELINE_FEATURE,
                data=config,
                baselines=baselines,
                metric=metric,
            )
        )
    d3_candidates = ["D2", *[arm for arm in configs if arm.startswith("D3_ratio_")]]
    d3_arm, d3, d3_summary = _best_arm(rows, configs, d3_candidates)
    threshold = float(protocol["selection"]["practical_threshold"])
    stable = (
        d3_summary["mean_normalized_improvement"] >= threshold
        and d3_summary["positive_seed_fraction"]
        >= float(protocol["selection"]["positive_seed_fraction"])
        and all(evaluator.external_admitted for evaluator in evaluators.values())
    )
    if not stable:
        d3_arm = "D0_abstain"
        d3 = NO_DATA
    summaries = _summaries(rows)
    return rows, {
        "D1": {"source_arm": "D1", "config": configs["D1"], "development": summaries["D1"]},
        "D2": {"source_arm": "D2", "config": configs["D2"], "development": summaries["D2"]},
        "D3": {
            "source_arm": d3_arm,
            "config": d3,
            "development": d3_summary,
            "stable_promotion_passed": stable,
        },
    }


def _factorial_configs(selected: dict[str, Any]) -> dict[str, dict[str, Any]]:
    result = {}
    for cell in ("000", "100", "010", "001", "110", "101", "011", "111"):
        result[cell] = {
            "model": selected["model"]["M3"]["config"] if cell[0] == "1" else BASELINE_MODEL,
            "feature": selected["feature"]["F3"]["config"] if cell[1] == "1" else BASELINE_FEATURE,
            "data": selected["data"]["D3"]["config"] if cell[2] == "1" else NO_DATA,
        }
    return result


def _evaluate_factorial(
    evaluators: dict[int, Evaluator],
    baselines: dict[int, float],
    metric: str,
    selected: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    configs = _factorial_configs(selected)
    rows = []
    for cell, config in configs.items():
        rows.extend(
            _evaluate_across(
                evaluators,
                arm=f"factorial_{cell}",
                model=config["model"],
                feature=config["feature"],
                data=config["data"],
                baselines=baselines,
                metric=metric,
            )
        )
    return rows, configs


def _build_split_evaluators(
    *,
    task: str,
    seeds: list[int],
    data_dir: Path,
    external: pd.DataFrame,
    admitted: bool,
    weight_policy: dict[str, Any] | None,
    protocol: dict[str, Any],
    feature_cache_dir: Path,
    n_jobs: int,
) -> dict[int, Evaluator]:
    benchmark = TASKS[task][0]
    group = admet_group(path=str(data_dir))
    evaluators = {}
    effective_external = external if admitted else external.iloc[:0].copy()
    cache = FeatureCache(feature_cache_dir, device="cpu")
    for seed in seeds:
        train_raw, valid_raw = group.get_train_valid_split(
            benchmark=benchmark, split_type="default", seed=int(seed)
        )
        train, _ = _standardized_tdc(train_raw)
        valid, _ = _standardized_tdc(valid_raw)
        evaluators[int(seed)] = Evaluator(
            task=task,
            seed=int(seed),
            train=train,
            predict_frame=valid,
            external=effective_external,
            external_admitted=admitted,
            external_weight_policy=weight_policy,
            protocol=protocol,
            feature_cache=cache,
            n_jobs=n_jobs,
        )
    return evaluators


def _baseline_rows(
    evaluators: dict[int, Evaluator], metric: str
) -> tuple[list[dict[str, Any]], dict[int, float]]:
    rows = []
    scores = {}
    for seed, evaluator in evaluators.items():
        result = evaluator.evaluate(BASELINE_MODEL, BASELINE_FEATURE, NO_DATA)
        scores[seed] = float(result["score"])
        rows.append(
            _row(
                seed=seed,
                arm="baseline",
                model=BASELINE_MODEL,
                feature=BASELINE_FEATURE,
                data=NO_DATA,
                baseline=scores[seed],
                result=result,
                metric=metric,
            )
        )
    return rows, scores


def _run_discovery(
    args: argparse.Namespace,
    task: str,
    protocol: dict[str, Any],
    external: pd.DataFrame,
    external_audit: dict[str, Any],
) -> dict[str, Any]:
    metric = TASKS[task][2]
    seeds = [int(value) for value in protocol["seeds"]["discovery"]]
    evaluators = _build_split_evaluators(
        task=task,
        seeds=seeds,
        data_dir=args.data_dir,
        external=external,
        admitted=external_audit["admitted"],
        weight_policy=external_audit.get("weight_policy"),
        protocol=protocol,
        feature_cache_dir=args.feature_cache_dir / task,
        n_jobs=args.n_jobs,
    )
    baseline_rows, baselines = _baseline_rows(evaluators, metric)
    model_rows, model_selected, model_meta = _model_discovery(evaluators, baselines, metric)
    feature_rows, feature_selected = _feature_discovery(
        evaluators, baselines, metric, protocol
    )
    data_rows, data_selected = _data_discovery(evaluators, baselines, metric, protocol)
    selected = {
        "model": model_selected,
        "feature": feature_selected,
        "data": data_selected,
    }
    factorial_rows, factorial_configs = _evaluate_factorial(
        evaluators, baselines, metric, selected
    )
    factorial_summary = _summaries(factorial_rows)
    threshold = float(protocol["routing"]["threshold"])
    best_cell = max(
        factorial_configs,
        key=lambda cell: factorial_summary[f"factorial_{cell}"]["mean_normalized_improvement"],
    )
    if factorial_summary[f"factorial_{best_cell}"]["mean_normalized_improvement"] < threshold:
        best_cell = "000"
    return {
        "protocol": PROTOCOL_NAME,
        "protocol_sha256": protocol["sha256"],
        "phase": "discovery",
        "task": task,
        "benchmark": TASKS[task][0],
        "task_type": TASKS[task][1],
        "metric": metric,
        "status": "completed",
        "seeds": seeds,
        "safety": {
            "test_structure_columns_loaded": ["Drug"],
            "test_labels_loaded": False,
            "selection_uses_test_results": False,
        },
        "external_audit": external_audit,
        "baseline_rows": baseline_rows,
        "model_axis": {"selected": model_selected, "metadata": model_meta, "rows": model_rows},
        "feature_axis": {"selected": feature_selected, "rows": feature_rows},
        "data_axis": {"selected": data_selected, "rows": data_rows},
        "selected": selected,
        "factorial": {
            "configs": factorial_configs,
            "rows": factorial_rows,
            "summaries": factorial_summary,
        },
        "routing": {
            "selected_cell": best_cell,
            "development_summary": factorial_summary[f"factorial_{best_cell}"],
        },
    }


def _selected_ladder_configs(discovery: dict[str, Any]) -> dict[str, dict[str, Any]]:
    selected = discovery["selected"]
    result = {
        "baseline": {"model": BASELINE_MODEL, "feature": BASELINE_FEATURE, "data": NO_DATA}
    }
    for level in ("M1", "M2", "M3"):
        result[level] = {
            "model": selected["model"][level]["config"],
            "feature": BASELINE_FEATURE,
            "data": NO_DATA,
        }
    for level in ("F1", "F2", "F3"):
        result[level] = {
            "model": BASELINE_MODEL,
            "feature": selected["feature"][level]["config"],
            "data": NO_DATA,
        }
    for level in ("D1", "D2", "D3"):
        result[level] = {
            "model": BASELINE_MODEL,
            "feature": BASELINE_FEATURE,
            "data": selected["data"][level]["config"],
        }
    return result


def _run_frozen_grid(
    *,
    evaluators: dict[int, Evaluator],
    baselines: dict[int, float],
    metric: str,
    discovery: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
    ladder_configs = _selected_ladder_configs(discovery)
    ladder_rows = []
    for arm, config in ladder_configs.items():
        ladder_rows.extend(
            _evaluate_across(
                evaluators,
                arm=arm,
                model=config["model"],
                feature=config["feature"],
                data=config["data"],
                baselines=baselines,
                metric=metric,
            )
        )
    factorial_rows, factorial_configs = _evaluate_factorial(
        evaluators, baselines, metric, discovery["selected"]
    )
    return ladder_rows, ladder_configs, factorial_rows, factorial_configs


def _run_confirmation(
    args: argparse.Namespace,
    task: str,
    protocol: dict[str, Any],
    discovery: dict[str, Any],
    external: pd.DataFrame,
    external_audit: dict[str, Any],
) -> dict[str, Any]:
    metric = TASKS[task][2]
    seeds = [int(value) for value in protocol["seeds"]["confirmation"]]
    evaluators = _build_split_evaluators(
        task=task,
        seeds=seeds,
        data_dir=args.data_dir,
        external=external,
        admitted=external_audit["admitted"],
        weight_policy=external_audit.get("weight_policy"),
        protocol=protocol,
        feature_cache_dir=args.feature_cache_dir / task,
        n_jobs=args.n_jobs,
    )
    baseline_rows, baselines = _baseline_rows(evaluators, metric)
    ladder_rows, ladder_configs, factorial_rows, factorial_configs = _run_frozen_grid(
        evaluators=evaluators,
        baselines=baselines,
        metric=metric,
        discovery=discovery,
    )
    return {
        "protocol": PROTOCOL_NAME,
        "protocol_sha256": protocol["sha256"],
        "phase": "confirmation",
        "task": task,
        "benchmark": TASKS[task][0],
        "task_type": TASKS[task][1],
        "metric": metric,
        "status": "completed",
        "seeds": seeds,
        "discovery_artifact": str((args.output_root / "discovery" / f"{task}.json").resolve()),
        "discovery_artifact_sha256": _sha256(args.output_root / "discovery" / f"{task}.json"),
        "safety": {
            "test_structure_columns_loaded": ["Drug"],
            "test_labels_loaded": False,
            "selection_uses_test_results": False,
            "confirmation_seeds_disjoint": True,
        },
        "external_audit": external_audit,
        "baseline_rows": baseline_rows,
        "ladder": {
            "configs": ladder_configs,
            "rows": ladder_rows,
            "summaries": _summaries(ladder_rows),
        },
        "factorial": {
            "configs": factorial_configs,
            "rows": factorial_rows,
            "summaries": _summaries(factorial_rows),
        },
        "routing": discovery["routing"],
    }


def _run_certification(
    args: argparse.Namespace,
    task: str,
    protocol: dict[str, Any],
    discovery: dict[str, Any],
    confirmation: dict[str, Any],
    external: pd.DataFrame,
    external_audit: dict[str, Any],
) -> dict[str, Any]:
    del confirmation
    benchmark, task_type, metric, _ = TASKS[task]
    train_raw = pd.read_csv(
        args.data_dir / "admet_group" / benchmark / "train_val.csv",
        usecols=["Drug", "Y"],
    )
    test_raw = pd.read_csv(
        args.data_dir / "admet_group" / benchmark / "test.csv",
        usecols=["Drug", "Y"],
    )
    train, _ = _standardized_tdc(train_raw)
    test = _standardize_structure_frame(test_raw[["Drug"]])
    test["Y"] = test_raw["Y"].to_numpy()
    seeds = [int(value) for value in protocol["seeds"]["certification_model_seeds"]]
    cache = FeatureCache(args.feature_cache_dir / task, device="cpu")
    effective_external = (
        external if external_audit["admitted"] else external.iloc[:0].copy()
    )
    evaluators = {
        seed: Evaluator(
            task=task,
            seed=seed,
            train=train,
            predict_frame=test,
            external=effective_external,
            external_admitted=external_audit["admitted"],
            external_weight_policy=external_audit.get("weight_policy"),
            protocol=protocol,
            feature_cache=cache,
            n_jobs=args.n_jobs,
            certification=True,
        )
        for seed in seeds
    }
    baseline_rows, baselines = _baseline_rows(evaluators, metric)
    ladder_rows, ladder_configs, factorial_rows, factorial_configs = _run_frozen_grid(
        evaluators=evaluators,
        baselines=baselines,
        metric=metric,
        discovery=discovery,
    )
    routed_cell = discovery["routing"]["selected_cell"]
    routed_rows = [
        {**row, "arm": "routed"}
        for row in factorial_rows
        if row["arm"] == f"factorial_{routed_cell}"
    ]
    return {
        "protocol": PROTOCOL_NAME,
        "protocol_sha256": protocol["sha256"],
        "phase": "certification",
        "task": task,
        "benchmark": benchmark,
        "task_type": task_type,
        "metric": metric,
        "status": "completed",
        "seeds": seeds,
        "discovery_artifact_sha256": _sha256(args.output_root / "discovery" / f"{task}.json"),
        "confirmation_artifact_sha256": _sha256(args.output_root / "confirmation" / f"{task}.json"),
        "safety": {
            "test_structure_columns_loaded": ["Drug"],
            "test_labels_loaded": True,
            "test_labels_loaded_only_after_frozen_confirmation": True,
            "test_results_used_for_selection": False,
        },
        "external_audit": external_audit,
        "baseline_rows": baseline_rows,
        "ladder": {
            "configs": ladder_configs,
            "rows": ladder_rows,
            "summaries": _summaries(ladder_rows),
        },
        "factorial": {
            "configs": factorial_configs,
            "rows": factorial_rows,
            "summaries": _summaries(factorial_rows),
        },
        "routing": {
            "selected_cell": routed_cell,
            "rows": routed_rows,
            "summary": _summaries(routed_rows)["routed"],
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("discovery", "confirmation", "certification"), required=True)
    parser.add_argument("--task", choices=tuple(TASKS), required=True)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data")
    parser.add_argument("--external-data", type=Path, default=DEFAULT_EXTERNAL)
    parser.add_argument(
        "--feature-cache-dir",
        type=Path,
        default=DEFAULT_OUTPUT / "artifacts/feature_cache",
    )
    parser.add_argument("--n-jobs", type=int, default=8)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.protocol = args.protocol.resolve()
    args.output_root = args.output_root.resolve()
    args.data_dir = args.data_dir.resolve()
    args.external_data = args.external_data.resolve()
    args.feature_cache_dir = args.feature_cache_dir.resolve()
    protocol = _load_protocol(args.protocol)
    task = args.task
    phase_dir = args.output_root / args.phase
    phase_dir.mkdir(parents=True, exist_ok=True)
    output = phase_dir / f"{task}.json"
    error = phase_dir / f"{task}.error.json"
    if output.exists() and not args.overwrite:
        print(json.dumps({"task": task, "phase": args.phase, "status": "skipped_existing"}))
        return
    benchmark = TASKS[task][0]
    test_structure_raw = pd.read_csv(
        args.data_dir / "admet_group" / benchmark / "test.csv", usecols=["Drug"]
    )
    test_structures = _standardize_structure_frame(test_structure_raw)
    external, external_audit = _load_external(
        task=task,
        path=args.external_data,
        protocol=protocol,
        test_structures=test_structures,
    )
    try:
        if args.phase == "discovery":
            payload = _run_discovery(args, task, protocol, external, external_audit)
        elif args.phase == "confirmation":
            discovery = _complete_artifact(
                args.output_root / "discovery" / f"{task}.json",
                phase="discovery",
                task=task,
                protocol_sha=protocol["sha256"],
            )
            payload = _run_confirmation(
                args, task, protocol, discovery, external, external_audit
            )
        else:
            discovery = _complete_artifact(
                args.output_root / "discovery" / f"{task}.json",
                phase="discovery",
                task=task,
                protocol_sha=protocol["sha256"],
            )
            confirmation = _complete_artifact(
                args.output_root / "confirmation" / f"{task}.json",
                phase="confirmation",
                task=task,
                protocol_sha=protocol["sha256"],
            )
            payload = _run_certification(
                args,
                task,
                protocol,
                discovery,
                confirmation,
                external,
                external_audit,
            )
        temporary = output.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        temporary.replace(output)
        error.unlink(missing_ok=True)
        print(
            json.dumps(
                {
                    "task": task,
                    "phase": args.phase,
                    "status": payload["status"],
                    "output": str(output),
                }
            )
        )
    except Exception as exc:
        failure = {
            "protocol": PROTOCOL_NAME,
            "protocol_sha256": protocol["sha256"],
            "phase": args.phase,
            "task": task,
            "status": "failed",
            "error_type": type(exc).__name__,
            "error": str(exc),
            "traceback": traceback.format_exc(),
            "safety": {
                "test_structure_columns_loaded": ["Drug"],
                "test_labels_loaded": args.phase == "certification",
            },
        }
        error.write_text(json.dumps(failure, indent=2, sort_keys=True) + "\n")
        raise


if __name__ == "__main__":
    main()
