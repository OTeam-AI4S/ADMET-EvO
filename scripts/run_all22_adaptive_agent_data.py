"""Agent-first adaptive external-data experiment for all 22 TDC ADMET tasks.

Development uses only a fixed LightGBM/Morgan agent controller on seeds 2-5 to
choose the smallest coreset within one standard error of the best result.
Agent confirmation uses untouched seeds 8-11. Classic models may run only after
the agent arm is frozen and confirmed, using exactly the same rows and weights.
Only the TDC test structure column is loaded to build an identity firewall;
test labels are never loaded.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs, RDLogger
from run_classic_ml_external_data_confirmation import (
    TASKS,
    _classification_model,
    _curate_external,
    _oriented_delta,
    _regression_model,
    _score,
    _sha256,
    _source_spec,
    _standardized_tdc,
)
from scipy import sparse
from tdc.benchmark_group import admet_group

from admet_agent.adaptive_external_data import (
    external_mass_capped_weights,
    molecular_scaffold,
    select_relevance_diverse_binary_coreset,
    select_relevance_diverse_regression_coreset,
)
from admet_agent.external_data_admission import require_bound_task_admission
from admet_agent.herg_data_scaling import (
    fingerprints_to_csr,
    morgan_fingerprints,
    near_neighbor_keep_mask,
)
from admet_agent.heldout_structure_firewall import (
    load_structure_only_inchikey_blocklist,
)

RDLogger.DisableLog("rdApp.*")

PROTOCOL = "all22-agent-first-adaptive-external-data-v2"
RATIOS = (0.5, 1.0, 2.0, 4.0, 8.0)
DEVELOPMENT_SEEDS = (2, 3, 4, 5)
CONFIRMATION_SEEDS = (8, 9, 10, 11)
AGENT_MODEL = "lightgbm"
CLASSIC_MODELS = ("random_forest", "linear_svm", "morgan_mlp")
NEAR_THRESHOLD = 0.80
MINIMUM_RELEVANCE = 0.20
PRACTICAL_DELTA = 0.005
MAX_CLASS_RATE_GAP = 0.20
MAX_MEDIAN_SHIFT_IQR = 0.75
MIN_IQR_RATIO = 1.0 / 3.0
MAX_IQR_RATIO = 3.0


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _prepare_task(args: argparse.Namespace, task: str) -> dict:
    benchmark, task_type, metric, _ = TASKS[task]
    source_args = SimpleNamespace(cohort_dir=args.cohort_dir)
    source_spec = _source_spec(source_args, task)
    admission = require_bound_task_admission(
        args.admission_report,
        task,
        source_path=source_spec["path"],
        source_name=source_spec["name"],
        augmentation_mode=args.augmentation_mode,
    )
    external, audit, _ = _curate_external(
        source_spec["path"],
        task_type=task_type,
        smiles_column=source_spec["smiles_column"],
        label_column=source_spec["label_column"],
        id_column=source_spec["id_column"],
        n_jobs=args.n_jobs,
        majority_cap=15_000,
    )
    discovery = pd.read_csv(
        args.data_dir / "admet_group" / benchmark / "train_val.csv",
        usecols=["Drug", "Y"],
    )
    discovery_standard, fallback_count = _standardized_tdc(discovery)
    discovery_exact_mask = external["standard_inchikey"].isin(
        set(discovery_standard["standard_inchikey"])
    )
    test_ids, test_firewall_audit = load_structure_only_inchikey_blocklist(
        args.data_dir / "admet_group" / benchmark / "test.csv"
    )
    test_exact_mask = external["standard_inchikey"].isin(test_ids)
    discovery_exact_exclusions = int(discovery_exact_mask.sum())
    test_exact_exclusions = int(test_exact_mask.sum())
    exact_mask = discovery_exact_mask | test_exact_mask
    external = external.loc[~exact_mask].reset_index(drop=True)
    parseable = external["standard_smiles"].map(
        lambda value: Chem.MolFromSmiles(str(value)) is not None
    )
    parse_exclusions = int((~parseable).sum())
    external = external.loc[parseable].reset_index(drop=True)
    external["scaffold"] = external["standard_smiles"].map(molecular_scaffold)
    minimum = 100 if task_type == "classification" else 30
    return {
        "task": task,
        "benchmark": benchmark,
        "task_type": task_type,
        "metric": metric,
        "admission": admission,
        "weight_policy": admission["weight_policy"],
        "source": {
            "name": source_spec["name"],
            "path": str(source_spec["path"]),
            "sha256": _sha256(source_spec["path"]),
            "lineage_status": source_spec["lineage_status"],
            "semantic_role": source_spec["semantic_role"],
        },
        "curation": {
            **audit,
            "exact_tdc_discovery_overlap_exclusions": discovery_exact_exclusions,
            "exact_tdc_test_structure_overlap_exclusions": test_exact_exclusions,
            **test_firewall_audit,
            "fingerprint_parse_exclusions": parse_exclusions,
            "post_exact_firewall_identities": len(external),
            "minimum_required_identities": minimum,
            "tdc_standardization_fallback_count": fallback_count,
        },
        "external": external,
        "external_fps": (
            morgan_fingerprints(external["standard_smiles"].tolist())
            if len(external) >= minimum
            else []
        ),
        "eligible_for_development": len(external) >= minimum,
    }


def _relevance(external_fps: list, train_fps: list) -> np.ndarray:
    return np.asarray(
        [
            max(DataStructs.BulkTanimotoSimilarity(fingerprint, train_fps))
            for fingerprint in external_fps
        ],
        dtype=float,
    )


def _predictive_score(
    *,
    task_type: str,
    metric: str,
    model_name: str,
    seed: int,
    n_jobs: int,
    train_x,
    train_y: np.ndarray,
    valid_x,
    valid_y: np.ndarray,
    sample_weight: np.ndarray | None = None,
) -> float:
    if task_type == "classification":
        model = _classification_model(model_name, seed, n_jobs)
        fitted = model.fit(train_x, train_y, sample_weight=sample_weight)
        prediction = (
            fitted.predict_proba(valid_x)[:, 1]
            if hasattr(fitted, "predict_proba")
            else fitted.decision_function(valid_x)
        )
    else:
        mean = float(np.mean(train_y))
        scale = float(np.std(train_y))
        if not math.isfinite(scale) or scale <= 0:
            scale = 1.0
        model = _regression_model(model_name, seed, n_jobs)
        fitted = model.fit(
            train_x,
            (train_y - mean) / scale,
            sample_weight=sample_weight,
        )
        prediction = fitted.predict(valid_x) * scale + mean
    return _score(metric, valid_y, np.asarray(prediction, dtype=float))


def _coreset(
    prepared: dict,
    relevance: np.ndarray,
    train_y: np.ndarray,
    ratio: float,
) -> tuple[np.ndarray, dict]:
    requested = max(1, int(round(ratio * len(train_y))))
    if prepared["task_type"] == "classification":
        return select_relevance_diverse_binary_coreset(
            prepared["external"],
            relevance,
            total_rows=requested,
            target_positive_rate=float(train_y.mean()),
            minimum_relevance=MINIMUM_RELEVANCE,
        )
    return select_relevance_diverse_regression_coreset(
        prepared["external"],
        relevance,
        total_rows=requested,
        reference_targets=train_y,
        minimum_relevance=MINIMUM_RELEVANCE,
    )


def _run_rows(
    prepared: dict,
    *,
    data_dir: Path,
    seeds: tuple[int, ...],
    ratios: tuple[float, ...],
    models: tuple[str, ...],
    n_jobs: int,
) -> list[dict]:
    group = admet_group(path=str(data_dir))
    external = prepared["external"]
    external_fps = prepared["external_fps"]
    external_x = fingerprints_to_csr(external_fps)
    dtype = int if prepared["task_type"] == "classification" else float
    external_y = external["Y"].to_numpy(dtype=dtype)
    rows: list[dict] = []
    for seed in seeds:
        train_raw, valid_raw = group.get_train_valid_split(
            benchmark=prepared["benchmark"], split_type="default", seed=seed
        )
        train, _ = _standardized_tdc(train_raw)
        valid, _ = _standardized_tdc(valid_raw)
        train_fps = morgan_fingerprints(train["standard_smiles"].tolist())
        valid_fps = morgan_fingerprints(valid["standard_smiles"].tolist())
        train_x = fingerprints_to_csr(train_fps)
        valid_x = fingerprints_to_csr(valid_fps)
        train_y = train["Y"].to_numpy(dtype=dtype)
        valid_y = valid["Y"].to_numpy(dtype=dtype)
        keep = near_neighbor_keep_mask(
            external_fps, valid_fps, threshold=NEAR_THRESHOLD
        )
        relevance = _relevance(external_fps, train_fps)
        relevance[~keep] = -1.0
        baseline = {
            model: _predictive_score(
                task_type=prepared["task_type"],
                metric=prepared["metric"],
                model_name=model,
                seed=seed,
                n_jobs=n_jobs,
                train_x=train_x,
                train_y=train_y,
                valid_x=valid_x,
                valid_y=valid_y,
            )
            for model in models
        }
        for ratio in ratios:
            selected, audit = _coreset(prepared, relevance, train_y, ratio)
            selected_ids = external.iloc[selected]["standard_inchikey"].tolist()
            signature = hashlib.sha256("\n".join(selected_ids).encode()).hexdigest()
            augmented_x = sparse.vstack([train_x, external_x[selected]], format="csr")
            augmented_y = np.concatenate([train_y, external_y[selected]])
            weights = external_mass_capped_weights(
                len(train_y),
                len(selected),
                external_mass_ratio=prepared["weight_policy"]["external_mass_ratio"],
                external_row_weight_cap=prepared["weight_policy"][
                    "external_row_weight_cap"
                ],
                external_to_min_native_weight_ratio_cap=prepared["weight_policy"][
                    "external_to_min_native_weight_ratio_cap"
                ],
            )
            for model in models:
                fit_started = time.perf_counter()
                augmented = _predictive_score(
                    task_type=prepared["task_type"],
                    metric=prepared["metric"],
                    model_name=model,
                    seed=seed,
                    n_jobs=n_jobs,
                    train_x=augmented_x,
                    train_y=augmented_y,
                    valid_x=valid_x,
                    valid_y=valid_y,
                    sample_weight=weights,
                )
                row = {
                    "seed": seed,
                    "ratio": ratio,
                    "model": model,
                    "tdc_train_rows": len(train_y),
                    "tdc_validation_rows": len(valid_y),
                    "near_neighbor_exclusions": int((~keep).sum()),
                    **{f"coreset_{key}": value for key, value in audit.items()},
                    "coreset_sha256": signature,
                    "tdc_total_sample_weight": float(weights[: len(train_y)].sum()),
                    "external_total_sample_weight": float(weights[len(train_y) :].sum()),
                    "baseline_score": baseline[model],
                    "augmented_score": augmented,
                    "oriented_delta": _oriented_delta(
                        prepared["metric"], baseline[model], augmented
                    ),
                    "fit_wall_seconds": time.perf_counter() - fit_started,
                }
                rows.append(row)
                print(json.dumps({"task": prepared["task"], **row}), flush=True)
    return rows


def _ratio_selection(rows: list[dict]) -> tuple[dict, dict]:
    summaries = {}
    passing = []
    for ratio in RATIOS:
        selected = [row for row in rows if row["ratio"] == ratio]
        delta = np.asarray([row["oriented_delta"] for row in selected], dtype=float)
        summary = {
            "baseline_mean": float(np.mean([row["baseline_score"] for row in selected])),
            "augmented_mean": float(np.mean([row["augmented_score"] for row in selected])),
            "oriented_delta_mean": float(delta.mean()),
            "oriented_delta_std": float(delta.std()),
            "standard_error": float(delta.std() / math.sqrt(len(delta))),
            "positive_seed_fraction": float((delta > 0).mean()),
        }
        if "coreset_target_positive_rate" in selected[0]:
            rate_gaps = np.asarray(
                [
                    abs(
                        row["coreset_selected_positive_rate"]
                        - row["coreset_target_positive_rate"]
                    )
                    for row in selected
                ],
                dtype=float,
            )
            summary.update(
                {
                    "maximum_class_rate_gap": float(rate_gaps.max()),
                    "maximum_allowed_class_rate_gap": MAX_CLASS_RATE_GAP,
                    "quality_distribution_passed": bool(
                        np.all(rate_gaps <= MAX_CLASS_RATE_GAP)
                    ),
                }
            )
        else:
            median_shifts = []
            iqr_ratios = []
            for row in selected:
                reference_iqr = row["coreset_reference_target_iqr"]
                selected_iqr = row["coreset_selected_target_iqr"]
                if reference_iqr <= 0:
                    median_shifts.append(float("inf"))
                    iqr_ratios.append(float("inf"))
                else:
                    median_shifts.append(
                        abs(
                            row["coreset_selected_target_median"]
                            - row["coreset_reference_target_median"]
                        )
                        / reference_iqr
                    )
                    iqr_ratios.append(selected_iqr / reference_iqr)
            summary.update(
                {
                    "maximum_median_shift_iqr": float(max(median_shifts)),
                    "maximum_allowed_median_shift_iqr": MAX_MEDIAN_SHIFT_IQR,
                    "minimum_iqr_ratio": float(min(iqr_ratios)),
                    "maximum_iqr_ratio": float(max(iqr_ratios)),
                    "allowed_iqr_ratio_interval": [MIN_IQR_RATIO, MAX_IQR_RATIO],
                    "quality_distribution_passed": bool(
                        max(median_shifts) <= MAX_MEDIAN_SHIFT_IQR
                        and min(iqr_ratios) >= MIN_IQR_RATIO
                        and max(iqr_ratios) <= MAX_IQR_RATIO
                    ),
                }
            )
        summary["development_passed"] = bool(
            summary["oriented_delta_mean"] >= PRACTICAL_DELTA
            and summary["positive_seed_fraction"] >= 0.75
            and summary["quality_distribution_passed"]
        )
        summaries[str(ratio)] = summary
        if summary["development_passed"]:
            passing.append((ratio, summary))
    if not passing:
        return summaries, {
            "status": "auto_abstain_no_stable_agent_gain",
            "selected_ratio": None,
            "rule": "smallest passing ratio within one best-ratio standard error",
        }
    best_ratio, best = max(passing, key=lambda item: item[1]["oriented_delta_mean"])
    floor = best["oriented_delta_mean"] - best["standard_error"]
    eligible = sorted(
        ratio for ratio, summary in passing if summary["oriented_delta_mean"] >= floor
    )
    return summaries, {
        "status": "selected",
        "selected_ratio": eligible[0],
        "best_ratio": best_ratio,
        "best_mean": best["oriented_delta_mean"],
        "one_standard_error_floor": floor,
        "eligible_ratios": eligible,
        "rule": "smallest passing ratio within one best-ratio standard error",
    }


def _paired_summary(rows: list[dict], model: str) -> dict:
    selected = [row for row in rows if row["model"] == model]
    baseline = np.asarray([row["baseline_score"] for row in selected], dtype=float)
    augmented = np.asarray([row["augmented_score"] for row in selected], dtype=float)
    delta = np.asarray([row["oriented_delta"] for row in selected], dtype=float)
    positive = float((delta > 0).mean())
    return {
        "baseline_mean": float(baseline.mean()),
        "baseline_std": float(baseline.std()),
        "augmented_mean": float(augmented.mean()),
        "augmented_std": float(augmented.std()),
        "oriented_delta_mean": float(delta.mean()),
        "oriented_delta_std": float(delta.std()),
        "positive_seed_fraction": positive,
        "promotion_passed": bool(delta.mean() >= PRACTICAL_DELTA and positive >= 0.75),
    }


def _base_result(args: argparse.Namespace, prepared: dict) -> dict:
    return {
        "protocol": PROTOCOL,
        "task": prepared["task"],
        "benchmark": prepared["benchmark"],
        "task_type": prepared["task_type"],
        "metric": prepared["metric"],
        "source": prepared["source"],
        "curation": prepared["curation"],
        "admission": prepared["admission"],
        "weight_policy": prepared["weight_policy"],
        "safety": {
            "tdc_test_loaded": True,
            "tdc_test_structures_loaded_for_firewall": True,
            "tdc_test_columns_loaded": ["Drug"],
            "tdc_test_labels_loaded": False,
            "test_labels_accessible": False,
        },
        "claim_tier": "controlled_train_validation_auxiliary_data",
    }


def run_development(args: argparse.Namespace, prepared: dict) -> dict:
    result = {
        **_base_result(args, prepared),
        "phase": "agent_development",
        "seeds": list(DEVELOPMENT_SEEDS),
        "ratios": list(RATIOS),
        "models": [AGENT_MODEL],
    }
    if not prepared["eligible_for_development"]:
        result.update(
            {
                "status": "auto_abstain_insufficient_curated_external_data",
                "per_seed_ratio_model": [],
                "selection": {"selected_ratio": None},
            }
        )
    else:
        rows = _run_rows(
            prepared,
            data_dir=args.data_dir,
            seeds=DEVELOPMENT_SEEDS,
            ratios=RATIOS,
            models=(AGENT_MODEL,),
            n_jobs=args.n_jobs,
        )
        summaries, selection = _ratio_selection(rows)
        result.update(
            {
                "status": selection["status"],
                "per_seed_ratio_model": rows,
                "ratio_summaries": summaries,
                "selection": selection,
            }
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return result


def _load_development(args: argparse.Namespace, task: str) -> tuple[Path, dict, float]:
    path = args.development_artifact.resolve()
    payload = json.loads(path.read_text())
    if not (
        payload.get("protocol") == PROTOCOL
        and payload.get("phase") == "agent_development"
        and payload.get("task") == task
        and payload.get("seeds") == list(DEVELOPMENT_SEEDS)
        and payload.get("safety", {}).get("tdc_test_loaded") is True
        and payload.get("safety", {}).get("tdc_test_structures_loaded_for_firewall") is True
        and payload.get("safety", {}).get("tdc_test_columns_loaded") == ["Drug"]
        and payload.get("safety", {}).get("tdc_test_labels_loaded") is False
        and payload.get("safety", {}).get("test_labels_accessible") is False
    ):
        raise RuntimeError("Invalid development artifact")
    ratio = payload.get("selection", {}).get("selected_ratio")
    if ratio is None:
        raise RuntimeError("Agent development abstained; confirmation is blocked")
    return path, payload, float(ratio)


def run_agent_confirmation(args: argparse.Namespace, prepared: dict) -> dict:
    dev_path, development, ratio = _load_development(args, prepared["task"])
    rows = _run_rows(
        prepared,
        data_dir=args.data_dir,
        seeds=CONFIRMATION_SEEDS,
        ratios=(ratio,),
        models=(AGENT_MODEL,),
        n_jobs=args.n_jobs,
    )
    summary = _paired_summary(rows, AGENT_MODEL)
    result = {
        **_base_result(args, prepared),
        "phase": "agent_independent_confirmation",
        "status": "promoted" if summary["promotion_passed"] else "fallback_baseline",
        "development_artifact": str(dev_path),
        "development_artifact_sha256": _file_sha256(dev_path),
        "selection": development["selection"],
        "selected_ratio": ratio,
        "seeds": list(CONFIRMATION_SEEDS),
        "models": [AGENT_MODEL],
        "per_seed_model": rows,
        "agent_summary": summary,
    }
    result["safety"]["confirmation_seeds_disjoint_from_development"] = True
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return result


def run_classic_development(args: argparse.Namespace, prepared: dict) -> dict:
    """Test classic-model transfer after the agent has frozen the data policy."""
    dev_path, development, ratio = _load_development(args, prepared["task"])
    rows = _run_rows(
        prepared,
        data_dir=args.data_dir,
        seeds=DEVELOPMENT_SEEDS,
        ratios=(ratio,),
        models=CLASSIC_MODELS,
        n_jobs=args.n_jobs,
    )
    summaries = {model: _paired_summary(rows, model) for model in CLASSIC_MODELS}
    positive = [
        model
        for model, summary in summaries.items()
        if summary["oriented_delta_mean"] > 0
    ]
    result = {
        **_base_result(args, prepared),
        "phase": "classic_model_development_generalization",
        "status": "generalized" if len(positive) >= 2 else "not_generalized",
        "development_artifact": str(dev_path),
        "development_artifact_sha256": _file_sha256(dev_path),
        "selected_ratio": ratio,
        "seeds": list(DEVELOPMENT_SEEDS),
        "models": list(CLASSIC_MODELS),
        "per_seed_model": rows,
        "model_summaries": summaries,
        "positive_mean_delta_models": positive,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return result


def run_classic_confirmation(args: argparse.Namespace, prepared: dict) -> dict:
    dev_path, development, ratio = _load_development(args, prepared["task"])
    agent_path = args.agent_confirmation_artifact.resolve()
    agent = json.loads(agent_path.read_text())
    if not (
        agent.get("protocol") == PROTOCOL
        and agent.get("task") == prepared["task"]
        and agent.get("phase") == "agent_independent_confirmation"
        and agent.get("agent_summary", {}).get("promotion_passed") is True
        and float(agent.get("selected_ratio")) == ratio
    ):
        raise RuntimeError("Classic confirmation requires a promoted frozen agent arm")
    rows = _run_rows(
        prepared,
        data_dir=args.data_dir,
        seeds=CONFIRMATION_SEEDS,
        ratios=(ratio,),
        models=CLASSIC_MODELS,
        n_jobs=args.n_jobs,
    )
    summaries = {model: _paired_summary(rows, model) for model in CLASSIC_MODELS}
    promoted = [model for model, value in summaries.items() if value["promotion_passed"]]
    result = {
        **_base_result(args, prepared),
        "phase": "classic_model_generalization",
        "status": "generalized" if len(promoted) >= 2 else "not_generalized",
        "development_artifact": str(dev_path),
        "development_artifact_sha256": _file_sha256(dev_path),
        "agent_confirmation_artifact": str(agent_path),
        "agent_confirmation_artifact_sha256": _file_sha256(agent_path),
        "selected_ratio": ratio,
        "seeds": list(CONFIRMATION_SEEDS),
        "models": list(CLASSIC_MODELS),
        "per_seed_model": rows,
        "model_summaries": summaries,
        "promoted_models": promoted,
        "robust_across_classic_models": len(promoted) >= 2,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=tuple(TASKS), required=True)
    parser.add_argument(
        "--phase",
        choices=(
            "development",
            "agent-confirmation",
            "classic-development",
            "classic-confirmation",
        ),
        required=True,
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--development-artifact", type=Path)
    parser.add_argument("--agent-confirmation-artifact", type=Path)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument(
        "--cohort-dir",
        type=Path,
        default=Path("data/external/chembl/chembl_37/tdc22_cohorts"),
    )
    parser.add_argument(
        "--admission-report",
        type=Path,
        default=Path(
            "reports/classic_ml_external_data_all22_audit_v2/admission_report_v2.json"
        ),
    )
    parser.add_argument(
        "--augmentation-mode",
        choices=("auto", "direct", "controlled_bridge"),
        default="auto",
    )
    parser.add_argument("--n-jobs", type=int, default=4)
    args = parser.parse_args()
    if args.phase != "development" and args.development_artifact is None:
        parser.error("confirmation phases require --development-artifact")
    if args.phase == "classic-confirmation" and args.agent_confirmation_artifact is None:
        parser.error("classic confirmation requires --agent-confirmation-artifact")
    return args


if __name__ == "__main__":
    parsed = parse_args()
    prepared_task = _prepare_task(parsed, parsed.task)
    if parsed.phase == "development":
        payload = run_development(parsed, prepared_task)
    elif parsed.phase == "agent-confirmation":
        payload = run_agent_confirmation(parsed, prepared_task)
    elif parsed.phase == "classic-development":
        payload = run_classic_development(parsed, prepared_task)
    else:
        payload = run_classic_confirmation(parsed, prepared_task)
    print(
        json.dumps(
            {
                "task": payload["task"],
                "phase": payload["phase"],
                "status": payload["status"],
                "selection": payload.get("selection"),
            },
            indent=2,
        )
    )
