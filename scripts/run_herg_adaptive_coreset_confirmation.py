"""Develop and independently confirm adaptive hERG external-data integration.

The protocol chooses the smallest relevant, diverse external coreset within one
standard error of the best development result. External rows have at most the
same total sample weight as TDC training rows. Development uses seeds 2-7;
confirmation uses untouched seeds 8-11. Only the TDC test Drug column is
loaded for the identity firewall; test labels are never loaded.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs, RDLogger
from run_classic_ml_external_data_confirmation import (
    _classification_model,
    _standardized_tdc,
)
from run_scaling_parallel_adapters import _curate_binary
from scipy import sparse
from sklearn.metrics import roc_auc_score
from tdc.benchmark_group import admet_group

from admet_agent.adaptive_external_data import (
    external_mass_capped_weights,
    molecular_scaffold,
    select_relevance_diverse_binary_coreset,
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

PROTOCOL = "herg-adaptive-quality-quantity-coreset-v2"
SOURCE = Path("external/ecvoi_v6_tdc_cache/herg_central.tab")
EXPECTED_SOURCE_SHA256 = (
    "5775a6900c3be6a2db65d357411fd3f768189de611dc2da3456dab4acbcb3ee8"
)
RATIOS = (0.5, 1.0, 2.0, 4.0, 8.0)
DEVELOPMENT_SEEDS = (2, 3, 4, 5, 6, 7)
CONFIRMATION_SEEDS = (8, 9, 10, 11)
MODELS = ("random_forest", "linear_svm", "morgan_mlp", "lightgbm")
MINIMUM_RELEVANCE = 0.20
NEAR_THRESHOLD = 0.80
PRACTICAL_DELTA = 0.005


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _prepare_external(data_dir: Path) -> tuple[pd.DataFrame, list, dict]:
    source = SOURCE.resolve()
    if _sha256(source) != EXPECTED_SOURCE_SHA256:
        raise RuntimeError("Frozen hERGCentral source SHA-256 mismatch")
    raw = pd.read_csv(source, sep="\t")
    external, audit = _curate_binary(
        raw,
        smiles_column="X",
        label_column="hERG_inhib",
        id_column="ID",
        majority_cap=15_000,
    )
    discovery = pd.read_csv(
        data_dir / "admet_group/herg/train_val.csv", usecols=["Drug", "Y"]
    )
    discovery_standard, fallback_count = _standardized_tdc(discovery)
    discovery_ids = set(discovery_standard["standard_inchikey"])
    test_ids, test_firewall_audit = load_structure_only_inchikey_blocklist(
        data_dir / "admet_group/herg/test.csv"
    )
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
    external["scaffold"] = external["standard_smiles"].map(molecular_scaffold)
    fingerprints = morgan_fingerprints(external["standard_smiles"].tolist())
    return external, fingerprints, {
        **audit.__dict__,
        "raw_rows": len(raw),
        "exact_tdc_discovery_overlap_exclusions": discovery_exact_exclusions,
        "exact_tdc_test_structure_overlap_exclusions": test_exact_exclusions,
        **test_firewall_audit,
        "fingerprint_parse_exclusions": parse_exclusions,
        "post_exact_firewall_identities": len(external),
        "tdc_standardization_fallback_count": fallback_count,
    }


def _train_relevance(candidate_fps: list, train_fps: list) -> np.ndarray:
    return np.asarray(
        [
            max(DataStructs.BulkTanimotoSimilarity(fingerprint, train_fps))
            for fingerprint in candidate_fps
        ],
        dtype=float,
    )


def _fit_score(
    model_name: str,
    seed: int,
    n_jobs: int,
    train_x,
    train_y: np.ndarray,
    valid_x,
    valid_y: np.ndarray,
    *,
    sample_weight: np.ndarray | None = None,
) -> float:
    model = _classification_model(model_name, seed, n_jobs)
    fitted = model.fit(train_x, train_y, sample_weight=sample_weight)
    if hasattr(fitted, "predict_proba"):
        prediction = fitted.predict_proba(valid_x)[:, 1]
    else:
        prediction = fitted.decision_function(valid_x)
    return float(roc_auc_score(valid_y, prediction))


def _summarize(rows: list[dict], ratios: tuple[float, ...]) -> dict:
    by_ratio: dict[str, dict] = {}
    for ratio in ratios:
        selected = [row for row in rows if row["ratio"] == ratio]
        deltas = np.asarray([row["oriented_delta"] for row in selected], dtype=float)
        model_summaries = {}
        promoted_models = []
        for model in MODELS:
            model_delta = np.asarray(
                [row["oriented_delta"] for row in selected if row["model"] == model],
                dtype=float,
            )
            positive_fraction = float((model_delta > 0).mean())
            passed = bool(
                model_delta.mean() >= PRACTICAL_DELTA
                and positive_fraction >= 2 / 3
            )
            if passed:
                promoted_models.append(model)
            model_summaries[model] = {
                "oriented_delta_mean": float(model_delta.mean()),
                "oriented_delta_std": float(model_delta.std()),
                "positive_seed_fraction": positive_fraction,
                "development_passed": passed,
            }
        by_ratio[str(ratio)] = {
            "oriented_delta_mean": float(deltas.mean()),
            "oriented_delta_std": float(deltas.std()),
            "standard_error": float(deltas.std() / math.sqrt(len(deltas))),
            "positive_seed_model_fraction": float((deltas > 0).mean()),
            "promoted_models": promoted_models,
            "robust_across_models": len(promoted_models) >= 2,
            "model_summaries": model_summaries,
        }
    return by_ratio


def _select_ratio(summaries: dict[str, dict]) -> dict:
    robust = [
        (float(ratio), summary)
        for ratio, summary in summaries.items()
        if summary["robust_across_models"]
        and summary["oriented_delta_mean"] >= PRACTICAL_DELTA
        and summary["positive_seed_model_fraction"] >= 0.60
    ]
    if not robust:
        return {
            "selected_ratio": None,
            "status": "no_robust_development_ratio",
            "rule": "smallest robust ratio within one best-ratio standard error",
        }
    best_ratio, best = max(robust, key=lambda item: item[1]["oriented_delta_mean"])
    floor = best["oriented_delta_mean"] - best["standard_error"]
    eligible = sorted(
        ratio for ratio, summary in robust if summary["oriented_delta_mean"] >= floor
    )
    return {
        "selected_ratio": eligible[0],
        "status": "selected",
        "best_development_ratio": best_ratio,
        "best_development_mean": best["oriented_delta_mean"],
        "best_development_standard_error": best["standard_error"],
        "one_standard_error_floor": floor,
        "eligible_ratios": eligible,
        "rule": "smallest robust ratio within one best-ratio standard error",
    }


def _run_rows(
    *,
    data_dir: Path,
    external: pd.DataFrame,
    external_fps: list,
    seeds: tuple[int, ...],
    ratios: tuple[float, ...],
    weight_policy: dict,
    n_jobs: int,
) -> list[dict]:
    external_x = fingerprints_to_csr(external_fps)
    external_y = external["Y"].to_numpy(dtype=int)
    group = admet_group(path=str(data_dir))
    rows: list[dict] = []
    for seed in seeds:
        train_raw, valid_raw = group.get_train_valid_split(
            benchmark="herg", split_type="default", seed=seed
        )
        train, _ = _standardized_tdc(train_raw)
        valid, _ = _standardized_tdc(valid_raw)
        train_fps = morgan_fingerprints(train["standard_smiles"].tolist())
        valid_fps = morgan_fingerprints(valid["standard_smiles"].tolist())
        train_x = fingerprints_to_csr(train_fps)
        valid_x = fingerprints_to_csr(valid_fps)
        train_y = train["Y"].to_numpy(dtype=int)
        valid_y = valid["Y"].to_numpy(dtype=int)
        firewall_keep = near_neighbor_keep_mask(
            external_fps, valid_fps, threshold=NEAR_THRESHOLD
        )
        relevance = _train_relevance(external_fps, train_fps)
        relevance[~firewall_keep] = -1.0
        baselines = {
            model: _fit_score(
                model,
                seed,
                n_jobs,
                train_x,
                train_y,
                valid_x,
                valid_y,
            )
            for model in MODELS
        }
        for ratio in ratios:
            selected, coreset_audit = select_relevance_diverse_binary_coreset(
                external,
                relevance,
                total_rows=max(1, int(round(ratio * len(train)))),
                target_positive_rate=float(train_y.mean()),
                minimum_relevance=MINIMUM_RELEVANCE,
            )
            augmented_x = sparse.vstack(
                [train_x, external_x[selected]], format="csr"
            )
            augmented_y = np.concatenate([train_y, external_y[selected]])
            weights = external_mass_capped_weights(
                len(train),
                len(selected),
                external_mass_ratio=weight_policy["external_mass_ratio"],
                external_row_weight_cap=weight_policy["external_row_weight_cap"],
                external_to_min_native_weight_ratio_cap=weight_policy[
                    "external_to_min_native_weight_ratio_cap"
                ],
            )
            for model in MODELS:
                fit_started = time.perf_counter()
                augmented = _fit_score(
                    model,
                    seed,
                    n_jobs,
                    augmented_x,
                    augmented_y,
                    valid_x,
                    valid_y,
                    sample_weight=weights,
                )
                row = {
                    "seed": seed,
                    "ratio": ratio,
                    "model": model,
                    "tdc_train_rows": len(train),
                    "tdc_validation_rows": len(valid),
                    "near_neighbor_exclusions": int((~firewall_keep).sum()),
                    **{f"coreset_{key}": value for key, value in coreset_audit.items()},
                    "external_total_sample_weight": float(weights[len(train) :].sum()),
                    "tdc_total_sample_weight": float(weights[: len(train)].sum()),
                    "baseline_score": baselines[model],
                    "augmented_score": augmented,
                    "oriented_delta": augmented - baselines[model],
                    "fit_wall_seconds": time.perf_counter() - fit_started,
                }
                rows.append(row)
                print(json.dumps(row), flush=True)
    return rows


def run_development(args: argparse.Namespace) -> dict:
    admission = require_bound_task_admission(
        args.admission_report,
        "hERG",
        source_path=SOURCE,
        source_name="hERGCentral_hERG_inhib",
        augmentation_mode="controlled_bridge",
    )
    started = time.perf_counter()
    external, external_fps, curation = _prepare_external(args.data_dir)
    rows = _run_rows(
        data_dir=args.data_dir,
        external=external,
        external_fps=external_fps,
        seeds=DEVELOPMENT_SEEDS,
        ratios=RATIOS,
        weight_policy=admission["weight_policy"],
        n_jobs=args.n_jobs,
    )
    summaries = _summarize(rows, RATIOS)
    selection = _select_ratio(summaries)
    result = {
        "protocol": PROTOCOL,
        "phase": "development_budget_selection",
        "task": "hERG",
        "source_sha256": EXPECTED_SOURCE_SHA256,
        "curation": curation,
        "seeds": list(DEVELOPMENT_SEEDS),
        "ratios": list(RATIOS),
        "models": list(MODELS),
        "minimum_train_relevance": MINIMUM_RELEVANCE,
        "near_neighbor_threshold": NEAR_THRESHOLD,
        "admission": admission,
        "weight_policy": admission["weight_policy"],
        "per_seed_ratio_model": rows,
        "ratio_summaries": summaries,
        "selection": selection,
        "safety": {
            "tdc_test_loaded": True,
            "tdc_test_structures_loaded_for_firewall": True,
            "tdc_test_columns_loaded": ["Drug"],
            "tdc_test_labels_loaded": False,
            "test_labels_accessible": False,
        },
        "wall_seconds": time.perf_counter() - started,
        "claim_boundary": "Development-only budget selection; no paper gain claim.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return result


def _confirmation_summary(rows: list[dict]) -> dict:
    summaries = {}
    promoted = []
    for model in MODELS:
        selected = [row for row in rows if row["model"] == model]
        baseline = np.asarray([row["baseline_score"] for row in selected])
        augmented = np.asarray([row["augmented_score"] for row in selected])
        delta = augmented - baseline
        positive_fraction = float((delta > 0).mean())
        passed = bool(
            delta.mean() >= PRACTICAL_DELTA and positive_fraction >= 0.75
        )
        if passed:
            promoted.append(model)
        summaries[model] = {
            "baseline_mean": float(baseline.mean()),
            "baseline_std": float(baseline.std()),
            "augmented_mean": float(augmented.mean()),
            "augmented_std": float(augmented.std()),
            "oriented_delta_mean": float(delta.mean()),
            "oriented_delta_std": float(delta.std()),
            "positive_seed_fraction": positive_fraction,
            "promotion_passed": passed,
        }
    return {
        "model_summaries": summaries,
        "promoted_models": promoted,
        "promoted_model_count": len(promoted),
        "robust_across_models": len(promoted) >= 2,
    }


def run_confirmation(args: argparse.Namespace) -> dict:
    admission = require_bound_task_admission(
        args.admission_report,
        "hERG",
        source_path=SOURCE,
        source_name="hERGCentral_hERG_inhib",
        augmentation_mode="controlled_bridge",
    )
    development_path = args.development_artifact.resolve()
    development = json.loads(development_path.read_text())
    if not (
        development.get("protocol") == PROTOCOL
        and development.get("phase") == "development_budget_selection"
        and development.get("seeds") == list(DEVELOPMENT_SEEDS)
        and development.get("safety", {}).get("tdc_test_loaded") is True
        and development.get("safety", {}).get("tdc_test_structures_loaded_for_firewall") is True
        and development.get("safety", {}).get("tdc_test_columns_loaded") == ["Drug"]
        and development.get("safety", {}).get("tdc_test_labels_loaded") is False
        and development.get("safety", {}).get("test_labels_accessible") is False
    ):
        raise RuntimeError("Invalid frozen development artifact")
    selected_ratio = development["selection"].get("selected_ratio")
    if selected_ratio is None:
        raise RuntimeError("Development found no robust ratio; confirmation is blocked")
    started = time.perf_counter()
    external, external_fps, curation = _prepare_external(args.data_dir)
    rows = _run_rows(
        data_dir=args.data_dir,
        external=external,
        external_fps=external_fps,
        seeds=CONFIRMATION_SEEDS,
        ratios=(float(selected_ratio),),
        weight_policy=admission["weight_policy"],
        n_jobs=args.n_jobs,
    )
    confirmation = _confirmation_summary(rows)
    result = {
        "protocol": PROTOCOL,
        "phase": "independent_confirmation",
        "task": "hERG",
        "source_sha256": EXPECTED_SOURCE_SHA256,
        "admission": admission,
        "weight_policy": admission["weight_policy"],
        "development_artifact": str(development_path),
        "development_artifact_sha256": _sha256(development_path),
        "selected_ratio": selected_ratio,
        "selection": development["selection"],
        "curation": curation,
        "seeds": list(CONFIRMATION_SEEDS),
        "models": list(MODELS),
        "per_seed_model": rows,
        **confirmation,
        "safety": {
            "tdc_test_loaded": True,
            "tdc_test_structures_loaded_for_firewall": True,
            "tdc_test_columns_loaded": ["Drug"],
            "tdc_test_labels_loaded": False,
            "test_labels_accessible": False,
            "confirmation_seeds_disjoint_from_development": True,
        },
        "wall_seconds": time.perf_counter() - started,
        "claim_boundary": (
            "Independent train/validation seed confirmation of a controlled bridge "
            "source. No official test or independent-source claim is made."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("development", "confirmation"), required=True)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--development-artifact", type=Path)
    parser.add_argument("--n-jobs", type=int, default=8)
    parser.add_argument(
        "--admission-report",
        type=Path,
        default=Path(
            "reports/classic_ml_external_data_all22_audit_v2/admission_report_v2.json"
        ),
    )
    args = parser.parse_args()
    if args.phase == "confirmation" and args.development_artifact is None:
        parser.error("--development-artifact is required for confirmation")
    return args


if __name__ == "__main__":
    parsed = parse_args()
    payload = (
        run_development(parsed)
        if parsed.phase == "development"
        else run_confirmation(parsed)
    )
    print(
        json.dumps(
            payload.get("selection", payload.get("model_summaries", {})), indent=2
        )
    )
