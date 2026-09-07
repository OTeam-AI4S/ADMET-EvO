"""Matched external-data ablation with each task's frozen v2.7 recipe.

This is the protocol-correct successor to ``run_all22_adaptive_agent_data.py``.
The v1 screen intentionally used one Morgan/LightGBM controller on every task;
this runner instead loads the immutable 22-task finalist mapping and keeps that
task-specific recipe identical between the native-only and augmented arms.

Development seeds select an external-data ratio.  Disjoint confirmation seeds
evaluate the selected ratio.  Only TDC test structures are loaded for the
identity firewall; test labels are never loaded.
"""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import sys
import time
import traceback
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from scipy.stats import rankdata
from sklearn.pipeline import Pipeline
from tdc.benchmark_group import admet_group

from admet_agent.adaptive_external_data import external_mass_capped_weights
from admet_agent.features import FeatureCache
from admet_agent.frozen_executor import FrozenRecipeExecutor
from admet_agent.frozen_recipe import parse_recipe
from admet_agent.schemas import ExperimentSpec
from admet_agent.selection import build_model, predict, score
from admet_agent.tasks import resolve_task
from admet_agent.herg_data_scaling import near_neighbor_keep_mask

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from run_all22_adaptive_agent_data import (  # noqa: E402
    CONFIRMATION_SEEDS,
    DEVELOPMENT_SEEDS,
    RATIOS,
    TASKS,
    _coreset,
    _prepare_task,
    _ratio_selection,
    _relevance,
)
from run_classic_ml_external_data_confirmation import (  # noqa: E402
    _oriented_delta,
    _standardized_tdc,
)

RDLogger.DisableLog("rdApp.*")

PROTOCOL = "all22-optimal-frozen-recipe-adaptive-external-data-v3"
DEFAULT_MAPPING = Path("runs/v27-maplight-gnn-freeze/full-22-task.mapping.json")
DEFAULT_OUTPUT = Path("reports/all22_optimal_recipe_adaptive_data_v3")
DEFAULT_ADMISSION = Path(
    "reports/classic_ml_external_data_all22_audit_v2/admission_report_v2.json"
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_mapping(path: Path) -> tuple[dict[str, dict[str, Any]], str]:
    resolved = path.resolve()
    mapping = json.loads(resolved.read_text(encoding="utf-8"))
    if not isinstance(mapping, dict) or set(mapping) != set(TASKS):
        missing = sorted(set(TASKS) - set(mapping)) if isinstance(mapping, dict) else []
        extra = sorted(set(mapping) - set(TASKS)) if isinstance(mapping, dict) else []
        raise ValueError(
            f"Frozen mapping must contain the canonical 22 tasks; missing={missing}, extra={extra}"
        )
    return mapping, _sha256(resolved)


def _recipe_label(frozen: dict[str, Any]) -> str:
    kind = frozen.get("kind", "single")
    if kind == "recipe":
        return f"recipe({frozen['recipe']})"
    if kind == "ensemble":
        return "ensemble(" + "+".join(member["model"] for member in frozen["members"]) + ")"
    return str(frozen["model"])


def _runtime_blocker(frozen: dict[str, Any], device: str, root: Path) -> str | None:
    if frozen.get("kind") != "recipe":
        return None
    experts = set(parse_recipe(str(frozen["recipe"])).expert_names)
    if "finetuned_unimol2" in experts and device != "cuda":
        return "frozen Uni-Mol2 fine-tuning requires an allocated CUDA device"
    if "maplight_gnn" in experts:
        python = root / ".venvs/maplight-gnn/bin/python"
        if not python.exists():
            return "frozen MapLight+GNN isolated runtime is unavailable"
    return None


class _FeaturePool:
    """One task-level feature matrix per representation tuple."""

    def __init__(self, cache: FeatureCache, smiles: list[str]):
        self.cache = cache
        self.smiles = sorted(set(str(value) for value in smiles))
        self.position = {value: index for index, value in enumerate(self.smiles)}
        self.matrices: dict[tuple[str, ...], np.ndarray] = {}

    def take(self, requested: list[str], representations: tuple[str, ...]) -> np.ndarray:
        key = tuple(representations)
        if key not in self.matrices:
            self.matrices[key] = self.cache.get_or_compute(self.smiles, key)
        indices = [self.position[str(value)] for value in requested]
        return self.matrices[key][indices]


def _fit_with_sample_weight(
    model: object,
    x: np.ndarray,
    y: np.ndarray,
    sample_weight: np.ndarray | None,
) -> object:
    if sample_weight is None:
        return model.fit(x, y)
    if isinstance(model, Pipeline):
        fit_params: dict[str, np.ndarray] = {}
        for name, step in model.steps:
            try:
                accepts_weight = "sample_weight" in inspect.signature(step.fit).parameters
            except (TypeError, ValueError):
                accepts_weight = False
            if accepts_weight:
                fit_params[f"{name}__sample_weight"] = sample_weight
        if not fit_params:
            raise TypeError("Frozen pipeline does not expose sample_weight")
        return model.fit(x, y, **fit_params)
    return model.fit(x, y, sample_weight=sample_weight)


def _predict_tabular_mapping(
    *,
    task_name: str,
    frozen: dict[str, Any],
    seed: int,
    fit_frame: pd.DataFrame,
    predict_frame: pd.DataFrame,
    sample_weight: np.ndarray | None,
    pool: _FeaturePool,
    device: str,
    n_jobs: int,
) -> np.ndarray:
    task = resolve_task(task_name)
    if frozen.get("kind") == "single":
        members = [{key: value for key, value in frozen.items() if key != "kind"}]
        weights = [1.0]
        aggregation = "probability"
    elif frozen.get("kind") == "ensemble":
        members = frozen["members"]
        weights = [float(value) for value in frozen["weights"]]
        aggregation = frozen.get("aggregation", "probability")
    else:
        raise ValueError("Tabular execution received a heterogeneous recipe")
    total = float(sum(weights))
    if total <= 0:
        raise ValueError("Frozen ensemble weights must have positive mass")
    weights = [value / total for value in weights]
    fit_smiles = fit_frame["standard_smiles"].astype(str).tolist()
    predict_smiles = predict_frame["standard_smiles"].astype(str).tolist()
    y_fit = fit_frame["Y"].to_numpy()
    combined: np.ndarray | None = None
    for member, member_weight in zip(members, weights, strict=True):
        experiment = ExperimentSpec.from_dict(
            {
                **member,
                "task": task_name,
                "seed": int(seed),
                "device": device,
                "n_jobs": int(n_jobs),
            }
        )
        x_fit = pool.take(fit_smiles, experiment.representations)
        x_predict = pool.take(predict_smiles, experiment.representations)
        model = build_model(experiment, task, y_fit)
        fitted = _fit_with_sample_weight(model, x_fit, y_fit, sample_weight)
        values = predict(fitted, x_predict, task)
        if aggregation == "rank":
            values = rankdata(values, method="average") / len(values)
        weighted = member_weight * np.asarray(values, dtype=float)
        combined = weighted if combined is None else combined + weighted
    if combined is None:
        raise RuntimeError("Frozen mapping produced no predictions")
    return combined


def _mass_capped_frame(
    native: pd.DataFrame,
    external: pd.DataFrame,
    *,
    external_mass_ratio: float,
    external_row_weight_cap: float,
    seed: int,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Downsample external rows for backends without sample-weight support."""

    if not 0 < external_mass_ratio <= 1:
        raise ValueError("external_mass_ratio must lie in (0, 1]")
    if not 0 < external_row_weight_cap <= 1:
        raise ValueError("external_row_weight_cap must lie in (0, 1]")
    effective_mass = min(
        external_row_weight_cap * len(external),
        external_mass_ratio * len(native),
    )
    target = min(len(external), int(np.floor(effective_mass + 1e-12)))
    if target < 1:
        raise RuntimeError("External weight policy leaves no materializable rows")
    generator = np.random.default_rng(seed)
    positions = np.sort(generator.choice(len(external), size=target, replace=False))
    external_fit = external.iloc[positions].reset_index(drop=True)
    combined = pd.concat([native.reset_index(drop=True), external_fit], ignore_index=True)
    return combined, {
        "mode": "deterministic_external_downsample_for_unweighted_backend",
        "native_unique_rows": len(native),
        "external_candidate_rows": len(external),
        "external_materialized_rows": target,
        "native_materialized_rows": len(native),
        "external_mass_ratio_cap": external_mass_ratio,
        "external_row_weight_cap": external_row_weight_cap,
        "target_effective_external_mass": float(effective_mass),
        "no_row_upsampling": True,
    }


def _predict_frozen(
    *,
    task_name: str,
    frozen: dict[str, Any],
    seed: int,
    fit_frame: pd.DataFrame,
    predict_frame: pd.DataFrame,
    sample_weight: np.ndarray | None,
    pool: _FeaturePool,
    cache: FeatureCache,
    artifact_dir: Path,
    device: str,
    n_jobs: int,
) -> np.ndarray:
    if frozen.get("kind") != "recipe":
        return _predict_tabular_mapping(
            task_name=task_name,
            frozen=frozen,
            seed=seed,
            fit_frame=fit_frame,
            predict_frame=predict_frame,
            sample_weight=sample_weight,
            pool=pool,
            device=device,
            n_jobs=n_jobs,
        )
    if sample_weight is not None:
        raise ValueError("Heterogeneous recipes require materialized source weights")
    protocol = {
        "model_size": "84m",
        "learning_rate": 1e-4,
        "epochs": 20,
        "batch_size": 16,
        "inner_folds": 5,
        **frozen.get("protocol", {}),
    }
    executor = FrozenRecipeExecutor(
        cache=cache,
        artifact_dir=artifact_dir,
        device=device,
        n_jobs=n_jobs,
    )
    return executor.predict(
        task=resolve_task(task_name),
        recipe_value=str(frozen["recipe"]),
        seed=seed,
        fit_frame=fit_frame,
        predict_smiles=predict_frame["standard_smiles"].astype(str).tolist(),
        smiles_col="standard_smiles",
        y_col="Y",
        protocol=protocol,
    )


def _task_rows(
    *,
    prepared: dict[str, Any],
    frozen: dict[str, Any],
    seeds: tuple[int, ...],
    ratios: tuple[float, ...],
    data_dir: Path,
    run_dir: Path,
    device: str,
    n_jobs: int,
) -> list[dict[str, Any]]:
    group = admet_group(path=str(data_dir))
    task = resolve_task(prepared["task"])
    external = prepared["external"]
    external_fps = prepared["external_fps"]
    discovery = pd.read_csv(
        data_dir / "admet_group" / prepared["benchmark"] / "train_val.csv",
        usecols=["Drug", "Y"],
    )
    discovery_standard, _ = _standardized_tdc(discovery)
    cache = FeatureCache(run_dir / "feature_cache", device=device)
    pool = _FeaturePool(
        cache,
        [
            *discovery_standard["standard_smiles"].astype(str).tolist(),
            *external["standard_smiles"].astype(str).tolist(),
        ],
    )
    rows: list[dict[str, Any]] = []
    for seed in seeds:
        train_raw, valid_raw = group.get_train_valid_split(
            benchmark=prepared["benchmark"], split_type="default", seed=seed
        )
        train, _ = _standardized_tdc(train_raw)
        valid, _ = _standardized_tdc(valid_raw)
        # Use the exact v1 Morgan relevance/firewall implementation.
        from admet_agent.herg_data_scaling import morgan_fingerprints

        train_fps = morgan_fingerprints(train["standard_smiles"].tolist())
        valid_fps = morgan_fingerprints(valid["standard_smiles"].tolist())
        keep = near_neighbor_keep_mask(external_fps, valid_fps, threshold=0.80)
        relevance = _relevance(external_fps, train_fps)
        relevance[~keep] = -1.0
        baseline_started = time.perf_counter()
        baseline_prediction = _predict_frozen(
            task_name=prepared["task"],
            frozen=frozen,
            seed=seed,
            fit_frame=train,
            predict_frame=valid,
            sample_weight=None,
            pool=pool,
            cache=cache,
            artifact_dir=run_dir / prepared["task"] / f"seed_{seed}" / "baseline",
            device=device,
            n_jobs=n_jobs,
        )
        baseline_score = score(valid["Y"].to_numpy(), baseline_prediction, task.metric)
        baseline_seconds = time.perf_counter() - baseline_started
        augmented_by_selected_rows: dict[
            tuple[int, ...], tuple[float, float, float]
        ] = {}
        for ratio in ratios:
            selected, audit = _coreset(prepared, relevance, train["Y"].to_numpy(), ratio)
            selected_key = tuple(int(value) for value in selected)
            selected_sha256 = hashlib.sha256(
                np.asarray(selected, dtype=np.int64).tobytes()
            ).hexdigest()
            selected_external = external.iloc[selected][["standard_smiles", "Y"]].copy()
            weighting_audit: dict[str, Any]
            if frozen.get("kind") == "recipe":
                fit_frame, weighting_audit = _mass_capped_frame(
                    train[["standard_smiles", "Y"]],
                    selected_external,
                    external_mass_ratio=prepared["weight_policy"][
                        "external_mass_ratio"
                    ],
                    external_row_weight_cap=prepared["weight_policy"][
                        "external_row_weight_cap"
                    ],
                    seed=seed * 100_000 + len(selected_key),
                )
                sample_weight = None
            else:
                fit_frame = pd.concat(
                    [train[["standard_smiles", "Y"]], selected_external],
                    ignore_index=True,
                )
                sample_weight = external_mass_capped_weights(
                    len(train),
                    len(selected_external),
                    external_mass_ratio=prepared["weight_policy"][
                        "external_mass_ratio"
                    ],
                    external_row_weight_cap=prepared["weight_policy"][
                        "external_row_weight_cap"
                    ],
                    external_to_min_native_weight_ratio_cap=prepared[
                        "weight_policy"
                    ]["external_to_min_native_weight_ratio_cap"],
                )
                weighting_audit = {
                    "mode": "native_one_external_uniform_exact_mass_cap",
                    "native_total_weight": float(sample_weight[: len(train)].sum()),
                    "external_total_weight": float(sample_weight[len(train) :].sum()),
                    "external_per_row_weight": float(sample_weight[len(train)]),
                }
            reused_from_ratio = None
            if selected_key in augmented_by_selected_rows:
                (
                    augmented_score,
                    augmented_seconds,
                    reused_from_ratio,
                ) = augmented_by_selected_rows[selected_key]
            else:
                augmented_started = time.perf_counter()
                augmented_prediction = _predict_frozen(
                    task_name=prepared["task"],
                    frozen=frozen,
                    seed=seed,
                    fit_frame=fit_frame,
                    predict_frame=valid,
                    sample_weight=sample_weight,
                    pool=pool,
                    cache=cache,
                    artifact_dir=(
                        run_dir
                        / prepared["task"]
                        / f"seed_{seed}"
                        / f"ratio_{ratio:g}"
                    ),
                    device=device,
                    n_jobs=n_jobs,
                )
                augmented_score = score(
                    valid["Y"].to_numpy(), augmented_prediction, task.metric
                )
                augmented_seconds = time.perf_counter() - augmented_started
                augmented_by_selected_rows[selected_key] = (
                    augmented_score,
                    augmented_seconds,
                    ratio,
                )
            row = {
                "seed": seed,
                "ratio": ratio,
                "model": "task_specific_frozen_v27_recipe",
                "frozen_recipe": _recipe_label(frozen),
                "tdc_train_rows": len(train),
                "tdc_validation_rows": len(valid),
                "near_neighbor_exclusions": int((~keep).sum()),
                **{f"coreset_{key}": value for key, value in audit.items()},
                "weighting": weighting_audit,
                "selected_rows_sha256": selected_sha256,
                "reused_equivalent_fit_from_ratio": reused_from_ratio,
                "baseline_score": baseline_score,
                "augmented_score": augmented_score,
                "oriented_delta": _oriented_delta(
                    task.metric, baseline_score, augmented_score
                ),
                "baseline_fit_wall_seconds": baseline_seconds,
                "augmented_fit_wall_seconds": (
                    0.0 if reused_from_ratio is not None else augmented_seconds
                ),
            }
            rows.append(row)
            print(json.dumps({"task": prepared["task"], **row}), flush=True)
    return rows


def _paired_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    baseline = np.asarray([row["baseline_score"] for row in rows], dtype=float)
    augmented = np.asarray([row["augmented_score"] for row in rows], dtype=float)
    delta = np.asarray([row["oriented_delta"] for row in rows], dtype=float)
    positive = float((delta > 0).mean())
    return {
        "baseline_mean": float(baseline.mean()),
        "baseline_std": float(baseline.std()),
        "augmented_mean": float(augmented.mean()),
        "augmented_std": float(augmented.std()),
        "oriented_delta_mean": float(delta.mean()),
        "oriented_delta_std": float(delta.std()),
        "positive_seed_fraction": positive,
        "promotion_passed": bool(delta.mean() >= 0.005 and positive >= 0.75),
    }


def _base_payload(
    *,
    prepared: dict[str, Any],
    frozen: dict[str, Any],
    mapping_path: Path,
    mapping_sha256: str,
) -> dict[str, Any]:
    return {
        "protocol": PROTOCOL,
        "task": prepared["task"],
        "benchmark": prepared["benchmark"],
        "task_type": prepared["task_type"],
        "metric": prepared["metric"],
        "frozen_mapping": str(mapping_path.resolve()),
        "frozen_mapping_sha256": mapping_sha256,
        "frozen_recipe": frozen,
        "frozen_recipe_label": _recipe_label(frozen),
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
            "same_recipe_in_baseline_and_augmented_arms": True,
        },
        "claim_tier": "controlled_train_validation_auxiliary_data",
    }


def _run_one(args: argparse.Namespace, task_name: str, mapping: dict, mapping_sha: str) -> dict:
    task_args = SimpleNamespace(
        cohort_dir=args.cohort_dir,
        data_dir=args.data_dir,
        admission_report=args.admission_report,
        augmentation_mode=args.augmentation_mode,
        n_jobs=args.n_jobs,
    )
    prepared = _prepare_task(task_args, task_name)
    frozen = mapping[task_name]
    blocker = _runtime_blocker(frozen, args.device, Path(__file__).resolve().parents[1])
    base = _base_payload(
        prepared=prepared,
        frozen=frozen,
        mapping_path=args.mapping,
        mapping_sha256=mapping_sha,
    )
    if blocker:
        return {**base, "phase": args.phase, "status": "blocked_runtime", "blocker": blocker}
    if not prepared["eligible_for_development"]:
        return {
            **base,
            "phase": args.phase,
            "status": "auto_abstain_insufficient_curated_external_data",
            "per_seed_ratio_model": [],
            "selection": {"selected_ratio": None},
        }
    if args.phase == "development":
        rows = _task_rows(
            prepared=prepared,
            frozen=frozen,
            seeds=DEVELOPMENT_SEEDS,
            ratios=RATIOS,
            data_dir=args.data_dir,
            run_dir=args.output_dir / "artifacts",
            device=args.device,
            n_jobs=args.n_jobs,
        )
        summaries, selection = _ratio_selection(rows)
        return {
            **base,
            "phase": "optimal_recipe_development",
            "status": selection["status"],
            "seeds": list(DEVELOPMENT_SEEDS),
            "ratios": list(RATIOS),
            "per_seed_ratio_model": rows,
            "ratio_summaries": summaries,
            "selection": selection,
        }
    development_path = args.development_dir / f"{task_name}.json"
    development = json.loads(development_path.read_text(encoding="utf-8"))
    if not (
        development.get("protocol") == PROTOCOL
        and development.get("task") == task_name
        and development.get("frozen_mapping_sha256") == mapping_sha
        and development.get("safety", {}).get("tdc_test_loaded") is True
        and development.get("safety", {}).get("tdc_test_structures_loaded_for_firewall") is True
        and development.get("safety", {}).get("tdc_test_columns_loaded") == ["Drug"]
        and development.get("safety", {}).get("tdc_test_labels_loaded") is False
        and development.get("safety", {}).get("test_labels_accessible") is False
    ):
        raise RuntimeError("Invalid optimal-recipe development artifact")
    ratio = development.get("selection", {}).get("selected_ratio")
    if ratio is None:
        return {
            **base,
            "phase": "optimal_recipe_independent_confirmation",
            "status": "not_run_development_abstained",
            "development_artifact": str(development_path.resolve()),
            "selection": development.get("selection", {}),
        }
    rows = _task_rows(
        prepared=prepared,
        frozen=frozen,
        seeds=CONFIRMATION_SEEDS,
        ratios=(float(ratio),),
        data_dir=args.data_dir,
        run_dir=args.output_dir / "artifacts",
        device=args.device,
        n_jobs=args.n_jobs,
    )
    paired = _paired_summary(rows)
    return {
        **base,
        "phase": "optimal_recipe_independent_confirmation",
        "status": "promoted" if paired["promotion_passed"] else "fallback_baseline",
        "development_artifact": str(development_path.resolve()),
        "development_artifact_sha256": _sha256(development_path),
        "selected_ratio": float(ratio),
        "seeds": list(CONFIRMATION_SEEDS),
        "per_seed_model": rows,
        "agent_summary": paired,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("development", "confirmation"), required=True)
    tasks = parser.add_mutually_exclusive_group(required=True)
    tasks.add_argument("--tasks", nargs="+", choices=tuple(TASKS))
    tasks.add_argument("--all", action="store_true")
    parser.add_argument("--mapping", type=Path, default=DEFAULT_MAPPING)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT / "development")
    parser.add_argument("--development-dir", type=Path, default=DEFAULT_OUTPUT / "development")
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument(
        "--cohort-dir",
        type=Path,
        default=Path("data/external/chembl/chembl_37/tdc22_cohorts"),
    )
    parser.add_argument("--admission-report", type=Path, default=DEFAULT_ADMISSION)
    parser.add_argument(
        "--augmentation-mode",
        choices=("auto", "direct", "controlled_bridge"),
        default="auto",
    )
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--n-jobs", type=int, default=8)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    mapping, mapping_sha = _load_mapping(args.mapping)
    selected_tasks = list(TASKS) if args.all else list(args.tasks)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for task_name in selected_tasks:
        output = args.output_dir / f"{task_name}.json"
        if output.exists() and not args.overwrite:
            print(json.dumps({"task": task_name, "status": "skipped_existing"}), flush=True)
            continue
        try:
            payload = _run_one(args, task_name, mapping, mapping_sha)
            output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
            print(
                json.dumps(
                    {
                        "task": task_name,
                        "status": payload["status"],
                        "selection": payload.get("selection"),
                    }
                ),
                flush=True,
            )
        except Exception as exc:
            error = {
                "protocol": PROTOCOL,
                "task": task_name,
                "phase": args.phase,
                "status": "execution_failed",
                "error_type": type(exc).__name__,
                "error": str(exc),
                "traceback": traceback.format_exc(),
                "safety": {
                    "tdc_test_loaded": True,
                    "tdc_test_structures_loaded_for_firewall": True,
                    "tdc_test_columns_loaded": ["Drug"],
                    "tdc_test_labels_loaded": False,
                    "test_labels_accessible": False,
                },
            }
            output.with_suffix(".error.json").write_text(
                json.dumps(error, indent=2, sort_keys=True) + "\n"
            )
            print(json.dumps({"task": task_name, "status": "execution_failed", "error": str(exc)}), flush=True)


if __name__ == "__main__":
    main()
