#!/usr/bin/env python3
"""Develop and confirm Evidence Packet weight/quantity configurations."""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from preflight_evidence_packet_v4 import preflight

from admet_agent.evidence_packet_loop import CANONICAL_TASKS
from admet_agent.evidence_packet_multiseed import (
    DEVELOPMENT_SEEDS,
    QUANTITY_FRACTIONS,
    WEIGHT_GRID,
    effective_external_sample_weights,
    pooled_normalized_improvement,
    quantity_rows,
    reconcile_external_records,
    select_external_positions,
    summarize_grid,
)
from admet_agent.evidence_packet_multiseed import (
    REPAIR_CONFIRMATION_SEEDS as CONFIRMATION_SEEDS,
)
from admet_agent.evidence_packet_splits import scaffold_train_validation_split
from admet_agent.evidence_packets import (
    prepare_evidence_packet,
    sha256_file,
    validate_evidence_packet,
)
from admet_agent.external_structure_firewall import load_label_blind_test_structures
from admet_agent.features import FeatureCache
from admet_agent.schemas import ExperimentSpec
from admet_agent.selection import build_model, predict, score
from admet_agent.structure_firewall_cache import (
    prepare_structure_firewall_reference_cached as prepare_structure_firewall_reference,
)
from admet_agent.tasks import resolve_task

PROTOCOL = "admet-evidence-packet-multiseed-weight-quantity-v5.3"
FEATURES = ("morgan",)
MODEL = "lightgbm"


def _fit_score(
    *,
    task_name: str,
    seed: int,
    x_fit: np.ndarray,
    y_fit: np.ndarray,
    x_validation: np.ndarray,
    y_validation: np.ndarray,
    sample_weight: np.ndarray,
    n_jobs: int,
) -> float:
    task = resolve_task(task_name)
    model_y = y_fit
    target_mean = 0.0
    target_scale = 1.0
    if task.task_type == "regression":
        target_mean = float(np.mean(y_fit))
        target_scale = float(np.std(y_fit))
        if not math.isfinite(target_scale) or target_scale <= 0:
            target_scale = 1.0
        model_y = (y_fit - target_mean) / target_scale
    experiment = ExperimentSpec.from_dict(
        {
            "task": task_name,
            "seed": seed,
            "representations": list(FEATURES),
            "model": MODEL,
            "hyperparameters": {},
            "imbalance": "none",
            "data_policy": "official",
            "device": "cpu",
            "n_jobs": n_jobs,
            "hypothesis": "Evidence Packet multi-seed weight/quantity comparison",
        }
    )
    model = build_model(experiment, task, np.asarray(model_y))
    if np.allclose(sample_weight, 1.0):
        fitted = model.fit(x_fit, np.asarray(model_y))
    else:
        try:
            fitted = model.fit(
                x_fit, np.asarray(model_y), sample_weight=sample_weight
            )
        except TypeError as error:
            raise TypeError("frozen model does not consume sample_weight") from error
    predictions = np.asarray(predict(fitted, x_validation, task), dtype=float)
    if task.task_type == "regression":
        predictions = predictions * target_scale + target_mean
    return float(score(y_validation, predictions, task.metric))


def _normalized_improvement(metric: str, baseline: float, challenger: float) -> float:
    denominator = max(abs(baseline), 1e-12)
    if metric == "mae":
        return float((baseline - challenger) / denominator)
    return float((challenger - baseline) / denominator)


def _packet_manifests(
    registry: Path, task_name: str, packet_ids: list[str] | None = None
) -> tuple[dict, list[Path], list[str]]:
    report = preflight(registry)
    if report["status"] == "failed_closed":
        raise RuntimeError("Evidence Packet registry failed closed")
    available = {
        item["packet_id"]: Path(item["manifest_path"])
        for item in report["packets"]
        if item["task"] == task_name
    }
    selected_ids = sorted(set(packet_ids or available))
    missing = sorted(set(selected_ids) - set(available))
    if missing:
        raise RuntimeError(f"requested packets are not active for {task_name}: {missing}")
    manifests = [available[packet_id] for packet_id in selected_ids]
    if not manifests:
        raise RuntimeError(f"no admitted Evidence Packet is registered for {task_name}")
    return report, manifests, selected_ids


def _prepare_external_once(
    *, task_name: str, manifests: list[Path], data_dir: Path
) -> tuple[pd.DataFrame, dict]:
    """Decontaminate against the full development pool once for every seed."""

    task = resolve_task(task_name)
    task_dir = data_dir / "admet_group" / task_name.lower()
    source = task_dir / "train_val.csv"
    development = pd.read_csv(source, usecols=["Drug", "Y"])
    train_smiles = development["Drug"].astype(str).tolist()
    test_path = task_dir / "test.csv"
    test_smiles = load_label_blind_test_structures(test_path)
    firewall_reference = prepare_structure_firewall_reference(
        train_smiles=train_smiles,
        validation_smiles=[],
        test_smiles=test_smiles,
    )
    prepared = []
    for manifest in manifests:
        packet = validate_evidence_packet(manifest)
        prepared.append(
            prepare_evidence_packet(
                packet,
                train_smiles=train_smiles,
                validation_smiles=[],
                test_path=test_path,
                firewall_reference=firewall_reference,
            )
        )
    admitted = [item for item in prepared if item.admitted]
    if not admitted:
        raise RuntimeError(f"all Evidence Packets were rejected for {task_name}")
    external_raw = pd.concat(
        [item.records for item in admitted], ignore_index=True, sort=False
    )
    external, reconciliation = reconcile_external_records(
        external_raw, task_type=task.task_type
    )
    if external.empty:
        raise RuntimeError("cross-paper reconciliation removed every external row")
    return external, {
        "scope": "full train_val development union before all seed splits",
        "train_val_path": str(source.resolve()),
        "train_val_sha256": sha256_file(source),
        "packet_audits": [item.audit for item in prepared],
        "cross_packet_reconciliation": reconciliation,
        "reconciled_base_weight_summary": {
            "minimum": float(external["evidence_base_weight"].min()),
            "maximum": float(external["evidence_base_weight"].max()),
            "mean": float(external["evidence_base_weight"].mean()),
        },
        "test_labels_loaded": False,
    }


def _evaluate_seed(
    *,
    task_name: str,
    seed: int,
    external: pd.DataFrame,
    data_dir: Path,
    cache_root: Path,
    weights: tuple[float, ...],
    fractions: tuple[float, ...],
    n_jobs: int,
) -> tuple[list[dict], dict]:
    task = resolve_task(task_name)
    task_dir = data_dir / "admet_group" / task_name.lower()
    source = task_dir / "train_val.csv"
    raw = pd.read_csv(source, usecols=["Drug", "Y"])
    train, validation, split_audit = scaffold_train_validation_split(raw, seed=seed)
    all_smiles = (
        train["Drug"].astype(str).tolist()
        + external["Drug"].astype(str).tolist()
        + validation["Drug"].astype(str).tolist()
    )
    features = FeatureCache(
        cache_root / task_name / f"seed{seed}", device="cpu"
    ).get_or_compute(all_smiles, FEATURES)
    native_end = len(train)
    external_end = native_end + len(external)
    x_native = features[:native_end]
    x_external = features[native_end:external_end]
    x_validation = features[external_end:]
    y_native = train["Y"].to_numpy(
        dtype=int if task.task_type == "classification" else float
    )
    y_external = external["Y"].to_numpy(
        dtype=int if task.task_type == "classification" else float
    )
    y_validation = validation["Y"].to_numpy(
        dtype=int if task.task_type == "classification" else float
    )
    baseline = _fit_score(
        task_name=task_name,
        seed=seed,
        x_fit=x_native,
        y_fit=y_native,
        x_validation=x_validation,
        y_validation=y_validation,
        sample_weight=np.ones(len(y_native), dtype=float),
        n_jobs=n_jobs,
    )
    subset_by_fraction = {}
    subset_audits = {}
    for fraction in fractions:
        requested = quantity_rows(len(external), fraction)
        positions, audit = select_external_positions(
            external,
            native_targets=y_native,
            task_type=task.task_type,
            total_rows=requested,
        )
        subset_by_fraction[fraction] = positions
        subset_audits[str(fraction)] = audit

    rows = []
    for fraction in fractions:
        positions = subset_by_fraction[fraction]
        for weight in weights:
            x_fit = np.vstack([x_native, x_external[positions]])
            y_fit = np.concatenate([y_native, y_external[positions]])
            selected_external_weights = effective_external_sample_weights(
                external, positions, weight
            )
            sample_weight = np.concatenate(
                [
                    np.ones(len(y_native), dtype=float),
                    selected_external_weights,
                ]
            )
            started = time.perf_counter()
            challenger = _fit_score(
                task_name=task_name,
                seed=seed,
                x_fit=x_fit,
                y_fit=y_fit,
                x_validation=x_validation,
                y_validation=y_validation,
                sample_weight=sample_weight,
                n_jobs=n_jobs,
            )
            normalized = _normalized_improvement(task.metric, baseline, challenger)
            rows.append(
                {
                    "task": task_name,
                    "seed": seed,
                    "metric": task.metric,
                    "weight": float(weight),
                    "quantity_fraction": float(fraction),
                    "native_train_rows": len(train),
                    "validation_rows": len(validation),
                    "external_available_rows": len(external),
                    "external_selected_rows": len(positions),
                    "external_base_weight_min": float(
                        external.iloc[positions]["evidence_base_weight"].min()
                    ),
                    "external_base_weight_max": float(
                        external.iloc[positions]["evidence_base_weight"].max()
                    ),
                    "external_weight_multiplier": float(weight),
                    "external_total_weight": float(selected_external_weights.sum()),
                    "external_to_native_mass_ratio": float(
                        selected_external_weights.sum() / len(y_native)
                    ),
                    "baseline_score": baseline,
                    "challenger_score": challenger,
                    "normalized_improvement": normalized,
                    "all22_mean_increment_if_only_task_changed": normalized / 22.0,
                    "fit_wall_seconds": time.perf_counter() - started,
                    "test_labels_loaded": False,
                }
            )
    audit = {
        "seed": seed,
        "split": split_audit,
        "train_val_path": str(source.resolve()),
        "train_val_sha256": sha256_file(source),
        "quantity_selection": subset_audits,
        "test_labels_loaded": False,
    }
    return rows, audit


def _write_output(path: Path, payload: dict) -> None:
    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(path)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def run_development(args: argparse.Namespace) -> dict:
    _report, manifests, active_packet_ids = _packet_manifests(
        args.registry.resolve(), args.task, args.packet_ids
    )
    external, external_audit = _prepare_external_once(
        task_name=args.task,
        manifests=manifests,
        data_dir=args.data_dir.resolve(),
    )
    rows = []
    audits = []
    started = time.perf_counter()
    for seed in DEVELOPMENT_SEEDS:
        seed_rows, audit = _evaluate_seed(
            task_name=args.task,
            seed=seed,
            external=external,
            data_dir=args.data_dir.resolve(),
            cache_root=args.feature_cache_root.resolve(),
            weights=WEIGHT_GRID,
            fractions=QUANTITY_FRACTIONS,
            n_jobs=args.n_jobs,
        )
        rows.extend(seed_rows)
        audits.append(audit)
        print(json.dumps({"completed_seed": seed, "rows": len(seed_rows)}), flush=True)
    summaries, selection = summarize_grid(rows)
    payload = {
        "protocol": PROTOCOL,
        "phase": "development",
        "task": args.task,
        "seeds": list(DEVELOPMENT_SEEDS),
        "weight_grid": list(WEIGHT_GRID),
        "weight_semantics": (
            "effective row weight = packet native_priority_external_weight "
            "* searched global multiplier; native row weight = 1.0"
        ),
        "quantity_fractions": list(QUANTITY_FRACTIONS),
        "registry_path": str(args.registry.resolve()),
        "registry_sha256": sha256_file(args.registry.resolve()),
        "packet_ids": active_packet_ids,
        "rows": rows,
        "seed_audits": audits,
        "global_external_audit": external_audit,
        "configuration_summaries": summaries,
        "selection": selection,
        "safety": {
            "validation_used_for_configuration_scoring": True,
            "validation_used_for_row_selection": False,
            "test_structures_used_only_for_decontamination": True,
            "test_labels_loaded": False,
            "legacy_bulk_pool_used": False,
        },
        "claim_boundary": "development-only configuration selection",
        "wall_seconds": time.perf_counter() - started,
    }
    _write_output(args.output, payload)
    return payload


def run_confirmation(args: argparse.Namespace) -> dict:
    if args.development_artifact is None:
        raise ValueError("confirmation requires --development-artifact")
    development_path = args.development_artifact.resolve()
    development = json.loads(development_path.read_text())
    selected = development.get("selection", {}).get("selected")
    if not (
        development.get("protocol") == PROTOCOL
        and development.get("phase") == "development"
        and development.get("task") == args.task
        and development.get("seeds") == list(DEVELOPMENT_SEEDS)
        and development.get("safety", {}).get("test_labels_loaded") is False
        and selected
    ):
        raise RuntimeError("invalid or unselected development artifact")
    _report, manifests, active_packet_ids = _packet_manifests(
        args.registry.resolve(), args.task, args.packet_ids
    )
    if development["registry_sha256"] != sha256_file(args.registry.resolve()):
        raise RuntimeError("registry changed after development selection")
    if development.get("packet_ids") != active_packet_ids:
        raise RuntimeError("packet subset changed after development selection")
    rows = []
    audits = []
    started = time.perf_counter()
    external, external_audit = _prepare_external_once(
        task_name=args.task,
        manifests=manifests,
        data_dir=args.data_dir.resolve(),
    )
    for seed in CONFIRMATION_SEEDS:
        seed_rows, audit = _evaluate_seed(
            task_name=args.task,
            seed=seed,
            external=external,
            data_dir=args.data_dir.resolve(),
            cache_root=args.feature_cache_root.resolve(),
            weights=(float(selected["weight"]),),
            fractions=(float(selected["quantity_fraction"]),),
            n_jobs=args.n_jobs,
        )
        rows.extend(seed_rows)
        audits.append(audit)
        print(json.dumps({"completed_confirmation_seed": seed}), flush=True)
    pooled_mean, values = pooled_normalized_improvement(
        resolve_task(args.task).metric,
        [row["baseline_score"] for row in rows],
        [row["challenger_score"] for row in rows],
    )
    summary = {
        "mean_normalized_improvement": pooled_mean,
        "normalized_improvement_aggregation": "ratio_of_mean_scores",
        "std_normalized_improvement": float(values.std(ddof=0)),
        "positive_seed_fraction": float((values > 0).mean()),
        "all22_mean_increment_if_only_task_changed": float(pooled_mean / 22.0),
        "confirmation_passed": bool(pooled_mean > 0 and (values > 0).mean() >= 2 / 3),
    }
    payload = {
        "protocol": PROTOCOL,
        "phase": "confirmation",
        "task": args.task,
        "seeds": list(CONFIRMATION_SEEDS),
        "selected_configuration": selected,
        "development_artifact": str(development_path),
        "development_artifact_sha256": sha256_file(development_path),
        "registry_path": str(args.registry.resolve()),
        "registry_sha256": sha256_file(args.registry.resolve()),
        "packet_ids": active_packet_ids,
        "rows": rows,
        "seed_audits": audits,
        "global_external_audit": external_audit,
        "confirmation_summary": summary,
        "safety": {
            "confirmation_seeds_used_during_development": False,
            "test_structures_used_only_for_decontamination": True,
            "test_labels_loaded": False,
            "legacy_bulk_pool_used": False,
        },
        "wall_seconds": time.perf_counter() - started,
    }
    _write_output(args.output, payload)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=("develop", "confirm"), required=True)
    parser.add_argument("--task", choices=CANONICAL_TASKS, required=True)
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--packet-ids", nargs="+")
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--development-artifact", type=Path)
    parser.add_argument("--feature-cache-root", type=Path, default=Path("checkpoints/evidence_packet_v5/features"))
    parser.add_argument("--n-jobs", type=int, default=8)
    args = parser.parse_args()
    resolve_task(args.task)
    result = run_development(args) if args.phase == "develop" else run_confirmation(args)
    print(json.dumps({"phase": result["phase"], "output": str(args.output), "task": args.task}, sort_keys=True))


if __name__ == "__main__":
    main()
