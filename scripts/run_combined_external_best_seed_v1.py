#!/usr/bin/env python3
"""Leakage-safe literature + admitted-database data-axis evaluation.

The development split is the only place where a configuration and model seed
may be selected.  Official test labels are loaded only by ``certification``
after the content-addressed freeze has been written.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = Path(__file__).resolve().parent
for value in (ROOT, SCRIPT_DIR):
    if str(value) not in sys.path:
        sys.path.insert(0, str(value))

import run_evidence_packet_frozen_recipe_v54 as v54  # noqa: E402
from admet_agent.evidence_packet_multiseed import (  # noqa: E402
    effective_external_sample_weights,
    quantity_rows,
    select_external_positions,
)
from admet_agent.evidence_packet_v54 import cap_external_weights  # noqa: E402
from admet_agent.external_structure_firewall import (  # noqa: E402
    apply_external_structure_firewall,
    load_label_blind_test_structures,
)
from admet_agent.features import FeatureCache  # noqa: E402
from admet_agent.selection import score  # noqa: E402
from admet_agent.structure_firewall_cache import (  # noqa: E402
    prepare_structure_firewall_reference_cached,
)
from admet_agent.tasks import resolve_task  # noqa: E402
from run_all22_optimal_recipe_adaptive_data import (  # noqa: E402
    _FeaturePool,
    _load_mapping,
    _predict_frozen,
    _recipe_label,
    _runtime_blocker,
)
from run_classic_ml_external_data_confirmation import (  # noqa: E402
    TASKS,
    _standardized_tdc,
)


PROTOCOL_NAME = "admet-combined-external-best-seed-v1"
DB_PACKET_ID = "admitted-db-v3-hergcentral"
DEFAULT_ROOT = ROOT / "reports/combined_external_best_seed_v1"
DEFAULT_PROTOCOL = DEFAULT_ROOT / "protocol.json"
DEFAULT_FREEZE = DEFAULT_ROOT / "freeze.json"
DEFAULT_POOL = ROOT / "data/ADMET_v3_clean_external_candidate_pool.csv"
DEFAULT_V54_PROTOCOL = ROOT / "reports/evidence_packet_data_axis_v54/protocol.json"
DEFAULT_V54_RESULTS = ROOT / "reports/evidence_packet_data_axis_v54"
DEFAULT_MAPPING = ROOT / "runs/v27-maplight-gnn-freeze/full-22-task.mapping.json"
DEFAULT_DATA = ROOT / "data"
DEFAULT_DB_PREFLIGHT = ROOT / "reports/all22_three_axis_ablation_v3/preflight.json"
DEFAULT_DB_SAFE = DEFAULT_ROOT / "data/herg_admitted_db_runtime_safe.csv"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, payload: dict[str, Any], *, overwrite: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not overwrite:
        raise FileExistsError(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def _config_id(source_set_id: str, quantity_fraction: float, weight: float) -> str:
    value = json.dumps(
        {
            "source_set_id": source_set_id,
            "quantity_fraction": quantity_fraction,
            "weight": weight,
        },
        sort_keys=True,
    )
    return "combined_" + hashlib.sha256(value.encode()).hexdigest()[:12]


def init_protocol(args: argparse.Namespace) -> dict[str, Any]:
    source = json.loads(args.v54_protocol.read_text())
    if source.get("protocol") != v54.PROTOCOL_NAME:
        raise RuntimeError("unexpected Evidence Packet protocol")
    preflight = json.loads(args.db_preflight.read_text())
    admitted_db_tasks = sorted(preflight.get("admitted_tasks", []))
    if admitted_db_tasks != ["hERG"]:
        raise RuntimeError(
            f"strict database preflight changed; expected only hERG, got {admitted_db_tasks}"
        )
    payload = json.loads(json.dumps(source))
    payload["protocol"] = PROTOCOL_NAME
    payload["frozen_at"] = datetime.now(timezone.utc).isoformat()
    payload["safety"].update(
        {
            "top_seed_selection_allowed": True,
            "top_seed_selection_data": "development validation only",
            "test_seed_selection_forbidden": True,
            "database_sources_fail_closed_unless_strictly_admitted": True,
        }
    )
    payload["database_axis"] = {
        "candidate_pool": str(args.pool.resolve()),
        "candidate_pool_sha256": _sha256(args.pool),
        "strict_preflight": str(args.db_preflight.resolve()),
        "strict_preflight_sha256": _sha256(args.db_preflight),
        "admitted_tasks": admitted_db_tasks,
        "rejected_tasks": sorted(set(TASKS) - set(admitted_db_tasks)),
        "hERG_weight_tier": "controlled_bridge_pending_lineage",
        "hERG_evidence_base_weight": 0.25,
    }
    herg = payload["tasks"]["hERG"]
    paper_ids = [item["packet_id"] for item in herg["packets"]]
    combined_ids = [*paper_ids, DB_PACKET_ID]
    herg["source_sets"] = [
        {"source_set_id": "paper_plus_admitted_db", "packet_ids": combined_ids}
    ]
    herg["candidate_configurations"] = []
    for fraction in (0.25, 0.50, 0.75, 1.00):
        for weight in (0.25, 0.50):
            herg["candidate_configurations"].append(
                {
                    "config_id": _config_id("paper_plus_admitted_db", fraction, weight),
                    "packet_ids": combined_ids,
                    "source_set_id": "paper_plus_admitted_db",
                    "quantity_fraction": fraction,
                    "weight": weight,
                    "shortlisted_from_v5_controller": False,
                }
            )
    payload["selection"] = {
        "candidate_set": "baseline plus external configurations",
        "rule": "maximum single development-validation normalized improvement",
        "tie_break": "smaller external fraction, smaller weight, smaller seed",
        "official_test_used": False,
        "reporting": "selected seed plus all development-seed mean and sample SD",
    }
    _write_json(args.protocol, payload, overwrite=args.overwrite)
    return {"output": str(args.protocol.resolve()), "sha256": _sha256(args.protocol)}


def _database_herg_frame(
    *, pool_path: Path, data_dir: Path
) -> tuple[pd.DataFrame, dict[str, Any]]:
    cache_path = DEFAULT_DB_SAFE
    cache_audit_path = cache_path.with_suffix(".audit.json")
    if cache_path.exists() and cache_audit_path.exists():
        audit = json.loads(cache_audit_path.read_text())
        if not (
            audit.get("source_sha256") == _sha256(pool_path)
            and audit.get("safe_csv_sha256") == _sha256(cache_path)
            and audit.get("test_labels_loaded") is False
            and audit.get("structure_firewall", {}).get("dataset_rejected") is False
            and audit.get("structure_firewall", {}).get("exact_test_overlap_rows") == 0
            and audit.get("structure_firewall", {}).get("near_test_neighbor_rows") == 5
        ):
            raise RuntimeError("cached hERG database firewall artifact failed verification")
        return pd.read_csv(cache_path), audit
    pool = pd.read_csv(pool_path, low_memory=False)
    raw = pool.loc[pool["endpoint"].eq("hERG")].copy()
    if raw.empty:
        raise RuntimeError("hERG is absent from the clean database pool")
    frame = pd.DataFrame(
        {
            "Drug": raw["standard_smiles"].astype(str),
            "Y": pd.to_numeric(raw["external_label"], errors="raise"),
            "evidence_packet_id": DB_PACKET_ID,
            "evidence_source_kind": "controlled_bridge_pending_lineage",
            "evidence_base_weight": 0.25,
            "external_source_name": raw["external_source_name"].astype(str),
        }
    )
    development_path = data_dir / "admet_group/herg/train_val.csv"
    test_path = data_dir / "admet_group/herg/test.csv"
    development = pd.read_csv(development_path, usecols=["Drug"])
    test_smiles = load_label_blind_test_structures(test_path)
    reference = prepare_structure_firewall_reference_cached(
        train_smiles=development["Drug"].astype(str).tolist(),
        validation_smiles=[],
        test_smiles=test_smiles,
    )
    safe, firewall = apply_external_structure_firewall(
        frame,
        train_smiles=development["Drug"].astype(str).tolist(),
        validation_smiles=[],
        test_smiles=test_smiles,
        same_source_test_overlap_limit=0.05,
        test_tanimoto_limit=0.90,
        reference=reference,
    )
    if firewall.dataset_rejected or safe.empty:
        raise RuntimeError("strictly admitted hERG database source failed runtime firewall")
    safe = safe.reset_index(drop=True)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    safe.to_csv(cache_path, index=False)
    audit = {
        "packet_id": DB_PACKET_ID,
        "source": str(pool_path.resolve()),
        "source_sha256": _sha256(pool_path),
        "rows_before_runtime_firewall": len(frame),
        "rows_after_runtime_firewall": len(safe),
        "runtime_exact_or_near_neighbor_exclusions": len(frame) - len(safe),
        "structure_firewall": firewall.to_dict(),
        "test_labels_loaded": False,
        "strict_preflight_admitted": True,
    }
    audit["safe_csv"] = str(cache_path.resolve())
    audit["safe_csv_sha256"] = _sha256(cache_path)
    _write_json(cache_audit_path, audit, overwrite=True)
    return safe, audit


def _prepared_sources(
    task: str,
    task_entry: dict[str, Any],
    *,
    data_dir: Path,
    pool: Path,
) -> tuple[dict[str, pd.DataFrame], dict[str, Any]]:
    prepared, audit = v54._verify_task_packets(task, task_entry, data_dir)
    if task == "hERG":
        database, database_audit = _database_herg_frame(
            pool_path=pool, data_dir=data_dir
        )
        prepared[DB_PACKET_ID] = database
        audit["database_audit"] = database_audit
    return prepared, audit


def development_herg(args: argparse.Namespace) -> dict[str, Any]:
    protocol = json.loads(args.protocol.read_text())
    if protocol.get("protocol") != PROTOCOL_NAME:
        raise RuntimeError("unexpected combined protocol")
    if _sha256(args.pool) != protocol["database_axis"]["candidate_pool_sha256"]:
        raise RuntimeError("database pool changed after freeze")
    task_entry = protocol["tasks"]["hERG"]
    mapping, mapping_sha = _load_mapping(args.mapping)
    if mapping_sha != protocol["model_axis"]["mapping_sha256"]:
        raise RuntimeError("frozen model mapping changed")
    frozen = mapping["hERG"]
    blocker = _runtime_blocker(frozen, args.device, ROOT)
    if blocker:
        raise RuntimeError(blocker)
    prepared, packet_audit = _prepared_sources(
        "hERG", task_entry, data_dir=args.data_dir, pool=args.pool
    )
    source_frames, source_audits = v54._source_frames(
        "hERG", task_entry, prepared
    )
    rows, split_audits = v54._evaluate(
        task_name="hERG",
        frozen=frozen,
        seeds=tuple(protocol["data_axis"]["development_seeds"]),
        configurations=task_entry["candidate_configurations"],
        source_frames=source_frames,
        data_dir=args.data_dir,
        cache_root=args.feature_cache,
        artifact_root=args.artifact_root / "development",
        device=args.device,
        n_jobs=args.n_jobs,
        maximum_mass_ratio=float(
            protocol["data_axis"]["maximum_external_to_native_mass_ratio"]
        ),
    )
    return {
        "protocol": PROTOCOL_NAME,
        "protocol_sha256": _sha256(args.protocol),
        "phase": "development",
        "task": "hERG",
        "status": "completed",
        "frozen_recipe": frozen,
        "frozen_recipe_label": _recipe_label(frozen),
        "seeds": protocol["data_axis"]["development_seeds"],
        "rows": rows,
        "packet_audit": packet_audit,
        "source_set_audits": source_audits,
        "split_audits": split_audits,
        "safety": {
            "test_structures_used_only_for_decontamination": True,
            "test_labels_loaded": False,
            "test_results_used_for_selection": False,
        },
    }


def _task_development_path(task: str, args: argparse.Namespace) -> Path:
    if task == "hERG":
        return args.output_root / "development/hERG.json"
    return args.v54_results / "development" / f"{task}.json"


def freeze(args: argparse.Namespace) -> dict[str, Any]:
    protocol = json.loads(args.protocol.read_text())
    if protocol.get("protocol") != PROTOCOL_NAME:
        raise RuntimeError("unexpected combined protocol")
    tasks: dict[str, Any] = {}
    for task in TASKS:
        entry = protocol["tasks"][task]
        if entry["status"] != "candidate_ready":
            tasks[task] = {
                "route": "baseline",
                "seed": int(protocol["data_axis"]["development_seeds"][0]),
                "reason": entry["reason"],
                "development_artifact": None,
            }
            continue
        path = _task_development_path(task, args)
        payload = json.loads(path.read_text())
        expected_protocol = PROTOCOL_NAME if task == "hERG" else v54.PROTOCOL_NAME
        if payload.get("protocol") != expected_protocol:
            raise RuntimeError(f"{task}: development protocol mismatch")
        if payload.get("safety", {}).get("test_labels_loaded") is not False:
            raise RuntimeError(f"{task}: development loaded test labels")
        rows = [
            row
            for row in payload.get("rows", [])
            if math.isfinite(float(row["normalized_improvement"]))
        ]
        if not rows:
            raise RuntimeError(f"{task}: no finite development rows")
        best_external = max(
            rows,
            key=lambda row: (
                float(row["normalized_improvement"]),
                -float(row["quantity_fraction"]),
                -float(row["weight"]),
                -int(row["seed"]),
            ),
        )
        best_baseline = max(
            rows,
            key=lambda row: (
                (float(row["baseline_score"]) if resolve_task(task).metric != "mae" else -float(row["baseline_score"])),
                -int(row["seed"]),
            ),
        )
        use_external = float(best_external["normalized_improvement"]) > 0
        selected = best_external if use_external else best_baseline
        selected_config = None
        if use_external:
            selected_config = next(
                item
                for item in entry["candidate_configurations"]
                if item["config_id"] == selected["config_id"]
            )
        same_config = [
            row
            for row in rows
            if use_external and row["config_id"] == selected["config_id"]
        ]
        values = [float(row["normalized_improvement"]) for row in same_config]
        tasks[task] = {
            "route": "external" if use_external else "baseline",
            "seed": int(selected["seed"]),
            "selected_configuration": selected_config,
            "best_development_normalized_improvement": float(
                best_external["normalized_improvement"]
            ),
            "selected_config_all_seed_mean": float(np.mean(values)) if values else None,
            "selected_config_all_seed_sample_sd": (
                float(np.std(values, ddof=1)) if len(values) > 1 else None
            ),
            "development_artifact": str(path.resolve()),
            "development_artifact_sha256": _sha256(path),
        }
    payload = {
        "protocol": PROTOCOL_NAME,
        "protocol_path": str(args.protocol.resolve()),
        "protocol_sha256": _sha256(args.protocol),
        "frozen_at": datetime.now(timezone.utc).isoformat(),
        "selection_data": "development validation only",
        "selection_rule": protocol["selection"],
        "test_labels_loaded": False,
        "test_results_used_for_selection": False,
        "tasks": tasks,
    }
    _write_json(args.freeze, payload, overwrite=args.overwrite)
    return {
        "output": str(args.freeze.resolve()),
        "sha256": _sha256(args.freeze),
        "external_tasks": sum(v["route"] == "external" for v in tasks.values()),
        "baseline_tasks": sum(v["route"] == "baseline" for v in tasks.values()),
    }


def _load_task_sources(
    task: str,
    protocol: dict[str, Any],
    *,
    data_dir: Path,
    pool: Path,
) -> tuple[dict[str, pd.DataFrame], dict[str, Any]]:
    entry = protocol["tasks"][task]
    prepared, packet_audit = _prepared_sources(
        task, entry, data_dir=data_dir, pool=pool
    )
    frames, source_audits = v54._source_frames(task, entry, prepared)
    return frames, {"packet_audit": packet_audit, "source_set_audits": source_audits}


def _certification_fit(
    *,
    task_name: str,
    frozen: dict[str, Any],
    seed: int,
    route: str,
    config: dict[str, Any] | None,
    source_frames: dict[str, pd.DataFrame] | None,
    data_dir: Path,
    feature_cache: Path,
    artifact_root: Path,
    device: str,
    n_jobs: int,
    maximum_mass_ratio: float,
) -> tuple[dict[str, Any], pd.DataFrame]:
    task = resolve_task(task_name)
    task_dir = data_dir / "admet_group" / task_name.lower()
    train_raw = pd.read_csv(task_dir / "train_val.csv", usecols=["Drug", "Y"])
    test_raw = pd.read_csv(task_dir / "test.csv", usecols=["Drug", "Y"])
    train, _ = _standardized_tdc(train_raw)
    test, _ = _standardized_tdc(test_raw)
    selected_export = pd.DataFrame()
    fit_frame = train[["standard_smiles", "Y"]].copy()
    sample_weight = None
    external_audit: dict[str, Any] = {"route": route, "selected_rows": 0}
    all_smiles: list[str] = []
    if route == "external":
        if config is None or source_frames is None:
            raise RuntimeError("external route lacks a frozen configuration")
        source = source_frames[config["source_set_id"]]
        standardized = v54._standardize_external(source)
        requested = quantity_rows(len(standardized), float(config["quantity_fraction"]))
        positions, selection_audit = select_external_positions(
            source,
            native_targets=train["Y"].to_numpy(),
            task_type=task.task_type,
            total_rows=requested,
        )
        selected = standardized.iloc[positions].reset_index(drop=True)
        selected_export = source.iloc[positions].reset_index(drop=True).copy()
        raw_weights = effective_external_sample_weights(
            source, positions, float(config["weight"])
        )
        weights, weight_audit = cap_external_weights(
            raw_weights,
            native_rows=len(train),
            maximum_mass_ratio=maximum_mass_ratio,
        )
        if frozen.get("kind") == "recipe":
            fit_frame, materialization = v54._materialize_recipe_weights(
                fit_frame,
                selected[["standard_smiles", "Y"]],
                weights,
                seed=seed,
                config_id=config["config_id"],
            )
            # Reconstruct the deterministic PPS positions so the exported final
            # external table contains exactly the rows materialized for fitting,
            # rather than the larger pre-weight candidate selection.
            total = float(weights.sum())
            floor = int(np.floor(total))
            digest = hashlib.sha256(f"{seed}|{config['config_id']}".encode()).digest()
            generator = np.random.default_rng(int.from_bytes(digest[:8], "little"))
            target = min(floor + int(generator.random() < (total - floor)), len(selected))
            if target:
                positions_materialized = np.sort(
                    generator.choice(
                        len(selected), size=target, replace=False, p=weights / weights.sum()
                    )
                )
                selected_export = selected_export.iloc[positions_materialized].reset_index(drop=True)
            else:
                selected_export = selected_export.iloc[:0].copy()
        else:
            fit_frame = pd.concat(
                [fit_frame, selected[["standard_smiles", "Y"]]], ignore_index=True
            )
            sample_weight = np.concatenate([np.ones(len(train)), weights])
            materialization = {
                "mode": "exact_sample_weight",
                "external_materialized_rows": len(selected),
            }
        all_smiles = standardized["standard_smiles"].astype(str).tolist()
        external_audit = {
            "route": route,
            "available_rows": len(source),
            "candidate_selected_rows": len(selected),
            "selected_rows": len(selected_export),
            "selection": selection_audit,
            "weighting": weight_audit,
            "materialization": materialization,
        }
    cache = FeatureCache(feature_cache / task_name, device=device)
    pool = _FeaturePool(
        cache,
        [
            *train["standard_smiles"].astype(str).tolist(),
            *test["standard_smiles"].astype(str).tolist(),
            *all_smiles,
        ],
    )
    started = time.perf_counter()
    predictions = _predict_frozen(
        task_name=task_name,
        frozen=frozen,
        seed=seed,
        fit_frame=fit_frame,
        predict_frame=test,
        sample_weight=sample_weight,
        pool=pool,
        cache=cache,
        artifact_dir=artifact_root / task_name / route,
        device=device,
        n_jobs=n_jobs,
    )
    result = {
        "score": float(score(test["Y"], predictions, task.metric)),
        "metric": task.metric,
        "test_rows": len(test),
        "fit_rows": len(fit_frame),
        "wall_seconds": time.perf_counter() - started,
        "external": external_audit,
    }
    return result, selected_export


def certification(args: argparse.Namespace) -> dict[str, Any]:
    protocol = json.loads(args.protocol.read_text())
    frozen_selection = json.loads(args.freeze.read_text())
    if frozen_selection.get("protocol_sha256") != _sha256(args.protocol):
        raise RuntimeError("freeze does not bind the current protocol")
    selection = frozen_selection["tasks"][args.task]
    mapping, mapping_sha = _load_mapping(args.mapping)
    if mapping_sha != protocol["model_axis"]["mapping_sha256"]:
        raise RuntimeError("frozen model mapping changed")
    frozen = mapping[args.task]
    blocker = _runtime_blocker(frozen, args.device, ROOT)
    if blocker:
        raise RuntimeError(blocker)
    source_frames = None
    source_audit: dict[str, Any] = {}
    if selection["route"] == "external":
        source_frames, source_audit = _load_task_sources(
            args.task, protocol, data_dir=args.data_dir, pool=args.pool
        )
    # Both arms use the same validation-selected seed and frozen recipe.
    baseline, _ = _certification_fit(
        task_name=args.task,
        frozen=frozen,
        seed=int(selection["seed"]),
        route="baseline",
        config=None,
        source_frames=None,
        data_dir=args.data_dir,
        feature_cache=args.feature_cache,
        artifact_root=args.artifact_root / "certification",
        device=args.device,
        n_jobs=args.n_jobs,
        maximum_mass_ratio=float(
            protocol["data_axis"]["maximum_external_to_native_mass_ratio"]
        ),
    )
    if selection["route"] == "baseline":
        chosen, selected_external = baseline, pd.DataFrame()
    else:
        chosen, selected_external = _certification_fit(
            task_name=args.task,
            frozen=frozen,
            seed=int(selection["seed"]),
            route=selection["route"],
            config=selection.get("selected_configuration"),
            source_frames=source_frames,
            data_dir=args.data_dir,
            feature_cache=args.feature_cache,
            artifact_root=args.artifact_root / "certification",
            device=args.device,
            n_jobs=args.n_jobs,
            maximum_mass_ratio=float(
                protocol["data_axis"]["maximum_external_to_native_mass_ratio"]
            ),
        )
    improvement = v54._normalized_improvement(
        chosen["metric"], baseline["score"], chosen["score"]
    )
    export_path = args.output_root / "selected_external" / f"{args.task}.csv"
    export_path.parent.mkdir(parents=True, exist_ok=True)
    if len(selected_external):
        selected_external.insert(0, "task", args.task)
        selected_external.to_csv(export_path, index=False)
    elif export_path.exists():
        export_path.unlink()
    return {
        "protocol": PROTOCOL_NAME,
        "protocol_sha256": _sha256(args.protocol),
        "freeze_path": str(args.freeze.resolve()),
        "freeze_sha256": _sha256(args.freeze),
        "phase": "certification",
        "task": args.task,
        "status": "completed",
        "frozen_recipe_label": _recipe_label(frozen),
        "selected_seed": int(selection["seed"]),
        "selected_route": selection["route"],
        "selected_configuration": selection.get("selected_configuration"),
        "baseline": baseline,
        "selected": chosen,
        "normalized_improvement": improvement,
        "selected_external_csv": str(export_path.resolve()) if len(selected_external) else None,
        "selected_external_sha256": _sha256(export_path) if len(selected_external) else None,
        "source_audit": source_audit,
        "safety": {
            "test_labels_loaded": True,
            "test_labels_loaded_only_after_content_addressed_freeze": True,
            "test_results_used_for_selection": False,
            "same_seed_and_recipe_in_baseline_and_selected_arms": True,
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--phase", choices=("init", "development-herg", "freeze", "certification"), required=True
    )
    parser.add_argument("--task", choices=tuple(TASKS))
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--freeze", type=Path, default=DEFAULT_FREEZE)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--v54-protocol", type=Path, default=DEFAULT_V54_PROTOCOL)
    parser.add_argument("--v54-results", type=Path, default=DEFAULT_V54_RESULTS)
    parser.add_argument("--db-preflight", type=Path, default=DEFAULT_DB_PREFLIGHT)
    parser.add_argument("--pool", type=Path, default=DEFAULT_POOL)
    parser.add_argument("--mapping", type=Path, default=DEFAULT_MAPPING)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA)
    parser.add_argument(
        "--feature-cache", type=Path, default=DEFAULT_ROOT / "artifacts/feature_cache"
    )
    parser.add_argument(
        "--artifact-root", type=Path, default=DEFAULT_ROOT / "artifacts/models"
    )
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--n-jobs", type=int, default=8)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    for name in (
        "protocol", "freeze", "output_root", "v54_protocol", "v54_results",
        "db_preflight", "pool", "mapping", "data_dir", "feature_cache", "artifact_root",
    ):
        setattr(args, name, getattr(args, name).resolve())
    output = args.output
    if output is None:
        if args.phase == "development-herg":
            output = args.output_root / "development/hERG.json"
        elif args.phase == "certification":
            if args.task is None:
                raise ValueError("certification requires --task")
            output = args.output_root / "certification" / f"{args.task}.json"
    try:
        if args.phase == "init":
            result = init_protocol(args)
        elif args.phase == "development-herg":
            result = development_herg(args)
            _write_json(output, result, overwrite=args.overwrite)
        elif args.phase == "freeze":
            result = freeze(args)
        else:
            result = certification(args)
            _write_json(output, result, overwrite=args.overwrite)
        print(json.dumps(result, sort_keys=True))
    except Exception as exc:
        if output is not None:
            _write_json(
                output.with_suffix(".error.json"),
                {
                    "protocol": PROTOCOL_NAME,
                    "phase": args.phase,
                    "task": args.task,
                    "status": "failed",
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "traceback": traceback.format_exc(),
                },
                overwrite=True,
            )
        raise


if __name__ == "__main__":
    main()
