#!/usr/bin/env python3
"""Matched data-axis evaluation of Evidence Packets with each task's frozen recipe."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = Path(__file__).resolve().parent
for value in (ROOT, SCRIPT_DIR):
    if str(value) not in sys.path:
        sys.path.insert(0, str(value))

from admet_agent.evidence_packet_multiseed import (  # noqa: E402
    effective_external_sample_weights,
    quantity_rows,
    reconcile_external_records,
    select_external_positions,
)
from admet_agent.evidence_packet_splits import scaffold_train_validation_split  # noqa: E402
from admet_agent.evidence_packet_v54 import (  # noqa: E402
    cap_external_weights,
    independent_gate,
    summarize_source_grid,
)
from admet_agent.evidence_packets import (  # noqa: E402
    TEST_TANIMOTO_LIMIT,
    prepare_evidence_packet,
    sha256_file,
    validate_evidence_packet,
)
from admet_agent.external_structure_firewall import (  # noqa: E402
    load_label_blind_test_structures,
)
from admet_agent.features import FeatureCache  # noqa: E402
from admet_agent.structure_firewall_cache import (  # noqa: E402
    prepare_structure_firewall_reference_cached as prepare_structure_firewall_reference,
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
    _standardized_tdc,
)

PROTOCOL_NAME = "admet-evidence-packet-v5.4-frozen-recipe-data-axis"


def _normalized_improvement(metric: str, baseline: float, challenger: float) -> float:
    denominator = max(abs(baseline), 1e-12)
    return float(
        (baseline - challenger) / denominator
        if metric == "mae"
        else (challenger - baseline) / denominator
    )


def _load_protocol(path: Path) -> tuple[dict, str]:
    resolved = path.resolve()
    payload = json.loads(resolved.read_text(encoding="utf-8"))
    if payload.get("protocol") != PROTOCOL_NAME:
        raise RuntimeError("unexpected v5.4 protocol")
    data = payload.get("data_axis", {})
    seed_groups = [
        tuple(data.get("development_seeds", ())),
        tuple(data.get("confirmation_seeds", ())),
        tuple(data.get("certification_seeds", ())),
    ]
    if any(not values for values in seed_groups):
        raise RuntimeError("all three seed groups must be frozen")
    if len(set().union(*map(set, seed_groups))) != sum(map(len, seed_groups)):
        raise RuntimeError("development, confirmation, and certification seeds overlap")
    if data.get("tanimoto_rejection_threshold") != TEST_TANIMOTO_LIMIT:
        raise RuntimeError("v5.4 protocol must freeze Tanimoto at 0.90")
    safety = payload.get("safety", {})
    if not (
        safety.get("test_labels_loaded") is False
        and safety.get("legacy_bulk_pool_used") is False
        and safety.get("old_tanimoto_0_80_flow_used") is False
    ):
        raise RuntimeError("v5.4 safety contract is incomplete")
    return payload, sha256_file(resolved)


def _verify_task_packets(
    task_name: str, task_entry: dict, data_dir: Path
) -> tuple[dict[str, pd.DataFrame], dict]:
    resolve_task(task_name)
    task_dir = data_dir / "admet_group" / task_name.lower()
    development_path = task_dir / "train_val.csv"
    development = pd.read_csv(development_path, usecols=["Drug", "Y"])
    test_path = task_dir / "test.csv"
    test_smiles = load_label_blind_test_structures(test_path)
    reference = prepare_structure_firewall_reference(
        train_smiles=development["Drug"].astype(str).tolist(),
        validation_smiles=[],
        test_smiles=test_smiles,
    )
    prepared_by_id: dict[str, pd.DataFrame] = {}
    audits = []
    for spec in task_entry["packets"]:
        manifest = ROOT / spec["manifest_path"]
        audit_path = ROOT / spec["audit_path"]
        if sha256_file(manifest) != spec["manifest_sha256"]:
            raise RuntimeError(f"manifest changed after protocol freeze: {manifest}")
        if sha256_file(audit_path) != spec["audit_sha256"]:
            raise RuntimeError(f"promotion audit changed after protocol freeze: {audit_path}")
        packet = validate_evidence_packet(manifest)
        if packet.task != task_name or sha256_file(packet.records_path) != spec["records_sha256"]:
            raise RuntimeError(f"packet binding changed after protocol freeze: {packet.packet_id}")
        prepared = prepare_evidence_packet(
            packet,
            train_smiles=development["Drug"].astype(str).tolist(),
            validation_smiles=[],
            test_path=test_path,
            firewall_reference=reference,
        )
        firewall = prepared.audit["structure_firewall"]
        if firewall.get("tanimoto_rejection_threshold") != TEST_TANIMOTO_LIMIT:
            raise RuntimeError("runtime Evidence Packet firewall is not Tanimoto 0.90")
        if prepared.audit.get("test_labels_loaded") is not False:
            raise RuntimeError("test label firewall failed")
        if not prepared.admitted:
            raise RuntimeError(f"frozen packet no longer passes admission: {packet.packet_id}")
        prepared_by_id[packet.packet_id] = prepared.records
        audits.append(prepared.audit)
    return prepared_by_id, {
        "train_val_path": str(development_path.resolve()),
        "train_val_sha256": sha256_file(development_path),
        "packet_audits": audits,
        "test_structures_used_only_for_decontamination": True,
        "test_labels_loaded": False,
    }


def _source_frames(
    task_name: str, task_entry: dict, prepared_by_id: dict[str, pd.DataFrame]
) -> tuple[dict[str, pd.DataFrame], dict[str, dict]]:
    task = resolve_task(task_name)
    frames = {}
    audits = {}
    for source_set in task_entry["source_sets"]:
        source_set_id = source_set["source_set_id"]
        raw = pd.concat(
            [prepared_by_id[packet_id] for packet_id in source_set["packet_ids"]],
            ignore_index=True,
            sort=False,
        )
        reconciled, audit = reconcile_external_records(raw, task_type=task.task_type)
        if reconciled.empty:
            raise RuntimeError(f"source set is empty after reconciliation: {source_set_id}")
        frames[source_set_id] = reconciled
        audits[source_set_id] = {
            "packet_ids": source_set["packet_ids"],
            "reconciliation": audit,
            "rows": len(reconciled),
        }
    return frames, audits


def _standardize_external(frame: pd.DataFrame) -> pd.DataFrame:
    standardized, _ = _standardized_tdc(frame[["Drug", "Y"]])
    standardized["evidence_base_weight"] = frame["evidence_base_weight"].to_numpy(
        dtype=float
    )
    standardized["evidence_packet_id"] = frame["evidence_packet_id"].astype(str).to_numpy()
    return standardized


def _materialize_recipe_weights(
    native: pd.DataFrame,
    external: pd.DataFrame,
    weights: np.ndarray,
    *,
    seed: int,
    config_id: str,
) -> tuple[pd.DataFrame, dict]:
    """Systematic PPS materialization for recipe backends without sample weights."""

    total = float(weights.sum())
    floor = int(np.floor(total))
    digest = hashlib.sha256(f"{seed}|{config_id}".encode()).digest()
    generator = np.random.default_rng(int.from_bytes(digest[:8], "little"))
    target = floor + int(generator.random() < (total - floor))
    target = min(target, len(external))
    if target == 0:
        return native.reset_index(drop=True).copy(), {
            "mode": "deterministic_systematic_pps_materialization",
            "external_materialized_rows": 0,
            "target_expected_external_mass": total,
            "selection_probability_weighted_by_packet_quality": True,
        }
    probabilities = weights / weights.sum()
    positions = np.sort(
        generator.choice(len(external), size=target, replace=False, p=probabilities)
    )
    combined = pd.concat(
        [native.reset_index(drop=True), external.iloc[positions].reset_index(drop=True)],
        ignore_index=True,
    )
    return combined, {
        "mode": "deterministic_systematic_pps_materialization",
        "external_materialized_rows": int(target),
        "target_expected_external_mass": total,
        "selection_probability_weighted_by_packet_quality": True,
        "selected_positions_sha256": hashlib.sha256(
            np.asarray(positions, dtype=np.int64).tobytes()
        ).hexdigest(),
    }


def _evaluate(
    *,
    task_name: str,
    frozen: dict,
    seeds: tuple[int, ...],
    configurations: list[dict],
    source_frames: dict[str, pd.DataFrame],
    data_dir: Path,
    cache_root: Path,
    artifact_root: Path,
    device: str,
    n_jobs: int,
    maximum_mass_ratio: float,
) -> tuple[list[dict], list[dict]]:
    task = resolve_task(task_name)
    raw = pd.read_csv(
        data_dir / "admet_group" / task_name.lower() / "train_val.csv",
        usecols=["Drug", "Y"],
    )
    external_standard = {
        source_set_id: _standardize_external(frame)
        for source_set_id, frame in source_frames.items()
    }
    all_external_smiles = sorted(
        {
            value
            for frame in external_standard.values()
            for value in frame["standard_smiles"].astype(str)
        }
    )
    full_native, _ = _standardized_tdc(raw)
    cache = FeatureCache(cache_root / task_name, device=device)
    pool = _FeaturePool(
        cache,
        [*full_native["standard_smiles"].astype(str).tolist(), *all_external_smiles],
    )
    rows: list[dict] = []
    split_audits: list[dict] = []
    for seed in seeds:
        train_raw, validation_raw, split_audit = scaffold_train_validation_split(raw, seed=seed)
        train, _ = _standardized_tdc(train_raw)
        validation, _ = _standardized_tdc(validation_raw)
        baseline_started = time.perf_counter()
        baseline_prediction = _predict_frozen(
            task_name=task_name,
            frozen=frozen,
            seed=seed,
            fit_frame=train,
            predict_frame=validation,
            sample_weight=None,
            pool=pool,
            cache=cache,
            artifact_dir=artifact_root / task_name / f"seed_{seed}" / "baseline",
            device=device,
            n_jobs=n_jobs,
        )
        from admet_agent.selection import score

        baseline_score = float(score(validation["Y"], baseline_prediction, task.metric))
        baseline_seconds = time.perf_counter() - baseline_started
        for config in configurations:
            source_set_id = config["source_set_id"]
            external_raw = source_frames[source_set_id]
            external = external_standard[source_set_id]
            requested = quantity_rows(len(external), float(config["quantity_fraction"]))
            positions, selection_audit = select_external_positions(
                external_raw,
                native_targets=train["Y"].to_numpy(),
                task_type=task.task_type,
                total_rows=requested,
            )
            selected = external.iloc[positions].reset_index(drop=True)
            raw_weights = effective_external_sample_weights(
                external_raw, positions, float(config["weight"])
            )
            external_weights, weight_audit = cap_external_weights(
                raw_weights,
                native_rows=len(train),
                maximum_mass_ratio=maximum_mass_ratio,
            )
            config_id = str(config["config_id"])
            if frozen.get("kind") == "recipe":
                fit_frame, materialization = _materialize_recipe_weights(
                    train[["standard_smiles", "Y"]],
                    selected[["standard_smiles", "Y"]],
                    external_weights,
                    seed=seed,
                    config_id=config_id,
                )
                sample_weight = None
            else:
                fit_frame = pd.concat(
                    [train[["standard_smiles", "Y"]], selected[["standard_smiles", "Y"]]],
                    ignore_index=True,
                )
                sample_weight = np.concatenate(
                    [np.ones(len(train), dtype=float), external_weights]
                )
                materialization = {
                    "mode": "exact_sample_weight",
                    "external_materialized_rows": len(selected),
                }
            started = time.perf_counter()
            prediction = _predict_frozen(
                task_name=task_name,
                frozen=frozen,
                seed=seed,
                fit_frame=fit_frame,
                predict_frame=validation,
                sample_weight=sample_weight,
                pool=pool,
                cache=cache,
                artifact_dir=artifact_root / task_name / f"seed_{seed}" / config_id,
                device=device,
                n_jobs=n_jobs,
            )
            challenger = float(score(validation["Y"], prediction, task.metric))
            normalized = _normalized_improvement(task.metric, baseline_score, challenger)
            row = {
                "task": task_name,
                "seed": int(seed),
                "config_id": config_id,
                "source_set_id": source_set_id,
                "packet_ids": config["packet_ids"],
                "weight": float(config["weight"]),
                "quantity_fraction": float(config["quantity_fraction"]),
                "native_train_rows": len(train),
                "validation_rows": len(validation),
                "external_available_rows": len(external),
                "external_selected_rows": len(selected),
                "external_to_native_mass_ratio": weight_audit[
                    "observed_external_to_native_mass_ratio"
                ],
                "weighting": weight_audit,
                "materialization": materialization,
                "quantity_selection": selection_audit,
                "baseline_score": baseline_score,
                "challenger_score": challenger,
                "normalized_improvement": normalized,
                "baseline_fit_wall_seconds": baseline_seconds,
                "challenger_fit_wall_seconds": time.perf_counter() - started,
                "test_labels_loaded": False,
            }
            rows.append(row)
            print(json.dumps(row, sort_keys=True), flush=True)
        split_audits.append({"seed": seed, "split": split_audit, "test_labels_loaded": False})
    return rows, split_audits


def _write(path: Path, payload: dict, overwrite: bool) -> None:
    resolved = path.resolve()
    resolved.parent.mkdir(parents=True, exist_ok=True)
    if resolved.exists() and not overwrite:
        raise FileExistsError(resolved)
    resolved.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _previous(path: Path, *, phase: str, task: str, protocol_sha: str) -> dict:
    payload = json.loads(path.resolve().read_text(encoding="utf-8"))
    if not (
        payload.get("protocol") == PROTOCOL_NAME
        and payload.get("phase") == phase
        and payload.get("task") == task
        and payload.get("protocol_sha256") == protocol_sha
        and payload.get("safety", {}).get("test_labels_loaded") is False
    ):
        raise RuntimeError(f"invalid {phase} artifact")
    return payload


def run(args: argparse.Namespace) -> dict:
    protocol, protocol_sha = _load_protocol(args.protocol)
    task_entry = protocol["tasks"][args.task]
    base = {
        "protocol": PROTOCOL_NAME,
        "protocol_path": str(args.protocol.resolve()),
        "protocol_sha256": protocol_sha,
        "phase": args.phase,
        "task": args.task,
        "model_mapping_path": str(args.mapping.resolve()),
        "model_mapping_sha256": sha256_file(args.mapping.resolve()),
        "safety": {
            "tanimoto_rejection_threshold": TEST_TANIMOTO_LIMIT,
            "test_structures_used_only_for_decontamination": True,
            "test_labels_loaded": False,
            "legacy_bulk_pool_used": False,
            "old_tanimoto_0_80_flow_used": False,
            "same_frozen_recipe_in_native_and_augmented_arms": True,
        },
    }
    if task_entry["status"] != "candidate_ready":
        return {
            **base,
            "status": "source_rejected_before_training",
            "reason": task_entry["reason"],
            "rows": [],
        }
    mapping, mapping_sha = _load_mapping(args.mapping)
    if mapping_sha != protocol["model_axis"]["mapping_sha256"]:
        raise RuntimeError("frozen task mapping changed after protocol freeze")
    frozen = mapping[args.task]
    blocker = _runtime_blocker(frozen, args.device, ROOT)
    if blocker:
        raise RuntimeError(blocker)
    base.update(
        {
            "frozen_recipe": frozen,
            "frozen_recipe_label": _recipe_label(frozen),
            "model_mapping_sha256": mapping_sha,
        }
    )
    prepared, packet_audit = _verify_task_packets(args.task, task_entry, args.data_dir)
    source_frames, source_audits = _source_frames(args.task, task_entry, prepared)
    data_policy = protocol["data_axis"]
    if args.phase == "development":
        configurations = task_entry["candidate_configurations"]
        seeds = tuple(data_policy["development_seeds"])
    else:
        if args.previous_artifact is None:
            raise ValueError(f"{args.phase} requires --previous-artifact")
        previous_phase = "development" if args.phase == "confirmation" else "confirmation"
        previous = _previous(
            args.previous_artifact,
            phase=previous_phase,
            task=args.task,
            protocol_sha=protocol_sha,
        )
        if args.phase == "confirmation":
            selected = previous.get("selection", {}).get("selected")
            if not selected:
                return {
                    **base,
                    "status": "not_run_development_abstained",
                    "previous_artifact": str(args.previous_artifact.resolve()),
                    "rows": [],
                }
            configurations = [
                item
                for item in task_entry["candidate_configurations"]
                if item["config_id"] == selected["config_id"]
            ]
            seeds = tuple(data_policy["confirmation_seeds"])
        else:
            if not previous.get("gate", {}).get("passed"):
                return {
                    **base,
                    "status": "rollback_confirmation_failed",
                    "previous_artifact": str(args.previous_artifact.resolve()),
                    "rows": [],
                }
            selected = previous["selected_configuration"]
            configurations = [selected]
            seeds = tuple(data_policy["certification_seeds"])
    rows, split_audits = _evaluate(
        task_name=args.task,
        frozen=frozen,
        seeds=seeds,
        configurations=configurations,
        source_frames=source_frames,
        data_dir=args.data_dir,
        cache_root=args.feature_cache_root,
        artifact_root=args.artifact_root / args.phase,
        device=args.device,
        n_jobs=args.n_jobs,
        maximum_mass_ratio=float(data_policy["maximum_external_to_native_mass_ratio"]),
    )
    common = {
        **base,
        "seeds": list(seeds),
        "rows": rows,
        "packet_audit": packet_audit,
        "source_set_audits": source_audits,
        "split_audits": split_audits,
    }
    if args.phase == "development":
        summaries, selection = summarize_source_grid(
            rows,
            minimum_positive_seed_fraction=float(
                data_policy["development_minimum_positive_seed_fraction"]
            ),
        )
        if selection.get("selected"):
            config = next(
                item
                for item in configurations
                if item["source_set_id"] == selection["selected"]["source_set_id"]
                and item["weight"] == selection["selected"]["weight"]
                and item["quantity_fraction"]
                == selection["selected"]["quantity_fraction"]
            )
            selection["selected"] = {**selection["selected"], "config_id": config["config_id"]}
        return {
            **common,
            "status": selection["status"],
            "configuration_summaries": summaries,
            "selection": selection,
        }
    values = [row["normalized_improvement"] for row in rows]
    if args.phase == "confirmation":
        gate = independent_gate(
            values,
            minimum_mean=float(
                data_policy["confirmation_minimum_mean_normalized_improvement"]
            ),
            minimum_positive_fraction=float(
                data_policy["confirmation_minimum_positive_seed_fraction"]
            ),
        )
        return {
            **common,
            "status": "confirmation_passed" if gate["passed"] else "fallback_baseline",
            "previous_artifact": str(args.previous_artifact.resolve()),
            "previous_artifact_sha256": sha256_file(args.previous_artifact.resolve()),
            "selected_configuration": configurations[0],
            "gate": gate,
        }
    gate = independent_gate(
        values,
        minimum_mean=float(data_policy["certification_minimum_mean_normalized_improvement"]),
        minimum_positive_fraction=float(
            data_policy["certification_minimum_positive_seed_fraction"]
        ),
    )
    return {
        **common,
        "status": "certified_active" if gate["passed"] else "rollback_certification_failed",
        "previous_artifact": str(args.previous_artifact.resolve()),
        "previous_artifact_sha256": sha256_file(args.previous_artifact.resolve()),
        "selected_configuration": configurations[0],
        "gate": gate,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=("development", "confirmation", "certification"), required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--mapping", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--previous-artifact", type=Path)
    parser.add_argument("--feature-cache-root", type=Path, default=Path("checkpoints/evidence_packet_v54/features"))
    parser.add_argument("--artifact-root", type=Path, default=Path("reports/evidence_packet_data_axis_v54/artifacts"))
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--n-jobs", type=int, default=8)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    started = time.perf_counter()
    try:
        payload = run(args)
        payload["wall_seconds"] = time.perf_counter() - started
        _write(args.output, payload, args.overwrite)
        print(json.dumps({"task": args.task, "phase": args.phase, "status": payload["status"]}, sort_keys=True))
    except Exception as exc:
        error = {
            "protocol": PROTOCOL_NAME,
            "task": args.task,
            "phase": args.phase,
            "status": "execution_failed",
            "error_type": type(exc).__name__,
            "error": str(exc),
            "traceback": traceback.format_exc(),
            "test_labels_loaded": False,
        }
        error_path = args.output.with_suffix(".error.json")
        _write(error_path, error, True)
        raise


if __name__ == "__main__":
    main()
