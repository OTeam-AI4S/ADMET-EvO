#!/usr/bin/env python3
"""Freeze validation-best DILI data recipe and certify once on official test."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = Path(__file__).resolve().parent
for value in (ROOT, SCRIPT_DIR):
    if str(value) not in sys.path:
        sys.path.insert(0, str(value))

import run_evidence_packet_frozen_recipe_v54 as v54  # noqa: E402
from run_all22_optimal_recipe_adaptive_data import (  # noqa: E402
    _load_mapping,
    _recipe_label,
    _runtime_blocker,
)
from run_combined_external_best_seed_v1 import (  # noqa: E402
    _certification_fit,
)


NAME = "dili-leakage-safe-best-seed-v1"
REPORT = ROOT / "reports/dili_leakage_safe_best_seed_v1"
PROTOCOL = ROOT / "reports/evidence_packet_data_axis_v54/protocol.json"
DEVELOPMENT = REPORT / "development/DILI.json"
FREEZE = REPORT / "freeze.json"
RESULT = REPORT / "certification/DILI.json"
MAPPING = ROOT / "runs/v27-maplight-gnn-freeze/full-22-task.mapping.json"
DATA = ROOT / "data"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, payload: dict, *, overwrite: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not overwrite:
        raise FileExistsError(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def merge(args: argparse.Namespace) -> dict:
    protocol = json.loads(args.protocol.read_text())
    expected_seeds = [int(value) for value in protocol["data_axis"]["development_seeds"]]
    shards = []
    for seed in expected_seeds:
        path = args.shard_root / f"seed_{seed}.json"
        payload = json.loads(path.read_text())
        if not (
            payload.get("protocol") == v54.PROTOCOL_NAME
            and payload.get("protocol_sha256") == sha256(args.protocol)
            and payload.get("task") == "DILI"
            and payload.get("status") == "completed_shard"
            and payload.get("seeds") == [seed]
            and payload.get("safety", {}).get("test_labels_loaded") is False
        ):
            raise RuntimeError(f"invalid DILI shard: {path}")
        shards.append((path, payload))
    first = shards[0][1]
    output = {
        "protocol": v54.PROTOCOL_NAME, "protocol_sha256": sha256(args.protocol),
        "phase": "development", "task": "DILI", "status": "completed",
        "frozen_recipe": first["frozen_recipe"],
        "frozen_recipe_label": first["frozen_recipe_label"],
        "seeds": expected_seeds,
        "rows": [row for _, payload in shards for row in payload["rows"]],
        "packet_audit": first["packet_audit"],
        "source_set_audits": first["source_set_audits"],
        "split_audits": [audit for _, payload in shards for audit in payload["split_audits"]],
        "shards": [{"path": str(path.resolve()), "sha256": sha256(path),
                    "seed": payload["seeds"][0]} for path, payload in shards],
        "safety": {"test_structures_used_only_for_decontamination": True,
                   "test_labels_loaded": False, "test_results_used_for_selection": False,
                   "parallel_seed_shards_content_addressed": True},
    }
    expected_rows = len(expected_seeds) * len(protocol["tasks"]["DILI"]["candidate_configurations"])
    if len(output["rows"]) != expected_rows:
        raise RuntimeError("merged DILI grid is incomplete")
    write_json(args.development, output, overwrite=args.overwrite)
    return {"development": str(args.development.resolve()),
            "sha256": sha256(args.development), "rows": len(output["rows"])}


def freeze(args: argparse.Namespace) -> dict:
    protocol = json.loads(args.protocol.read_text())
    development = json.loads(args.development.read_text())
    if not (
        protocol.get("protocol") == v54.PROTOCOL_NAME
        and development.get("protocol") == v54.PROTOCOL_NAME
        and development.get("task") == "DILI"
        and development.get("phase") == "development"
        and development.get("safety", {}).get("test_labels_loaded") is False
        and development.get("protocol_sha256") == sha256(args.protocol)
    ):
        raise RuntimeError("DILI development artifact failed protocol/safety binding")
    rows = [
        row
        for row in development.get("rows", [])
        if math.isfinite(float(row["normalized_improvement"]))
    ]
    if not rows:
        raise RuntimeError("no finite DILI development rows")
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
        key=lambda row: (float(row["baseline_score"]), -int(row["seed"])),
    )
    use_external = float(best_external["normalized_improvement"]) > 0
    selected = best_external if use_external else best_baseline
    config = None
    if use_external:
        config = next(
            item
            for item in protocol["tasks"]["DILI"]["candidate_configurations"]
            if item["config_id"] == selected["config_id"]
        )
    same_config = [
        row for row in rows if use_external and row["config_id"] == selected["config_id"]
    ]
    deltas = [float(row["normalized_improvement"]) for row in same_config]
    packet_audit = development["packet_audit"]["packet_audits"][0]
    firewall = packet_audit["structure_firewall"]
    if not (
        packet_audit["admitted_for_training"]
        and firewall["exact_train_overlap_rows"] == 0
        and firewall["exact_validation_overlap_rows"] == 0
        and firewall["exact_test_overlap_rows"] == 0
        and firewall["near_test_neighbor_rows"] == 0
        and firewall["dataset_rejected"] is False
    ):
        raise RuntimeError("DILI leakage firewall is not clean")
    payload = {
        "protocol": NAME,
        "frozen_at": datetime.now(timezone.utc).isoformat(),
        "source_protocol": str(args.protocol.resolve()),
        "source_protocol_sha256": sha256(args.protocol),
        "development_artifact": str(args.development.resolve()),
        "development_artifact_sha256": sha256(args.development),
        "task": "DILI",
        "route": "external" if use_external else "baseline",
        "selected_seed": int(selected["seed"]),
        "selected_configuration": config,
        "best_development_normalized_improvement": float(
            best_external["normalized_improvement"]
        ),
        "selected_config_all_seed_mean": float(np.mean(deltas)) if deltas else None,
        "selected_config_all_seed_sample_sd": (
            float(np.std(deltas, ddof=1)) if len(deltas) > 1 else None
        ),
        "selection_rule": "maximum single development-validation normalized improvement; baseline retained if no external row is positive",
        "tie_break": "smaller quantity fraction, smaller weight, smaller seed",
        "packet_id": packet_audit["packet_id"],
        "packet_records_sha256": packet_audit["records_sha256"],
        "leakage_firewall": firewall,
        "test_labels_loaded": False,
        "test_results_used_for_selection": False,
    }
    write_json(args.freeze, payload, overwrite=args.overwrite)
    return {"freeze": str(args.freeze.resolve()), "freeze_sha256": sha256(args.freeze), **payload}


def certify(args: argparse.Namespace) -> dict:
    protocol = json.loads(args.protocol.read_text())
    frozen_selection = json.loads(args.freeze.read_text())
    if not (
        frozen_selection.get("protocol") == NAME
        and frozen_selection.get("source_protocol_sha256") == sha256(args.protocol)
        and frozen_selection.get("development_artifact_sha256") == sha256(args.development)
        and frozen_selection.get("test_results_used_for_selection") is False
    ):
        raise RuntimeError("DILI freeze failed content-address verification")
    mapping, mapping_sha = _load_mapping(args.mapping)
    if mapping_sha != protocol["model_axis"]["mapping_sha256"]:
        raise RuntimeError("frozen model mapping changed")
    model = mapping["DILI"]
    blocker = _runtime_blocker(model, args.device, ROOT)
    if blocker:
        raise RuntimeError(blocker)
    prepared, packet_audit = v54._verify_task_packets(
        "DILI", protocol["tasks"]["DILI"], args.data_dir
    )
    source_frames, source_audits = v54._source_frames(
        "DILI", protocol["tasks"]["DILI"], prepared
    )
    seed = int(frozen_selection["selected_seed"])
    baseline, _ = _certification_fit(
        task_name="DILI", frozen=model, seed=seed, route="baseline", config=None,
        source_frames=None, data_dir=args.data_dir, feature_cache=args.feature_cache,
        artifact_root=args.artifact_root, device=args.device, n_jobs=args.n_jobs,
        maximum_mass_ratio=float(protocol["data_axis"]["maximum_external_to_native_mass_ratio"]),
    )
    if frozen_selection["route"] == "external":
        selected, external = _certification_fit(
            task_name="DILI", frozen=model, seed=seed, route="external",
            config=frozen_selection["selected_configuration"], source_frames=source_frames,
            data_dir=args.data_dir, feature_cache=args.feature_cache,
            artifact_root=args.artifact_root, device=args.device, n_jobs=args.n_jobs,
            maximum_mass_ratio=float(protocol["data_axis"]["maximum_external_to_native_mass_ratio"]),
        )
    else:
        selected, external = baseline, pd.DataFrame()
    normalized = v54._normalized_improvement(
        baseline["metric"], baseline["score"], selected["score"]
    )
    external_path = ROOT / "data/external/DILI_leakage_safe_best_seed_v1.csv"
    if len(external):
        external = external.copy()
        external.insert(0, "task", "DILI")
        external_path.parent.mkdir(parents=True, exist_ok=True)
        external.to_csv(external_path, index=False)
    elif external_path.exists():
        external_path.unlink()
    payload = {
        "protocol": NAME,
        "task": "DILI",
        "status": "completed",
        "freeze": str(args.freeze.resolve()),
        "freeze_sha256": sha256(args.freeze),
        "selected_seed": seed,
        "selected_route": frozen_selection["route"],
        "selected_configuration": frozen_selection["selected_configuration"],
        "frozen_model": _recipe_label(model),
        "baseline": baseline,
        "selected": selected,
        "official_test_normalized_improvement": normalized,
        "final_external_data": str(external_path.resolve()) if len(external) else None,
        "final_external_data_sha256": sha256(external_path) if len(external) else None,
        "final_external_rows": len(external),
        "packet_audit": packet_audit,
        "source_set_audits": source_audits,
        "safety": {
            "historical_dilirank_database_used": False,
            "test_labels_loaded": True,
            "test_labels_loaded_only_after_freeze": True,
            "test_results_used_for_selection": False,
            "same_seed_and_model_in_baseline_and_external_arms": True,
            "external_exact_test_overlap": 0,
            "external_tanimoto_ge_0_90_test_overlap": 0,
        },
    }
    write_json(args.result, payload, overwrite=args.overwrite)
    table = pd.DataFrame([
        {
            "Task": "DILI",
            "Metric": baseline["metric"],
            "Model": _recipe_label(model),
            "Best Validation Seed": seed,
            "Route": frozen_selection["route"],
            "External Rows Used": len(external),
            "Official Test Baseline": baseline["score"],
            "Official Test Selected": selected["score"],
            "Official Test Normalized Improvement": normalized,
            "Exact Test Overlap": 0,
            "Tanimoto >= 0.90 Test Overlap": 0,
        }
    ])
    csv_path = REPORT / "DILI_leakage_safe_best_seed_v1_results.csv"
    xlsx_path = REPORT / "DILI_leakage_safe_best_seed_v1_results.xlsx"
    table.to_csv(csv_path, index=False)
    table.to_excel(xlsx_path, index=False)
    audit = {
        "protocol": NAME,
        "certification_sha256": sha256(args.result),
        "results_csv": str(csv_path.resolve()),
        "results_csv_sha256": sha256(csv_path),
        "results_xlsx": str(xlsx_path.resolve()),
        "results_xlsx_sha256": sha256(xlsx_path),
        "final_external_data": payload["final_external_data"],
        "final_external_data_sha256": payload["final_external_data_sha256"],
        "explicit_external_test_exact_overlap": 0,
        "explicit_external_test_tanimoto_ge_0_90_overlap": 0,
        "test_results_used_for_selection": False,
        "pretrained_model_corpus_overlap_guarantee": False,
    }
    audit_path = REPORT / "DILI_leakage_safe_best_seed_v1_results.audit.json"
    audit_path.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("merge", "freeze", "certification"), required=True)
    parser.add_argument("--protocol", type=Path, default=PROTOCOL)
    parser.add_argument("--development", type=Path, default=DEVELOPMENT)
    parser.add_argument("--shard-root", type=Path, default=REPORT / "development_shards")
    parser.add_argument("--freeze", type=Path, default=FREEZE)
    parser.add_argument("--result", type=Path, default=RESULT)
    parser.add_argument("--mapping", type=Path, default=MAPPING)
    parser.add_argument("--data-dir", type=Path, default=DATA)
    parser.add_argument("--feature-cache", type=Path, default=REPORT / "artifacts/features")
    parser.add_argument("--artifact-root", type=Path, default=REPORT / "artifacts/models")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--n-jobs", type=int, default=8)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    for field in ("protocol", "development", "shard_root", "freeze", "result", "mapping", "data_dir", "feature_cache", "artifact_root"):
        setattr(args, field, getattr(args, field).resolve())
    output = merge(args) if args.phase == "merge" else freeze(args) if args.phase == "freeze" else certify(args)
    print(json.dumps(output, sort_keys=True))


if __name__ == "__main__":
    main()
