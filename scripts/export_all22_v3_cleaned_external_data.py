#!/usr/bin/env python3
"""Export molecule-level external data from the fixed v3 structure firewall."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pandas as pd
from rdkit import Chem
from tdc.benchmark_group import admet_group

from admet_agent.adaptive_external_data import molecular_scaffold
from admet_agent.herg_data_scaling import (
    morgan_fingerprints,
    near_neighbor_keep_mask,
)
from admet_agent.heldout_structure_firewall import (
    load_structure_only_inchikey_blocklist,
)
from run_all22_adaptive_agent_data import (
    TASKS,
    _coreset,
    _relevance,
)
from run_all22_optimal_recipe_adaptive_data import PROTOCOL
from run_classic_ml_external_data_confirmation import (
    _curate_external,
    _source_spec,
    _standardized_tdc,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_POOL = ROOT / "data/ADMET_v3_clean_external_candidate_pool.csv"
DEFAULT_SELECTED = ROOT / "data/ADMET_v3_best_seed_selected_external_data.csv"
DEVELOPMENT = ROOT / "reports/all22_optimal_recipe_adaptive_data_v3/development"


def _sha256(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def _clean_task(
    task: str,
    *,
    data_dir: Path,
    cohort_dir: Path,
    n_jobs: int,
) -> tuple[pd.DataFrame, dict[str, Any], dict[str, Any]]:
    benchmark, task_type, metric, _ = TASKS[task]
    source_spec = _source_spec(SimpleNamespace(cohort_dir=cohort_dir), task)
    external, source_audit, _ = _curate_external(
        source_spec["path"],
        task_type=task_type,
        smiles_column=source_spec["smiles_column"],
        label_column=source_spec["label_column"],
        id_column=source_spec["id_column"],
        n_jobs=n_jobs,
        majority_cap=15_000,
    )
    discovery = pd.read_csv(
        data_dir / "admet_group" / benchmark / "train_val.csv",
        usecols=["Drug", "Y"],
    )
    discovery_standard, fallback_count = _standardized_tdc(discovery)
    discovery_ids = set(discovery_standard["standard_inchikey"])
    test_ids, test_audit = load_structure_only_inchikey_blocklist(
        data_dir / "admet_group" / benchmark / "test.csv"
    )
    discovery_mask = external["standard_inchikey"].isin(discovery_ids)
    test_mask = external["standard_inchikey"].isin(test_ids)
    external = external.loc[~(discovery_mask | test_mask)].reset_index(drop=True)
    parseable = external["standard_smiles"].map(
        lambda value: Chem.MolFromSmiles(str(value)) is not None
    )
    parse_exclusions = int((~parseable).sum())
    external = external.loc[parseable].reset_index(drop=True)
    external["scaffold"] = external["standard_smiles"].map(molecular_scaffold)
    external["curated_external_index"] = np.arange(len(external), dtype=np.int64)

    identities = set(external["standard_inchikey"])
    if identities & discovery_ids:
        raise RuntimeError(f"{task}: cleaned external still overlaps train/validation")
    if identities & test_ids:
        raise RuntimeError(f"{task}: cleaned external still overlaps hidden test")
    if external["standard_inchikey"].duplicated().any():
        raise RuntimeError(f"{task}: cleaned external contains duplicate identities")

    source = {
        "name": source_spec["name"],
        "path": str(source_spec["path"]),
        "sha256": _sha256(source_spec["path"]),
        "semantic_role": source_spec["semantic_role"],
        "lineage_status": source_spec["lineage_status"],
    }
    audit = {
        **source_audit,
        "exact_tdc_discovery_overlap_exclusions": int(discovery_mask.sum()),
        "exact_tdc_test_structure_overlap_exclusions": int(test_mask.sum()),
        "fingerprint_parse_exclusions": parse_exclusions,
        "post_exact_firewall_identities": len(external),
        "tdc_standardization_fallback_count": fallback_count,
        **test_audit,
        "post_firewall_discovery_overlap": 0,
        "post_firewall_test_overlap": 0,
        "post_firewall_duplicate_identities": 0,
    }
    return external, source, audit


def _best_row(payload: dict[str, Any]) -> dict[str, Any] | None:
    rows = [
        row
        for row in payload.get("per_seed_ratio_model", [])
        if math.isfinite(float(row["oriented_delta"]))
    ]
    if not rows:
        return None
    return max(
        rows,
        key=lambda row: (
            float(row["oriented_delta"]),
            -float(row["ratio"]),
            -int(row["seed"]),
        ),
    )


def _task_export_frame(
    external: pd.DataFrame,
    *,
    task: str,
    source: dict[str, Any],
    mode: str,
    selected: np.ndarray | None = None,
    relevance: np.ndarray | None = None,
    best: dict[str, Any] | None = None,
) -> pd.DataFrame:
    if selected is None:
        frame = external.copy()
    else:
        frame = external.iloc[selected].copy()
        frame["selected_row_order"] = np.arange(1, len(frame) + 1, dtype=np.int64)
        frame["train_relevance_tanimoto"] = relevance[selected]
    benchmark, task_type, metric, _ = TASKS[task]
    frame = frame.rename(columns={"Y": "external_label"})
    frame.insert(0, "endpoint", task)
    frame.insert(1, "benchmark", benchmark)
    frame.insert(2, "metric", metric)
    frame.insert(3, "task_type", task_type)
    frame.insert(4, "export_mode", mode)
    if best is not None:
        frame.insert(5, "best_seed", int(best["seed"]))
        frame.insert(6, "best_ratio", float(best["ratio"]))
        frame.insert(7, "best_oriented_delta", float(best["oriented_delta"]))
    frame["external_source_name"] = source["name"]
    frame["external_source_path"] = source["path"]
    frame["external_source_sha256"] = source["sha256"]
    frame["external_source_role"] = source["semantic_role"]
    frame["external_source_lineage_status"] = source["lineage_status"]
    ordered = [
        "endpoint",
        "benchmark",
        "metric",
        "task_type",
        "export_mode",
    ]
    if best is not None:
        ordered.extend(["best_seed", "best_ratio", "best_oriented_delta"])
    ordered.extend(
        [
            "selected_row_order",
            "curated_external_index",
            "source_record_id",
            "standard_smiles",
            "standard_inchikey",
            "external_label",
            "replicate_count",
            "scaffold",
            "train_relevance_tanimoto",
            "external_source_name",
            "external_source_path",
            "external_source_sha256",
            "external_source_role",
            "external_source_lineage_status",
        ]
    )
    for column in ordered:
        if column not in frame:
            frame[column] = pd.NA
    return frame.loc[:, ordered]


def _write_frame(frame: pd.DataFrame, output: Path, *, first: bool) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(
        output,
        mode="w" if first else "a",
        header=first,
        index=False,
    )


def export_pool(args: argparse.Namespace) -> dict[str, Any]:
    output = args.output or DEFAULT_POOL
    audits: dict[str, Any] = {}
    first = True
    total = 0
    for task in TASKS:
        external, source, audit = _clean_task(
            task,
            data_dir=args.data_dir,
            cohort_dir=args.cohort_dir,
            n_jobs=args.n_jobs,
        )
        frame = _task_export_frame(
            external,
            task=task,
            source=source,
            mode="clean_candidate_pool",
        )
        if len(frame):
            _write_frame(frame, output, first=first)
            first = False
        total += len(frame)
        audits[task] = audit
        print(json.dumps({"task": task, "clean_rows": len(frame)}), flush=True)
    if first:
        raise RuntimeError("No cleaned external rows were available")
    return {
        "protocol": PROTOCOL,
        "mode": "clean_candidate_pool",
        "output": str(output.resolve()),
        "output_sha256": _sha256(output),
        "rows": total,
        "tasks": audits,
        "test_labels_loaded": False,
    }


def export_selected(args: argparse.Namespace) -> dict[str, Any]:
    output = args.output or DEFAULT_SELECTED
    group = admet_group(path=str(args.data_dir))
    audits: dict[str, Any] = {}
    first = True
    total = 0
    for task in TASKS:
        artifact = DEVELOPMENT / f"{task}.json"
        error = DEVELOPMENT / f"{task}.error.json"
        if error.exists() or not artifact.exists():
            raise RuntimeError(f"{task}: completed v3 artifact is unavailable")
        payload = json.loads(artifact.read_text(encoding="utf-8"))
        if payload.get("protocol") != PROTOCOL:
            raise RuntimeError(f"{task}: development protocol mismatch")
        safety = payload.get("safety", {})
        if not (
            safety.get("tdc_test_columns_loaded") == ["Drug"]
            and safety.get("tdc_test_labels_loaded") is False
        ):
            raise RuntimeError(f"{task}: development firewall audit failed")
        best = _best_row(payload)
        external, source, clean_audit = _clean_task(
            task,
            data_dir=args.data_dir,
            cohort_dir=args.cohort_dir,
            n_jobs=args.n_jobs,
        )
        if best is None:
            audits[task] = {
                "status": payload.get("status"),
                "selected_rows": 0,
                "cleaning": clean_audit,
            }
            continue

        benchmark = TASKS[task][0]
        train_raw, valid_raw = group.get_train_valid_split(
            benchmark=benchmark,
            split_type="default",
            seed=int(best["seed"]),
        )
        train, _ = _standardized_tdc(train_raw)
        valid, _ = _standardized_tdc(valid_raw)
        external_fps = morgan_fingerprints(external["standard_smiles"].tolist())
        train_fps = morgan_fingerprints(train["standard_smiles"].tolist())
        valid_fps = morgan_fingerprints(valid["standard_smiles"].tolist())
        keep = near_neighbor_keep_mask(external_fps, valid_fps, threshold=0.80)
        relevance = _relevance(external_fps, train_fps)
        relevance[~keep] = -1.0
        prepared = {
            "task": task,
            "task_type": TASKS[task][1],
            "external": external,
        }
        selected, coreset_audit = _coreset(
            prepared,
            relevance,
            train["Y"].to_numpy(),
            float(best["ratio"]),
        )
        selected_sha256 = hashlib.sha256(
            np.asarray(selected, dtype=np.int64).tobytes()
        ).hexdigest()
        if selected_sha256 != best["selected_rows_sha256"]:
            raise RuntimeError(f"{task}: reconstructed selected-row hash mismatch")
        frame = _task_export_frame(
            external,
            task=task,
            source=source,
            mode="best_seed_selected_external",
            selected=selected,
            relevance=relevance,
            best=best,
        )
        if len(frame):
            _write_frame(frame, output, first=first)
            first = False
        total += len(frame)
        audits[task] = {
            "status": payload.get("status"),
            "best_seed": int(best["seed"]),
            "best_ratio": float(best["ratio"]),
            "best_oriented_delta": float(best["oriented_delta"]),
            "selected_rows": len(frame),
            "selected_rows_sha256": selected_sha256,
            "near_neighbor_exclusions": int((~keep).sum()),
            "coreset": coreset_audit,
            "cleaning": clean_audit,
        }
        print(
            json.dumps(
                {
                    "task": task,
                    "best_seed": int(best["seed"]),
                    "best_ratio": float(best["ratio"]),
                    "selected_rows": len(frame),
                }
            ),
            flush=True,
        )
    if first:
        raise RuntimeError("No best-seed selected rows were available")
    return {
        "protocol": PROTOCOL,
        "mode": "best_seed_selected_external",
        "selection_rule": "maximum single observed oriented delta",
        "output": str(output.resolve()),
        "output_sha256": _sha256(output),
        "rows": total,
        "tasks": audits,
        "test_labels_loaded": False,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("pool", "selected"), required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data")
    parser.add_argument(
        "--cohort-dir",
        type=Path,
        default=ROOT / "data/external/chembl/chembl_37/tdc22_cohorts",
    )
    parser.add_argument("--n-jobs", type=int, default=8)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = export_pool(args) if args.mode == "pool" else export_selected(args)
    output = Path(result["output"])
    audit = output.with_suffix(output.suffix + ".audit.json")
    audit.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
