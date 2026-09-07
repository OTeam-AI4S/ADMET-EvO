#!/usr/bin/env python3
"""Validate all external-data routes without fitting a model or loading test labels."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

import run_all22_three_axis_ablation as core
import run_all22_three_axis_ablation_v2 as v3
from run_classic_ml_external_data_confirmation import TASKS


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--protocol",
        type=Path,
        default=ROOT / "reports/all22_three_axis_ablation_v3/protocol.json",
    )
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data")
    parser.add_argument(
        "--external-data",
        type=Path,
        default=ROOT / "data/ADMET_v3_clean_external_candidate_pool.csv",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "reports/all22_three_axis_ablation_v3/preflight.json",
    )
    args = parser.parse_args()
    core.PROTOCOL_NAME = v3.PROTOCOL_NAME
    protocol = core._load_protocol(args.protocol.resolve())
    tasks = {}
    for task, (benchmark, _, _, _) in TASKS.items():
        raw = pd.read_csv(
            args.data_dir / "admet_group" / benchmark / "test.csv",
            usecols=["Drug"],
        )
        structures = core._standardize_structure_frame(raw)
        external, audit = core._load_external(
            task=task,
            path=args.external_data.resolve(),
            protocol=protocol,
            test_structures=structures,
        )
        tasks[task] = {
            "admitted": bool(audit["admitted"]),
            "admission_status": audit.get("admission_status"),
            "post_firewall_rows": int(audit.get("post_firewall_rows", 0)),
            "weight_policy": audit.get("weight_policy"),
            "fail_closed_reason": audit.get("admission_error"),
            "test_structure_columns_loaded": audit.get(
                "test_structure_columns_loaded"
            ),
            "test_labels_loaded": audit.get("test_labels_loaded"),
        }
        if len(external) and not audit["admitted"]:
            raise RuntimeError(f"{task}: non-admitted external rows escaped quarantine")
        if audit["admitted"] and not audit.get("admission", {}).get(
            "audit_artifact_verified"
        ):
            raise RuntimeError(f"{task}: admitted source is not audit-bound")
    admitted = sorted(task for task, value in tasks.items() if value["admitted"])
    payload = {
        "protocol": v3.PROTOCOL_NAME,
        "protocol_sha256": protocol["sha256"],
        "task_count": len(tasks),
        "admitted_tasks": admitted,
        "fail_closed_tasks": sorted(set(tasks) - set(admitted)),
        "tasks": tasks,
        "model_fit": False,
        "test_labels_loaded": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"output": str(args.output), "admitted_tasks": admitted}))


if __name__ == "__main__":
    main()
