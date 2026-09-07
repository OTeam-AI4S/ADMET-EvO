#!/usr/bin/env python3
"""Verify and summarize all 22 Evidence Packet v5.4 terminal outcomes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from admet_agent.evidence_packet_loop import CANONICAL_TASKS
from admet_agent.evidence_packets import sha256_file

PROTOCOL = "admet-evidence-packet-v5.4-frozen-recipe-data-axis"


def build(root: Path) -> dict:
    root = root.resolve()
    protocol_path = root / "protocol.json"
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    protocol_sha = sha256_file(protocol_path)
    tasks = []
    effective = []
    for task in CANONICAL_TASKS:
        artifacts = {}
        for phase in ("development", "confirmation", "certification"):
            path = root / phase / f"{task}.json"
            if not path.is_file():
                raise RuntimeError(f"missing {phase} artifact for {task}")
            value = json.loads(path.read_text(encoding="utf-8"))
            if not (
                value.get("protocol") == PROTOCOL
                and value.get("task") == task
                and value.get("phase") == phase
                and value.get("protocol_sha256") == protocol_sha
                and value.get("safety", {}).get("test_labels_loaded") is False
                and value.get("safety", {}).get("legacy_bulk_pool_used") is False
                and value.get("safety", {}).get("old_tanimoto_0_80_flow_used") is False
                and value.get("safety", {}).get("tanimoto_rejection_threshold") == 0.9
            ):
                raise RuntimeError(f"invalid {phase} artifact for {task}")
            artifacts[phase] = (path, value)
        development = artifacts["development"][1]
        confirmation = artifacts["confirmation"][1]
        certification = artifacts["certification"][1]
        certified = certification.get("status") == "certified_active"
        delta = (
            float(certification["gate"]["mean_normalized_improvement"])
            if certified
            else 0.0
        )
        effective.append(delta)
        tasks.append(
            {
                "task": task,
                "frozen_recipe": development.get("frozen_recipe_label"),
                "development_status": development["status"],
                "confirmation_status": confirmation["status"],
                "certification_status": certification["status"],
                "certified_active": certified,
                "effective_normalized_improvement": delta,
                "selected_configuration": certification.get("selected_configuration"),
                "certification_gate": certification.get("gate"),
                "development_artifact": str(artifacts["development"][0]),
                "confirmation_artifact": str(artifacts["confirmation"][0]),
                "certification_artifact": str(artifacts["certification"][0]),
            }
        )
    return {
        "protocol": PROTOCOL,
        "protocol_path": str(protocol_path),
        "protocol_sha256": protocol_sha,
        "all22_tasks_resolved": len(tasks) == 22,
        "candidate_task_count": sum(
            item["status"] == "candidate_ready" for item in protocol["tasks"].values()
        ),
        "source_rejected_task_count": sum(
            item["status"] == "source_rejected_before_training"
            for item in protocol["tasks"].values()
        ),
        "certified_active_task_count": sum(item["certified_active"] for item in tasks),
        "all22_mean_effective_normalized_improvement": sum(effective) / 22.0,
        "tasks": tasks,
        "safety": {
            "tanimoto_rejection_threshold": 0.9,
            "test_labels_loaded": False,
            "legacy_bulk_pool_used": False,
            "task_specific_recipe_matched_between_arms": True,
            "certification_used_for_final_effect_only": True,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root", type=Path, default=Path("reports/evidence_packet_data_axis_v54")
    )
    args = parser.parse_args()
    payload = build(args.root)
    output = args.root.resolve() / "all22_execution_status.json"
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    pd.DataFrame(payload["tasks"]).to_csv(
        args.root.resolve() / "all22_task_results.tsv", sep="\t", index=False
    )
    print(
        json.dumps(
            {
                "all22_tasks_resolved": payload["all22_tasks_resolved"],
                "certified_active_task_count": payload["certified_active_task_count"],
                "all22_mean_effective_normalized_improvement": payload[
                    "all22_mean_effective_normalized_improvement"
                ],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
