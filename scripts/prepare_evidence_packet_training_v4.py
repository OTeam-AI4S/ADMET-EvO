#!/usr/bin/env python3
"""Build a seed-42 task training table exclusively from registered packets."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from tdc.benchmark_group import admet_group

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from preflight_evidence_packet_v4 import DEFAULT_REGISTRY, preflight

from admet_agent.evidence_packet_splits import (
    scaffold_train_validation_split,
)
from admet_agent.evidence_packets import (
    assemble_evidence_training_data,
    prepare_evidence_packet,
    validate_evidence_packet,
)
from admet_agent.tasks import resolve_task


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", required=True)
    parser.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--weight-mode",
        choices=("paper_equal", "native_priority"),
        default="paper_equal",
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    task = resolve_task(args.task)
    registry_report = preflight(args.registry)
    if not registry_report["formal_experiment_allowed"]:
        raise RuntimeError(
            "Evidence Packet v4 failed closed: no valid modeling-ready packets"
        )
    packet_entries = [
        entry for entry in registry_report["packets"] if entry["task"] == task.name
    ]
    if not packet_entries:
        raise RuntimeError(
            f"No registered Evidence Packet is available for {task.name}"
        )

    benchmark = task.name.lower()
    group = admet_group(path=str(args.data_dir))
    train_validation = group.get(benchmark)["train_val"]
    train, validation, split_audit = scaffold_train_validation_split(
        train_validation, seed=args.seed
    )
    test_path = args.data_dir / "admet_group" / benchmark / "test.csv"
    prepared = []
    for entry in packet_entries:
        packet = validate_evidence_packet(entry["manifest_path"])
        result = prepare_evidence_packet(
            packet,
            train_smiles=train["Drug"].astype(str).tolist(),
            validation_smiles=validation["Drug"].astype(str).tolist(),
            test_path=test_path,
        )
        prepared.append(result)

    training, merge_audit = assemble_evidence_training_data(
        train, prepared, weight_mode=args.weight_mode
    )
    if merge_audit["external_rows"] == 0:
        raise RuntimeError(
            "All registered packets were rejected or emptied by the firewall"
        )
    output = args.output or (
        ROOT
        / "reports/evidence_packet_data_axis_v4/prepared"
        / f"{task.name}.seed{args.seed}.{args.weight_mode}.csv"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    training.to_csv(output, index=False)
    audit = {
        "protocol": "admet-evidence-packet-training-preparation-v4.1",
        "task": task.name,
        "seed": args.seed,
        "development_split": "Auto-Research-Molprop scaffold partition",
        "split": split_audit,
        "output": str(output.resolve()),
        "merge": merge_audit,
        "packets": [item.audit for item in prepared],
        "validation_labels_used_for_packet_selection_or_curation": False,
        "test_labels_loaded": False,
        "legacy_bulk_pool_used": False,
    }
    audit_path = output.with_suffix(output.suffix + ".audit.json")
    audit_path.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                "output": str(output),
                "audit": str(audit_path),
                "external_rows": merge_audit["external_rows"],
            }
        )
    )


if __name__ == "__main__":
    main()
