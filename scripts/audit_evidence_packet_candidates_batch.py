#!/usr/bin/env python3
"""Audit several candidate packets while reusing task-bound structure references."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from admet_agent.evidence_packets import (
    prepare_evidence_packet,
    validate_evidence_packet,
)
from admet_agent.external_structure_firewall import (
    StructureFirewallReference,
    load_label_blind_test_structures,
    prepare_structure_firewall_reference,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifests", type=Path, nargs="+", required=True)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    args = parser.parse_args()

    references: dict[
        str, tuple[list[str], Path, StructureFirewallReference]
    ] = {}
    outputs = []
    for manifest_path in args.manifests:
        packet = validate_evidence_packet(
            manifest_path, require_modeling_ready=False
        )
        if packet.task not in references:
            task_dir = args.data_dir / "admet_group" / packet.task.lower()
            development = pd.read_csv(
                task_dir / "train_val.csv", usecols=["Drug", "Y"]
            )
            train_smiles = development["Drug"].astype(str).tolist()
            test_path = task_dir / "test.csv"
            test_smiles = load_label_blind_test_structures(test_path)
            reference = prepare_structure_firewall_reference(
                train_smiles=train_smiles,
                validation_smiles=[],
                test_smiles=test_smiles,
            )
            references[packet.task] = (train_smiles, test_path, reference)
        train_smiles, test_path, reference = references[packet.task]
        prepared = prepare_evidence_packet(
            packet,
            train_smiles=train_smiles,
            validation_smiles=[],
            test_path=test_path,
            firewall_reference=reference,
        )
        payload = {
            "protocol": "admet-evidence-packet-candidate-audit-v1",
            "packet_id": packet.packet_id,
            "task": packet.task,
            "full_development_union_used": True,
            "audit": prepared.audit,
            "promotion_eligible": bool(prepared.admitted),
            "test_labels_loaded": False,
        }
        output = packet.manifest_path.parent / "candidate_firewall_audit.json"
        if output.exists():
            raise FileExistsError(output)
        output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        outputs.append(
            {
                "packet_id": packet.packet_id,
                "task": packet.task,
                "promotion_eligible": prepared.admitted,
                "admitted_rows": prepared.audit["admitted_rows"],
                "output": str(output),
            }
        )
    print(json.dumps({"candidate_audits": outputs}, sort_keys=True))


if __name__ == "__main__":
    main()
