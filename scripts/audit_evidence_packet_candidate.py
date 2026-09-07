#!/usr/bin/env python3
"""Run the full label-blind structural audit before packet promotion."""

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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    packet = validate_evidence_packet(
        args.manifest, require_modeling_ready=False
    )
    task_dir = args.data_dir / "admet_group" / packet.task.lower()
    development = pd.read_csv(
        task_dir / "train_val.csv", usecols=["Drug", "Y"]
    )
    prepared = prepare_evidence_packet(
        packet,
        train_smiles=development["Drug"].astype(str).tolist(),
        validation_smiles=[],
        test_path=task_dir / "test.csv",
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
    args.output.resolve().write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n"
    )
    print(
        json.dumps(
            {
                "packet_id": packet.packet_id,
                "promotion_eligible": prepared.admitted,
                "admitted_rows": prepared.audit["admitted_rows"],
                "output": str(args.output),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
