#!/usr/bin/env python3
"""Audit EPA/Helman rat-oral LD50 data against the Zhu benchmark lineage."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from admet_agent.evidence_packets import sha256_file
from admet_agent.external_structure_firewall import (
    apply_external_structure_firewall,
    load_label_blind_test_structures,
)
from admet_agent.structure_firewall_cache import (
    prepare_structure_firewall_reference_cached,
)

PROTOCOL = "helman2019-ld50-lineage-audit-v1"
SOURCE_REPOSITORY = "https://github.com/i-shah/genra-py"
SOURCE_COMMIT = "3e9b68f3f739319e9d2fb2218270af01cc430a7d"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-dir",
        type=Path,
        default=Path("data/external/epa_niceatm/helman2019"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "reports/evidence_packet_data_axis_v5/"
            "ld50_helman2019_lineage_audit.json"
        ),
    )
    args = parser.parse_args()
    values_path = args.source_dir / "small_acute_processed.csv"
    structures_path = args.source_dir / "smi_acute.xlsx"
    values = pd.read_csv(values_path)
    structures = pd.concat(
        pd.read_excel(structures_path, sheet_name=None).values(),
        ignore_index=True,
    )
    candidate = values.merge(
        structures[["DTXSID", "QSAR_READY_SMILES"]],
        left_on="dsstox_sid",
        right_on="DTXSID",
        validate="one_to_one",
    ).rename(columns={"QSAR_READY_SMILES": "Drug"})

    task_dir = Path("data/admet_group/ld50_zhu")
    train_smiles = pd.read_csv(
        task_dir / "train_val.csv", usecols=["Drug"]
    )["Drug"].astype(str).tolist()
    test_smiles = load_label_blind_test_structures(task_dir / "test.csv")
    reference = prepare_structure_firewall_reference_cached(
        train_smiles=train_smiles,
        validation_smiles=[],
        test_smiles=test_smiles,
    )
    _, audit = apply_external_structure_firewall(
        candidate,
        train_smiles=train_smiles,
        validation_smiles=[],
        test_smiles=test_smiles,
        reference=reference,
        same_source_test_overlap_limit=0.05,
        test_tanimoto_limit=0.90,
    )
    payload = {
        "protocol": PROTOCOL,
        "task": "LD50_Zhu",
        "candidate": "Helman et al. 2019 rat acute oral LD50",
        "source_repository": SOURCE_REPOSITORY,
        "source_commit": SOURCE_COMMIT,
        "source_files": {
            str(values_path): sha256_file(values_path),
            str(structures_path): sha256_file(structures_path),
        },
        "source_lineage": (
            "EPA/NICEATM multi-source aggregation includes TEST/ChemIDplus-family "
            "records and is not independent of the Zhu benchmark family"
        ),
        "full_development_union_used": True,
        "audit": audit.to_dict(),
        "promotion_eligible": False,
        "decision": "reject_whole_candidate_before_training",
        "test_labels_loaded": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
