#!/usr/bin/env python3
"""Audit the Korzekwa/Nagar 2017 VDss supplement against native identities."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from xml.etree import ElementTree as ET
from zipfile import ZipFile

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
TASK_ROOT = ROOT / "data/admet_group/vdss_lombardo"
DEFAULT_OUTPUT = (
    ROOT
    / "reports/evidence_packet_data_axis_v5"
    / "vdss_korzekwa2017_source_lineage_audit.json"
)
WORD_NS = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def supplement_rows(path: Path) -> list[dict[str, object]]:
    with ZipFile(path) as archive:
        root = ET.fromstring(archive.read("word/document.xml"))
    rows: list[dict[str, object]] = []
    for table_row in root.findall(".//w:tr", WORD_NS):
        cells = []
        for cell in table_row.findall("./w:tc", WORD_NS):
            cells.append(
                "".join(
                    node.text or "" for node in cell.findall(".//w:t", WORD_NS)
                ).strip()
            )
        if len(cells) < 3 or not cells[0]:
            continue
        try:
            observed_vdss_l_per_kg = float(cells[2])
        except ValueError:
            continue
        rows.append(
            {
                "drug_name": cells[0],
                "observed_vdss_l_per_kg": observed_vdss_l_per_kg,
            }
        )
    return rows


def native_names() -> tuple[set[str], set[str]]:
    train = pd.read_csv(TASK_ROOT / "train_val.csv", usecols=["Drug_ID"])
    test = pd.read_csv(TASK_ROOT / "test.csv", usecols=["Drug_ID"])
    normalize = lambda series: set(series.astype(str).str.strip().str.casefold())
    return normalize(train["Drug_ID"]), normalize(test["Drug_ID"])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("supplement_docx", type=Path)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    rows = supplement_rows(args.supplement_docx)
    if len(rows) != 63:
        raise RuntimeError(f"expected 63 Table S2 records, found {len(rows)}")
    train_names, test_names = native_names()
    train_overlap = sorted(
        row["drug_name"]
        for row in rows
        if str(row["drug_name"]).strip().casefold() in train_names
    )
    test_overlap = sorted(
        row["drug_name"]
        for row in rows
        if str(row["drug_name"]).strip().casefold() in test_names
    )
    union_overlap = sorted(set(train_overlap) | set(test_overlap))
    payload = {
        "protocol": "admet-evidence-packet-source-lineage-audit-v5.1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "task": "VDss_Lombardo",
        "candidate_source": {
            "title": "Drug Distribution Part 2. Predicting Volume of Distribution from Plasma Protein Binding and Membrane Partitioning",
            "doi": "10.1007/s11095-016-2086-y",
            "pmcid": "PMC5285473",
            "supplement": "Table S2",
            "supplement_sha256": sha256_file(args.supplement_docx),
        },
        "candidate_record_count": len(rows),
        "native_name_exact_overlap": {
            "train_count": len(train_overlap),
            "test_count": len(test_overlap),
            "union_count": len(union_overlap),
            "union_fraction": len(union_overlap) / len(rows),
            "drug_names": union_overlap,
        },
        "decision": "rejected_before_structure_resolution_or_training",
        "reason": "All 63 candidate drug identities are already present by exact normalized name in the native Lombardo train/test union; an admitted independent training row cannot remain after the mandatory exact-identity firewall.",
        "test_labels_loaded": False,
        "external_labels_used_for_training": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                "output": str(args.output),
                "candidate_records": len(rows),
                "native_union_name_overlap": len(union_overlap),
                "decision": payload["decision"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
