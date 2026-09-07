#!/usr/bin/env python3
"""Materialize a strict temporal FDA DILIrank 2.0 Evidence Packet."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from admet_agent.evidence_packets import SCHEMA_VERSION, sha256_file

FDA_URL = (
    "https://www.fda.gov/science-research/liver-toxicity-knowledge-base-ltkb/"
    "drug-induced-liver-injury-rank-dilirank-20-dataset"
)
PAPER_DOI = "10.1016/j.drudis.2025.104485"


def _normalized_risk(value: object) -> str:
    return str(value).strip().lower().replace("^", "")


def materialize(
    workbook: Path,
    structures_path: Path,
    output_root: Path,
) -> dict:
    packet_id = "fda-dilirank2-2025-new-strict"
    packet_dir = output_root.resolve() / packet_id
    packet_dir.mkdir(parents=True, exist_ok=False)

    raw = pd.read_excel(workbook, sheet_name="version 2", header=1)
    expected = {
        "LTKBID",
        "CompoundName",
        "SeverityClass",
        "LabelSection",
        "vDILI-Concern",
        "Comment",
    }
    if set(raw.columns) != expected:
        raise RuntimeError(f"unexpected DILIrank 2.0 columns: {list(raw.columns)}")
    raw["normalized_risk"] = raw["vDILI-Concern"].map(_normalized_risk)
    is_new = raw["Comment"].astype(str).str.strip().str.casefold().eq("new")
    is_most = raw["normalized_risk"].str.contains("most-dili-concern", regex=False)
    is_no = raw["normalized_risk"].str.contains("no-dili-concern", regex=False)
    selected = raw.loc[is_new & (is_most | is_no)].copy()
    selected["Y"] = is_most.loc[selected.index].astype(int)

    structures = pd.read_csv(structures_path, sep="\t", dtype={"source_record_id": str})
    required_structure_columns = [
        "source_record_id",
        "name",
        "PubChem_CID",
        "X",
    ]
    if not set(required_structure_columns).issubset(structures.columns):
        raise RuntimeError("DILIrank structure table is missing required provenance fields")
    structures = structures[required_structure_columns].copy()
    if structures.duplicated(["source_record_id", "name"]).any():
        raise RuntimeError("structure resolver contains duplicate FDA identity keys")

    joined = selected.merge(
        structures,
        left_on=["LTKBID", "CompoundName"],
        right_on=["source_record_id", "name"],
        how="left",
        validate="one_to_one",
        indicator=True,
    )
    resolved = joined.loc[joined["_merge"].eq("both") & joined["X"].notna()].copy()
    records = resolved.rename(
        columns={
            "X": "Drug",
            "LTKBID": "fda_ltkb_id",
            "CompoundName": "compound_name",
            "vDILI-Concern": "source_risk_class",
            "SeverityClass": "source_severity_class",
            "LabelSection": "source_label_section",
            "Comment": "source_version_comment",
        }
    )[
        [
            "Drug",
            "Y",
            "source_record_id",
            "fda_ltkb_id",
            "compound_name",
            "PubChem_CID",
            "source_risk_class",
            "source_severity_class",
            "source_label_section",
            "source_version_comment",
        ]
    ]
    if len(selected) != 141 or len(records) != 129:
        raise RuntimeError(
            f"expected 141 strict-new FDA rows and 129 resolved rows, got "
            f"{len(selected)} and {len(records)}"
        )
    if set(records["Y"]) != {0, 1}:
        raise RuntimeError("strict FDA DILI packet must contain both classes")

    shared = output_root.resolve() / "_shared_raw"
    shared.mkdir(parents=True, exist_ok=True)
    shared_workbook = shared / "fda_dilirank2_2025.xlsx"
    if shared_workbook.exists():
        if sha256_file(shared_workbook) != sha256_file(workbook):
            raise RuntimeError("existing shared FDA workbook has a different checksum")
    else:
        shutil.copy2(workbook, shared_workbook)

    source_rows_path = packet_dir / "source_rows.csv"
    selected.to_csv(source_rows_path, index=False)
    records_path = packet_dir / "records.csv"
    records.to_csv(records_path, index=False)
    now = datetime.now(timezone.utc).isoformat()
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "packet_id": packet_id,
        "task": "DILI",
        "task_type": "classification",
        "source": {
            "kind": "fda_regulatory_dataset",
            "name": "FDA Liver Toxicity Knowledge Base DILIrank 2.0",
            "title": "Drug Induced Liver Injury Rank (DILIrank) Dataset Version 2.0",
            "citation_url": FDA_URL,
            "citation_type": "fda_document",
            "stable_identifier": (
                "FDA:DILIrank2.0; doi:10.1016/j.drudis.2025.104485; PMID:41005561"
            ),
            "license_or_access_terms": "Public FDA scientific dataset; facts used with source attribution",
            "independence_status": "verified_independent",
            "independence_evidence": (
                "Temporal-record independence: only records explicitly marked New in version 2.0 are retained; all Unchanged and Revised records from the pre-existing FDA lineage are excluded before structure decontamination."
            ),
            "peer_reviewed_publication_year": 2025,
            "raw_download_url": FDA_URL,
            "raw_local_path": str(shared_workbook.relative_to(ROOT)),
            "raw_sha256": sha256_file(shared_workbook),
            "structure_resolution_source": "PubChem PUG REST by FDA compound name",
            "structure_resolution_url": "https://pubchem.ncbi.nlm.nih.gov/docs/pug-rest",
            "structure_resolution_sha256": sha256_file(structures_path),
            "source_rows_sha256": sha256_file(source_rows_path),
        },
        "assay": {
            "endpoint_definition": "High versus no human drug-induced-liver-injury risk",
            "assay_type": "FDA regulatory-label and peer-reviewed-literature adjudication",
            "organism": "Homo sapiens",
            "biological_system": "Clinical and postmarketing human DILI evidence",
            "source_record_locator": "DILIrank 2.0 workbook, version 2 sheet, LTKBID rows marked New",
        },
        "label": {
            "source_value_definition": "FDA vDILI-Concern category",
            "mapping_rule": (
                "Y=1 only for vMost-DILI-concern; Y=0 only for vNo-DILI-concern; vLess-DILI-concern and Ambiguous-DILI-concern excluded"
            ),
            "direction": "Y=1 denotes high DILI risk; Y=0 denotes no DILI concern",
            "transform": "strict high-risk-versus-no-risk binary mapping; no imputation",
        },
        "review": {
            "status": "review_required",
            "reviewer": "ADMET-EvO Evidence Packet data-steward review",
            "reviewed_at": now,
            "endpoint_match": "direct",
            "native_priority_external_weight": 0.8,
            "notes": (
                "FDA primary evidence with a strict temporal-new filter and TDC-aligned high-versus-no-risk mapping. Promotion requires the full-development and label-blind test structure firewall."
            ),
        },
        "records": {
            "path": "records.csv",
            "row_count": len(records),
            "sha256": sha256_file(records_path),
        },
    }
    manifest_path = packet_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    audit = {
        "protocol": "admet-fda-dilirank2-temporal-new-materialization-v1",
        "packet_id": packet_id,
        "raw_version2_rows": len(raw),
        "new_rows_all_risk_classes": int(is_new.sum()),
        "strict_new_most_or_no_rows": len(selected),
        "resolved_records": len(records),
        "unresolved_records": len(selected) - len(records),
        "positive_records": int(records["Y"].sum()),
        "negative_records": int((records["Y"] == 0).sum()),
        "excluded_less_or_ambiguous": int(
            (is_new & ~(is_most | is_no)).sum()
        ),
        "excluded_unchanged_or_revised": int((~is_new).sum()),
        "workbook_sha256": sha256_file(shared_workbook),
        "structure_resolution_sha256": sha256_file(structures_path),
        "records_sha256": sha256_file(records_path),
        "test_labels_loaded": False,
        "review_status": "review_required",
    }
    (packet_dir / "materialization_audit.json").write_text(
        json.dumps(audit, indent=2, sort_keys=True) + "\n"
    )
    return audit


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--workbook",
        type=Path,
        default=ROOT / "data/external/fda/dilirank2/dilirank2.xlsx",
    )
    parser.add_argument(
        "--structures",
        type=Path,
        default=ROOT / "data/external/fda/dilirank2/dilirank2_structures.tab",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=ROOT / "data/evidence_packets_v4/packets",
    )
    args = parser.parse_args()
    print(
        json.dumps(
            materialize(args.workbook, args.structures, args.output_root),
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
