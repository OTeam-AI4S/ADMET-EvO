#!/usr/bin/env python3
"""Materialize the Dorn/PDSP PubChem AID 376 hERG packet candidate."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from admet_agent.evidence_packets import SCHEMA_VERSION, sha256_file


def materialize(raw_path: Path, description_path: Path, output_dir: Path) -> dict:
    raw_path = raw_path.resolve()
    description_path = description_path.resolve()
    raw = pd.read_csv(raw_path, low_memory=False)
    required = {
        "PUBCHEM_SID",
        "PUBCHEM_EXT_DATASOURCE_SMILES",
        "PUBCHEM_ACTIVITY_OUTCOME",
        "Percent HERG blockade",
    }
    if not required.issubset(raw):
        raise ValueError(f"PubChem CSV missing columns: {sorted(required - set(raw))}")
    assay_rows = raw.loc[raw["PUBCHEM_SID"].notna()].copy()
    outcomes = assay_rows["PUBCHEM_ACTIVITY_OUTCOME"].astype(str)
    labeled = assay_rows.loc[outcomes.isin(["Active", "Inactive"])].copy()
    labeled["Y"] = (
        labeled["PUBCHEM_ACTIVITY_OUTCOME"].astype(str) == "Active"
    ).astype(int)
    labeled["source_record_id"] = labeled["PUBCHEM_SID"].map(
        lambda value: f"PubChem:AID376:SID{int(value)}"
    )
    records = labeled.rename(
        columns={"PUBCHEM_EXT_DATASOURCE_SMILES": "Drug"}
    )[["Drug", "Y", "source_record_id"]]
    if records["source_record_id"].duplicated().any():
        raise ValueError("PubChem SID is not unique")

    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    raw_output = output_dir / "raw_pubchem_aid376.csv"
    raw_output.write_bytes(raw_path.read_bytes())
    description_output = output_dir / "raw_pubchem_aid376_description.json"
    description_output.write_bytes(description_path.read_bytes())
    records_path = output_dir / "records.csv"
    records.to_csv(records_path, index=False)

    packet_id = "dorn2005-herg-aid376"
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "packet_id": packet_id,
        "task": "hERG",
        "task_type": "classification",
        "source": {
            "kind": "independent_peer_reviewed_assay",
            "name": "PDSP hERG fluorescence assay, PubChem AID 376",
            "title": "Evaluation of a high-throughput fluorescence assay method for HERG potassium channel inhibition",
            "citation_url": "https://doi.org/10.1177/1087057104272045",
            "citation_type": "doi",
            "stable_identifier": "doi:10.1177/1087057104272045; PMID:15964935; PubChem:AID376",
            "license_or_access_terms": "Factual PubChem BioAssay extraction with DOI/PubMed/PubChem citation; PubChem data-use terms apply",
            "independence_status": "verified_independent",
            "independence_evidence": "The PDSP/Dorn 2005 HEK-293 fluorescence campaign is a distinct experimental assay from the Wang 2016 literature-curated 648-compound TDC benchmark. Because an older assay could still have been cited by a later compilation, modeling-ready promotion remains conditional on whole-development and label-blind test structure firewalls.",
            "peer_reviewed_publication_year": 2005,
            "raw_download_url": "https://pubchem.ncbi.nlm.nih.gov/rest/pug/assay/aid/376/CSV",
            "raw_description_url": "https://pubchem.ncbi.nlm.nih.gov/rest/pug/assay/aid/376/description/JSON",
            "raw_assay_result_rows": len(assay_rows),
            "raw_sha256": sha256_file(raw_output),
            "raw_description_sha256": sha256_file(description_output),
        },
        "assay": {
            "endpoint_definition": "Direct blockade of the human KCNH2/hERG potassium channel",
            "assay_type": "FlexStation II 96-well membrane-potential fluorescence assay",
            "organism": "Homo sapiens",
            "biological_system": "HEK-293 cells stably expressing human hERG; duplicate compound measurements with terfenadine positive control",
            "source_record_locator": "PubChem AID 376; one record per PubChem SID; official assay protocol and outcome fields",
        },
        "label": {
            "source_value_definition": "PubChem activity outcome; assay hits are compounds with at least 20% hERG blockade",
            "mapping_rule": "Y=1 only for PUBCHEM_ACTIVITY_OUTCOME Active (official >=20% blocker call); Y=0 only for Inactive; missing outcomes excluded",
            "direction": "Y=1 denotes hERG blockade and Y=0 denotes inactive",
            "transform": "Deterministic official Active/Inactive mapping; continuous fluorescence measurements are not relabeled and no benchmark labels are used",
        },
        "review": {
            "status": "review_required",
            "reviewer": "ADMET-EvO Evidence Packet data-steward review",
            "reviewed_at": datetime.now(timezone.utc).isoformat(),
            "endpoint_match": "direct",
            "native_priority_external_weight": 0.75,
            "notes": "Official single-assay extraction. Full structural firewall and later-source lineage risk review are required before modeling_ready promotion.",
        },
        "records": {
            "path": "records.csv",
            "row_count": len(records),
            "sha256": sha256_file(records_path),
        },
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    audit = {
        "aid": 376,
        "task": "hERG",
        "raw_sha256": sha256_file(raw_output),
        "raw_description_sha256": sha256_file(description_output),
        "raw_assay_rows": len(assay_rows),
        "active_rows": int((outcomes == "Active").sum()),
        "inactive_rows": int((outcomes == "Inactive").sum()),
        "labeled_records": len(records),
        "records_sha256": sha256_file(records_path),
        "review_status": "review_required",
    }
    (output_dir / "materialization_audit.json").write_text(
        json.dumps(audit, indent=2, sort_keys=True) + "\n"
    )
    return audit


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-csv", type=Path, required=True)
    parser.add_argument("--description-json", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    print(
        json.dumps(
            materialize(args.raw_csv, args.description_json, args.output_dir),
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
