#!/usr/bin/env python3
"""Materialize one reviewed Gonzalez-2021 NCATS CYP Evidence Packet candidate."""

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

ASSAYS = {
    1645840: {
        "task": "CYP2D6_Veith",
        "isoform": "CYP2D6",
        "target": "Cytochrome P450 family 2 subfamily D member 6",
    },
    1645841: {
        "task": "CYP3A4_Veith",
        "isoform": "CYP3A4",
        "target": "Cytochrome P450 family 3 subfamily A member 4",
    },
    1645842: {
        "task": "CYP2C9_Veith",
        "isoform": "CYP2C9",
        "target": "Cytochrome P450 family 2 subfamily C member 9",
    },
}


def materialize(raw_path: Path, output_dir: Path, aid: int) -> dict:
    config = ASSAYS[aid]
    raw_path = raw_path.resolve()
    raw = pd.read_csv(raw_path, low_memory=False)
    required = {
        "PUBCHEM_SID",
        "PUBCHEM_EXT_DATASOURCE_SMILES",
        "PUBCHEM_ACTIVITY_OUTCOME",
        "PUBCHEM_ACTIVITY_SCORE",
    }
    if not required.issubset(raw):
        raise ValueError(f"PubChem CSV missing columns: {sorted(required - set(raw))}")
    assay_rows = raw.loc[raw["PUBCHEM_SID"].notna()].copy()
    outcome = assay_rows["PUBCHEM_ACTIVITY_OUTCOME"].astype(str)
    labeled = assay_rows.loc[outcome.isin(["Active", "Inactive"])].copy()
    labeled["Y"] = (
        labeled["PUBCHEM_ACTIVITY_OUTCOME"].astype(str) == "Active"
    ).astype(int)
    labeled["source_record_id"] = labeled["PUBCHEM_SID"].map(
        lambda value: f"PubChem:AID{aid}:SID{int(value)}"
    )
    records = labeled.rename(
        columns={"PUBCHEM_EXT_DATASOURCE_SMILES": "Drug"}
    )[["Drug", "Y", "source_record_id"]]
    if records["source_record_id"].duplicated().any():
        raise ValueError("PubChem SID is not unique")

    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    raw_output = output_dir / f"raw_pubchem_aid{aid}.csv"
    if raw_path != raw_output:
        raw_output.write_bytes(raw_path.read_bytes())
    records_path = output_dir / "records.csv"
    records.to_csv(records_path, index=False)
    packet_id = f"gonzalez2021-{config['isoform'].lower()}-aid{aid}"
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "packet_id": packet_id,
        "task": config["task"],
        "task_type": "classification",
        "source": {
            "kind": "independent_peer_reviewed_assay",
            "name": f"Gonzalez et al. 2021 NCATS {config['isoform']} P450-Glo qHTS",
            "title": "Development of Robust Quantitative Structure-Activity Relationship Models for CYP2C9, CYP2D6, and CYP3A4 Catalysis and Inhibition",
            "citation_url": "https://doi.org/10.1124/dmd.120.000320",
            "citation_type": "doi",
            "stable_identifier": f"doi:10.1124/dmd.120.000320; PMID:34183376; PubChem:AID{aid}",
            "license_or_access_terms": "Factual PubChem BioAssay extraction with DOI/PubMed/PubChem citation; PubChem data-use terms apply",
            "independence_status": "verified_independent",
            "independence_evidence": "Distinct 2021 assay campaign and PubChem AID from the Veith 2009 benchmark campaign; same NCATS institutional lineage is disclosed. Modeling-ready promotion remains conditional on the full label-blind structure-firewall audit.",
            "peer_reviewed_publication_year": 2021,
            "raw_download_url": f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/assay/aid/{aid}/CSV",
            "raw_assay_result_rows": len(assay_rows),
            "raw_sha256": sha256_file(raw_output),
        },
        "assay": {
            "endpoint_definition": f"Direct inhibition of human {config['isoform']} enzyme activity",
            "assay_type": f"P450-Glo {config['isoform']} luminescent confirmatory qHTS",
            "organism": "Homo sapiens",
            "biological_system": f"recombinant human {config['isoform']} enzyme-substrate/NADPH P450-Glo system",
            "source_record_locator": f"PubChem AID {aid}; one record per PubChem SID; deposited 2021-03-04",
        },
        "label": {
            "source_value_definition": "PubChem assay activity outcome derived from replicate dose-response curve classes and activity score",
            "mapping_rule": f"Y=1 only when PUBCHEM_ACTIVITY_OUTCOME is Active; Y=0 only when outcome is Inactive; {int((outcome == 'Inconclusive').sum())} Inconclusive rows are excluded",
            "direction": f"Y=1 denotes {config['isoform']} antagonism/inhibition; Y=0 denotes inactive",
            "transform": "Deterministic Active/Inactive mapping; no potency imputation and no use of benchmark or test labels",
        },
        "review": {
            "status": "review_required",
            "reviewer": "ADMET-EvO Evidence Packet data-steward review",
            "reviewed_at": datetime.now(timezone.utc).isoformat(),
            "endpoint_match": "direct",
            "native_priority_external_weight": 0.75,
            "notes": "Materialized from the complete official PubChem assay. Full-cohort firewall and lineage audit are required before modeling_ready promotion.",
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
        "aid": aid,
        "task": config["task"],
        "raw_path": str(raw_output),
        "raw_sha256": sha256_file(raw_output),
        "raw_assay_rows": len(assay_rows),
        "active_rows": int((outcome == "Active").sum()),
        "inactive_rows": int((outcome == "Inactive").sum()),
        "inconclusive_rows": int((outcome == "Inconclusive").sum()),
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
    parser.add_argument("--aid", type=int, choices=sorted(ASSAYS), required=True)
    parser.add_argument("--raw-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(materialize(args.raw_csv, args.output_dir, args.aid), sort_keys=True))


if __name__ == "__main__":
    main()
