#!/usr/bin/env python3
"""Materialize source-stratified PPBR packets from Sutherland et al. 2023."""

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

ARTICLE_DOI = "10.1038/s41467-023-40064-9"
ARTICLE_TITLE = (
    "A preclinical secondary pharmacology resource illuminates "
    "target-adverse drug reaction associations of marketed drugs"
)
SHEET = "S Data 2"
SOURCES = {
    "NIBR curation": {
        "packet_id": "sutherland2023-ppbr-nibr-curation",
        "weight": 0.75,
        "quality": (
            "Authors describe this component as manual primary-literature PPB "
            "curation; exact, uncensored human PPB values only."
        ),
    },
    "Smit et al.": {
        "packet_id": "sutherland2023-ppbr-smit",
        "weight": 0.65,
        "quality": (
            "Peer-reviewed secondary source retained as its own component; "
            "exact, uncensored human PPB values only."
        ),
    },
    "DrugCentral": {
        "packet_id": "sutherland2023-ppbr-drugcentral",
        "weight": 0.50,
        "quality": (
            "Public curated drug-resource component, disclosed molecule by "
            "molecule in the peer-reviewed supplement."
        ),
    },
    "Pharmapendium": {
        "packet_id": "sutherland2023-ppbr-pharmapendium",
        "weight": 0.25,
        "quality": (
            "Heterogeneous commercial-source component republished in the "
            "peer-reviewed supplement; deliberately assigned the lowest weight."
        ),
    },
}


def _records_for_source(frame: pd.DataFrame, source_name: str) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    source_rows = frame.loc[frame["PPB source"].eq(source_name)].copy()
    source_rows["PPB %"] = pd.to_numeric(source_rows["PPB %"], errors="coerce")
    exact = source_rows.loc[
        source_rows["drugcentral smiles"].notna()
        & source_rows["PPB %"].notna()
        & source_rows["PPB qualifier"].eq("=")
        & source_rows["PPB %"].between(0.0, 100.0, inclusive="both")
    ].copy()
    if exact.empty:
        raise ValueError(f"no exact PPB records for {source_name}")
    exact["Y"] = exact["PPB %"].astype(float)
    exact["source_record_id"] = [
        f"PMC10356841:SData2:xlsx-row{int(index) + 4}:{source_name}"
        for index in exact.index
    ]
    records = exact.rename(
        columns={
            "drugcentral smiles": "Drug",
            "drugcentral name": "compound_name",
            "drugcentral struct id": "drugcentral_struct_id",
            "drugcentral inchikey": "reported_inchikey",
            "PPB %": "source_ppb_percent",
            "PPB source": "ppb_source",
            "PPB qualifier": "ppb_qualifier",
        }
    )[
        [
            "Drug",
            "Y",
            "source_record_id",
            "compound_name",
            "drugcentral_struct_id",
            "reported_inchikey",
            "source_ppb_percent",
            "ppb_source",
            "ppb_qualifier",
        ]
    ]
    if records["source_record_id"].duplicated().any():
        raise ValueError(f"duplicate source locator for {source_name}")
    counts = {
        "source_rows": len(source_rows),
        "exact_records": len(records),
        "censored_or_incomplete_rows_excluded": len(source_rows) - len(records),
    }
    return exact, records, counts


def materialize(workbook: Path, output_root: Path) -> list[dict]:
    workbook = workbook.resolve()
    frame = pd.read_excel(workbook, sheet_name=SHEET, header=2)
    required = {
        "drugcentral struct id",
        "drugcentral name",
        "drugcentral inchikey",
        "drugcentral smiles",
        "PPB source",
        "PPB qualifier",
        "PPB %",
    }
    if not required.issubset(frame):
        raise ValueError(f"supplement missing columns: {sorted(required - set(frame))}")

    output_root = output_root.resolve()
    shared_dir = output_root / "_shared_raw"
    shared_dir.mkdir(parents=True, exist_ok=True)
    shared_raw = shared_dir / "sutherland2023_supplementary_data_1_15.xlsx"
    if shared_raw.exists():
        if sha256_file(shared_raw) != sha256_file(workbook):
            raise ValueError("shared Sutherland workbook exists with a different hash")
    else:
        shutil.copyfile(workbook, shared_raw)

    outputs = []
    for source_name, config in SOURCES.items():
        exact, records, counts = _records_for_source(frame, source_name)
        packet_id = str(config["packet_id"])
        packet_dir = output_root / packet_id
        packet_dir.mkdir(parents=True, exist_ok=False)
        source_rows_path = packet_dir / "source_rows.csv"
        exact.to_csv(source_rows_path, index=True, index_label="supplement_row_index")
        records_path = packet_dir / "records.csv"
        records.to_csv(records_path, index=False)
        manifest = {
            "schema_version": SCHEMA_VERSION,
            "packet_id": packet_id,
            "task": "PPBR_AZ",
            "task_type": "regression",
            "source": {
                "kind": "independent_peer_reviewed_assay",
                "name": f"Sutherland 2023 human PPB component: {source_name}",
                "title": ARTICLE_TITLE,
                "citation_url": f"https://doi.org/{ARTICLE_DOI}",
                "citation_type": "doi",
                "stable_identifier": (
                    f"doi:{ARTICLE_DOI}; PMID:37468498; PMCID:PMC10356841; "
                    f"Supplementary Data 2; PPB source={source_name}"
                ),
                "license_or_access_terms": (
                    "Nature Communications article and molecule-level supplement "
                    "are distributed under CC BY 4.0."
                ),
                "independence_status": "verified_independent",
                "independence_evidence": (
                    "Novartis-authored drug curation is institutionally distinct "
                    "from the AstraZeneca PPBR benchmark. Exact and near benchmark "
                    "structures are removed by the sealed label-blind firewall."
                ),
                "peer_reviewed_publication_year": 2023,
                "raw_download_url": (
                    "https://www.ebi.ac.uk/europepmc/webservices/rest/"
                    "PMC10356841/supplementaryFiles"
                ),
                "raw_local_path": str(shared_raw.relative_to(ROOT)),
                "raw_sha256": sha256_file(shared_raw),
                "source_rows_sha256": sha256_file(source_rows_path),
                "component_quality_assessment": str(config["quality"]),
            },
            "assay": {
                "endpoint_definition": "Percentage of drug bound to human plasma proteins",
                "assay_type": (
                    "Human plasma protein binding measurements curated from "
                    f"{source_name}; experimental method may vary by originating study"
                ),
                "organism": "Homo sapiens",
                "biological_system": "Human plasma; literature/resource measurement compilation",
                "source_record_locator": (
                    f"PMC10356841 Supplementary Data 2 rows with PPB source exactly "
                    f"{source_name!r} and PPB qualifier '='"
                ),
            },
            "label": {
                "source_value_definition": "Reported exact human plasma protein binding percentage",
                "mapping_rule": "Y equals PPB % only when qualifier is '='; censored and missing values are excluded",
                "direction": "Higher Y denotes a higher percentage bound to plasma proteins",
                "transform": "identity; no imputation, midpoint conversion, or benchmark-label use",
                "unit": "percent bound",
            },
            "review": {
                "status": "review_required",
                "reviewer": "ADMET-EvO Evidence Packet data-steward review",
                "reviewed_at": datetime.now(timezone.utc).isoformat(),
                "endpoint_match": "direct",
                "native_priority_external_weight": float(config["weight"]),
                "notes": (
                    f"Source-stratified extraction. {config['quality']} Promotion "
                    "requires full structural firewall and lineage review."
                ),
            },
            "records": {
                "path": "records.csv",
                "row_count": len(records),
                "sha256": sha256_file(records_path),
            },
        }
        manifest_path = packet_dir / "manifest.json"
        manifest_path.write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n"
        )
        audit = {
            "packet_id": packet_id,
            "task": "PPBR_AZ",
            "ppb_source": source_name,
            "native_priority_external_weight": float(config["weight"]),
            **counts,
            "minimum_y": float(records["Y"].min()),
            "maximum_y": float(records["Y"].max()),
            "records_sha256": sha256_file(records_path),
            "source_rows_sha256": sha256_file(source_rows_path),
            "raw_sha256": sha256_file(shared_raw),
            "review_status": "review_required",
        }
        (packet_dir / "materialization_audit.json").write_text(
            json.dumps(audit, indent=2, sort_keys=True) + "\n"
        )
        outputs.append(audit)
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--supplement-xlsx", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    results = materialize(args.supplement_xlsx, args.output_root)
    print(json.dumps({"packet_count": len(results), "packets": results}, sort_keys=True))


if __name__ == "__main__":
    main()
