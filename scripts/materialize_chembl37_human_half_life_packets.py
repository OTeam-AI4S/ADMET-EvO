#!/usr/bin/env python3
"""Materialize three original human clinical-PK half-life paper packets."""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from admet_agent.evidence_packets import SCHEMA_VERSION, sha256_file


@dataclass(frozen=True)
class Spec:
    packet_id: str
    doi: str
    assay_ids: tuple[int, ...]
    reconciliation: str
    base_weight: float
    quality_note: str


SPECS = (
    Spec(
        "chembl37-s013420-2009-human-terminal-half-life",
        "10.1128/aac.00567-09",
        (656848,),
        "single_exact_record",
        0.85,
        "Original single-dose oral clinical PK study in healthy subjects",
    ),
    Spec(
        "chembl37-nemonoxacin-2010-human-terminal-half-life",
        "10.1128/aac.00682-09",
        (645486, 645487, 645488, 645489, 645490, 645491, 645492),
        "median_across_prespecified_single_ascending_doses",
        0.80,
        "Original randomized single-ascending-dose clinical PK study; one molecule reconciled across all seven dose cohorts",
    ),
    Spec(
        "chembl37-maribavir-2008-human-terminal-half-life",
        "10.1128/aac.00951-07",
        (668118,),
        "single_exact_control_arm_record",
        0.80,
        "Original healthy-adult clinical PK study; only the maribavir-alone control-arm record was retained",
    ),
)

QUERY = """
select act.activity_id, act.standard_type, act.standard_value,
       act.standard_units, act.standard_relation, act.standard_text_value,
       act.activity_comment, ass.assay_id, ass.chembl_id as assay_chembl_id,
       ass.description, ass.assay_organism, doc.doc_id, doc.doi,
       doc.pubmed_id, doc.title, doc.year, md.chembl_id as molecule_chembl_id,
       cs.canonical_smiles
from activities act
join assays ass on ass.assay_id=act.assay_id
join docs doc on doc.doc_id=act.doc_id
join molecule_dictionary md on md.molregno=act.molregno
join compound_structures cs on cs.molregno=act.molregno
where ass.assay_id in ({placeholders})
  and coalesce(act.potential_duplicate,0)=0
  and act.data_validity_comment is null
order by act.activity_id
"""


def _smiles(value: object) -> str | None:
    molecule = Chem.MolFromSmiles(str(value))
    if molecule is None:
        return None
    return Chem.MolToSmiles(molecule, canonical=True, isomericSmiles=True)


def _materialize(
    connection: sqlite3.Connection,
    spec: Spec,
    output_root: Path,
    archive: Path,
    archive_sha256: str,
) -> dict:
    packet_dir = output_root / spec.packet_id
    packet_dir.mkdir(parents=True, exist_ok=False)
    query = QUERY.format(placeholders=",".join("?" for _ in spec.assay_ids))
    source = pd.read_sql_query(query, connection, params=list(spec.assay_ids))
    if source.empty or set(source["assay_id"].astype(int)) != set(spec.assay_ids):
        raise RuntimeError(f"missing assay rows for {spec.packet_id}")
    if {str(value).strip().lower() for value in source["doi"].dropna()} != {
        spec.doi
    }:
        raise RuntimeError(f"DOI mismatch for {spec.packet_id}")
    source_rows_path = packet_dir / "source_rows.csv"
    source.to_csv(source_rows_path, index=False)

    numeric = pd.to_numeric(source["standard_value"], errors="coerce")
    exact = source["standard_relation"].astype(str).str.strip().eq("=")
    units = source["standard_units"].astype(str).str.strip().eq("hr")
    kind = source["standard_type"].astype(str).str.strip().str.casefold().eq("t1/2")
    organism = source["assay_organism"].astype(str).str.strip().eq("Homo sapiens")
    selected = source.loc[
        exact & units & kind & organism & np.isfinite(numeric) & numeric.gt(0)
    ].copy()
    selected["source_half_life_hr"] = numeric.loc[selected.index].astype(float)
    selected["Drug"] = selected["canonical_smiles"].map(_smiles)
    selected = selected.loc[selected["Drug"].notna()].copy()
    if selected.empty or selected["Drug"].nunique() != 1:
        raise RuntimeError(f"{spec.packet_id} must bind exactly one molecule")
    first_selected = selected.iloc[0]
    records = pd.DataFrame(
        {
            "Drug": [first_selected["Drug"]],
            "Y": [float(selected["source_half_life_hr"].median())],
            "source_record_id": [
                "CHEMBL37:activities:"
                + ",".join(str(int(value)) for value in selected["activity_id"])
            ],
            "source_half_life_hr_min": [
                float(selected["source_half_life_hr"].min())
            ],
            "source_half_life_hr_max": [
                float(selected["source_half_life_hr"].max())
            ],
            "source_measurement_count": [len(selected)],
            "assay_ids": [",".join(str(value) for value in spec.assay_ids)],
            "molecule_chembl_id": [first_selected["molecule_chembl_id"]],
        }
    )
    records_path = packet_dir / "records.csv"
    records.to_csv(records_path, index=False)
    now = datetime.now(timezone.utc).isoformat()
    first = source.iloc[0]
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "packet_id": spec.packet_id,
        "task": "Half_Life_Obach",
        "task_type": "regression",
        "source": {
            "kind": "independent_peer_reviewed_assay",
            "name": "Original peer-reviewed human clinical PK study indexed in frozen ChEMBL 37",
            "title": str(first["title"]).strip(),
            "citation_url": f"https://doi.org/{spec.doi}",
            "citation_type": "doi",
            "stable_identifier": (
                f"doi:{spec.doi}; ChEMBL37 assays:"
                + ",".join(str(value) for value in spec.assay_ids)
            ),
            "license_or_access_terms": "ChEMBL data are available under CC BY-SA 3.0; original paper cited",
            "independence_status": "verified_independent",
            "independence_evidence": "Original clinical-study DOI differs from the Obach benchmark source; only explicit human terminal/elimination half-life records are retained before structure decontamination.",
            "peer_reviewed_publication_year": int(first["year"]),
            "raw_download_url": "https://ftp.ebi.ac.uk/pub/databases/chembl/ChEMBLdb/releases/chembl_37/",
            "raw_local_path": str(archive.relative_to(ROOT)),
            "raw_sha256": archive_sha256,
            "source_rows_sha256": sha256_file(source_rows_path),
            "data_access_note": "Activity rows re-extracted from frozen ChEMBL 37 SQLite; citation points to the original clinical PK paper",
        },
        "assay": {
            "endpoint_definition": "Human terminal or elimination half-life in hours",
            "assay_type": "Original human clinical pharmacokinetic measurement",
            "organism": "Homo sapiens",
            "biological_system": "In vivo human plasma pharmacokinetics",
            "source_record_locator": "ChEMBL 37 activity_id rows and assay_id fields bound in source_rows.csv",
        },
        "label": {
            "source_value_definition": "Exact measured terminal/elimination half-life",
            "mapping_rule": "Retain exact positive T1/2 values in hours; reconcile repeated dose-cohort measurements by the within-paper median",
            "direction": "Larger Y denotes longer human elimination half-life",
            "transform": spec.reconciliation,
            "unit": "hr",
        },
        "review": {
            "status": "review_required",
            "reviewer": "ADMET-EvO Evidence Packet data-steward review",
            "reviewed_at": now,
            "endpoint_match": "direct",
            "native_priority_external_weight": spec.base_weight,
            "notes": f"{spec.quality_note}. Full structure firewall is mandatory before training.",
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
        "protocol": "chembl37-original-human-clinical-half-life-materialization-v1",
        "packet_id": spec.packet_id,
        "task": "Half_Life_Obach",
        "doi": spec.doi,
        "assay_ids": list(spec.assay_ids),
        "raw_rows": len(source),
        "exact_measurements": len(selected),
        "modeling_records": len(records),
        "reconciliation": spec.reconciliation,
        "modeling_value_hr": float(records["Y"].iloc[0]),
        "records_sha256": sha256_file(records_path),
        "source_rows_sha256": sha256_file(source_rows_path),
        "archive_sha256": archive_sha256,
        "review_status": "review_required",
        "test_labels_loaded": False,
    }
    (packet_dir / "materialization_audit.json").write_text(
        json.dumps(audit, indent=2, sort_keys=True) + "\n"
    )
    return audit


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    root = Path("data/external/chembl/chembl_37")
    parser.add_argument(
        "--database",
        type=Path,
        default=root / "chembl_37/chembl_37_sqlite/chembl_37.db",
    )
    parser.add_argument(
        "--archive", type=Path, default=root / "chembl_37_sqlite.tar.gz"
    )
    parser.add_argument(
        "--cohort-manifest", type=Path, default=root / "tdc22_cohorts_manifest.json"
    )
    parser.add_argument(
        "--output-root", type=Path, default=Path("data/evidence_packets_v4/packets")
    )
    args = parser.parse_args()
    archive_sha256 = json.loads(args.cohort_manifest.read_text())["database_sha256"]
    connection = sqlite3.connect(f"file:{args.database.resolve()}?mode=ro", uri=True)
    connection.execute("pragma query_only=on")
    outputs = [
        _materialize(
            connection,
            spec,
            args.output_root.resolve(),
            args.archive.resolve(),
            archive_sha256,
        )
        for spec in SPECS
    ]
    connection.close()
    print(json.dumps({"packets": outputs}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
