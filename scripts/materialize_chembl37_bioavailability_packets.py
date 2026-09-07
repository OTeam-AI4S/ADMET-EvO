#!/usr/bin/env python3
"""Materialize four direct oral-bioavailability paper assay packets."""

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
    assay_id: int
    base_weight: float
    quality_note: str


SPECS = (
    Spec(
        "chembl37-nr2b2019-rat-oral-f",
        "10.1016/j.bmcl.2019.02.017",
        1821345,
        0.70,
        "Sprague-Dawley rat cassette dosing at 1 mg/kg",
    ),
    Spec(
        "chembl37-schisto2021-mouse-oral-f",
        "10.1371/journal.pntd.0009490",
        1998362,
        0.60,
        "Five-compound cassette in three Swiss albino male mice; lower base weight reflects sparse cassette design",
    ),
    Spec(
        "chembl37-prmt5-2021-rat-oral-f",
        "10.1021/acs.jmedchem.0c02083",
        2100045,
        0.75,
        "Direct rat oral bioavailability measurements",
    ),
    Spec(
        "chembl37-arginase2021-mouse-oral-f",
        "10.1021/acsmedchemlett.1c00195",
        2130270,
        0.75,
        "C57BL/6 mouse oral bioavailability at 10 mg/kg",
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
where ass.assay_id=?
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
    source = pd.read_sql_query(QUERY, connection, params=[spec.assay_id])
    if source.empty or set(source["assay_id"].astype(int)) != {spec.assay_id}:
        raise RuntimeError(f"missing assay {spec.assay_id}")
    if {str(value).strip().lower() for value in source["doi"].dropna()} != {
        spec.doi
    }:
        raise RuntimeError(f"DOI mismatch for {spec.packet_id}")
    source_rows_path = packet_dir / "source_rows.csv"
    source.to_csv(source_rows_path, index=False)

    numeric = pd.to_numeric(source["standard_value"], errors="coerce")
    exact = source["standard_relation"].astype(str).str.strip().eq("=")
    units = source["standard_units"].astype(str).str.strip().eq("%")
    kind = source["standard_type"].astype(str).str.strip().str.casefold().eq("f")
    physical = numeric.between(0.0, 100.0, inclusive="both")
    selected = source.loc[exact & units & kind & physical & np.isfinite(numeric)].copy()
    selected["source_f_percent"] = numeric.loc[selected.index].astype(float)
    selected["Y"] = (selected["source_f_percent"] >= 20.0).astype(int)
    selected["Drug"] = selected["canonical_smiles"].map(_smiles)
    selected = selected.loc[selected["Drug"].notna()].copy()
    selected["source_record_id"] = selected["activity_id"].map(
        lambda value: f"CHEMBL37:activity:{int(value)}"
    )
    records = selected.rename(columns={"description": "assay_description"})[
        [
            "Drug",
            "Y",
            "source_record_id",
            "source_f_percent",
            "activity_id",
            "assay_id",
            "assay_chembl_id",
            "molecule_chembl_id",
            "assay_organism",
            "assay_description",
        ]
    ]
    if records.empty or set(records["Y"]) != {0, 1}:
        raise RuntimeError(f"{spec.packet_id} must retain both F classes")
    if records["Drug"].duplicated().any():
        raise RuntimeError(f"{spec.packet_id} has duplicate paper structures")
    records_path = packet_dir / "records.csv"
    records.to_csv(records_path, index=False)
    now = datetime.now(timezone.utc).isoformat()
    first = source.iloc[0]
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "packet_id": spec.packet_id,
        "task": "Bioavailability_Ma",
        "task_type": "classification",
        "source": {
            "kind": "independent_peer_reviewed_assay",
            "name": "Original peer-reviewed oral-bioavailability assay indexed in frozen ChEMBL 37",
            "title": str(first["title"]).strip(),
            "citation_url": f"https://doi.org/{spec.doi}",
            "citation_type": "doi",
            "stable_identifier": f"doi:{spec.doi}; ChEMBL37 assay:{spec.assay_id}",
            "license_or_access_terms": "ChEMBL data are available under CC BY-SA 3.0; original paper cited",
            "independence_status": "verified_independent",
            "independence_evidence": (
                "Original paper DOI differs from the Ma benchmark source; only its explicit direct oral-F assay is retained, followed by exact/near structure decontamination."
            ),
            "peer_reviewed_publication_year": int(first["year"]),
            "raw_download_url": "https://ftp.ebi.ac.uk/pub/databases/chembl/ChEMBLdb/releases/chembl_37/",
            "raw_local_path": str(archive.relative_to(ROOT)),
            "raw_sha256": archive_sha256,
            "source_rows_sha256": sha256_file(source_rows_path),
            "data_access_note": "Activity rows re-extracted from frozen ChEMBL 37 SQLite; citation points to original paper",
        },
        "assay": {
            "endpoint_definition": "Oral bioavailability as measured F percent",
            "assay_type": str(first["description"]).strip(),
            "organism": str(first["assay_organism"]).strip(),
            "biological_system": "In vivo oral and intravenous pharmacokinetic comparison",
            "source_record_locator": f"ChEMBL 37 assay_id {spec.assay_id}; bound activity_id rows in source_rows.csv",
        },
        "label": {
            "source_value_definition": "Exact measured oral bioavailability F percent",
            "mapping_rule": "Y=1 when 20<=F<=100%; Y=0 when 0<=F<20%; nonnumeric, interval-censored, and F>100% rows excluded",
            "direction": "Y=1 denotes bioavailability at least 20%; Y=0 denotes lower oral bioavailability",
            "transform": "Prerecorded 20% binary cutoff matching the TDC cohort extraction; no imputation",
        },
        "review": {
            "status": "review_required",
            "reviewer": "ADMET-EvO Evidence Packet data-steward review",
            "reviewed_at": now,
            "endpoint_match": "direct",
            "native_priority_external_weight": spec.base_weight,
            "notes": f"{spec.quality_note}. One original paper and one direct assay; full structure firewall required before training.",
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
        "protocol": "chembl37-original-paper-oral-bioavailability-materialization-v1",
        "packet_id": spec.packet_id,
        "task": "Bioavailability_Ma",
        "doi": spec.doi,
        "assay_id": spec.assay_id,
        "raw_rows": len(source),
        "modeling_records": len(records),
        "positive_records": int(records["Y"].sum()),
        "negative_records": int((records["Y"] == 0).sum()),
        "excluded_nonnumeric_or_nonexact": int((~(exact & np.isfinite(numeric))).sum()),
        "excluded_outside_physical_percent_range": int(
            (exact & np.isfinite(numeric) & ~physical).sum()
        ),
        "minimum_f_percent": float(records["source_f_percent"].min()),
        "maximum_f_percent": float(records["source_f_percent"].max()),
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
