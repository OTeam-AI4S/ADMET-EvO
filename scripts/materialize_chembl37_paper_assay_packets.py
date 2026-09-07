#!/usr/bin/env python3
"""Materialize vetted ChEMBL-indexed original-paper assay packets.

The bulk ChEMBL endpoint cohorts are never used as training data.  Every spec
below was manually reviewed and binds one peer-reviewed paper to an explicit
assay panel.  Raw activity fields are re-read from frozen ChEMBL 37, and the
resulting packets must still pass the full Evidence Packet structure firewall.
"""

from __future__ import annotations

import argparse
import hashlib
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
class PacketSpec:
    packet_id: str
    task: str
    task_type: str
    doi: str
    assay_ids: tuple[int, ...]
    base_weight: float
    endpoint: str
    assay_type: str
    organism: str
    biological_system: str
    source_value_definition: str
    mapping_rule: str
    direction: str
    transform: str
    unit: str | None = None
    mode: str = "regression_exact"


SPECS = (
    PacketSpec(
        packet_id="chembl37-kim2021-ames-panel",
        task="AMES",
        task_type="classification",
        doi="10.1016/j.bmcl.2020.127662",
        assay_ids=(2024093, 2024094, 2024095, 2024096),
        base_weight=0.85,
        endpoint="Compound-level Ames bacterial mutagenicity",
        assay_type="Salmonella Typhimurium TA98/TA100 Ames tests with and without liver S9",
        organism="Salmonella enterica subsp. enterica serovar Typhimurium",
        biological_system="TA98 and TA100 bacterial reverse-mutation assays, +/- S9",
        source_value_definition="ChEMBL-curated paper activity_comment Toxic/Non-Toxic",
        mapping_rule="Y=1 if any reported TA98/TA100 +/-S9 assay is Toxic; Y=0 only if all reported assays are Non-Toxic",
        direction="Y=1 denotes mutagenic; Y=0 denotes non-mutagenic",
        transform="Exact text mapping followed by paper-level any-positive aggregation across four preregistered assay conditions",
        mode="ames_panel",
    ),
    PacketSpec(
        packet_id="chembl37-urea2013-ames-panel",
        task="AMES",
        task_type="classification",
        doi="10.1016/j.bmcl.2012.10.076",
        assay_ids=(886807, 886808, 886871, 886872),
        base_weight=0.85,
        endpoint="Compound-level Ames bacterial mutagenicity",
        assay_type="Salmonella Typhimurium TA98/TAMix Ames tests with and without S9",
        organism="Salmonella enterica subsp. enterica serovar Typhimurium",
        biological_system="TA98 and mixed-strain bacterial reverse-mutation assays, +/- S9",
        source_value_definition="ChEMBL-curated paper standard_text_value Toxic/Not toxic",
        mapping_rule="Y=1 if any reported TA98/TAMix +/-S9 assay is Toxic; Y=0 only if all reported assays are Not toxic",
        direction="Y=1 denotes mutagenic; Y=0 denotes non-mutagenic",
        transform="Exact text mapping followed by paper-level any-positive aggregation across four preregistered assay conditions",
        mode="ames_panel",
    ),
    PacketSpec(
        packet_id="chembl37-granberg2024-hlm-clint",
        task="Clearance_Microsome_AZ",
        task_type="regression",
        doi="10.1021/acs.jmedchem.3c02184",
        assay_ids=(2442728,),
        base_weight=0.80,
        endpoint="Intrinsic clearance in human liver microsomes",
        assay_type="Human liver microsomal metabolic-stability assay at 1 uM, LC-MS/MS",
        organism="Homo sapiens",
        biological_system="Human liver microsomes, measured up to 30 minutes",
        source_value_definition="Exact ChEMBL standard_value for CL in mL.min-1.g-1",
        mapping_rule="Keep finite relation='=' measurements only; exclude all interval-censored '<' and '>' rows",
        direction="Higher Y denotes faster intrinsic microsomal clearance",
        transform="Clip exact source values to the native TDC reporting range [3,150] after preserving source value",
        unit="mL.min-1.g-1",
        mode="clearance_exact",
    ),
    PacketSpec(
        packet_id="chembl37-lombardo2002-elogd74",
        task="Lipophilicity_AstraZeneca",
        task_type="regression",
        doi="10.1021/jm0100990",
        assay_ids=(16545,),
        base_weight=0.80,
        endpoint="Experimental octanol/water distribution coefficient at pH 7.4",
        assay_type="ElogD(oct) experimental lipophilicity assay",
        organism="Not applicable",
        biological_system="Octanol/aqueous buffer distribution at pH 7.4",
        source_value_definition="Exact ChEMBL LogD7.4 standard_value",
        mapping_rule="Keep finite relation='=' experimental LogD7.4 measurements only",
        direction="Higher Y denotes greater lipophilicity",
        transform="Identity transform on log10 distribution coefficient at pH 7.4",
        unit="log10(octanol/water distribution coefficient)",
    ),
    PacketSpec(
        packet_id="chembl37-granberg2024-kinetic-logs",
        task="Solubility_AqSolDB",
        task_type="regression",
        doi="10.1021/acs.jmedchem.3c02184",
        assay_ids=(2442720,),
        base_weight=0.70,
        endpoint="Kinetic aqueous solubility at pH 7.4",
        assay_type="24-hour phosphate-buffer kinetic-solubility assay by LC-MS/MS",
        organism="Not applicable",
        biological_system="Phosphate buffer at pH 7.4",
        source_value_definition="Exact ChEMBL Log S standard_value",
        mapping_rule="Keep finite relation='=' experimental Log S measurements only",
        direction="Higher Y denotes greater molar aqueous solubility",
        transform="Identity transform on source log10 mol/L",
        unit="log10 mol/L",
    ),
)

RAW_QUERY = """
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
order by ass.assay_id, act.activity_id
"""


def _canonical_smiles(value: object) -> str | None:
    molecule = Chem.MolFromSmiles(str(value))
    if molecule is None:
        return None
    return Chem.MolToSmiles(molecule, canonical=True, isomericSmiles=True)


def _label_text(row: pd.Series) -> str:
    values = [row.get("standard_text_value"), row.get("activity_comment")]
    return " | ".join(str(value).strip().casefold() for value in values if pd.notna(value))


def _ames_records(source: pd.DataFrame, spec: PacketSpec) -> tuple[pd.DataFrame, dict]:
    work = source.copy()
    work["source_label_text"] = work.apply(_label_text, axis=1)
    negative = work["source_label_text"].isin({"not toxic", "non-toxic"})
    positive = work["source_label_text"].eq("toxic")
    work = work.loc[negative | positive].copy()
    work["assay_y"] = positive.loc[work.index].astype(int)
    work["Drug"] = work["canonical_smiles"].map(_canonical_smiles)
    work = work.loc[work["Drug"].notna()].copy()
    grouped = []
    for smiles, group in work.groupby("Drug", sort=True):
        labels = sorted(set(group["assay_y"].astype(int)))
        y = max(labels)
        digest = hashlib.sha256(smiles.encode()).hexdigest()[:20]
        grouped.append(
            {
                "Drug": smiles,
                "Y": y,
                "source_record_id": f"CHEMBL37:{spec.packet_id}:structure:{digest}",
                "molecule_chembl_ids": "|".join(sorted(set(group["molecule_chembl_id"]))),
                "source_activity_ids": "|".join(str(value) for value in sorted(group["activity_id"])),
                "source_assay_ids": "|".join(str(value) for value in sorted(set(group["assay_id"]))),
                "source_assay_labels": "|".join(str(value) for value in sorted(group["assay_y"])),
                "aggregation_rule": "any assay-positive => Y=1; otherwise Y=0",
            }
        )
    records = pd.DataFrame(grouped)
    audit = {
        "raw_activity_rows": len(source),
        "exact_mappable_label_rows": len(work),
        "unmapped_label_rows": len(source) - len(work),
        "aggregated_compounds": len(records),
        "positive_compounds": int(records["Y"].sum()),
        "negative_compounds": int((records["Y"] == 0).sum()),
        "compounds_with_assay_label_conflict_resolved_any_positive": int(
            sum(
                group["assay_y"].nunique() > 1
                for _, group in work.groupby("Drug", sort=False)
            )
        ),
    }
    return records, audit


def _regression_records(source: pd.DataFrame, spec: PacketSpec) -> tuple[pd.DataFrame, dict]:
    work = source.copy()
    numeric = pd.to_numeric(work["standard_value"], errors="coerce")
    exact = work["standard_relation"].astype(str).str.strip().eq("=")
    finite = pd.Series(np.isfinite(numeric), index=work.index)
    work = work.loc[exact & finite].copy()
    work["source_value"] = numeric.loc[work.index].astype(float)
    work["Y"] = work["source_value"]
    if spec.mode == "clearance_exact":
        units = work["standard_units"].astype(str).str.strip().eq("mL.min-1.g-1")
        work = work.loc[units & work["source_value"].gt(0)].copy()
        work["Y"] = work["source_value"].clip(3.0, 150.0)
    work["Drug"] = work["canonical_smiles"].map(_canonical_smiles)
    work = work.loc[work["Drug"].notna()].copy()
    work["source_record_id"] = work["activity_id"].map(
        lambda value: f"CHEMBL37:activity:{int(value)}"
    )
    records = work.rename(columns={"description": "assay_description"})[
        [
            "Drug",
            "Y",
            "source_record_id",
            "source_value",
            "standard_relation",
            "standard_units",
            "activity_id",
            "assay_id",
            "assay_chembl_id",
            "molecule_chembl_id",
            "assay_description",
        ]
    ]
    audit = {
        "raw_activity_rows": len(source),
        "exact_finite_rows": int((exact & finite).sum()),
        "excluded_interval_censored_or_nonfinite_rows": int((~(exact & finite)).sum()),
        "modeling_records": len(records),
        "source_value_min": float(records["source_value"].min()),
        "source_value_max": float(records["source_value"].max()),
        "modeled_y_min": float(records["Y"].min()),
        "modeled_y_max": float(records["Y"].max()),
        "rows_clipped_to_native_range": int((records["Y"] != records["source_value"]).sum()),
    }
    return records, audit


def _read_source(connection: sqlite3.Connection, spec: PacketSpec) -> pd.DataFrame:
    placeholders = ",".join("?" for _ in spec.assay_ids)
    frame = pd.read_sql_query(
        RAW_QUERY.format(placeholders=placeholders),
        connection,
        params=spec.assay_ids,
    )
    if frame.empty:
        raise RuntimeError(f"no ChEMBL activities found for {spec.packet_id}")
    observed_assays = set(frame["assay_id"].astype(int))
    if observed_assays != set(spec.assay_ids):
        raise RuntimeError(
            f"{spec.packet_id} assay mismatch: {observed_assays} != {set(spec.assay_ids)}"
        )
    observed_dois = {str(value).strip().lower() for value in frame["doi"].dropna()}
    if observed_dois != {spec.doi}:
        raise RuntimeError(f"{spec.packet_id} DOI mismatch: {observed_dois}")
    return frame


def _materialize_one(
    connection: sqlite3.Connection,
    spec: PacketSpec,
    output_root: Path,
    archive_path: Path,
    archive_sha256: str,
) -> dict:
    packet_dir = output_root / spec.packet_id
    packet_dir.mkdir(parents=True, exist_ok=False)
    source = _read_source(connection, spec)
    source_rows_path = packet_dir / "source_rows.csv"
    source.to_csv(source_rows_path, index=False)
    if spec.mode == "ames_panel":
        records, audit = _ames_records(source, spec)
    else:
        records, audit = _regression_records(source, spec)
    if records.empty:
        raise RuntimeError(f"no modeling records for {spec.packet_id}")
    records_path = packet_dir / "records.csv"
    records.to_csv(records_path, index=False)
    title = str(source.iloc[0]["title"]).strip()
    years = sorted({int(value) for value in source["year"].dropna()})
    year = years[0] if len(years) == 1 else None
    assay_ids = ", ".join(str(value) for value in spec.assay_ids)
    now = datetime.now(timezone.utc).isoformat()
    label = {
        "source_value_definition": spec.source_value_definition,
        "mapping_rule": spec.mapping_rule,
        "direction": spec.direction,
        "transform": spec.transform,
    }
    if spec.unit is not None:
        label["unit"] = spec.unit
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "packet_id": spec.packet_id,
        "task": spec.task,
        "task_type": spec.task_type,
        "source": {
            "kind": "independent_peer_reviewed_assay",
            "name": "Original peer-reviewed assay records indexed in frozen ChEMBL 37",
            "title": title,
            "citation_url": f"https://doi.org/{spec.doi}",
            "citation_type": "doi",
            "stable_identifier": f"doi:{spec.doi}; ChEMBL37 assays:{assay_ids}",
            "license_or_access_terms": "ChEMBL data are available under CC BY-SA 3.0; original paper cited",
            "independence_status": "verified_independent",
            "independence_evidence": (
                "Original paper DOI differs from the canonical TDC benchmark source lineage; "
                "the packet is restricted to the explicitly listed direct assay IDs, and exact/near "
                "benchmark structures are removed by the downstream firewall."
            ),
            "peer_reviewed_publication_year": year,
            "raw_download_url": "https://ftp.ebi.ac.uk/pub/databases/chembl/ChEMBLdb/releases/chembl_37/",
            "raw_local_path": str(archive_path.relative_to(ROOT)),
            "raw_sha256": archive_sha256,
            "source_rows_sha256": sha256_file(source_rows_path),
            "data_access_note": "Activity rows re-extracted from frozen ChEMBL 37 SQLite; citation points to original paper",
        },
        "assay": {
            "endpoint_definition": spec.endpoint,
            "assay_type": spec.assay_type,
            "organism": spec.organism,
            "biological_system": spec.biological_system,
            "source_record_locator": f"ChEMBL 37 assay_id(s) {assay_ids}; bound activity_id rows in source_rows.csv",
        },
        "label": label,
        "review": {
            "status": "review_required",
            "reviewer": "ADMET-EvO Evidence Packet data-steward review",
            "reviewed_at": now,
            "endpoint_match": "direct",
            "native_priority_external_weight": spec.base_weight,
            "notes": (
                "Small original-paper direct-assay packet. The heterogeneous ChEMBL endpoint cohort "
                "remains quarantined. Promotion requires the full-development and label-blind test structure firewall."
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
    payload = {
        "protocol": "chembl37-original-paper-direct-assay-materialization-v1",
        "packet_id": spec.packet_id,
        "task": spec.task,
        "doi": spec.doi,
        "assay_ids": list(spec.assay_ids),
        **audit,
        "records_sha256": sha256_file(records_path),
        "source_rows_sha256": sha256_file(source_rows_path),
        "archive_sha256": archive_sha256,
        "review_status": "review_required",
        "test_labels_loaded": False,
    }
    (packet_dir / "materialization_audit.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n"
    )
    return payload


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
        "--output-root",
        type=Path,
        default=Path("data/evidence_packets_v4/packets"),
    )
    parser.add_argument("--packet-ids", nargs="*", default=[])
    args = parser.parse_args()
    extraction = json.loads(args.cohort_manifest.read_text())
    archive_sha256 = extraction["database_sha256"]
    selected = [
        spec for spec in SPECS if not args.packet_ids or spec.packet_id in args.packet_ids
    ]
    unknown = set(args.packet_ids) - {spec.packet_id for spec in SPECS}
    if unknown:
        raise ValueError(f"unknown packet ids: {sorted(unknown)}")
    connection = sqlite3.connect(f"file:{args.database.resolve()}?mode=ro", uri=True)
    connection.execute("pragma query_only=on")
    outputs = [
        _materialize_one(
            connection,
            spec,
            args.output_root.resolve(),
            args.archive.resolve(),
            archive_sha256,
        )
        for spec in selected
    ]
    connection.close()
    print(json.dumps({"packets": outputs}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
