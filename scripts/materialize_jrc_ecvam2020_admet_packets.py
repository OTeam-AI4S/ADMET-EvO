#!/usr/bin/env python3
"""Materialize uniform JRC human PPB and hepatocyte-clearance packets."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from rdkit import Chem

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from admet_agent.evidence_packets import SCHEMA_VERSION, sha256_file

DATASET_DOI = "10.2905/JRC.SR09628"
DATASET_TITLE = (
    "EURL ECVAM in vitro hepatocyte clearance and blood plasma protein "
    "binding dataset for 77 chemicals"
)
RAW_URL = (
    "https://jeodpp.jrc.ec.europa.eu/ftp/jrc-opendata/EURL-ECVAM/"
    "datasets/InVitroHepatocyte/"
    "EURLECVAM_JRC_insilico_HEPCL_ProteinBinding.zip"
)


def _first_valid_smiles(value: object) -> str | None:
    for candidate in str(value).splitlines():
        candidate = candidate.strip()
        molecule = Chem.MolFromSmiles(candidate)
        if candidate and molecule is not None:
            return Chem.MolToSmiles(molecule, canonical=True, isomericSmiles=True)
    return None


def _structure_table(workbook: Path) -> pd.DataFrame:
    frame = pd.read_excel(
        workbook, sheet_name="Chemical_List_LogKow_PERCEPTA"
    )
    frame = frame.rename(
        columns={
            frame.columns[0]: "seq_no",
            frame.columns[1]: "structure_compound_name",
            frame.columns[2]: "cas_number",
            frame.columns[4]: "reported_smiles",
        }
    )
    frame["seq_no"] = pd.to_numeric(frame["seq_no"], errors="coerce")
    frame["Drug"] = frame["reported_smiles"].map(_first_valid_smiles)
    frame = frame.loc[frame["seq_no"].notna() & frame["Drug"].notna()].copy()
    frame["seq_no"] = frame["seq_no"].astype(int)
    if frame["seq_no"].duplicated().any():
        raise ValueError("JRC structure sheet contains duplicate sequence numbers")
    return frame[
        ["seq_no", "Drug", "structure_compound_name", "cas_number", "reported_smiles"]
    ]


def _clearance_records(
    workbook: Path, structures: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    source = pd.read_excel(workbook, sheet_name="HepaStab successes only")
    source.columns = [str(column).strip() for column in source.columns]
    source["seq_no"] = pd.to_numeric(
        source.iloc[:, 0].astype(str).str.extract(r"(\d+)")[0], errors="coerce"
    )
    label_col = "CL int cells (µL/min/106 cells)"
    source[label_col] = pd.to_numeric(source[label_col], errors="coerce")
    selected = source.loc[source["seq_no"].notna() & source[label_col].gt(0)].copy()
    selected["seq_no"] = selected["seq_no"].astype(int)
    selected = selected.merge(structures, on="seq_no", how="inner", validate="many_to_one")
    selected["source_clint_ul_min_1e6_cells"] = selected[label_col].astype(float)
    selected["Y"] = selected["source_clint_ul_min_1e6_cells"].clip(3.0, 150.0)
    selected["source_record_id"] = [
        f"JRC-SR09628:HepaStab:xlsx-row{int(index) + 2}:seq{seq}"
        for index, seq in zip(selected.index, selected["seq_no"])
    ]
    records = selected.rename(
        columns={
            "Compound Name": "compound_name",
            "Study conc\n(µM)": "study_concentration_uM",
        }
    )[
        [
            "Drug",
            "Y",
            "source_record_id",
            "seq_no",
            "compound_name",
            "cas_number",
            "study_concentration_uM",
            "source_clint_ul_min_1e6_cells",
        ]
    ]
    audit = {
        "source_sheet_rows": len(source),
        "positive_numeric_rows": int(source[label_col].gt(0).sum()),
        "records_with_structure": len(records),
        "unique_sequence_numbers": int(records["seq_no"].nunique()),
        "rows_clipped_to_native_floor_3": int(
            (records["source_clint_ul_min_1e6_cells"] < 3.0).sum()
        ),
        "rows_clipped_to_native_ceiling_150": int(
            (records["source_clint_ul_min_1e6_cells"] > 150.0).sum()
        ),
    }
    return selected, records, audit


def _ppbr_records(
    workbook: Path, structures: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    source = pd.read_excel(workbook, sheet_name="ProtBind successes only")
    source.columns = [str(column).strip() for column in source.columns]
    source["seq_no"] = pd.to_numeric(source.iloc[:, 0], errors="coerce")
    for column in ("% Bound in Plasma", "SD.1", "% Recovery"):
        source[column] = pd.to_numeric(source[column], errors="coerce")
    quality = (
        source["seq_no"].notna()
        & source["% Bound in Plasma"].between(0.0, 100.0, inclusive="both")
        & source["% Recovery"].between(70.0, 130.0, inclusive="both")
        & source["SD.1"].le(20.0)
    )
    selected = source.loc[quality].copy()
    selected["seq_no"] = selected["seq_no"].astype(int)
    selected = selected.merge(structures, on="seq_no", how="inner", validate="one_to_one")
    selected["Y"] = selected["% Bound in Plasma"].astype(float)
    selected["source_record_id"] = selected["seq_no"].map(
        lambda seq: f"JRC-SR09628:ProtBind:seq{seq}"
    )
    records = selected.rename(
        columns={
            "Compound_ID": "compound_name",
            "% Bound in Plasma": "source_ppb_percent_bound",
            "SD.1": "source_ppb_sd",
            "% Recovery": "source_recovery_percent",
        }
    )[
        [
            "Drug",
            "Y",
            "source_record_id",
            "seq_no",
            "compound_name",
            "cas_number",
            "source_ppb_percent_bound",
            "source_ppb_sd",
            "source_recovery_percent",
        ]
    ]
    audit = {
        "source_sheet_rows": len(source),
        "physically_bounded_rows": int(
            source["% Bound in Plasma"].between(0.0, 100.0, inclusive="both").sum()
        ),
        "strict_quality_rows": int(quality.sum()),
        "records_with_structure": len(records),
        "quality_rule": "0<=percent_bound<=100; 70<=recovery<=130; SD<=20",
    }
    return selected, records, audit


def _write_packet(
    *,
    output_root: Path,
    shared_raw: Path,
    packet_id: str,
    task: str,
    selected: pd.DataFrame,
    records: pd.DataFrame,
    extraction_audit: dict,
    assay: dict,
    label: dict,
) -> dict:
    packet_dir = output_root / packet_id
    packet_dir.mkdir(parents=True, exist_ok=False)
    source_rows_path = packet_dir / "source_rows.csv"
    selected.to_csv(source_rows_path, index=False)
    records_path = packet_dir / "records.csv"
    records.to_csv(records_path, index=False)
    now = datetime.now(timezone.utc).isoformat()
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "packet_id": packet_id,
        "task": task,
        "task_type": "regression",
        "source": {
            "kind": "regulatory_scientific_assay_report",
            "name": "European Commission JRC / EURL ECVAM human ADME assay",
            "title": DATASET_TITLE,
            "citation_url": f"https://doi.org/{DATASET_DOI}",
            "citation_type": "doi",
            "stable_identifier": f"doi:{DATASET_DOI}; JRC dataset SR09628",
            "license_or_access_terms": (
                "European Commission reuse notice; anonymous public access; "
                "source acknowledgement required"
            ),
            "independence_status": "verified_independent",
            "independence_evidence": (
                "Uniform assay commissioned by the European Commission JRC and "
                "performed by GVK Biosciences, independent of AstraZeneca."
            ),
            "peer_reviewed_publication_year": 2020,
            "raw_download_url": RAW_URL,
            "raw_local_path": str(shared_raw.relative_to(ROOT)),
            "raw_sha256": sha256_file(shared_raw),
            "source_rows_sha256": sha256_file(source_rows_path),
        },
        "assay": assay,
        "label": label,
        "review": {
            "status": "review_required",
            "reviewer": "ADMET-EvO Evidence Packet data-steward review",
            "reviewed_at": now,
            "endpoint_match": "direct",
            "native_priority_external_weight": 0.75,
            "notes": (
                "Uniform public regulatory-science assay. Promotion requires the "
                "full-development and label-blind test structural firewall."
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
        "packet_id": packet_id,
        "task": task,
        **extraction_audit,
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
    return audit


def materialize(workbook: Path, output_root: Path) -> list[dict]:
    workbook = workbook.resolve()
    output_root = output_root.resolve()
    structures = _structure_table(workbook)
    shared_dir = output_root / "_shared_raw"
    shared_dir.mkdir(parents=True, exist_ok=True)
    shared_raw = shared_dir / "jrc_ecvam_sr09628_admet_77_summary.xlsx"
    if shared_raw.exists():
        if sha256_file(shared_raw) != sha256_file(workbook):
            raise ValueError("shared JRC workbook exists with a different hash")
    else:
        shutil.copyfile(workbook, shared_raw)

    clearance_source, clearance_records, clearance_audit = _clearance_records(
        workbook, structures
    )
    ppbr_source, ppbr_records, ppbr_audit = _ppbr_records(workbook, structures)
    common_system = "Homo sapiens"
    return [
        _write_packet(
            output_root=output_root,
            shared_raw=shared_raw,
            packet_id="jrc-ecvam2020-clearance-hepatocyte",
            task="Clearance_Hepatocyte_AZ",
            selected=clearance_source,
            records=clearance_records,
            extraction_audit=clearance_audit,
            assay={
                "endpoint_definition": "Intrinsic metabolic clearance in pooled human hepatocytes",
                "assay_type": "Substrate-depletion hepatocyte stability assay with LC-MS/MS bioanalysis",
                "organism": common_system,
                "biological_system": "Pooled human hepatocytes, 1 million cells/mL, Krebs-Henseleit buffer",
                "source_record_locator": "JRC SR09628 workbook, HepaStab successes only sheet",
            },
            label={
                "source_value_definition": "CLint in µL/min/10^6 cells at the reported study concentration",
                "mapping_rule": "positive numeric CLint retained; Y clipped to the TDC-native reporting range [3,150]",
                "direction": "Higher Y denotes faster intrinsic hepatocyte clearance",
                "transform": "Y=min(150,max(3,source CLint)); replicate concentrations collapse by median after firewall",
                "unit": "µL/min/10^6 cells",
            },
        ),
        _write_packet(
            output_root=output_root,
            shared_raw=shared_raw,
            packet_id="jrc-ecvam2020-ppbr",
            task="PPBR_AZ",
            selected=ppbr_source,
            records=ppbr_records,
            extraction_audit=ppbr_audit,
            assay={
                "endpoint_definition": "Percentage bound to human plasma proteins",
                "assay_type": "Five-hour equilibrium dialysis with LC-MS/MS bioanalysis and three replicates",
                "organism": common_system,
                "biological_system": "Human K2EDTA plasma and phosphate-buffered saline at pH 7.4",
                "source_record_locator": "JRC SR09628 workbook, ProtBind successes only sheet; strict recovery/SD filter",
            },
            label={
                "source_value_definition": "Measured percentage bound in human plasma",
                "mapping_rule": "Y=percent bound only for physical values with 70-130% recovery and SD<=20",
                "direction": "Higher Y denotes a higher fraction bound to plasma proteins",
                "transform": "identity; no imputation or benchmark-label use",
                "unit": "percent bound",
            },
        ),
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary-xlsx", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    results = materialize(args.summary_xlsx, args.output_root)
    print(json.dumps({"packet_count": len(results), "packets": results}, sort_keys=True))


if __name__ == "__main__":
    main()
