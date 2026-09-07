#!/usr/bin/env python3
"""Materialize per-paper Caco-2 packets from the PeerJ 1405 CC-BY supplement."""

from __future__ import annotations

import argparse
import json
import math
import re
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from admet_agent.evidence_packets import SCHEMA_VERSION, sha256_file

SOURCES = {
    "[Bergström, 2003 #46306]": ("bergstrom2003-caco2-peerj1405", "10.1021/jm020986i", "12570377", "Absorption classification of oral drugs based on molecular surface properties"),
    "[Corti, 2006 #11470]": ("corti2006-caco2-peerj1405", "10.1016/j.ejps.2005.11.005", "16364612", "Development and evaluation of an in vitro method for prediction of human drug absorption—II"),
    "[Peng, 2014 #50571]": ("peng2014-caco2-peerj1405", "10.1016/j.ejps.2014.02.008", "24576578", "Applications of a 7-day Caco-2 cell model in drug discovery and development"),
    "[Marino, 2005 #50602]": ("marino2005-caco2-peerj1405", "10.1016/j.ijpharm.2005.03.008", "15907606", "Validation of the 96 well Caco-2 cell culture model for high throughput permeability assessment of discovery compounds"),
    "[Avdeef, 2005 #48244]": ("avdeef2005-caco2-peerj1405", "10.1016/j.ejps.2004.11.011", "15734300", "Caco-2 permeability of weakly basic drugs predicted with the double-sink PAMPA pKa(flux) method"),
    "[Hayeshi, 2008 #17597]": ("hayeshi2008-caco2-peerj1405", "10.1016/j.ejps.2008.08.004", "18782614", "Comparison of drug transporter gene expression and functionality in Caco-2 cells from 10 different laboratories"),
    "[Wang, 2010 #24148]": ("wang2010-caco2-peerj1405", "10.1002/chir.20843", "19575464", "Stereoselective transport and uptake of propranolol across human intestinal Caco-2 cell monolayers"),
    "[Uchida, 2009 #50555]": ("uchida2009-caco2-peerj1405", "10.1016/j.vascn.2008.10.006", "19049886", "A modified fast (4 day) 96-well plate Caco-2 permeability assay"),
    "[Skolnik, 2010 #50606]": ("skolnik2010-caco2-peerj1405", "10.1002/jps.22080", "20166204", "Towards prediction of in vivo intestinal absorption using a 96-well Caco-2 assay"),
    "[Lin, 2011 #50607]": ("lin2011-caco2-peerj1405", "10.1124/dmd.110.034629", "21051535", "Attenuation of intestinal absorption by major efflux transporters: quantitative tools and strategies using a Caco-2 model"),
}


def materialize(supplement: Path, output_root: Path) -> list[dict]:
    supplement = supplement.resolve()
    source = pd.read_excel(supplement)
    required = {"Name", "SMILES", "Caco-2 Papp * 10^6 cm/s", "Reference number", "Endnote", "Data exist"}
    if not required.issubset(source):
        raise ValueError(f"supplement missing columns: {sorted(required - set(source))}")
    outputs = []
    for endnote, (packet_id, doi, pmid, title) in SOURCES.items():
        selected = source.loc[source["Data exist"].eq(1) & source["Endnote"].eq(endnote)].copy()
        selected["Caco-2 Papp * 10^6 cm/s"] = pd.to_numeric(selected["Caco-2 Papp * 10^6 cm/s"], errors="coerce")
        selected = selected.loc[selected["SMILES"].notna() & selected["Caco-2 Papp * 10^6 cm/s"].gt(0)].copy()
        if selected.empty:
            raise ValueError(f"no positive measured rows for {endnote}")
        selected["Y"] = selected["Caco-2 Papp * 10^6 cm/s"].map(lambda value: math.log10(float(value)) - 6.0)
        selected["source_record_id"] = [f"PeerJ1405:xls-row{index + 2}:{pmid}" for index in selected.index]
        records = selected.rename(columns={"SMILES": "Drug", "Caco-2 Papp * 10^6 cm/s": "source_papp_1e6_cm_s"})[["Drug", "Y", "source_record_id", "Name", "source_papp_1e6_cm_s", "Reference number", "Endnote"]]
        if records["source_record_id"].duplicated().any():
            raise ValueError(f"duplicate record locator for {endnote}")

        packet_dir = output_root.resolve() / packet_id
        packet_dir.mkdir(parents=True, exist_ok=False)
        raw_path = packet_dir / "raw_peerj1405_supplement.xls"
        shutil.copyfile(supplement, raw_path)
        extracted_path = packet_dir / "source_rows.csv"
        selected.to_csv(extracted_path, index=True, index_label="supplement_row_index")
        records_path = packet_dir / "records.csv"
        records.to_csv(records_path, index=False)
        manifest = {
            "schema_version": SCHEMA_VERSION,
            "packet_id": packet_id,
            "task": "Caco2_Wang",
            "task_type": "regression",
            "source": {
                "kind": "independent_peer_reviewed_assay",
                "name": f"{endnote.strip('[]')} Caco-2 measurements, transcribed in O'Hagan and Kell 2015",
                "title": title,
                "citation_url": f"https://doi.org/{doi}",
                "citation_type": "doi",
                "stable_identifier": f"doi:{doi}; PMID:{pmid}; transcription doi:10.7717/peerj.1405",
                "license_or_access_terms": "The molecule-level transcription is from the PeerJ 2015 CC BY 4.0 supplement; the original assay paper and PeerJ compilation are cited.",
                "independence_status": "verified_independent",
                "independence_evidence": "Publication-specific experimental Caco-2 cohort, not the Wang benchmark table. Later compilation reuse is controlled by full-development and label-blind test structure firewalls.",
                "peer_reviewed_publication_year": int(re.search(r"(19|20)\d{2}", packet_id).group()),
                "raw_download_url": "https://www.ebi.ac.uk/europepmc/webservices/rest/PMC4655101/supplementaryFiles",
                "transcription_citation": "doi:10.7717/peerj.1405; PMID:26618081; PMCID:PMC4655101",
                "raw_sha256": sha256_file(raw_path),
                "source_rows_sha256": sha256_file(extracted_path),
            },
            "assay": {
                "endpoint_definition": "Apical-to-basolateral apparent permeability across a human Caco-2 cell monolayer",
                "assay_type": "Publication-specific Caco-2 cell monolayer permeability measurement",
                "organism": "Homo sapiens",
                "biological_system": "Caco-2 human colon epithelial cell monolayer; paper-specific culture and plate conditions",
                "source_record_locator": f"PeerJ 1405 supplement rows with Endnote exactly {endnote}; original DOI {doi}",
            },
            "label": {
                "source_value_definition": "Reported Caco-2 Papp in 10^-6 cm/s",
                "mapping_rule": "For each positive measured Papp, Y=log10(Papp * 10^-6 cm/s)",
                "direction": "Higher Y denotes higher apparent permeability",
                "transform": "Y=log10(source_papp_1e6_cm_s)-6; no imputation or benchmark-label use",
                "unit": "log10(cm/s)",
            },
            "review": {
                "status": "review_required",
                "reviewer": "ADMET-EvO Evidence Packet data-steward review",
                "reviewed_at": datetime.now(timezone.utc).isoformat(),
                "endpoint_match": "direct",
                "native_priority_external_weight": 0.75,
                "notes": "Per-paper extraction from a CC-BY transcription. Promotion requires full structure firewall and source-lineage review.",
            },
            "records": {"path": "records.csv", "row_count": len(records), "sha256": sha256_file(records_path)},
        }
        (packet_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
        audit = {"packet_id": packet_id, "task": "Caco2_Wang", "endnote": endnote, "rows": len(records), "minimum_y": float(records["Y"].min()), "maximum_y": float(records["Y"].max()), "records_sha256": sha256_file(records_path), "source_rows_sha256": sha256_file(extracted_path), "raw_sha256": sha256_file(raw_path), "review_status": "review_required"}
        (packet_dir / "materialization_audit.json").write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n")
        outputs.append(audit)
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--supplement-xls", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    results = materialize(args.supplement_xls, args.output_root)
    print(json.dumps({"packet_count": len(results), "packets": results}, sort_keys=True))


if __name__ == "__main__":
    main()
