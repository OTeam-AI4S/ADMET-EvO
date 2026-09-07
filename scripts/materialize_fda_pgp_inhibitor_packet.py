#!/usr/bin/env python3
"""Materialize FDA clinical P-gp inhibitors as a positive Evidence Packet."""

from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import quote

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from admet_agent.evidence_packets import SCHEMA_VERSION, sha256_file

FDA_URL = (
    "https://www.fda.gov/drugs/drug-interactions-labeling/"
    "healthcare-professionals-fdas-examples-drugs-interact-cyp-enzymes-and-"
    "transporter-systems"
)
PUBCHEM_URL = (
    "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/{name}/property/"
    "ConnectivitySMILES,CanonicalSMILES,InChIKey/JSON"
)


def _clean_name(value: object) -> str:
    text = re.sub(r"\s+", " ", str(value)).strip()
    return re.sub(r"(?<=[A-Za-z])\d+$", "", text)


def _load_cache(path: Path) -> dict[str, dict]:
    if not path.exists():
        return {}
    return {
        row["name"]: row
        for row in (
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        )
    }


def _resolve(session: requests.Session, name: str, retries: int) -> dict:
    url = PUBCHEM_URL.format(name=quote(name, safe=""))
    last_status = None
    for attempt in range(retries + 1):
        response = session.get(url, timeout=30)
        last_status = response.status_code
        if response.status_code == 200:
            prop = response.json()["PropertyTable"]["Properties"][0]
            smiles = prop.get("ConnectivitySMILES") or prop.get("CanonicalSMILES")
            if smiles:
                return {
                    "name": name,
                    "status": "resolved",
                    "cid": prop.get("CID"),
                    "smiles": smiles,
                    "inchikey": prop.get("InChIKey"),
                    "url": url,
                }
            break
        if response.status_code not in {429, 500, 502, 503, 504}:
            break
        time.sleep(min(8.0, 0.5 * 2**attempt))
    return {
        "name": name,
        "status": "unresolved",
        "http_status": last_status,
        "url": url,
    }


def materialize(
    html: Path,
    output_root: Path,
    cache_path: Path,
    *,
    request_interval: float,
    retries: int,
) -> dict:
    table = pd.read_html(html)[0]
    mask = table["TRNSP INH"].fillna("").astype(str).str.contains(
        "P-gp", case=False, regex=False
    )
    selected = table.loc[
        mask, ["Drug or Other Substance", "TRNSP INH"]
    ].copy()
    selected["clean_name"] = selected["Drug or Other Substance"].map(_clean_name)
    combination = selected["clean_name"].str.contains(" and ", case=False, regex=False)
    excluded_combinations = selected.loc[combination, "clean_name"].tolist()
    selected = selected.loc[~combination].copy()

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache = _load_cache(cache_path)
    session = requests.Session()
    session.headers["User-Agent"] = (
        "ADMET-EvO academic FDA evidence curation; PubChem structure resolution"
    )
    with cache_path.open("a", encoding="utf-8") as handle:
        for name in sorted(set(selected["clean_name"]) - set(cache)):
            row = _resolve(session, name, retries)
            handle.write(json.dumps(row, sort_keys=True) + "\n")
            handle.flush()
            cache[name] = row
            time.sleep(request_interval)
    selected["resolution"] = selected["clean_name"].map(cache)
    selected["resolution_status"] = selected["resolution"].map(
        lambda row: row["status"]
    )
    resolved = selected.loc[selected["resolution_status"].eq("resolved")].copy()
    resolved["Drug"] = resolved["resolution"].map(lambda row: row["smiles"])
    resolved["PubChem_CID"] = resolved["resolution"].map(lambda row: row.get("cid"))
    resolved["PubChem_InChIKey"] = resolved["resolution"].map(
        lambda row: row.get("inchikey")
    )
    resolved["Y"] = 1
    resolved["source_record_id"] = "FDA-PGP-INH:" + resolved["clean_name"]
    records = resolved.rename(
        columns={
            "Drug or Other Substance": "fda_display_name",
            "TRNSP INH": "fda_transporter_class",
        }
    )[
        [
            "Drug",
            "Y",
            "source_record_id",
            "fda_display_name",
            "clean_name",
            "fda_transporter_class",
            "PubChem_CID",
            "PubChem_InChIKey",
        ]
    ]
    if records.empty:
        raise RuntimeError("no single-ingredient FDA P-gp inhibitor resolved")

    output_root = output_root.resolve()
    shared = output_root / "_shared_raw"
    shared.mkdir(parents=True, exist_ok=True)
    shared_html = shared / "fda_current_cyp_transporter_interactions.html"
    if not shared_html.exists():
        shutil.copy2(html, shared_html)
    elif sha256_file(shared_html) != sha256_file(html):
        raise RuntimeError("existing shared FDA interaction page differs")
    shared_cache = shared / "fda_pgp_inhibitor_pubchem_resolution.jsonl"
    shutil.copy2(cache_path, shared_cache)

    packet_id = "fda-current-clinical-pgp-inhibitors"
    packet_dir = output_root / packet_id
    packet_dir.mkdir(parents=True, exist_ok=False)
    source_rows_path = packet_dir / "source_rows.csv"
    selected.drop(columns=["resolution"]).to_csv(source_rows_path, index=False)
    records_path = packet_dir / "records.csv"
    records.to_csv(records_path, index=False)
    now = datetime.now(timezone.utc).isoformat()
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "packet_id": packet_id,
        "task": "Pgp_Broccatelli",
        "task_type": "classification",
        "source": {
            "kind": "fda_regulatory_dataset",
            "name": "FDA examples of drugs interacting with CYP enzymes and transporter systems",
            "title": "P-gp Clinical Inhibitors",
            "citation_url": FDA_URL,
            "citation_type": "fda_document",
            "stable_identifier": "FDA:CYP-transporter-table:P-gp-inhibitors",
            "license_or_access_terms": "Public FDA regulatory information; facts used with source attribution",
            "independence_status": "verified_independent",
            "independence_evidence": (
                "FDA clinical transporter evidence is independent of the Broccatelli literature compilation; exact and near structural overlap is removed before training."
            ),
            "raw_download_url": FDA_URL,
            "raw_local_path": str(shared_html.relative_to(ROOT)),
            "raw_sha256": sha256_file(shared_html),
            "structure_resolution_source": "PubChem PUG REST by FDA drug name",
            "structure_resolution_url": "https://pubchem.ncbi.nlm.nih.gov/docs/pug-rest",
            "structure_resolution_sha256": sha256_file(shared_cache),
            "source_rows_sha256": sha256_file(source_rows_path),
        },
        "assay": {
            "endpoint_definition": "P-glycoprotein inhibition",
            "assay_type": "FDA clinical transporter interaction evidence",
            "organism": "Homo sapiens",
            "biological_system": "Clinical P-gp-mediated drug transport",
            "source_record_locator": "FDA Table 1, TRNSP INH entries containing P-gp inhibitor",
        },
        "label": {
            "source_value_definition": "FDA P-gp inhibitor category",
            "mapping_rule": "Y=1 only when FDA lists the single-ingredient drug as a P-gp inhibitor",
            "direction": "Y=1 denotes confirmed P-gp inhibition",
            "transform": "positive-only direct regulatory evidence; absence is not treated as negative",
        },
        "review": {
            "status": "review_required",
            "reviewer": "ADMET-EvO Evidence Packet data-steward review",
            "reviewed_at": now,
            "endpoint_match": "direct",
            "native_priority_external_weight": 0.85,
            "notes": (
                "FDA definition requires in-vitro P-gp inhibition evidence plus clinically meaningful substrate AUC increase. Combination products are excluded. Promotion requires the full-development and label-blind test structure firewall."
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
        "protocol": "admet-fda-clinical-pgp-inhibitor-materialization-v1",
        "packet_id": packet_id,
        "fda_pgp_inhibitor_rows": int(mask.sum()),
        "excluded_combination_products": excluded_combinations,
        "single_ingredient_rows": len(selected),
        "resolved_records": len(records),
        "unresolved_records": len(selected) - len(records),
        "native_priority_external_weight": 0.85,
        "raw_sha256": sha256_file(shared_html),
        "structure_resolution_sha256": sha256_file(shared_cache),
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
    parser.add_argument("--html", type=Path, required=True)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=ROOT / "data/evidence_packets_v4/packets",
    )
    parser.add_argument(
        "--cache",
        type=Path,
        default=ROOT / "data/external/fda/pgp/pubchem_resolution.jsonl",
    )
    parser.add_argument("--request-interval", type=float, default=0.20)
    parser.add_argument("--retries", type=int, default=3)
    args = parser.parse_args()
    print(
        json.dumps(
            materialize(
                args.html,
                args.output_root,
                args.cache,
                request_interval=args.request_interval,
                retries=args.retries,
            ),
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
