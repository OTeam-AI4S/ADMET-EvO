#!/usr/bin/env python3
"""Materialize FDA clinical CYP substrate tables as tiered Evidence Packets."""

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
TASKS = {
    "2C9": "CYP2C9_Substrate_CarbonMangels",
    "2D6": "CYP2D6_Substrate_CarbonMangels",
    "3A": "CYP3A4_Substrate_CarbonMangels",
}
TIERS = {
    "sensitive": ("CYP SENS SUB", 0.90),
    "moderate-sensitive": ("CYP Mod SENS SUB", 0.75),
}


def _clean_name(value: object) -> str:
    text = re.sub(r"\s+", " ", str(value)).strip()
    return re.sub(r"(?<=[A-Za-z])\d+$", "", text)


def _load_cache(path: Path) -> dict[str, dict]:
    if not path.exists():
        return {}
    output = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            output[row["name"]] = row
    return output


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


def _selected_rows(html: Path) -> dict[tuple[str, str], pd.DataFrame]:
    tables = pd.read_html(html)
    if not tables or "Drug or Other Substance" not in tables[0].columns:
        raise RuntimeError("FDA CYP table was not found in the supplied HTML")
    table = tables[0].copy()
    table["clean_name"] = table["Drug or Other Substance"].map(_clean_name)
    output = {}
    for isoform in TASKS:
        pattern = rf"(?<![0-9A-Z]){re.escape(isoform)}(?![0-9A-Z])"
        for tier, (column, _) in TIERS.items():
            values = table[column].fillna("").astype(str)
            mask = values.str.contains(pattern, case=False, regex=True) & values.str.contains(
                "substrate", case=False, regex=False
            )
            selected = table.loc[
                mask,
                ["Drug or Other Substance", "clean_name", column],
            ].copy()
            selected = selected.rename(columns={column: "fda_substrate_class"})
            selected["isoform"] = isoform
            selected["evidence_tier"] = tier
            if selected["clean_name"].duplicated().any():
                raise RuntimeError(f"duplicate FDA drug name for {isoform}/{tier}")
            output[(isoform, tier)] = selected
    return output


def materialize(
    html: Path,
    output_root: Path,
    cache_path: Path,
    *,
    request_interval: float,
    retries: int,
) -> list[dict]:
    cohorts = _selected_rows(html)
    all_names = sorted(
        {name for frame in cohorts.values() for name in frame["clean_name"]}
    )
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache = _load_cache(cache_path)
    session = requests.Session()
    session.headers["User-Agent"] = (
        "ADMET-EvO academic FDA evidence curation; PubChem structure resolution"
    )
    with cache_path.open("a", encoding="utf-8") as handle:
        for index, name in enumerate(
            [item for item in all_names if item not in cache], start=1
        ):
            row = _resolve(session, name, retries)
            handle.write(json.dumps(row, sort_keys=True) + "\n")
            handle.flush()
            cache[name] = row
            if index % 20 == 0:
                print(json.dumps({"resolved_requests": index}), flush=True)
            time.sleep(request_interval)

    output_root = output_root.resolve()
    shared = output_root / "_shared_raw"
    shared.mkdir(parents=True, exist_ok=True)
    shared_html = shared / "fda_current_cyp_transporter_interactions.html"
    if shared_html.exists():
        if sha256_file(shared_html) != sha256_file(html):
            raise RuntimeError("existing shared FDA CYP page has a different checksum")
    else:
        shutil.copy2(html, shared_html)
    shared_cache = shared / "fda_cyp_substrate_pubchem_resolution.jsonl"
    if shared_cache.exists():
        if sha256_file(shared_cache) != sha256_file(cache_path):
            raise RuntimeError("existing shared FDA CYP structure cache differs")
    else:
        shutil.copy2(cache_path, shared_cache)

    audits = []
    now = datetime.now(timezone.utc).isoformat()
    for (isoform, tier), source_rows in cohorts.items():
        base_weight = TIERS[tier][1]
        packet_id = f"fda-cyp-ddi-current-{isoform.lower()}-{tier}"
        packet_dir = output_root / packet_id
        packet_dir.mkdir(parents=True, exist_ok=False)
        source_rows = source_rows.copy()
        source_rows["resolution"] = source_rows["clean_name"].map(cache)
        source_rows["resolution_status"] = source_rows["resolution"].map(
            lambda row: row["status"]
        )
        resolved = source_rows.loc[
            source_rows["resolution_status"].eq("resolved")
        ].copy()
        resolved["Drug"] = resolved["resolution"].map(lambda row: row["smiles"])
        resolved["PubChem_CID"] = resolved["resolution"].map(
            lambda row: row.get("cid")
        )
        resolved["PubChem_InChIKey"] = resolved["resolution"].map(
            lambda row: row.get("inchikey")
        )
        resolved["Y"] = 1
        resolved["source_record_id"] = (
            "FDA-CYP-DDI:" + isoform + ":" + tier + ":" + resolved["clean_name"]
        )
        records = resolved.rename(
            columns={"Drug or Other Substance": "fda_display_name"}
        )[
            [
                "Drug",
                "Y",
                "source_record_id",
                "fda_display_name",
                "clean_name",
                "isoform",
                "evidence_tier",
                "fda_substrate_class",
                "PubChem_CID",
                "PubChem_InChIKey",
            ]
        ]
        if records.empty:
            raise RuntimeError(f"no structures resolved for {packet_id}")
        source_rows_path = packet_dir / "source_rows.csv"
        source_rows.drop(columns=["resolution"]).to_csv(source_rows_path, index=False)
        records_path = packet_dir / "records.csv"
        records.to_csv(records_path, index=False)
        manifest = {
            "schema_version": SCHEMA_VERSION,
            "packet_id": packet_id,
            "task": TASKS[isoform],
            "task_type": "classification",
            "source": {
                "kind": "fda_regulatory_dataset",
                "name": "FDA examples of drugs interacting with CYP enzymes and transporter systems",
                "title": "CYP Enzyme- and Transporter System-Based Clinical Substrates, Inhibitors, or Inducers",
                "citation_url": FDA_URL,
                "citation_type": "fda_document",
                "stable_identifier": f"FDA:CYP-DDI-table:{isoform}:{tier}",
                "license_or_access_terms": "Public FDA regulatory information; facts used with source attribution",
                "independence_status": "verified_independent",
                "independence_evidence": (
                    "FDA clinical DDI evidence is independent of the Carbon-Mangels literature compilation; exact and near structural overlap is removed before training."
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
                "endpoint_definition": f"Clinical {isoform} substrate status",
                "assay_type": "Clinical drug-drug-interaction AUC evidence",
                "organism": "Homo sapiens",
                "biological_system": "Clinical CYP-mediated metabolism",
                "source_record_locator": f"FDA Table 1, {isoform} {tier} substrate entries",
            },
            "label": {
                "source_value_definition": f"FDA {tier} clinical substrate category for CYP{isoform}",
                "mapping_rule": f"Y=1 only when FDA lists the drug as a {tier} CYP{isoform} substrate",
                "direction": "Y=1 denotes confirmed substrate status",
                "transform": "positive-only direct regulatory evidence; absence is not treated as negative",
            },
            "review": {
                "status": "review_required",
                "reviewer": "ADMET-EvO Evidence Packet data-steward review",
                "reviewed_at": now,
                "endpoint_match": "direct",
                "native_priority_external_weight": base_weight,
                "notes": (
                    "Clinical FDA substrate evidence. Sensitive and moderate-sensitive tiers are separate so reliability survives multi-packet fusion. Promotion requires the full-development and label-blind test structure firewall."
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
            "protocol": "admet-fda-cyp-clinical-substrate-materialization-v1",
            "packet_id": packet_id,
            "task": TASKS[isoform],
            "isoform": isoform,
            "evidence_tier": tier,
            "fda_rows": len(source_rows),
            "resolved_records": len(records),
            "unresolved_records": len(source_rows) - len(records),
            "native_priority_external_weight": base_weight,
            "raw_sha256": sha256_file(shared_html),
            "structure_resolution_sha256": sha256_file(shared_cache),
            "records_sha256": sha256_file(records_path),
            "test_labels_loaded": False,
            "review_status": "review_required",
        }
        (packet_dir / "materialization_audit.json").write_text(
            json.dumps(audit, indent=2, sort_keys=True) + "\n"
        )
        audits.append(audit)
    return audits


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
        default=ROOT / "data/external/fda/cyp_interactions/pubchem_resolution.jsonl",
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
