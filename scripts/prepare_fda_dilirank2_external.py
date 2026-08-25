"""Resolve FDA DILIrank 2.0 drug names to PubChem structures.

The resolver is resumable: every PubChem response is appended to JSONL before
the next request.  The primary binary mapping is frozen as Most/Less concern
positive, No concern negative, and Ambiguous excluded.  This mapped cohort is
still bridge-grade until its label alignment and source lineage are cleared.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path
from urllib.parse import quote

import pandas as pd
import requests

PUBCHEM = (
    "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/"
    "{name}/property/CanonicalSMILES,InChIKey/JSON"
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_cache(path: Path) -> dict[str, dict]:
    if not path.exists():
        return {}
    rows = {}
    for line in path.read_text().splitlines():
        if line.strip():
            row = json.loads(line)
            rows[row["name"]] = row
    return rows


def _label(value: object) -> int | None:
    text = str(value).strip().lower().replace("^", "")
    if "ambiguous" in text:
        return None
    if "no-dili" in text:
        return 0
    if "most-dili" in text or "less-dili" in text:
        return 1
    raise ValueError(f"Unrecognized DILIrank class: {value!r}")


def _resolve(session: requests.Session, name: str, retries: int) -> dict:
    url = PUBCHEM.format(name=quote(name, safe=""))
    last = None
    for attempt in range(retries + 1):
        response = session.get(url, timeout=30)
        last = response
        if response.status_code == 200:
            prop = response.json()["PropertyTable"]["Properties"][0]
            smiles = prop.get("ConnectivitySMILES") or prop.get("CanonicalSMILES")
            if not smiles:
                return {"name": name, "status": "missing_smiles", "url": url}
            return {
                "name": name,
                "status": "resolved",
                "cid": prop.get("CID"),
                "smiles": smiles,
                "inchikey": prop.get("InChIKey"),
                "url": url,
            }
        if response.status_code not in {429, 500, 502, 503, 504}:
            break
        time.sleep(min(8.0, 0.5 * 2**attempt))
    return {
        "name": name,
        "status": "unresolved",
        "http_status": None if last is None else last.status_code,
        "url": url,
    }


def run(args: argparse.Namespace) -> dict:
    raw = pd.read_excel(args.input, sheet_name="version 2", header=1)
    raw = raw.rename(
        columns={
            "LTKBID": "source_record_id",
            "CompoundName": "name",
            "vDILI-Concern": "risk_class",
        }
    )
    raw = raw[["source_record_id", "name", "risk_class"]].dropna(subset=["name"])
    raw["Y"] = raw["risk_class"].map(_label)
    eligible = raw.dropna(subset=["Y"]).copy()
    eligible["Y"] = eligible["Y"].astype(int)

    args.cache.parent.mkdir(parents=True, exist_ok=True)
    cache = _load_cache(args.cache)
    pending = [name for name in eligible["name"].astype(str).unique() if name not in cache]
    if args.max_requests is not None:
        pending = pending[: args.max_requests]
    session = requests.Session()
    session.headers["User-Agent"] = "ADMET-Agent academic data curation; contact=local-research"
    with args.cache.open("a") as handle:
        for index, name in enumerate(pending, start=1):
            row = _resolve(session, name, args.retries)
            handle.write(json.dumps(row, sort_keys=True) + "\n")
            handle.flush()
            cache[name] = row
            if index % 50 == 0:
                print(json.dumps({"resolved_requests": index, "remaining": len(pending)-index}))
            time.sleep(args.request_interval)

    joined = eligible.copy()
    joined["resolution"] = joined["name"].astype(str).map(cache)
    joined = joined.loc[joined["resolution"].notna()].copy()
    joined["resolution_status"] = joined["resolution"].map(lambda row: row["status"])
    resolved = joined.loc[joined["resolution_status"] == "resolved"].copy()
    resolved["X"] = resolved["resolution"].map(lambda row: row["smiles"])
    resolved["PubChem_CID"] = resolved["resolution"].map(lambda row: row.get("cid"))
    output = resolved[
        ["source_record_id", "name", "risk_class", "PubChem_CID", "X", "Y"]
    ]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(args.output, sep="\t", index=False)
    manifest = {
        "protocol": "fda-dilirank2-pubchem-resolution-v1",
        "input": str(args.input.resolve()),
        "input_sha256": _sha256(args.input),
        "raw_rows": len(raw),
        "eligible_non_ambiguous_rows": len(eligible),
        "resolved_rows": len(output),
        "unresolved_or_pending_rows": len(eligible) - len(output),
        "positive_rate": float(output["Y"].mean()) if len(output) else None,
        "mapping": {"Most/Less-DILI-concern": 1, "No-DILI-concern": 0, "Ambiguous": "excluded"},
        "source_urls": [
            "https://www.fda.gov/science-research/liver-toxicity-knowledge-base-ltkb/drug-induced-liver-injury-rank-dilirank-20-dataset",
            "https://pubchem.ncbi.nlm.nih.gov/docs/pug-rest",
        ],
    }
    args.manifest.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    root = Path("data/external/fda/dilirank2")
    parser.add_argument("--input", type=Path, default=root / "dilirank2.xlsx")
    parser.add_argument("--cache", type=Path, default=root / "pubchem_resolution.jsonl")
    parser.add_argument("--output", type=Path, default=root / "dilirank2_structures.tab")
    parser.add_argument("--manifest", type=Path, default=root / "manifest.json")
    parser.add_argument("--request-interval", type=float, default=0.25)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--max-requests", type=int)
    return parser.parse_args()


if __name__ == "__main__":
    print(json.dumps(run(parse_args()), indent=2))
