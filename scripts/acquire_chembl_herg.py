"""Acquire current-API ChEMBL hERG IC50 records into a compressed raw ledger."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import time
import urllib.parse
import urllib.request
from pathlib import Path

BASE = "https://www.ebi.ac.uk/chembl/api/data/activity.json"
FIELDS = (
    "activity_id,assay_chembl_id,assay_type,canonical_smiles,"
    "data_validity_comment,document_chembl_id,molecule_chembl_id,"
    "potential_duplicate,standard_flag,standard_relation,standard_type,"
    "standard_units,standard_value,target_chembl_id,target_organism"
)


def acquire(output: Path, *, limit: int = 250) -> dict:
    common_query = {
        "target_chembl_id": "CHEMBL240",
        "standard_type": "IC50",
        "standard_units": "nM",
        "only": FIELDS,
        "limit": limit,
    }
    records = []
    pages = 0
    total_expected = None
    offset = 0
    while total_expected is None or offset < total_expected:
        url = f"{BASE}?{urllib.parse.urlencode({**common_query, 'offset': offset})}"
        for attempt in range(5):
            try:
                request = urllib.request.Request(url, headers={"User-Agent": "ADMET-Agent/1.0"})
                with urllib.request.urlopen(request, timeout=90) as response:
                    payload = json.load(response)
                break
            except Exception:
                if attempt == 4:
                    raise
                time.sleep(2**attempt)
        records.extend(payload["activities"])
        pages += 1
        meta = payload["page_meta"]
        total_expected = int(meta["total_count"])
        if int(meta["offset"]) != offset:
            raise RuntimeError("ChEMBL returned a different offset than requested")
        offset += len(payload["activities"])
        if not payload["activities"] and offset < total_expected:
            raise RuntimeError("ChEMBL returned an empty page before total_count")
        print(f"acquired {offset}/{total_expected}", flush=True)
    if total_expected is None or len(records) != total_expected:
        raise RuntimeError(f"Pagination incomplete: got {len(records)}, expected {total_expected}")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    with gzip.open(temporary, "wt", encoding="utf-8", newline="\n") as handle:
        for record in sorted(records, key=lambda item: int(item["activity_id"])):
            handle.write(json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n")
    temporary.replace(output)
    checksum = hashlib.sha256(output.read_bytes()).hexdigest()
    return {
        "source": "ChEMBL current data API",
        "query": f"{BASE}?target_chembl_id=CHEMBL240&standard_type=IC50&standard_units=nM&only={FIELDS}",
        "page_limit": limit,
        "pages": pages,
        "records": len(records),
        "raw_gzip_sha256": checksum,
        "raw_path": str(output),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("data/external/chembl/herg_ic50.raw.jsonl.gz"))
    parser.add_argument("--manifest", type=Path, default=Path("data/external/chembl/herg_ic50.acquisition.json"))
    args = parser.parse_args()
    manifest = acquire(args.output)
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
