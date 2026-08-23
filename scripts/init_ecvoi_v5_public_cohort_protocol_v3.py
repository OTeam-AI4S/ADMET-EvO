"""Freeze the fingerprint compatibility-only v3 public cohort protocol."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    predecessor_path=Path("reports/ecvoi_v5_public_cohort_protocol_v2.json")
    payload=json.loads(predecessor_path.read_text())
    payload.pop("sha256",None)
    payload["version"]="ecvoi-v5-public-cohort-protocol-v3"
    payload["aborted_predecessor_file_sha256"]=_sha256(predecessor_path)
    payload["fingerprint_compatibility_adapter_scope"]=(
        "raw_smiles_deduplication_and_unparseable_protected_fingerprint_exact_identity_only"
    )
    implementation=[
        Path("admet_agent/cohort_governance.py"),
        Path("admet_agent/cohort_governance_v2.py"),
        Path("admet_agent/cohort_governance_v3.py"),
        Path("scripts/prepare_ecvoi_v5_public_cohort.py"),
        Path("scripts/prepare_ecvoi_v5_public_cohort_v2.py"),
        Path("scripts/prepare_ecvoi_v5_public_cohort_v3.py"),
        Path("tests/test_cohort_governance.py"),
    ]
    payload["implementation_sha256"]={str(path):_sha256(path) for path in implementation}
    encoded=json.dumps(payload,indent=2,sort_keys=True)+"\n"
    payload["sha256"]=hashlib.sha256(encoded.encode()).hexdigest()
    output=Path("reports/ecvoi_v5_public_cohort_protocol_v3.json")
    output.write_text(json.dumps(payload,indent=2,sort_keys=True)+"\n")
    print(payload["sha256"])


if __name__=="__main__":
    main()
