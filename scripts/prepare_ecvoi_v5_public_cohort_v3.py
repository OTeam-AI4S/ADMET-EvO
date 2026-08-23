"""Fingerprint compatibility recovery for EC-VOI v5 public cohort v2."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from admet_agent.cohort_governance_v2 import standardize_records
from admet_agent.cohort_governance_v3 import protected_structure_firewall
from scripts import prepare_ecvoi_v5_public_cohort as v1
from scripts import prepare_ecvoi_v5_public_cohort_v2 as v2


def parse_args() -> argparse.Namespace:
    parser=argparse.ArgumentParser()
    parser.add_argument("--protocol",type=Path,default=Path("reports/ecvoi_v5_public_cohort_protocol_v3.json"))
    parser.add_argument("--output",type=Path,default=Path("reports/ecvoi_v5_public_cohort_v3.csv"))
    parser.add_argument("--manifest",type=Path,default=Path("reports/ecvoi_v5_public_cohort_manifest_v3.json"))
    return parser.parse_args()


def run(args: argparse.Namespace) -> dict:
    v1.standardize_records = standardize_records
    v1.protected_structure_firewall = protected_structure_firewall
    payload = v2.run(args)
    payload["version"] = "ecvoi-v5-public-cohort-manifest-v3"
    payload["fingerprint_compatibility_adapter_scope"] = (
        "raw_smiles_deduplication_and_unparseable_protected_fingerprint_exact_identity_only"
    )
    args.manifest.write_text(json.dumps(payload,indent=2,sort_keys=True)+"\n")
    return payload


if __name__=="__main__":
    print(json.dumps(run(parse_args()),indent=2,sort_keys=True))
