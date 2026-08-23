"""Compatibility-only recovery for EC-VOI v5 public cohort preparation v1."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from admet_agent.cohort_governance_v2 import standardize_records
from scripts import prepare_ecvoi_v5_public_cohort as v1


def parse_args() -> argparse.Namespace:
    parser=argparse.ArgumentParser()
    parser.add_argument("--protocol",type=Path,default=Path("reports/ecvoi_v5_public_cohort_protocol_v2.json"))
    parser.add_argument("--output",type=Path,default=Path("reports/ecvoi_v5_public_cohort_v2.csv"))
    parser.add_argument("--manifest",type=Path,default=Path("reports/ecvoi_v5_public_cohort_manifest_v2.json"))
    return parser.parse_args()


def run(args: argparse.Namespace) -> dict:
    v1.standardize_records = standardize_records
    payload = v1.run(args)
    payload["version"] = "ecvoi-v5-public-cohort-manifest-v2"
    payload["compatibility_adapter_scope"] = (
        "chemistry_group_parse_failure_counted_as_invalid_structure_only"
    )
    args.manifest.write_text(json.dumps(payload,indent=2,sort_keys=True)+"\n")
    return payload


if __name__=="__main__":
    print(json.dumps(run(parse_args()),indent=2,sort_keys=True))
