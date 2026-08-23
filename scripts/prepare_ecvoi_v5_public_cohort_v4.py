"""Identity-component compatibility recovery for public cohort v3."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from admet_agent.cohort_governance_v2 import standardize_records
from admet_agent.cohort_governance_v3 import protected_structure_firewall
from admet_agent.cohort_governance_v4 import assign_identity_locked_folds
from scripts import prepare_ecvoi_v5_public_cohort as v1
from scripts import prepare_ecvoi_v5_public_cohort_v3 as v3


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, default=Path("reports/ecvoi_v5_public_cohort_protocol_v4.json"))
    parser.add_argument("--output", type=Path, default=Path("reports/ecvoi_v5_public_cohort_v4.csv"))
    parser.add_argument("--manifest", type=Path, default=Path("reports/ecvoi_v5_public_cohort_manifest_v4.json"))
    return parser.parse_args()


def run(args: argparse.Namespace) -> dict:
    v1.standardize_records = standardize_records
    v1.protected_structure_firewall = protected_structure_firewall
    v1.assign_identity_locked_folds = assign_identity_locked_folds
    payload = v3.run(args)
    frame = pd.read_csv(args.output, usecols=["inchikey", "chemistry_group"])
    multiplicity = frame.drop_duplicates().groupby("inchikey")["chemistry_group"].nunique()
    payload["version"] = "ecvoi-v5-public-cohort-manifest-v4"
    payload["identity_component_adapter"] = {
        "identities_joining_multiple_chemistry_groups": int((multiplicity > 1).sum()),
        "labels_used_for_component_construction": False,
        "policy_or_metric_results_used": False,
    }
    args.manifest.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return payload


if __name__ == "__main__":
    print(json.dumps(run(parse_args()), indent=2, sort_keys=True))
