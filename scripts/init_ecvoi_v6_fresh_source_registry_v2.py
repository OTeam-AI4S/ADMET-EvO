"""Extend v1 with SMILES-native replacements after a pre-outcome schema failure."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


def main() -> None:
    parent_path = Path("reports/ecvoi_v6_fresh_source_registry_v1.json")
    parent = json.loads(parent_path.read_text())
    replacements = [
        {
            "loader": "DeepChemRaw", "dataset": "BACE", "source_id": "bace_literature_collection_2016",
            "max_endpoints": 2, "smiles_column": "mol", "url": "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/bace.csv",
            "license_status": "MoleculeNet_CC_BY_NC_3.0_research_only_upstream_specific_unspecified",
        },
        {
            "loader": "DeepChemRaw", "dataset": "SIDER", "source_id": "kuhn_sider_2015",
            "max_endpoints": 4, "smiles_column": "smiles", "url": "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/sider.csv.gz",
            "license_status": "MoleculeNet_CC_BY_NC_3.0_research_only_upstream_specific_unspecified",
        },
        {
            "loader": "DeepChemRaw", "dataset": "MUV", "source_id": "rohrer_baumann_muv_2009",
            "max_endpoints": 4, "smiles_column": "smiles", "url": "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/muv.csv.gz",
            "license_status": "MoleculeNet_CC_BY_NC_3.0_research_only_upstream_specific_unspecified",
        },
    ]
    payload = {
        **{key: value for key, value in parent.items() if key != "sha256"},
        "version": "ecvoi-v6-fresh-source-registry-v2",
        "frozen_at": "2026-08-23",
        "supersedes": {
            "registry_self_sha256": parent["sha256"],
            "reason": "pre-outcome QM representation schema failure",
            "action_outcomes_seen": False,
        },
        "candidate_records_loaded_before_freeze": True,
        "candidate_action_outcomes_available_before_freeze": False,
        "candidates": parent["candidates"] + replacements,
        "quarantined_before_v2": ["QM7b", "QM8", "QM9"],
        "replacement_documentation": {
            "DeepChem_loaders": "https://github.com/deepchem/deepchem/tree/master/deepchem/molnet/load_function",
            "MoleculeNet_paper": "https://doi.org/10.1039/C7SC02664A",
            "paper_license": "CC BY-NC 3.0",
            "raw_redistribution": "forbidden by this project; local research modeling and derived statistics only",
            "accessed": "2026-08-23",
        },
    }
    payload["license_policy"] = {
        **parent["license_policy"],
        "MoleculeNet_CC_BY_NC_3.0_research_only_upstream_specific_unspecified": (
            "local non-commercial research modeling only; do not redistribute raw rows; disclose source-specific ambiguity"
        ),
    }
    encoded = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    payload["sha256"] = hashlib.sha256(encoded.encode()).hexdigest()
    output = Path("reports/ecvoi_v6_fresh_source_registry_v2.json")
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(payload["sha256"])


if __name__ == "__main__":
    main()
