"""Freeze strong boosting/AutoML controls before any action outcomes exist."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    predecessor = Path("reports/ecvoi_v5_public_action_screen_protocol_v1.json")
    payload = json.loads(predecessor.read_text())
    payload.pop("sha256", None)
    payload["version"] = "ecvoi-v5-public-action-screen-protocol-v2"
    payload["superseded_pre_execution_protocol_file_sha256"] = _sha256(predecessor)
    payload["supersession_reason"] = (
        "add fixed LightGBM and nested expanded-AutoML controls required by the publication objective before any action score was generated"
    )
    payload["actions"]["boosting_anchor"] = {
        "role": "fixed_strong_control",
        "representation": ["morgan_r2_2048", "avalon_1024", "erg", "rdkit2d"],
        "model": "LightGBM fixed candidate 2",
        "outer_fit_cost": 1,
    }
    payload["actions"]["automl_expanded"] = {
        "role": "nested_strong_AutoML_control",
        "representation": ["morgan_r2_2048", "avalon_1024", "erg", "rdkit2d"],
        "candidate_count": 4,
        "group_disjoint_inner_folds": 5,
        "outer_fit_cost": 21,
        "outer_validation_labels_used_for_selection": False,
    }
    payload["screen_integrity_gate"]["expected_result_rows"] = 415
    implementation = [
        Path("admet_agent/public_action_matrix.py"),
        Path("admet_agent/imbalance_action.py"),
        Path("scripts/run_ecvoi_v5_public_action_screen.py"),
        Path("tests/test_public_action_matrix.py"),
    ]
    payload["implementation_sha256"] = {str(path): _sha256(path) for path in implementation}
    encoded = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    payload["sha256"] = hashlib.sha256(encoded.encode()).hexdigest()
    output = Path("reports/ecvoi_v5_public_action_screen_protocol_v2.json")
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(payload["sha256"])


if __name__ == "__main__":
    main()
