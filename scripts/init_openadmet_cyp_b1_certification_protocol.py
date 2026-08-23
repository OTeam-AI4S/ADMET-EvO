"""Freeze the one-time B1 certification gate before reading holdout targets."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    recipe_path = Path("reports/openadmet_cyp_frozen_recipe_b1_v1.json")
    stage_b_path = Path("runs/openadmet-cyp-stage-b/b1_cv_result.json")
    recipe = json.loads(recipe_path.read_text())
    stage_b = json.loads(stage_b_path.read_text())
    output_dir = Path("runs/openadmet-cyp-b1-certification-v1")
    if output_dir.exists():
        raise RuntimeError("Refusing to freeze certification protocol after outputs exist")
    payload = {
        "version": "openadmet-cyp-b1-certification-protocol-v1",
        "frozen_at": "2026-08-23",
        "recipe_path": str(recipe_path.resolve()),
        "recipe_file_sha256": _sha256(recipe_path),
        "recipe_sha256": recipe["recipe_sha256"],
        "stage_b_result_sha256": _sha256(stage_b_path),
        "stage_b_macro_st_rae": stage_b["mean_macro_st_rae"],
        "stage_b_macro_mcc": stage_b["mean_macro_mcc"],
        "gates": {
            "macro_st_rae_max": stage_b["mean_macro_st_rae"] + 0.10,
            "macro_mcc_min": stage_b["mean_macro_mcc"] - 0.10,
            "worst_endpoint_st_rae_max": 1.25,
            "worst_endpoint_mcc_min": 0.0
        },
        "gate_policy": "all checks must pass before blind inference",
        "seeds": [1, 2, 3],
        "certification_partition_opened_at_freeze": False,
        "blind_test_labels_loaded": False,
        "leaderboard_feedback_used": False,
        "selection_reopened": False,
        "executor_sha256": _sha256(Path("scripts/run_openadmet_cyp_frozen_b1.py"))
    }
    encoded = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    payload["sha256"] = hashlib.sha256(encoded.encode()).hexdigest()
    output = Path("reports/openadmet_cyp_b1_certification_protocol_v1.json")
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(payload["sha256"])


if __name__ == "__main__":
    main()
