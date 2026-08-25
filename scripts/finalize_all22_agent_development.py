"""Apply the frozen label/target-distribution gate to development artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from run_all22_adaptive_agent_data import _ratio_selection


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for source in sorted(args.input_dir.glob("*.json")):
        if source.name.endswith(".error.json"):
            continue
        payload = json.loads(source.read_text())
        if payload.get("phase") != "agent_development":
            continue
        rows = payload.get("per_seed_ratio_model", [])
        if rows:
            summaries, selection = _ratio_selection(rows)
            payload["ratio_summaries"] = summaries
            payload["selection"] = selection
            payload["status"] = selection["status"]
        payload["distribution_quality_gate"] = {
            "classification_max_absolute_class_rate_gap": 0.20,
            "regression_max_median_shift_in_reference_iqr": 0.75,
            "regression_allowed_selected_to_reference_iqr_ratio": [
                1.0 / 3.0,
                3.0,
            ],
            "uses_validation_labels": False,
            "applied_before_confirmation": True,
        }
        target = args.output_dir / source.name
        target.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        print(
            json.dumps(
                {
                    "task": payload["task"],
                    "status": payload["status"],
                    "selected_ratio": payload.get("selection", {}).get(
                        "selected_ratio"
                    ),
                }
            ),
            flush=True,
        )


if __name__ == "__main__":
    main()
