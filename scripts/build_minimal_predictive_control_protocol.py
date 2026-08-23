"""Freeze the streamlined TDC-22 three-arm predictive control."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from admet_agent.source_snapshot import production_source_manifest
from admet_agent.tasks import TASKS


def _digest(payload: object) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def main() -> None:
    root = Path(__file__).parents[1]
    runner = root / "scripts/run_minimal_predictive_control.py"
    tasks = [task.name for task in TASKS.values()]
    body = {
        "protocol_version": "tdc22-minimal-three-arm-predictive-control-v1",
        "objective": "Compare diagnosis-driven minimal experiments with scalar model search after prediction qualification has already passed.",
        "tasks": tasks,
        "arms": {
            "streamlined_agent": {"screen_configs": 2, "evaluation_winners": 1},
            "expanded_automl": {"screen_configs": 2, "evaluation_winners": 1},
            "strong_fixed": {"screen_configs": 0, "evaluation_winners": 1},
        },
        "discovery_seed": 11,
        "evaluation_seed": 12,
        "maximum_logical_fits": 154,
        "exact_physical_fit_reuse_across_arms": True,
        "normalization": {"mae": "negative_mae_over_discovery_target_std", "other": "raw_oriented_metric"},
        "agent_vs_expanded_automl_noninferiority_margin": -0.01,
        "bootstrap_replicates": 10000,
        "default_unimol2": False,
        "default_committee": False,
        "external_data": False,
        "max_minutes_per_physical_fit": 30,
        "source_sha256": production_source_manifest()["source_sha256"],
        "runner_sha256": hashlib.sha256(runner.read_bytes()).hexdigest(),
        "test_loaded": False,
        "test_feedback_to_discovery": False,
    }
    output = root / "runs/minimal-predictive-control-v1/protocol.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps({**body, "sha256": _digest(body)}, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
