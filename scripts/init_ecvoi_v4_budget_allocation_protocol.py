"""Freeze prospective budget allocation before confirmation outputs are read."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


FILES = (
    "admet_agent/ecvoi_v4_budget_policy.py",
    "admet_agent/ecvoi_v4_policy.py",
    "scripts/analyze_ecvoi_v4_budget_allocation.py",
)


def _digest(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--matched-protocol", type=Path, required=True)
    parser.add_argument("--screen-registry", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    matched = json.loads(args.matched_protocol.read_text())
    body = {
        "version": "ecvoi-v4-prospective-budget-allocation-v1",
        "matched_protocol_sha256": matched["sha256"],
        "screen_registry_sha256": hashlib.sha256(args.screen_registry.read_bytes()).hexdigest(),
        "training_seeds": [17],
        "evaluation_seeds": [29, 43],
        "budget_fractions": [0.25, 0.5, 0.75, 1.0],
        "primary_budget_fraction": 0.5,
        "policy_options": {
            "harm_margin": 0.005,
            "harm_probability_limit": 0.5,
            "cost_penalty": 0.00025,
        },
        "shuffle_seed": 20260823,
        "implementation_sha256": {name: hashlib.sha256(Path(name).read_bytes()).hexdigest() for name in FILES},
        "test_loaded": False,
        "leaderboard_loaded": False,
    }
    value = {**body, "sha256": _digest(body)}
    args.output.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    print(json.dumps(value, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
