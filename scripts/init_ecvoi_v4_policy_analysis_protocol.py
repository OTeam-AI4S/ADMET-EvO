"""Freeze EC-VOI v4 matched-state policy analysis before outcomes are read."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


FILES = (
    "admet_agent/ecvoi_v4_policy.py",
    "scripts/analyze_ecvoi_v4_matched_states.py",
)


def _digest(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def build_protocol(matched_state_protocol: Path) -> dict:
    matched = json.loads(matched_state_protocol.read_text(encoding="utf-8"))
    body = {
        "version": "ecvoi-v4-matched-state-policy-analysis-v1",
        "matched_state_protocol_sha256": matched["sha256"],
        "model": "standardized BayesianRidge with action-by-diagnosis interactions",
        "policy_options": {
            "harm_margin": 0.005,
            "harm_probability_limit": 0.5,
            "cost_penalty": 0.00025,
        },
        "shuffle_seed": 20260823,
        "random_seed": 20260823,
        "implementation_sha256": {
            name: hashlib.sha256(Path(name).read_bytes()).hexdigest() for name in FILES
        },
        "advancement_checks": [
            "target_endpoint_firewall",
            "full_nonnegative_mean_effect",
            "full_harm_no_worse_than_no_diagnosis_and_rules",
            "full_regret_below_random",
            "diagnosis_identifiable_against_shuffle",
        ],
        "gate_requires_all_three_seeds": True,
        "test_loaded": False,
        "leaderboard_loaded": False,
    }
    return {**body, "sha256": _digest(body)}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--matched-state-protocol", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    value = build_protocol(args.matched_state_protocol)
    args.output.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    print(json.dumps(value, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
