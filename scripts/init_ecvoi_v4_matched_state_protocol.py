"""Freeze the development-only EC-VOI v4 matched-state protocol."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from admet_agent.matched_state import ACTION_NAMES, MATCHED_STATE_NAMES


IMPLEMENTATION_FILES = (
    "admet_agent/matched_state.py",
    "admet_agent/data_governance.py",
    "admet_agent/runner.py",
    "admet_agent/selection.py",
    "scripts/run_ecvoi_v4_matched_states.py",
)


def _digest(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def build_protocol() -> dict:
    body = {
        "version": "ecvoi-v4-matched-state-development-v1",
        "scope": "development_only_mechanism_evidence",
        "tasks": ["BBB_Martins", "CYP2C9_Veith", "CYP2D6_Veith"],
        "sealed_confirmation_tasks": 19,
        "states": list(MATCHED_STATE_NAMES),
        "actions": list(ACTION_NAMES),
        "screen_seeds": [17],
        "confirmation_seeds": [29, 43],
        "max_minutes_per_fit": 15,
        "state_contract": {
            "native": "unaltered discovery train split",
            "imbalance_mild": "retain 50% of minority-class discovery rows",
            "imbalance_severe": "retain 25% of minority-class discovery rows",
            "domain_shift_mild": "remove closest 20% train structures per class using validation structures without labels",
            "domain_shift_severe": "remove closest 40% train structures per class using validation structures without labels",
        },
        "action_contract": {
            "baseline": "ExtraTrees on Morgan, no class weighting",
            "feature_rich": "same ExtraTrees with Morgan+Avalon+ErG+RDKit2D",
            "imbalance_aware": "same ExtraTrees on Morgan with class weighting",
        },
        "confirmation_unlock": "screen execution integrity only; no outcome-based selection",
        "policy_evaluation": {
            "unit": "endpoint-state",
            "target_firewall": "leave one canonical endpoint out",
            "comparators": [
                "no_diagnosis",
                "shuffled_diagnosis",
                "fixed_rules",
                "seeded_random",
                "stop_baseline",
            ],
            "primary_outcomes": [
                "selected_action_oriented_effect",
                "selection_regret",
                "harmful_intervention_rate",
                "gain_per_fit_second",
            ],
            "practical_effect": 0.005,
            "harm_margin": 0.005,
        },
        "implementation_sha256": {
            name: hashlib.sha256(Path(name).read_bytes()).hexdigest()
            for name in IMPLEMENTATION_FILES
        },
        "test_loaded": False,
        "leaderboard_loaded": False,
        "test_feedback_to_policy": False,
    }
    return {**body, "sha256": _digest(body)}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    protocol = build_protocol()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(protocol, indent=2, sort_keys=True) + "\n")
    print(json.dumps(protocol, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
