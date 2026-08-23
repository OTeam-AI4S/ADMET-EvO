"""Locked capability controls for deterministic hidden-fault replay.

These controls are intentionally conservative in their claims.  They quantify
which gains come from having an endpoint/data guard at all and reveal whether
the current detector exceeds a complete hand-coded rule policy.
"""

from __future__ import annotations

import hashlib
from dataclasses import replace
from typing import Any

import numpy as np

from .hidden_fault_benchmark import (
    SUPPORTED_FAULTS,
    FaultChallengeResult,
    score_hidden_faults,
)

FAULT_CONTROL_PROTOCOL = "locked-hidden-fault-policy-controls-v1"
SCHEMA_GUARD_FAULTS = frozenset(
    {
        "invalid_molecular_structure",
        "invalid_target",
        "conflicting_class_labels",
        "high_variance_replicate_measurements",
        "replicate_measurements",
        "exact_split_overlap",
    }
)


def _case_f1(row: FaultChallengeResult) -> float:
    expected = set(row.expected_faults)
    detected = set(row.detected_faults)
    if not expected:
        return float(not detected)
    tp = len(expected & detected)
    precision = tp / max(1, len(detected))
    recall = tp / len(expected)
    return 2 * precision * recall / (precision + recall) if precision + recall else 0.0


def _seeded_random_fault(challenge_id: str, policy_seed: int) -> str:
    ordered = sorted(SUPPORTED_FAULTS)
    digest = hashlib.sha256(f"{policy_seed}:{challenge_id}".encode()).digest()
    return ordered[int.from_bytes(digest[:8], "big") % len(ordered)]


def apply_fault_control(
    rows: list[FaultChallengeResult],
    arm: str,
    *,
    policy_seed: int = 20260820,
) -> list[FaultChallengeResult]:
    """Apply a preregistered capability mask without consulting expected labels."""

    if arm not in {
        "agent_contract_guard",
        "complete_rules_guard",
        "expanded_automl_schema_guard",
        "conventional_automl_no_guard",
        "agent_no_hardness_diagnosis",
        "seeded_random_fault",
    }:
        raise ValueError(f"Unknown fault-control arm {arm!r}")
    controlled = []
    for row in rows:
        if arm in {"agent_contract_guard", "complete_rules_guard"}:
            detected = row.detected_faults
            proposed = row.remediations_proposed
            valid = row.valid_remediations
        elif arm == "expanded_automl_schema_guard":
            detected = tuple(
                fault for fault in row.detected_faults if fault in SCHEMA_GUARD_FAULTS
            )
            # Give the control full credit for a standard remediation attached
            # to each schema fault.  This is deliberately a strong baseline.
            proposed = len(detected)
            valid = len(detected)
        elif arm in {"conventional_automl_no_guard", "agent_no_hardness_diagnosis"}:
            detected = ()
            proposed = valid = 0
        else:
            detected = (_seeded_random_fault(row.challenge_id, policy_seed),)
            proposed = 1
            # Validity is scored after reveal, as it is for every arm.
            valid = int(bool(set(detected) & set(row.expected_faults)))
        controlled.append(
            replace(
                row,
                detected_faults=detected,
                remediations_proposed=proposed,
                valid_remediations=valid,
                detection_seconds=(
                    0.0 if arm != "agent_contract_guard" else row.detection_seconds
                ),
            )
        )
    return controlled


def _paired_bootstrap(
    agent: list[FaultChallengeResult],
    control: list[FaultChallengeResult],
    *,
    seed: int = 20260820,
) -> dict[str, Any]:
    if [row.challenge_id for row in agent] != [row.challenge_id for row in control]:
        raise ValueError("Fault-control rows must be aligned by challenge ID")
    differences = np.asarray(
        [_case_f1(left) - _case_f1(right) for left, right in zip(agent, control)],
        dtype=float,
    )
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(differences), size=(10000, len(differences)))
    means = differences[indices].mean(axis=1)
    return {
        "mean_agent_minus_control_case_f1": float(differences.mean()),
        "bootstrap_95_ci": [
            float(np.quantile(means, 0.025)),
            float(np.quantile(means, 0.975)),
        ],
        "agent_better_cases": int(np.sum(differences > 0)),
        "tied_cases": int(np.sum(differences == 0)),
        "agent_worse_cases": int(np.sum(differences < 0)),
        "independent_unit": "challenge_case",
    }


def compare_fault_controls(
    rows: list[FaultChallengeResult],
    *,
    policy_seed: int = 20260820,
) -> dict[str, Any]:
    if not rows:
        raise ValueError("Fault challenge rows are required")
    arms = (
        "agent_contract_guard",
        "complete_rules_guard",
        "expanded_automl_schema_guard",
        "conventional_automl_no_guard",
        "agent_no_hardness_diagnosis",
        "seeded_random_fault",
    )
    controlled = {
        arm: apply_fault_control(rows, arm, policy_seed=policy_seed) for arm in arms
    }
    agent = controlled["agent_contract_guard"]
    return {
        "protocol": FAULT_CONTROL_PROTOCOL,
        "claim_scope": "locked_replay_not_independent_blind_evaluation",
        "policy_seed": policy_seed,
        "test_loaded": False,
        "arms": {
            arm: {
                "score": score_hidden_faults(values),
                "agent_comparison": (
                    None
                    if arm == "agent_contract_guard"
                    else _paired_bootstrap(agent, values)
                ),
            }
            for arm, values in controlled.items()
        },
        "interpretation": {
            "automl_boundary": (
                "The replay measures the value of explicit data/endpoint guards, "
                "not predictive model quality."
            ),
            "rule_boundary": (
                "Agent and complete-rules equality means current fault discovery "
                "is not evidence of learned or general scientific reasoning."
            ),
            "upgrade_requirement": (
                "Use independently authored, unseen composite faults and require "
                "evidence-grounded remediation plans to exceed the complete-rules control."
            ),
        },
    }


__all__ = [
    "FAULT_CONTROL_PROTOCOL",
    "SCHEMA_GUARD_FAULTS",
    "apply_fault_control",
    "compare_fault_controls",
]
