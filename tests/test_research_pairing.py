from __future__ import annotations

import pytest

from admet_agent.research_pairing import resolve_action_pair
from admet_agent.schemas import TaskProfile


def _profile(task_type: str = "classification") -> TaskProfile:
    return TaskProfile(
        task="BBB_Martins",
        task_type=task_type,
        metric="auroc" if task_type == "classification" else "mae",
        n_train=1000,
        n_valid=200,
        positive_rate_train=0.2 if task_type == "classification" else None,
        target_mean=0.0,
        target_std=1.0,
        invalid_smiles_train=0,
        unique_scaffolds_train=100,
        unseen_valid_scaffold_rate=0.5,
    )


@pytest.mark.parametrize(
    ("action_id", "axis", "changed"),
    [
        ("feature-domain-shift", "feature", {"representations"}),
        ("model-estimator-family", "model", {"model"}),
        ("model-imbalance-aware", "model", {"imbalance"}),
        ("data-quality-filter", "data", {"data_policy"}),
    ],
)
def test_resolves_supported_family_to_one_axis_pair(action_id, axis, changed):
    resolved = resolve_action_pair(
        {"action_id": action_id, "axis": axis, "scientific_risk": 0.1},
        _profile(),
        ordered_seeds=(1, 2, 3),
        context_sha256="a" * 64,
        task_fit_budget_seconds=5400,
        task_candidate_budget=8,
    )
    assert resolved is not None
    assert resolved.action_pair.changed_fields() == changed
    assert resolved.action_pair.candidate_charge == 2
    assert resolved.action_pair.predicted_pair_fit_seconds == 1800


@pytest.mark.parametrize(
    "action_id,axis",
    [
        ("data-leakage-safe", "data"),
        ("data-external-quarantine", "data"),
        ("replicate-multiseed", "replicate"),
        ("stop-no-defensible-action", "stop"),
        ("unknown", "feature"),
    ],
)
def test_unresolved_or_episode_actions_fail_closed(action_id, axis):
    assert (
        resolve_action_pair(
            {"action_id": action_id, "axis": axis, "scientific_risk": 0.1},
            _profile(),
            ordered_seeds=(1, 2, 3),
            context_sha256="a" * 64,
            task_fit_budget_seconds=5400,
            task_candidate_budget=8,
        )
        is None
    )


def test_pair_rejected_when_whole_pair_does_not_fit_budget():
    assert (
        resolve_action_pair(
            {
                "action_id": "feature-domain-shift",
                "axis": "feature",
                "scientific_risk": 0.1,
            },
            _profile(),
            ordered_seeds=(1, 2, 3),
            context_sha256="a" * 64,
            task_fit_budget_seconds=1799,
            task_candidate_budget=8,
        )
        is None
    )
