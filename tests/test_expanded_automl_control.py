from __future__ import annotations

import json
import math

from admet_agent.committee import proposal_fingerprint
from admet_agent.guard import validate_experiment
from admet_agent.policy_brains import (
    EXPANDED_DATA_POLICIES,
    EXPANDED_FEATURE_VIEWS,
    AgentNoDataBrain,
    AgentNoHardnessBrain,
    AgentNoMemoryBrain,
    AutoMLControlBrain,
    ExpandedSpaceAutoMLControlBrain,
    benchmark_brain,
)
from admet_agent.schemas import RunResult, TaskProfile


def classification_profile() -> TaskProfile:
    return TaskProfile(
        task="HIA_Hou",
        task_type="classification",
        metric="auroc",
        n_train=100,
        n_valid=20,
        positive_rate_train=0.2,
        target_mean=0.2,
        target_std=0.4,
        invalid_smiles_train=0,
        unique_scaffolds_train=50,
        unseen_valid_scaffold_rate=0.8,
    )


def _fingerprints(items) -> list[str]:
    return [proposal_fingerprint(item) for item in items]


def _history_from_specs(specs, scores=None) -> list[RunResult]:
    scores = scores or [0.6 + 0.05 * index for index in range(len(specs))]
    return [
        RunResult(
            run_id=f"run-{index}",
            task=item.task,
            seed=1,
            model=item.model,
            metric="auroc",
            validation_score=scores[index],
            elapsed_seconds=1.0 + index,
            representations=item.representations,
            hyperparameters=item.hyperparameters,
            hypothesis=item.hypothesis,
            data_policy=item.data_policy,
        )
        for index, item in enumerate(specs)
    ]


def _contains_forbidden_test_field(value) -> bool:
    if isinstance(value, dict):
        return any(
            str(key).lower()
            in {"test", "test_labels", "y_test", "test_loaded", "test_score"}
            or _contains_forbidden_test_field(item)
            for key, item in value.items()
        )
    if isinstance(value, (list, tuple)):
        return any(_contains_forbidden_test_field(item) for item in value)
    return False


def test_candidate_pool_exposes_agent_primitive_union_without_context(
    monkeypatch,
):
    monkeypatch.setattr(
        "admet_agent.policy_brains.is_model_available", lambda _name: True
    )
    monkeypatch.setattr(
        "admet_agent.policy_brains.is_representation_available", lambda _name: True
    )
    brain = ExpandedSpaceAutoMLControlBrain(max_candidates=4, policy_seed=7)

    pool = brain.candidate_pool(classification_profile())

    assert {item.model for item in pool} == {
        "lightgbm",
        "xgboost",
        "extra_trees",
        "random_forest",
        "catboost",
        "svm_rbf",
        "balanced_extra_trees",
    }
    assert {item.representations for item in pool} == set(EXPANDED_FEATURE_VIEWS)
    assert {item.data_policy for item in pool} == set(EXPANDED_DATA_POLICIES)
    assert {item.imbalance for item in pool} == {"none", "class_weight"}
    assert any(item.hyperparameters for item in pool)
    assert any(not item.hyperparameters for item in pool)
    assert len(_fingerprints(pool)) == len(set(_fingerprints(pool)))
    for item in pool:
        validate_experiment(item)


def test_scientific_context_is_ignored_and_fixed_seed_is_reproducible(monkeypatch):
    monkeypatch.setattr(
        "admet_agent.policy_brains.is_representation_available", lambda _name: True
    )
    left = ExpandedSpaceAutoMLControlBrain(max_candidates=4, policy_seed=19)
    right = ExpandedSpaceAutoMLControlBrain(max_candidates=4, policy_seed=19)
    left.set_context(
        cross_task_evidence=[{"model": "catboost"}],
        hardness_card={"dimensions": {"domain_shift": {"severity": 1.0}}},
        research_plan={"selected_action_id": "feature-domain-shift"},
        action_memory={"supported": ["unimol2"]},
    )
    right.set_context(
        cross_task_evidence=[],
        hardness_card={"dimensions": {"imbalance": {"severity": 0.0}}},
        research_plan={"selected_action_id": "stop-no-defensible-action"},
        action_memory={},
    )

    left_proposals = left.propose(classification_profile(), [])
    right_proposals = right.propose(classification_profile(), [])

    assert _fingerprints(left_proposals) == _fingerprints(right_proposals)
    assert left.last_decision.to_dict() == right.last_decision.to_dict()
    audit = left.last_decision.to_dict()
    assert audit["uses_hardness"] is False
    assert audit["uses_research_plan"] is False
    assert audit["uses_action_memory"] is False
    assert audit["uses_matched_pair_priority"] is False


def test_acquisition_is_invariant_to_pool_and_history_order(monkeypatch):
    monkeypatch.setattr(
        "admet_agent.policy_brains.is_representation_available",
        lambda name: name != "unimol2",
    )
    profile = classification_profile()
    seed_brain = ExpandedSpaceAutoMLControlBrain(max_candidates=4, policy_seed=31)
    seeds = seed_brain.candidate_pool(profile)[:4]
    history = _history_from_specs(seeds, [0.61, 0.77, 0.66, 0.70])

    class ReversePoolBrain(ExpandedSpaceAutoMLControlBrain):
        def candidate_pool(self, profile):
            return list(reversed(super().candidate_pool(profile)))

    normal = ExpandedSpaceAutoMLControlBrain(max_candidates=4, policy_seed=31)
    reversed_pool = ReversePoolBrain(max_candidates=4, policy_seed=31)

    normal_result = normal.propose(profile, history)
    reversed_result = reversed_pool.propose(profile, list(reversed(history)))

    assert _fingerprints(normal_result) == _fingerprints(reversed_result)
    assert normal.last_decision.selected == reversed_pool.last_decision.selected


def test_later_round_uses_validation_scores_but_not_history_input_order(
    monkeypatch,
):
    monkeypatch.setattr(
        "admet_agent.policy_brains.is_representation_available",
        lambda name: name != "unimol2",
    )
    profile = classification_profile()
    source = ExpandedSpaceAutoMLControlBrain(max_candidates=4, policy_seed=43)
    anchors = source.candidate_pool(profile)[:4]
    history_a = _history_from_specs(anchors, [0.95, 0.50, 0.49, 0.48])
    history_b = _history_from_specs(anchors, [0.48, 0.49, 0.50, 0.95])

    first = ExpandedSpaceAutoMLControlBrain(max_candidates=4, policy_seed=43)
    first_reordered = ExpandedSpaceAutoMLControlBrain(max_candidates=4, policy_seed=43)
    second = ExpandedSpaceAutoMLControlBrain(max_candidates=4, policy_seed=43)

    proposals_a = first.propose(profile, history_a)
    proposals_a_reordered = first_reordered.propose(profile, list(reversed(history_a)))
    proposals_b = second.propose(profile, history_b)

    assert _fingerprints(proposals_a) == _fingerprints(proposals_a_reordered)
    assert _fingerprints(proposals_a) != _fingerprints(proposals_b)


def test_budget_unique_configs_audit_and_no_test_fields(monkeypatch):
    monkeypatch.setattr(
        "admet_agent.policy_brains.is_representation_available", lambda _name: True
    )
    profile = classification_profile()
    brain = ExpandedSpaceAutoMLControlBrain(max_candidates=3, policy_seed=53)

    first = brain.propose(profile, [])
    first_audit = brain.last_decision.to_dict()
    history = _history_from_specs(first)
    second = brain.propose(profile, history)
    second_audit = brain.last_decision.to_dict()

    assert len(first) == len(second) == 3
    assert all(item.budget.max_candidates == 3 for item in first + second)
    assert len(set(_fingerprints(first + second))) == 6
    assert first_audit["round_index"] == 1
    assert second_audit["round_index"] == 2
    assert second_audit["duplicate_candidates_rejected"] >= len(first)
    assert [item["fingerprint"] for item in first_audit["selected"]] == (
        _fingerprints(first)
    )
    assert all(
        math.isfinite(item["acquisition"]["value"])
        and 0.0 <= item["acquisition"]["normalized_cost"] <= 1.0
        for item in second_audit["selected"]
    )
    assert not _contains_forbidden_test_field(first_audit)
    assert not _contains_forbidden_test_field(second_audit)
    assert all(
        "no scientific hypothesis or causal attribution" in item.hypothesis
        for item in first + second
    )
    json.dumps(first_audit, sort_keys=True)


def test_factory_exposes_expanded_control_without_changing_conventional_automl():
    expanded = benchmark_brain("expanded_automl", max_candidates=2, policy_seed=61)
    conventional = benchmark_brain("automl", max_candidates=2, policy_seed=61)

    assert isinstance(expanded, ExpandedSpaceAutoMLControlBrain)
    assert isinstance(conventional, AutoMLControlBrain)
    assert not isinstance(conventional, ExpandedSpaceAutoMLControlBrain)
    proposals = conventional.propose(classification_profile(), [])
    assert len(proposals) == 2
    assert all(
        item.representations == ("morgan", "avalon", "erg", "rdkit2d")
        for item in proposals
    )
    assert all(item.data_policy == "official" for item in proposals)
    assert all("AutoML control" in item.hypothesis for item in proposals)


def test_factory_exposes_executable_agent_ablations_with_locked_context_removals():
    no_memory = benchmark_brain("agent_no_memory", max_candidates=2)
    no_hardness = benchmark_brain("agent_no_hardness", max_candidates=2)
    no_data = benchmark_brain("agent_no_data", max_candidates=2)
    assert isinstance(no_memory, AgentNoMemoryBrain)
    assert isinstance(no_hardness, AgentNoHardnessBrain)
    assert isinstance(no_data, AgentNoDataBrain)

    plan = {
        "selected_action_id": "data-action",
        "actions": [
            {
                "action_id": "data-action",
                "axis": "data",
                "falsification_test": "matched data pair",
            },
            {
                "action_id": "model-action",
                "axis": "model",
                "falsification_test": "matched model pair",
            },
        ],
    }
    hardness = {"dimensions": {"data_quality": {"severity": 1.0}}}
    memory = [{"model": "catboost"}]
    no_memory.set_context(
        cross_task_evidence=memory,
        hardness_card=hardness,
        research_plan=plan,
    )
    no_hardness.set_context(
        cross_task_evidence=memory,
        hardness_card=hardness,
        research_plan=plan,
    )
    no_data.set_context(
        cross_task_evidence=memory,
        hardness_card=hardness,
        research_plan=plan,
    )
    assert no_memory.cross_task_evidence == []
    assert no_hardness.hardness_card == {}
    assert no_hardness.research_plan == {}
    assert all(
        action["axis"] != "data"
        for action in no_data.effective_research_plan["actions"]
    )
    assert no_data.effective_research_plan["selected_action_id"] == "model-action"
