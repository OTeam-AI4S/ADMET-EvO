from __future__ import annotations

import pytest

from admet_agent.committee import (
    CommitteeBrain,
    DataStewardProposer,
    DeterministicArbiter,
    ModelResearcherProposer,
    ProposalAssessment,
    ProposalEnvelope,
    SkepticProposer,
    proposal_fingerprint,
)
from admet_agent.policy_brains import benchmark_brain
from admet_agent.schemas import Budget, ExperimentSpec, RunResult, TaskProfile


def profile(metric: str = "auroc") -> TaskProfile:
    task_type = "regression" if metric == "mae" else "classification"
    return TaskProfile(
        task="Caco2_Wang" if task_type == "regression" else "HIA_Hou",
        task_type=task_type,
        metric=metric,
        n_train=100,
        n_valid=20,
        positive_rate_train=None if task_type == "regression" else 0.2,
        target_mean=0.1,
        target_std=0.2,
        invalid_smiles_train=0,
        unique_scaffolds_train=50,
        unseen_valid_scaffold_rate=0.8,
    )


def spec(
    model: str,
    *,
    task: str = "HIA_Hou",
    data_policy: str = "official",
    hyperparameters: dict | None = None,
    budget: int = 9,
) -> ExperimentSpec:
    return ExperimentSpec(
        task=task,
        seed=1,
        representations=("morgan", "rdkit2d"),
        model=model,
        imbalance="class_weight" if task == "HIA_Hou" else "none",
        data_policy=data_policy,
        hyperparameters=hyperparameters or {},
        budget=Budget(max_candidates=budget),
        hypothesis=f"test {model}",
    )


def envelope(
    model: str,
    proposer_id: str,
    *,
    role: str = "model_researcher",
    guard_risk: float = 0.0,
    expected_gain: float = 0.5,
    confidence: float = 0.8,
    compute_cost: float = 0.1,
    scientific_risk: float = 0.1,
    matched_group: str | None = None,
    pair_role: str | None = None,
    matched_axis: str | None = None,
    experiment: ExperimentSpec | None = None,
) -> ProposalEnvelope:
    return ProposalEnvelope(
        spec=experiment or spec(model),
        proposer_id=proposer_id,
        proposer_role=role,
        assessment=ProposalAssessment(
            expected_gain=expected_gain,
            confidence=confidence,
            compute_cost=compute_cost,
            scientific_risk=scientific_risk,
            guard_risk=guard_risk,
            matched_group=matched_group,
            pair_role=pair_role,
            matched_axis=matched_axis,
        ),
    )


class StubBrain:
    def __init__(self, proposals, *, fail: bool = False):
        self.proposals = proposals
        self.fail = fail
        self.context = None
        self.calls = 0

    def set_context(self, **context):
        self.context = context

    def propose(self, profile, history):
        self.calls += 1
        if self.fail:
            raise RuntimeError("planned failure")
        return list(self.proposals)

    def select_finalist(self, profile, results):
        raise AssertionError("committee finalist selection must be centralized")


def test_fingerprint_is_scientific_not_operational_identity():
    baseline = spec("extra_trees")
    operational_variant = ExperimentSpec.from_dict(
        {
            **baseline.to_dict(),
            "seed": 99,
            "device": "cuda",
            "n_jobs": 1,
            "hypothesis": "different prose",
            "budget": {"max_candidates": 1, "max_minutes": 5},
        }
    )
    scientific_variant = ExperimentSpec.from_dict(
        {**baseline.to_dict(), "data_policy": "quality_filtered"}
    )

    assert proposal_fingerprint(baseline) == proposal_fingerprint(operational_variant)
    assert proposal_fingerprint(baseline) != proposal_fingerprint(scientific_variant)


def test_arbiter_rejects_guard_violations_and_high_declared_risk():
    leaked = spec(
        "extra_trees", hyperparameters={"y_test": [0, 1]}
    )
    decision = DeterministicArbiter(max_candidates=3, max_guard_risk=0.25).arbitrate(
        [
            envelope("extra_trees", "leaky", experiment=leaked),
            envelope("xgboost", "risky", guard_risk=0.5),
            envelope("lightgbm", "safe"),
        ]
    )

    assert [item.spec.model for item in decision.accepted] == ["lightgbm"]
    reasons = {item.proposer_id: item.reason for item in decision.rejected}
    assert reasons["risky"] == "guard_risk_exceeded"
    assert reasons["leaky"].startswith("guard_violation:")
    assert "forbidden" in reasons["leaky"]


def test_duplicate_resolution_prefers_guard_then_utility_and_records_disagreement():
    same = spec("extra_trees")
    high_utility = envelope(
        "extra_trees",
        "model",
        guard_risk=0.1,
        expected_gain=1.0,
        confidence=1.0,
        experiment=same,
    )
    safer = envelope(
        "extra_trees",
        "skeptic",
        role="skeptic",
        guard_risk=0.0,
        expected_gain=0.1,
        confidence=0.1,
        experiment=same,
    )

    decision = DeterministicArbiter(max_candidates=1).arbitrate(
        [high_utility, safer]
    )

    assert decision.accepted[0].proposer_id == "skeptic"
    assert any(item.reason == "duplicate_of:skeptic" for item in decision.rejected)
    assert any(item.kind == "assessment" for item in decision.disagreements)


def test_matched_treatment_control_is_atomic_and_unmatched_action_is_rejected():
    control = envelope(
        "extra_trees", "skeptic", role="skeptic", matched_group="model-a", pair_role="control",
        matched_axis="model"
    )
    treatment = envelope(
        "xgboost", "model", matched_group="model-a", pair_role="treatment",
        matched_axis="model"
    )
    unmatched = envelope(
        "lightgbm", "data", role="data_steward", matched_group="data-b", pair_role="treatment",
        matched_axis="data"
    )

    decision = DeterministicArbiter(max_candidates=2).arbitrate(
        [unmatched, treatment, control]
    )

    assert [item.assessment.pair_role for item in decision.accepted] == [
        "control",
        "treatment",
    ]
    assert {item.spec.model for item in decision.accepted} == {
        "extra_trees",
        "xgboost",
    }
    assert any(
        item.proposer_id == "data" and item.reason == "unmatched_pair:data-b"
        for item in decision.rejected
    )

    too_small = DeterministicArbiter(max_candidates=1).arbitrate(
        [treatment, control]
    )
    assert not too_small.accepted
    assert [item.reason for item in too_small.rejected] == [
        "budget_exhausted",
        "budget_exhausted",
    ]


def test_unit_order_is_guard_then_utility_then_fingerprint_and_order_invariant():
    lower_risk = envelope(
        "extra_trees",
        "low-risk",
        guard_risk=0.0,
        expected_gain=0.01,
        confidence=0.01,
    )
    high_value = envelope(
        "xgboost",
        "high-value",
        guard_risk=0.1,
        expected_gain=1.0,
        confidence=1.0,
    )
    utility_a = envelope(
        "lightgbm", "utility-a", guard_risk=0.2, expected_gain=0.7, confidence=0.8
    )
    utility_b = envelope(
        "random_forest", "utility-b", guard_risk=0.2, expected_gain=0.3, confidence=0.5
    )
    arbiter = DeterministicArbiter(max_candidates=3)

    forward = arbiter.arbitrate([utility_b, high_value, lower_risk, utility_a])
    reverse = arbiter.arbitrate([utility_a, lower_risk, high_value, utility_b])

    forward_ids = [item.proposer_id for item in forward.accepted]
    assert forward_ids == ["low-risk", "high-value", "utility-a"]
    assert forward_ids == [item.proposer_id for item in reverse.accepted]
    assert any(
        item.proposer_id == "utility-b" and item.reason == "budget_exhausted"
        for item in forward.rejected
    )

    tied_a = envelope("catboost", "tie-a")
    tied_b = envelope("svm_rbf", "tie-b")
    tied = DeterministicArbiter(max_candidates=1).arbitrate([tied_b, tied_a])
    expected = min((tied_a, tied_b), key=lambda item: item.fingerprint)
    assert tied.accepted[0].fingerprint == expected.fingerprint


def test_committee_wraps_three_logical_agents_without_expanding_budget():
    duplicate = spec("extra_trees", budget=9)
    data_brain = StubBrain([duplicate])
    model_brain = StubBrain([duplicate, spec("xgboost", budget=9)])
    skeptic_brain = StubBrain([spec("lightgbm", budget=9)])
    committee = CommitteeBrain(
        [
            DataStewardProposer(data_brain, proposer_id="data"),
            ModelResearcherProposer(model_brain, proposer_id="model"),
            SkepticProposer(skeptic_brain, proposer_id="skeptic"),
        ],
        max_candidates=2,
    )
    committee.set_context(cross_task_evidence=[], hardness_card={})

    proposals = committee.propose(profile(), [])

    assert len(proposals) == 2
    assert len({proposal_fingerprint(item) for item in proposals}) == 2
    assert all(item.budget.max_candidates == 2 for item in proposals)
    assert data_brain.calls == model_brain.calls == skeptic_brain.calls == 1
    assert data_brain.context == {
        "cross_task_evidence": [],
        "hardness_card": {},
    }
    assert committee.last_decision is not None
    assert committee.last_decision.disagreements
    assert any(
        item.reason.startswith("duplicate_of:")
        for item in committee.last_decision.rejected
    )


def test_duplicate_singleton_cannot_erase_matched_control_identity():
    control_spec = spec("extra_trees")
    treatment_spec = ExperimentSpec.from_dict(
        {
            **control_spec.to_dict(),
            "representations": ["morgan", "rdkit2d", "avalon"],
        }
    )
    control = envelope(
        "extra_trees",
        "paired-control",
        expected_gain=0.1,
        confidence=0.5,
        matched_group="feature-pair",
        pair_role="control",
        matched_axis="feature",
        experiment=control_spec,
    )
    treatment = envelope(
        "extra_trees",
        "paired-treatment",
        matched_group="feature-pair",
        pair_role="treatment",
        matched_axis="feature",
        experiment=treatment_spec,
    )
    higher_utility_singleton = envelope(
        "extra_trees",
        "skeptic-singleton",
        expected_gain=0.9,
        confidence=0.9,
        experiment=control_spec,
    )

    decision = DeterministicArbiter(max_candidates=2).arbitrate(
        [higher_utility_singleton, treatment, control]
    )

    assert {item.proposer_id for item in decision.accepted} == {
        "paired-control",
        "paired-treatment",
    }
    assert any(
        item.proposer_id == "skeptic-singleton"
        and item.reason == "duplicate_of:paired-control"
        for item in decision.rejected
    )


def test_committee_isolates_proposer_failure_and_selects_finalist_centrally():
    broken = StubBrain([], fail=True)
    healthy = StubBrain([spec("extra_trees")])
    committee = CommitteeBrain(
        [
            DataStewardProposer(broken, proposer_id="broken"),
            ModelResearcherProposer(healthy, proposer_id="healthy"),
        ],
        max_candidates=1,
    )

    assert [item.model for item in committee.propose(profile(), [])] == ["extra_trees"]
    assert "planned failure" in committee.drain_errors()[0]

    classification_results = [
        RunResult("b", "HIA_Hou", 1, "extra_trees", "auroc", 0.8, 1.0),
        RunResult("a", "HIA_Hou", 1, "xgboost", "auroc", 0.8, 1.0),
        RunResult("w", "HIA_Hou", 1, "lightgbm", "auroc", 0.9, 1.0),
    ]
    assert committee.select_finalist(profile(), classification_results) == "w"
    assert committee.last_finalist_audit["ranking"] == ("w", "a", "b")

    regression_results = [
        RunResult("bad", "Caco2_Wang", 1, "extra_trees", "mae", 0.4, 1.0),
        RunResult("good", "Caco2_Wang", 1, "xgboost", "mae", 0.2, 1.0),
    ]
    assert committee.select_finalist(profile("mae"), regression_results) == "good"
    with pytest.raises(ValueError, match="without completed runs"):
        committee.select_finalist(profile(), [])


def test_production_committee_ignores_incidental_data_flags_and_keeps_budget():
    committee = benchmark_brain("committee", max_candidates=4)
    committee.set_context(
        cross_task_evidence=[],
        hardness_card={
            "dimensions": {
                "split_leakage_risk": {"severity": 0.001},
                "data_quality": {"severity": 0.04},
                "domain_shift": {"severity": 0.7, "confidence": 0.9},
            }
        },
        research_plan={
            "selected_action_id": "feature-domain-shift",
            "actions": [],
        },
    )

    proposals = committee.propose(profile(), [])

    assert len(proposals) <= 4
    assert all(item.data_policy == "official" for item in proposals)
    assert len({proposal_fingerprint(item) for item in proposals}) == len(proposals)


def test_production_committee_preserves_selected_data_pair_atomically():
    committee = benchmark_brain("committee", max_candidates=4)
    committee.set_context(
        cross_task_evidence=[],
        hardness_card={
            "dimensions": {
                "split_leakage_risk": {"severity": 0.0},
                "data_quality": {"severity": 0.08},
                "domain_shift": {"severity": 0.2, "confidence": 0.9},
            }
        },
        research_plan={
            "selected_action_id": "data-quality-filter",
            "actions": [],
        },
    )

    proposals = committee.propose(profile(), [])

    policies = [item.data_policy for item in proposals]
    assert policies.count("quality_filtered") == 1
    assert "official" in policies
    treatment = next(item for item in proposals if item.data_policy == "quality_filtered")
    control = next(
        item
        for item in proposals
        if item.data_policy == "official"
        and item.model == treatment.model
        and item.representations == treatment.representations
        and item.hyperparameters == treatment.hyperparameters
    )
    assert control.imbalance == treatment.imbalance


def test_production_committee_does_not_displace_imbalance_matched_pair():
    committee = benchmark_brain("committee", max_candidates=4)
    committee.set_context(
        cross_task_evidence=[],
        hardness_card={
            "dimensions": {
                "imbalance": {"severity": 0.7},
                "split_leakage_risk": {"severity": 0.0},
                "data_quality": {"severity": 0.0},
                "domain_shift": {"severity": 0.2, "confidence": 0.9},
            }
        },
        research_plan={
            "selected_action_id": "model-imbalance-aware",
            "actions": [],
        },
    )

    proposals = committee.propose(profile(), [])

    models = {item.model for item in proposals}
    assert "balanced_extra_trees" in models
    assert "extra_trees" in models
    balanced = next(item for item in proposals if item.model == "balanced_extra_trees")
    control = next(item for item in proposals if item.model == "extra_trees")
    assert balanced.representations == control.representations
    assert balanced.data_policy == control.data_policy


def test_envelope_requires_complete_pair_metadata_and_finite_scores():
    with pytest.raises(ValueError, match="supplied together"):
        ProposalAssessment(matched_group="x")
    with pytest.raises(ValueError, match="finite"):
        ProposalAssessment(expected_gain=float("nan"))


def test_declared_pair_must_change_exactly_one_registered_axis():
    control = envelope(
        "extra_trees",
        "control",
        matched_group="confounded",
        pair_role="control",
        matched_axis="model",
    )
    confounded_spec = ExperimentSpec.from_dict(
        {
            **spec("xgboost").to_dict(),
            "representations": ["morgan"],
        }
    )
    treatment = envelope(
        "xgboost",
        "treatment",
        matched_group="confounded",
        pair_role="treatment",
        matched_axis="model",
        experiment=confounded_spec,
    )

    decision = DeterministicArbiter(max_candidates=2).arbitrate(
        [control, treatment]
    )

    assert not decision.accepted
    assert {item.reason for item in decision.rejected} == {
        "invalid_matched_pair:confounded"
    }
