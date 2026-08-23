from __future__ import annotations

from admet_agent.research_episode import (
    EpisodeBudget,
    PairExecution,
    run_episode,
)
from admet_agent.research_policy_v2 import (
    ActionPair,
    MatchedEffectObservation,
    MetricValuedECVOIPolicy,
    metric_valued_policy_for_variant,
)


def _pair() -> ActionPair:
    return ActionPair(
        pair_id="pair-a",
        action_id="feature-domain-shift",
        axis="feature",
        treatment={
            "task": "BBB_Martins",
            "representations": ["morgan", "rdkit2d"],
        },
        control={"task": "BBB_Martins", "representations": ["morgan"]},
        ordered_seeds=(1, 2, 3),
        predicted_pair_fit_seconds=60,
        candidate_charge=2,
        task_fit_budget_seconds=120,
        task_candidate_budget=4,
        practical_effect_threshold=0.005,
        effect_standardization_scale=0.005,
        predicted_native_observation_se=0.005,
        scientific_risk=0.0,
        context_sha256="a" * 64,
    )


def _observation() -> MatchedEffectObservation:
    return MatchedEffectObservation(
        endpoint_id="BBB_Martins",
        pair_id="pair-a",
        action_id="feature-domain-shift",
        native_oriented_delta=0.02,
        native_standard_error=0.001,
        effect_standardization_scale=0.005,
        context_similarity=1.0,
        endpoint_local=True,
        transport_eligible=True,
        test_derived=False,
        leaderboard_derived=False,
        single_axis_verified=True,
        complete_pair=True,
        provenance_sha256="b" * 64,
    )


def test_production_policy_ranks_metric_valued_pair():
    decision = MetricValuedECVOIPolicy().rank([_pair()], [_observation()])
    assert decision["policy_version"] == "metric-valued-ecvoi-v2"
    assert decision["decision"] == "execute_pair"
    assert decision["selected_pair_id"] == "pair-a"


def test_preregistered_policy_variants_change_only_named_term():
    assert (
        metric_valued_policy_for_variant("ecvoi-v2-no-voi").information_weight
        == 0
    )
    assert (
        metric_valued_policy_for_variant("ecvoi-v2-no-cost").compute_cost_weight
        == 0
    )
    assert (
        metric_valued_policy_for_variant("ecvoi-v2-no-risk").scientific_risk_weight
        == 0
    )
    assert (
        metric_valued_policy_for_variant("ecvoi-v2-no-context").transport_scope
        == "global"
    )
    assert (
        metric_valued_policy_for_variant(
            "ecvoi-v3"
        ).allow_one_endpoint_local_exploration
        is True
    )


def test_v3_allows_one_safe_local_exploration_then_restores_robust_gate():
    policy = metric_valued_policy_for_variant("ecvoi-v3")
    first = policy.rank([_pair()], [])
    assert first["decision"] == "execute_pair"
    assert first["decision_basis"] == "one_safe_endpoint_local_exploration"

    negative = MatchedEffectObservation(
        endpoint_id="BBB_Martins",
        pair_id="observed-pair",
        action_id="feature-domain-shift",
        native_oriented_delta=-0.02,
        native_standard_error=0.001,
        effect_standardization_scale=0.005,
        context_similarity=1.0,
        endpoint_local=True,
        transport_eligible=True,
        test_derived=False,
        leaderboard_derived=False,
        single_axis_verified=True,
        complete_pair=True,
        provenance_sha256="c" * 64,
    )
    second = policy.rank(
        [_pair()], [negative], current_episode_local_evidence=True
    )
    assert second["decision"] == "stop"
    assert second["decision_basis"] == "robust_utility"


def test_production_episode_reserves_pair_before_executor():
    observed_decisions = []

    def executor(action, decision):
        observed_decisions.append(decision)
        return PairExecution(
            pair_id=action.pair_id,
            action_id=action.action_id,
            treatment_run_ids=("t1", "t2", "t3"),
            control_run_ids=("c1", "c2", "c3"),
            ordered_seeds=(1, 2, 3),
            actual_fit_seconds=50,
            observation=_observation(),
        )

    result = run_episode(
        policy=MetricValuedECVOIPolicy(),
        actions=[_pair()],
        prior_observations=[_observation()],
        budget=EpisodeBudget(
            max_pairs=1, max_candidate_fits=2, max_fit_seconds=60
        ),
        executor=executor,
    )
    assert observed_decisions[0]["emitted_before_execution"] is True
    assert result.used_pairs == 1
    assert result.ledger[0]["event_type"] == "decision_emitted"
    assert result.ledger[1]["event_type"] == "pair_budget_reserved"
