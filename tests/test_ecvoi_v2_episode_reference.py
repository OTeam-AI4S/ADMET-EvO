from scripts.ecvoi_v2_episode_reference import (
    EpisodeBudget,
    PairExecution,
    run_episode,
)
from scripts.ecvoi_v2_reference import (
    ActionPair,
    MatchedEffectObservation,
    MetricValuedECVOIPolicy,
)

SHA = "a" * 64


def _pair(
    action_id: str, *, pair_id: str | None = None, seconds: float = 10.0
) -> ActionPair:
    control = {
        "task": "AMES",
        "metric": "auroc",
        "split": "discovery",
        "representations": ["morgan"],
    }
    pair_id = pair_id or f"{action_id}-pair"
    return ActionPair(
        pair_id=pair_id,
        action_id=action_id,
        axis="feature",
        treatment={**control, "representations": ["morgan", pair_id]},
        control=control,
        ordered_seeds=(1, 2, 3),
        predicted_pair_fit_seconds=seconds,
        candidate_charge=6,
        task_fit_budget_seconds=100.0,
        task_candidate_budget=18,
        practical_effect_threshold=0.01,
        effect_standardization_scale=0.01,
        predicted_native_observation_se=0.003,
        scientific_risk=0.0,
        context_sha256=SHA,
    )


def _observation(
    action_id: str, pair_id: str, delta: float = 0.03
) -> MatchedEffectObservation:
    return MatchedEffectObservation(
        endpoint_id="AMES-local",
        pair_id=pair_id,
        action_id=action_id,
        native_oriented_delta=delta,
        native_standard_error=0.002,
        effect_standardization_scale=0.01,
        context_similarity=1.0,
        endpoint_local=True,
        transport_eligible=False,
        test_derived=False,
        leaderboard_derived=False,
        single_axis_verified=True,
        complete_pair=True,
        provenance_sha256=SHA,
    )


def _complete(action: ActionPair, _: dict) -> PairExecution:
    return PairExecution(
        pair_id=action.pair_id,
        action_id=action.action_id,
        treatment_run_ids=tuple(f"{action.action_id}-t-{seed}" for seed in action.ordered_seeds),
        control_run_ids=tuple(f"{action.action_id}-c-{seed}" for seed in action.ordered_seeds),
        ordered_seeds=action.ordered_seeds,
        actual_fit_seconds=action.predicted_pair_fit_seconds - 1,
        observation=_observation(action.action_id, action.pair_id),
    )


def test_episode_emits_decision_reserves_pair_then_updates_posterior():
    result = run_episode(
        policy=MetricValuedECVOIPolicy(),
        actions=[_pair("a"), _pair("b")],
        prior_observations=[],
        budget=EpisodeBudget(2, 12, 20),
        executor=_complete,
    )
    assert result.executed_pair_ids == ("a-pair", "b-pair")
    assert result.executed_action_ids == ("a", "b")
    assert len(result.observations) == 2
    assert result.used_candidate_fits == 12
    assert [row["event_type"] for row in result.ledger[:3]] == [
        "decision_emitted",
        "pair_budget_reserved",
        "pair_execution_complete",
    ]
    assert result.ledger[2]["posterior_updated"] is True


def test_incomplete_pair_consumes_budget_but_cannot_update_posterior():
    def incomplete(action: ActionPair, _: dict) -> PairExecution:
        return PairExecution(
            pair_id=action.pair_id,
            action_id=action.action_id,
            treatment_run_ids=("t-1", "t-2"),
            control_run_ids=("c-1", "c-2"),
            ordered_seeds=(1, 2),
            actual_fit_seconds=4,
            observation=None,
            failure_reason="third_seed_timeout",
        )

    result = run_episode(
        policy=MetricValuedECVOIPolicy(),
        actions=[_pair("a")],
        prior_observations=[],
        budget=EpisodeBudget(1, 6, 10),
        executor=incomplete,
    )
    assert result.observations == ()
    assert result.used_pairs == 1
    assert result.used_candidate_fits == 6
    assert result.ledger[2]["event_type"] == "pair_execution_failed"
    assert result.ledger[2]["posterior_updated"] is False


def test_completed_pair_updates_remaining_pair_in_same_action_family():
    result = run_episode(
        policy=MetricValuedECVOIPolicy(),
        actions=[
            _pair("feature-family", pair_id="pair-a"),
            _pair("feature-family", pair_id="pair-b"),
        ],
        prior_observations=[],
        budget=EpisodeBudget(2, 12, 20),
        executor=_complete,
    )
    second_decision = result.ledger[3]["decision"]
    assert second_decision["selected_pair_id"] == "pair-b"
    assert second_decision["ranked_pairs"][0]["posterior"]["admitted_endpoints"] == 1
    assert second_decision["ranked_pairs"][0]["posterior"]["evidence_source"] == "endpoint_local"


def test_budget_that_cannot_fit_a_complete_pair_stops_without_execution():
    called = False

    def forbidden_executor(action: ActionPair, decision: dict) -> PairExecution:
        nonlocal called
        called = True
        return _complete(action, decision)

    result = run_episode(
        policy=MetricValuedECVOIPolicy(),
        actions=[_pair("a", seconds=10)],
        prior_observations=[],
        budget=EpisodeBudget(1, 5, 9),
        executor=forbidden_executor,
    )
    assert called is False
    assert result.stop_reason == "budget_exhausted_before_complete_pair"
    assert result.used_pairs == 0
    assert [row["event_type"] for row in result.ledger] == ["episode_stopped"]
