import pytest

from scripts.ecvoi_v2_reference import (
    ActionPair,
    MatchedEffectObservation,
    MetricValuedECVOIPolicy,
)

SHA = "a" * 64


def _pair(action_id="feature", *, cost=10.0, risk=0.0, treatment=None):
    control = {
        "task": "AMES",
        "metric": "auroc",
        "split": "discovery",
        "representations": ["morgan"],
        "model": "lightgbm",
    }
    treatment = treatment or {**control, "representations": ["morgan", "rdkit2d"]}
    return ActionPair(
        pair_id=f"{action_id}-pair",
        action_id=action_id,
        axis="feature",
        treatment=treatment,
        control=control,
        ordered_seeds=(1, 2, 3),
        predicted_pair_fit_seconds=cost,
        candidate_charge=2,
        task_fit_budget_seconds=100.0,
        task_candidate_budget=20,
        practical_effect_threshold=0.01,
        effect_standardization_scale=0.01,
        predicted_native_observation_se=0.005,
        scientific_risk=risk,
        context_sha256=SHA,
    )


def _observation(
    endpoint,
    delta,
    *,
    pair_id="feature-pair",
    similarity=1.0,
    local=False,
    eligible=True,
):
    return MatchedEffectObservation(
        endpoint_id=endpoint,
        pair_id=pair_id,
        action_id="feature",
        native_oriented_delta=delta,
        native_standard_error=0.003,
        effect_standardization_scale=0.01,
        context_similarity=similarity,
        endpoint_local=local,
        transport_eligible=eligible,
        test_derived=False,
        leaderboard_derived=False,
        single_axis_verified=True,
        complete_pair=True,
        provenance_sha256=SHA,
    )


def test_action_pair_rejects_multi_axis_and_protected_changes():
    control = {
        "task": "AMES",
        "metric": "auroc",
        "split": "discovery",
        "representations": ["morgan"],
        "model": "lightgbm",
    }
    with pytest.raises(ValueError, match="also changes"):
        _pair(treatment={**control, "representations": ["rdkit2d"], "model": "rf"})
    with pytest.raises(ValueError, match="protected"):
        _pair(treatment={**control, "split": "test", "representations": ["rdkit2d"]})


def test_metric_posterior_uses_endpoint_as_unit_and_context_weights():
    observations = [
        _observation("local", 0.2, local=True),
        _observation("near", 0.2, similarity=1.0),
        _observation("far", -0.2, similarity=0.01),
    ]
    contextual = MetricValuedECVOIPolicy().posterior("feature", observations)
    global_value = MetricValuedECVOIPolicy(transport_scope="global").posterior(
        "feature", observations
    )
    assert contextual.mean > global_value.mean
    assert contextual.effective_endpoints == pytest.approx(2.01)
    with pytest.raises(ValueError, match="duplicate endpoint/pair"):
        MetricValuedECVOIPolicy().posterior(
            "feature", observations + [_observation("near", 0.1)]
        )


def test_multiple_pairs_on_one_endpoint_remain_one_statistical_unit():
    observations = [
        _observation("same-endpoint", 0.03, pair_id="pair-a", local=True),
        _observation("same-endpoint", 0.01, pair_id="pair-b", local=True),
    ]
    posterior = MetricValuedECVOIPolicy().posterior("feature", observations)
    assert posterior.admitted_endpoints == 1
    assert posterior.effective_endpoints == pytest.approx(1.0)


def test_test_or_leaderboard_evidence_is_rejected_at_construction():
    with pytest.raises(RuntimeError, match="forbidden"):
        MatchedEffectObservation(
            endpoint_id="x",
            pair_id="feature-pair",
            action_id="feature",
            native_oriented_delta=0.1,
            native_standard_error=0.1,
            effect_standardization_scale=0.01,
            context_similarity=1.0,
            endpoint_local=False,
            transport_eligible=True,
            test_derived=True,
            leaderboard_derived=False,
            single_axis_verified=True,
            complete_pair=True,
            provenance_sha256=SHA,
        )


def test_pair_cost_and_risk_change_ranking_and_stop_is_explicit():
    positive = [_observation("a", 0.25, local=True), _observation("b", 0.2)]
    cheap = _pair("cheap", cost=5.0)
    expensive = _pair("expensive", cost=90.0, risk=0.8)
    # Give both action IDs the same effect evidence.
    evidence = positive + [
        MatchedEffectObservation(
            **{
                **item.__dict__,
                "action_id": "cheap",
            }
        )
        for item in positive
    ] + [
        MatchedEffectObservation(
            **{
                **item.__dict__,
                "action_id": "expensive",
            }
        )
        for item in positive
    ]
    decision = MetricValuedECVOIPolicy().rank([expensive, cheap], evidence)
    assert decision["decision"] == "execute_pair"
    assert decision["selected_action_id"] == "cheap"
    assert decision["selected_pair_id"] == "cheap-pair"
    assert decision["emitted_before_execution"] is True

    stop = MetricValuedECVOIPolicy(
        information_weight=0.0,
        compute_cost_weight=1.0,
        scientific_risk_weight=1.0,
    ).rank([_pair("stop-me", cost=100.0, risk=1.0)], [])
    assert stop["decision"] == "stop"
    assert stop["selected_action_id"] is None
    assert stop["selected_pair_id"] is None


def test_multiple_pairs_can_share_one_action_family_posterior():
    first = _pair("feature")
    second = ActionPair(
        **{
            **first.__dict__,
            "pair_id": "feature-alternative-pair",
            "treatment": {
                **first.control,
                "representations": ["morgan", "avalon"],
            },
        }
    )
    evidence = [_observation("endpoint", 0.03, local=True)]
    decision = MetricValuedECVOIPolicy().rank([second, first], evidence)
    assert {row["action_id"] for row in decision["ranked_pairs"]} == {"feature"}
    assert {row["pair_id"] for row in decision["ranked_pairs"]} == {
        "feature-pair",
        "feature-alternative-pair",
    }


def test_native_metric_effects_are_compared_only_after_explicit_standardization():
    action = _pair("feature")
    observation = _observation("endpoint", 0.02, local=True)
    posterior = MetricValuedECVOIPolicy().posterior("feature", [observation])
    row = MetricValuedECVOIPolicy().evaluate(action, [observation])
    assert observation.standardized_oriented_delta() == pytest.approx(2.0)
    assert action.standardized_effect_threshold() == pytest.approx(1.0)
    assert posterior.mean > action.standardized_effect_threshold()
    assert row["effect_unit"] == "discovery_standardized_oriented_paired_delta"


def test_no_memory_excludes_transport_but_keeps_local_evidence():
    observations = [
        _observation("local", 0.2, local=True),
        _observation("transport", 0.5, local=False),
    ]
    posterior = MetricValuedECVOIPolicy(transport_scope="none").posterior(
        "feature", observations
    )
    assert posterior.admitted_endpoints == 1
    assert posterior.excluded_observations == 1
    assert posterior.evidence_source == "endpoint_local"
