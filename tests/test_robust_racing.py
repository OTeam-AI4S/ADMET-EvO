from __future__ import annotations

import hashlib
import json
import math

import pytest

from admet_agent.robust_racing import (
    RobustFutilityPolicy,
    choose_incumbent,
    evaluate_futility,
    evaluate_race,
)
from admet_agent.schemas import RunResult


def run(
    config: str,
    seed: int,
    score: float,
    seconds: float,
    *,
    status: str = "completed",
) -> RunResult:
    return RunResult(
        run_id=f"{config}-{seed}",
        task="HIA_Hou",
        seed=seed,
        model="extra_trees",
        metric="auroc",
        validation_score=score,
        elapsed_seconds=seconds,
        status=status,
    )


def race_rows(
    incumbent_scores=(0.80, 0.82),
    candidate_scores=(0.79, 0.80),
    incumbent_seconds=(1.0, 1.0),
    candidate_seconds=(3.0, 3.0),
):
    return {
        "incumbent": {
            seed: run("incumbent", seed, score, seconds)
            for seed, score, seconds in zip((1, 2), incumbent_scores, incumbent_seconds)
        },
        "candidate": {
            seed: run("candidate", seed, score, seconds)
            for seed, score, seconds in zip((1, 2), candidate_scores, candidate_seconds)
        },
    }


def test_maximize_candidate_stops_only_when_all_three_thresholds_hold():
    rows = race_rows()
    decision = evaluate_futility("candidate", "incumbent", rows, maximize=True)
    assert decision.decision == "stop"
    assert decision.reason == "futility_thresholds_met"
    assert decision.oriented_gaps == pytest.approx((0.01, 0.02))
    assert decision.mean_oriented_gap == pytest.approx(0.015)
    assert decision.time_ratio == pytest.approx(3.0)
    assert decision.worse_every_seed is True
    assert decision.gap_threshold_met is True
    assert decision.time_threshold_met is True


def test_mae_uses_lower_is_better_orientation():
    rows = race_rows(
        incumbent_scores=(0.40, 0.42),
        candidate_scores=(0.41, 0.44),
    )
    decision = evaluate_futility("candidate", "incumbent", rows, maximize=False)
    assert decision.decision == "stop"
    assert decision.oriented_gaps == pytest.approx((0.01, 0.02))


@pytest.mark.parametrize(
    ("rows", "failed_criterion"),
    [
        (
            race_rows(candidate_scores=(0.81, 0.79)),
            "worse_every_seed",
        ),
        (
            race_rows(candidate_scores=(0.797, 0.817)),
            "gap_threshold_met",
        ),
        (
            race_rows(candidate_seconds=(2.9, 2.9)),
            "time_threshold_met",
        ),
    ],
)
def test_each_futility_requirement_is_necessary(rows, failed_criterion):
    decision = evaluate_futility("candidate", "incumbent", rows, maximize=True)
    assert decision.decision == "continue"
    assert getattr(decision, failed_criterion) is False


def test_gap_and_time_thresholds_are_inclusive_at_exact_boundary():
    rows = race_rows(
        incumbent_scores=(0.005, 0.005),
        candidate_scores=(0.0, 0.0),
        incumbent_seconds=(1.0, 1.0),
        candidate_seconds=(3.0, 3.0),
    )
    decision = evaluate_futility("candidate", "incumbent", rows, maximize=True)
    assert decision.decision == "stop"
    assert decision.mean_oriented_gap == pytest.approx(0.005)
    assert decision.time_ratio == pytest.approx(3.0)


def test_structurally_supplied_protected_id_cannot_be_stopped():
    rows = race_rows()
    decision = evaluate_futility(
        "candidate",
        "incumbent",
        rows,
        maximize=True,
        protected_ids=frozenset({"candidate"}),
    )
    assert decision.decision == "continue"
    assert decision.reason == "protected_selected_action_pair"
    assert decision.protected is True
    assert decision.gap_threshold_met is True
    assert decision.time_threshold_met is True


def test_incumbent_is_never_stopped():
    rows = race_rows()
    decision = evaluate_futility("incumbent", "incumbent", rows, maximize=True)
    assert decision.decision == "continue"
    assert decision.reason == "incumbent"


def test_incumbent_tie_breaks_by_time_then_lexical_id():
    rows = {
        "slow": {1: run("slow", 1, 0.8, 2), 2: run("slow", 2, 0.8, 2)},
        "z-fast": {
            1: run("z-fast", 1, 0.8, 1),
            2: run("z-fast", 2, 0.8, 1),
        },
        "a-fast": {
            1: run("a-fast", 1, 0.8, 1),
            2: run("a-fast", 2, 0.8, 1),
        },
    }
    assert choose_incumbent(rows, maximize=True) == "a-fast"
    assert choose_incumbent(rows, maximize=False) == "a-fast"


def test_incumbent_selection_respects_metric_direction():
    rows = {
        "high": {1: run("high", 1, 0.9, 1), 2: run("high", 2, 0.8, 1)},
        "low": {1: run("low", 1, 0.4, 1), 2: run("low", 2, 0.5, 1)},
    }
    assert choose_incumbent(rows, maximize=True) == "high"
    assert choose_incumbent(rows, maximize=False) == "low"


def test_missing_seed_holds_the_global_barrier():
    rows = race_rows()
    rows["candidate"].pop(2)
    decisions = evaluate_race(rows, maximize=True)
    assert choose_incumbent(rows, maximize=True) is None
    assert {item.decision for item in decisions} == {"defer"}


@pytest.mark.parametrize(
    ("candidate_runs", "reason"),
    [
        ({1: run("c", 1, 0.7, 3)}, "candidate_missing_checkpoint_seed"),
        (
            {
                1: run("c", 1, 0.7, 3),
                2: run("c", 2, 0.7, 3, status="failed"),
            },
            "candidate_failed_checkpoint_fit",
        ),
        (
            {1: run("c", 1, math.nan, 3), 2: run("c", 2, 0.7, 3)},
            "candidate_nonfinite_score",
        ),
        (
            {1: run("c", 1, 0.7, math.inf), 2: run("c", 2, 0.7, 3)},
            "candidate_invalid_elapsed",
        ),
    ],
)
def test_missing_failed_and_nonfinite_candidate_evidence_defers(candidate_runs, reason):
    rows = race_rows()
    rows["candidate"] = candidate_runs
    decision = evaluate_futility("candidate", "incumbent", rows, maximize=True)
    assert decision.decision == "defer"
    assert decision.reason == reason
    assert decision.mean_oriented_gap is None


def test_invalid_incumbent_evidence_defers():
    rows = race_rows()
    rows["incumbent"].pop(2)
    decision = evaluate_futility("candidate", "incumbent", rows, maximize=True)
    assert decision.decision == "defer"
    assert decision.reason == "incumbent_missing_checkpoint_seed"


def test_race_is_mapping_order_independent_and_hash_stable():
    rows = race_rows()
    forward = evaluate_race(rows, maximize=True)
    reverse = evaluate_race(dict(reversed(list(rows.items()))), maximize=True)
    assert [item.to_dict() for item in forward] == [item.to_dict() for item in reverse]
    assert all(
        item.decision_id == item.to_dict()["decision_sha256"] for item in forward
    )


def test_decision_hash_changes_when_protection_changes():
    rows = race_rows()
    stopped = evaluate_futility("candidate", "incumbent", rows, maximize=True)
    protected = evaluate_futility(
        "candidate",
        "incumbent",
        rows,
        maximize=True,
        protected_ids=frozenset({"candidate"}),
    )
    assert stopped.decision_sha256 != protected.decision_sha256


def test_to_dict_hash_covers_the_complete_decision_payload():
    decision = evaluate_futility("candidate", "incumbent", race_rows(), maximize=True)
    payload = decision.to_dict()
    claimed = payload.pop("decision_sha256")
    actual = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    assert claimed == actual


def test_race_with_no_valid_incumbent_defers_every_candidate():
    rows = {
        "failed": {
            1: run("failed", 1, 0.7, 1, status="failed"),
            2: run("failed", 2, 0.7, 1),
        }
    }
    decisions = evaluate_race(rows, maximize=True)
    assert choose_incumbent(rows, maximize=True) is None
    assert len(decisions) == 1
    assert decisions[0].decision == "defer"


@pytest.mark.parametrize(
    "kwargs",
    [
        {"checkpoint_seeds": (1, 1)},
        {"minimum_oriented_mean_gap": 0},
        {"minimum_time_ratio": 1},
        {"policy_version": ""},
    ],
)
def test_invalid_policy_is_rejected(kwargs):
    with pytest.raises(ValueError):
        RobustFutilityPolicy(**kwargs)
