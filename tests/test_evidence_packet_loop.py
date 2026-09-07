from pathlib import Path

import pytest

from admet_agent.evidence_packet_loop import (
    CANONICAL_TASKS,
    EvidencePacketLoopStore,
    evaluate_packet_trial,
    weakest_tasks,
)


def _scores(default: float = 0.0) -> dict[str, float]:
    return {task: default for task in CANONICAL_TASKS}


def test_weak_task_targeting_uses_bottom_two_or_three():
    scores = _scores(0.1)
    scores["hERG"] = -0.2
    scores["DILI"] = -0.1
    scores["HIA_Hou"] = -0.05
    assert weakest_tasks(scores, count=2) == ("hERG", "DILI")
    assert weakest_tasks(scores, count=3) == ("hERG", "DILI", "HIA_Hou")


def test_trial_requires_one_packet_mutation_and_all22_scores():
    scores = _scores()
    with pytest.raises(ValueError, match="add, remove, or replace one"):
        evaluate_packet_trial(
            incumbent_packet_ids=[],
            challenger_packet_ids=["a", "b"],
            incumbent_scores=scores,
            challenger_scores=scores,
            packet_task_by_id={"a": "hERG", "b": "DILI"},
        )
    incomplete = dict(scores)
    incomplete.pop("hERG")
    with pytest.raises(ValueError, match="exactly all 22"):
        weakest_tasks(incomplete)


def test_new_packet_must_target_current_weak_task():
    scores = _scores()
    scores["hERG"] = -0.2
    scores["DILI"] = -0.1
    scores["HIA_Hou"] = -0.05
    with pytest.raises(ValueError, match="bottom-2/3"):
        evaluate_packet_trial(
            incumbent_packet_ids=[],
            challenger_packet_ids=["packet-a"],
            incumbent_scores=scores,
            challenger_scores=scores,
            packet_task_by_id={"packet-a": "AMES"},
        )


def test_frozen_acquisition_targets_override_zero_baseline_tie():
    scores = _scores()
    decision = evaluate_packet_trial(
        incumbent_packet_ids=[],
        challenger_packet_ids=["packet-a"],
        incumbent_scores=scores,
        challenger_scores=scores,
        packet_task_by_id={"packet-a": "CYP3A4_Veith"},
        acquisition_target_tasks=("AMES", "CYP3A4_Veith", "CYP2C9_Veith"),
    )
    assert decision.target_tasks == ("AMES", "CYP3A4_Veith", "CYP2C9_Veith")
    assert decision.targeted_weak_task is True


def test_manual_score_submission_is_disabled(tmp_path: Path):
    scores = _scores()
    scores["hERG"] = -0.1
    store = EvidencePacketLoopStore(tmp_path / "state.json")
    store.initialize(packet_ids=[], scores=scores)
    with pytest.raises(RuntimeError, match="manual score submission is disabled"):
        store.submit_trial(
            challenger_packet_ids=["packet-a"],
            challenger_scores=scores,
            packet_task_by_id={"packet-a": "hERG"},
        )
