import pytest

from admet_agent.evidence_router import EvidenceRouter
from admet_agent.runner import ADMETAgent
from admet_agent.schemas import RunResult, TaskProfile


def _profile(metric="mae"):
    return TaskProfile(
        task="Example",
        task_type="regression" if metric == "mae" else "classification",
        metric=metric,
        n_train=100,
        n_valid=20,
        positive_rate_train=None,
        target_mean=0.0,
        target_std=1.0,
        invalid_smiles_train=0,
        unique_scaffolds_train=80,
        unseen_valid_scaffold_rate=1.0,
    )


def _run(run_id, score, std, seconds, representations=("morgan",), model="lightgbm"):
    return RunResult(
        run_id=run_id,
        task="Example",
        seed=1,
        model=model,
        metric="mae",
        validation_score=score,
        validation_std=std,
        elapsed_seconds=seconds,
        representations=representations,
    )


def test_cost_aware_lcb_rejects_unreliable_expensive_3d_gain():
    cheap = _run("cheap", 0.41, 0.005, 10)
    expensive = _run("expensive", 0.40, 0.08, 1000, ("unimol2",))

    decision = EvidenceRouter(uncertainty_penalty=0.5, cost_penalty=0.01).select(
        _profile(), [expensive, cheap]
    )

    assert decision["selected_run_id"] == "cheap"
    assert decision["selected_action"] == "stop_2d"
    assert decision["best_cheap_run_id"] == "cheap"
    assert decision["candidates"][1]["action"] == "acquire_3d"


def test_lcb_handles_maximize_metrics_and_identifies_fusion():
    single = _run("single", 0.80, 0.01, 10)
    blend = _run("blend", 0.84, 0.01, 20, model="ensemble")
    single = RunResult(**{**single.to_dict(), "metric": "auroc"})
    blend = RunResult(**{**blend.to_dict(), "metric": "auroc"})

    decision = EvidenceRouter(uncertainty_penalty=0.5, cost_penalty=0.0).select(
        _profile("auroc"), [single, blend]
    )

    assert decision["selected_run_id"] == "blend"
    assert decision["selected_action"] == "fuse"
    assert decision["policy"] == "lcb"


def test_agent_finalist_branch_uses_router_and_returns_audit():
    agent = object.__new__(ADMETAgent)
    agent.selection_policy = "cost-aware-lcb"
    agent.evidence_router = EvidenceRouter(
        uncertainty_penalty=0.5, cost_penalty=0.01
    )
    cheap = _run("cheap", 0.41, 0.005, 10)
    expensive = _run("expensive", 0.40, 0.08, 1000, ("unimol2",))

    selected, audit = agent._select_finalist(_profile(), [expensive, cheap])

    assert selected == "cheap"
    assert audit is not None
    assert audit["policy"] == "cost_aware_lcb"
    assert audit["selected_action"] == "stop_2d"


@pytest.mark.parametrize("field", ["uncertainty_penalty", "cost_penalty"])
def test_router_rejects_negative_penalties(field):
    with pytest.raises(ValueError, match=field):
        EvidenceRouter(**{field: -0.1})
