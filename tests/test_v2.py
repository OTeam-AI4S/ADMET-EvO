import json
import sys
import types

import numpy as np
import pandas as pd
import pytest

from admet_agent.brain import HybridBrain, RuleBasedBrain
from admet_agent.experiment_db import ExperimentDatabase
from admet_agent.features import MolecularFeaturizer
from admet_agent.guard import ProtocolViolation, validate_experiment
from admet_agent.runner import ADMETAgent, _percentile_ranks
from admet_agent.schemas import ExperimentSpec, RunResult, TaskProfile
from admet_agent.tasks import resolve_task


def regression_profile(task="Caco2_Wang"):
    return TaskProfile(
        task=task, task_type="regression", metric="mae", n_train=50, n_valid=10,
        positive_rate_train=None, target_mean=0.0, target_std=1.0,
        invalid_smiles_train=0, unique_scaffolds_train=30,
        unseen_valid_scaffold_rate=0.7,
    )


def test_all_v2_representations_compute_finite_values():
    names = (
        "morgan", "morgan_r3", "morgan_count", "avalon", "erg",
        "maccs", "rdkit_fp", "atom_pair", "torsion", "rdkit2d",
    )
    for name in names:
        values = MolecularFeaturizer((name,)).transform(["CCO", "c1ccccc1"])
        assert values.shape[0] == 2
        assert values.shape[1] > 0
        assert np.isfinite(values).all()


@pytest.mark.parametrize("current_api", [False, True])
def test_unimol2_representation_preserves_rows_and_supports_both_apis(
    monkeypatch, current_api
):
    calls = {}

    class FakeUniMolRepr:
        def __init__(self, **kwargs):
            calls["init"] = kwargs

        def get_repr(self, smiles, return_atomic_reprs=False):
            calls["smiles"] = smiles
            calls["return_atomic_reprs"] = return_atomic_reprs
            values = [[float(index), float(len(value))] for index, value in enumerate(smiles)]
            return values if current_api else {"cls_repr": values}

    module = types.ModuleType("unimol_tools")
    module.UniMolRepr = FakeUniMolRepr
    monkeypatch.setitem(sys.modules, "unimol_tools", module)

    values = MolecularFeaturizer(("unimol2",), device="cpu").transform(
        ["CCO", "not-a-smiles", "CC"]
    )

    assert values.shape == (3, 2)
    assert np.array_equal(values[1], np.zeros(2, dtype=np.float32))
    assert calls["smiles"] == ["CCO", "CC"]
    assert calls["init"]["model_name"] == "unimolv2"
    assert calls["init"]["model_size"] == "84m"
    assert calls["init"]["use_cuda"] is False
    assert calls["return_atomic_reprs"] is False


def test_rule_brain_requires_transport_eligible_memory_for_unimol2(monkeypatch):
    monkeypatch.setattr("admet_agent.brain.is_representation_available", lambda name: True)
    brain = RuleBasedBrain(max_candidates=2)
    brain.set_context(
        cross_task_evidence=[],
        hardness_card={
            "dimensions": {
                "domain_shift": {"severity": 0.8, "confidence": 0.8}
            }
        },
        research_plan={
            "selected_action_id": "feature-domain-shift",
            "actions": [
                {
                    "action_id": "feature-domain-shift",
                    "prior_evidence": {
                        "routing_eligible": True,
                        "mean_oriented_delta": 0.01,
                        "posterior_support_probability": 0.6,
                    },
                }
            ],
        },
    )
    proposals = brain.propose(regression_profile(), [])
    assert any("unimol2" in item.representations for item in proposals)


def test_guard_rejects_duplicates_and_bad_types():
    with pytest.raises(ProtocolViolation, match="duplicates"):
        validate_experiment(ExperimentSpec(
            task="Caco2_Wang", seed=1, representations=("morgan", "morgan")
        ))
    with pytest.raises(ProtocolViolation, match="numeric"):
        validate_experiment(ExperimentSpec(
            task="Caco2_Wang", seed=1,
            hyperparameters={"n_estimators": "many"},
        ))


def test_sqlite_memory_returns_cross_task_recipe(tmp_path):
    database = ExperimentDatabase(tmp_path / "experiments.sqlite")
    for task, model, value in (
        ("Caco2_Wang", "random_forest", 0.2),
        ("Caco2_Wang", "extra_trees", 0.4),
        ("PPBR_AZ", "random_forest", 0.3),
        ("PPBR_AZ", "extra_trees", 0.5),
    ):
        profile = regression_profile(task)
        spec = ExperimentSpec(task=task, seed=1, model=model)
        result = RunResult(
            f"{task}-{model}", task, 1, model, "mae", value, 1.0
        )
        database.record(profile, spec, result)
    insights = database.cross_task_insights("regression", "mae")
    assert insights[0]["model"] == "random_forest"
    assert insights[0]["evidence_tasks"] == 2
    assert database.summary() == {"runs": 4, "tasks": 2, "completed": 4}


def test_evolutionary_brain_proposes_new_feature_hypotheses():
    profile = regression_profile()
    brain = RuleBasedBrain(max_candidates=3)
    first = brain.propose(profile, [])
    history = [
        RunResult(
            f"run-{index}", profile.task, 1, item.model, profile.metric,
            0.3 + index / 100, 1.0, representations=item.representations,
            hyperparameters=item.hyperparameters,
        )
        for index, item in enumerate(first)
    ]
    second = brain.propose(profile, history)
    assert second
    first_signatures = {
        (item.model, item.representations, tuple(sorted(item.hyperparameters.items())))
        for item in first
    }
    assert all(
        (item.model, item.representations, tuple(sorted(item.hyperparameters.items())))
        not in first_signatures
        for item in second
    )


def test_ensemble_normalization_and_test_freeze(tmp_path):
    member = ExperimentSpec(task="Caco2_Wang", seed=1).to_dict()
    normalized = ADMETAgent._normalize_frozen(
        __import__("admet_agent.tasks", fromlist=["resolve_task"]).resolve_task("Caco2_Wang"),
        {"kind": "ensemble", "members": [member, member], "weights": [2, 1]},
    )
    assert normalized["weights"] == pytest.approx([2 / 3, 1 / 3])
    assert normalized["aggregation"] == "probability"

    ranked = ADMETAgent._normalize_frozen(
        __import__("admet_agent.tasks", fromlist=["resolve_task"]).resolve_task("HIA_Hou"),
        {
            "kind": "ensemble",
            "members": [
                {**member, "task": "HIA_Hou"},
                {**member, "task": "HIA_Hou"},
            ],
            "weights": [1, 1],
            "aggregation": "rank",
        },
    )
    assert ranked["aggregation"] == "rank"

    agent = object.__new__(ADMETAgent)
    agent.run_dir = tmp_path
    agent._lock_finalists(["Caco2_Wang"], {"Caco2_Wang": normalized})
    changed = json.loads(json.dumps(normalized))
    changed["weights"] = [0.5, 0.5]
    with pytest.raises(RuntimeError, match="firewall"):
        agent._lock_finalists(["Caco2_Wang"], {"Caco2_Wang": changed})


def test_percentile_ranks_average_ties():
    assert _percentile_ranks(np.asarray([3.0, 1.0, 3.0, 2.0])).tolist() == pytest.approx(
        [0.875, 0.25, 0.875, 0.5]
    )



def test_official_finalize_requires_five_unique_seeds():
    agent = object.__new__(ADMETAgent)
    with pytest.raises(ValueError, match="five unique"):
        agent.finalize(["Caco2_Wang"], [1, 2, 3], {"Caco2_Wang": "random_forest"})


class StubPlanner:
    def propose(self, profile, history):
        return [
            ExperimentSpec(
                task=profile.task,
                seed=1,
                model="lightgbm",
                representations=("morgan", "rdkit2d"),
                hypothesis="stub GPT candidate",
            )
        ]

    def select_finalist(self, profile, results):
        raise AssertionError("Hybrid finalist selection must be metric-deterministic")


def test_hybrid_brain_uses_broad_baseline_then_combines_planners():
    profile = regression_profile()
    brain = HybridBrain(StubPlanner())
    first = brain.propose(profile, [])
    assert {"xgboost", "lightgbm", "random_forest"}.issubset(
        {item.model for item in first}
    )
    history = [
        RunResult(
            f"run-{index}", profile.task, 1, item.model, profile.metric,
            0.3 + index / 100, 1.0, representations=item.representations,
            hyperparameters=item.hyperparameters,
        )
        for index, item in enumerate(first)
    ]
    second = brain.propose(profile, history)
    assert second[0].hypothesis == "stub GPT candidate"
    assert len(second) <= 6



def test_feature_cache_is_order_invariant(tmp_path):
    from admet_agent.features import FeatureCache

    cache = FeatureCache(tmp_path)
    first = cache.get_or_compute(["CCO", "CC", "CCO"], ("morgan",))
    second = cache.get_or_compute(["CC", "CCO"], ("morgan",))
    assert len(list(tmp_path.glob("*.npy"))) == 1
    assert np.array_equal(first[0], second[1])
    assert np.array_equal(first[1], second[0])
    assert np.array_equal(first[0], first[2])



def test_single_finalist_internal_kind_is_not_an_experiment_field():
    task = __import__("admet_agent.tasks", fromlist=["resolve_task"]).resolve_task("Caco2_Wang")
    normalized = ADMETAgent._normalize_frozen(task, "random_forest")
    member = {key: value for key, value in normalized.items() if key != "kind"}
    assert ExperimentSpec.from_dict(member).model == "random_forest"


def test_recipe_finalist_normalization_freezes_protocol():
    task = __import__("admet_agent.tasks", fromlist=["resolve_task"]).resolve_task("DILI")
    normalized = ADMETAgent._normalize_frozen(
        task,
        {
            "kind": "recipe",
            "recipe": "raw(2d_lightgbm@0.50+finetuned_unimol2@0.50)",
        },
    )

    assert normalized["kind"] == "recipe"
    assert normalized["protocol"] == {
        "model_size": "84m",
        "learning_rate": 1e-4,
        "epochs": 20,
        "batch_size": 16,
        "inner_folds": 5,
    }


def test_recipe_runner_passes_only_prediction_smiles_to_executor(monkeypatch, tmp_path):
    calls = {}

    class FakeExecutor:
        def __init__(self, **kwargs):
            calls["init"] = kwargs

        def predict(self, **kwargs):
            calls["predict"] = kwargs
            return np.asarray([0.25, 0.75])

    monkeypatch.setattr("admet_agent.runner.FrozenRecipeExecutor", FakeExecutor)
    agent = object.__new__(ADMETAgent)
    agent.cache = None
    agent.run_dir = tmp_path
    agent.device = "cpu"
    agent.n_jobs = 1
    train = pd.DataFrame({"Drug": ["CC", "CCC"], "Y": [0, 1]})
    valid = pd.DataFrame({"Drug": ["CO"], "Y": [1]})
    test = pd.DataFrame({"Drug": ["CN", "CF"], "Y": [99, 99]})
    task = resolve_task("DILI")
    frozen = ADMETAgent._normalize_frozen(
        task, {"kind": "recipe", "recipe": "finetuned_unimol2"}
    )

    result = agent._predict_frozen(task, frozen, 1, train, valid, test, "Drug", "Y")

    assert result.tolist() == [0.25, 0.75]
    assert calls["predict"]["predict_smiles"] == ["CN", "CF"]
    assert "test" not in calls["predict"]
