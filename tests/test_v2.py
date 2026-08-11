import json

import numpy as np
import pytest

from admet_agent.brain import HybridBrain, RuleBasedBrain
from admet_agent.experiment_db import ExperimentDatabase
from admet_agent.features import MolecularFeaturizer
from admet_agent.guard import ProtocolViolation, validate_experiment
from admet_agent.runner import ADMETAgent
from admet_agent.schemas import ExperimentSpec, RunResult, TaskProfile


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

    agent = object.__new__(ADMETAgent)
    agent.run_dir = tmp_path
    agent._lock_finalists(["Caco2_Wang"], {"Caco2_Wang": normalized})
    changed = json.loads(json.dumps(normalized))
    changed["weights"] = [0.5, 0.5]
    with pytest.raises(RuntimeError, match="firewall"):
        agent._lock_finalists(["Caco2_Wang"], {"Caco2_Wang": changed})



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
