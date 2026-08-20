import pytest

from admet_agent.brain import RuleBasedBrain
from admet_agent.guard import ProtocolViolation, validate_experiment
from admet_agent.openai_brain import Hyperparameters
from admet_agent.schemas import ExperimentSpec, RunResult, TaskProfile


def test_hyperparameters_are_filtered_per_model():
    values = Hyperparameters(
        n_estimators=500, min_samples_leaf=2, max_features=0.7,
        n_members=None, majority_ratio=None,
        criterion="squared_error", bootstrap=True, max_samples=0.8,
        iterations=700, depth=7, learning_rate=0.04, l2_leaf_reg=3.0,
        random_strength=1.0, max_depth=6, subsample=0.8,
        colsample_bytree=0.7, min_child_weight=2.0, reg_alpha=0.1,
        reg_lambda=2.0, num_leaves=31, min_child_samples=15,
        C=10.0, gamma="scale", epsilon=0.1,
    )
    assert set(values.compact("random_forest")) == {
        "n_estimators", "min_samples_leaf", "max_features",
        "criterion", "bootstrap", "max_samples",
    }
    assert set(values.compact("catboost")) == {
        "iterations", "depth", "learning_rate", "l2_leaf_reg", "random_strength",
    }


def test_guard_rejects_model_specific_and_incompatible_parameters():
    with pytest.raises(ProtocolViolation):
        validate_experiment(ExperimentSpec(
            task="Caco2_Wang", seed=1, model="catboost",
            hyperparameters={"n_estimators": 400},
        ))
    with pytest.raises(ProtocolViolation):
        validate_experiment(ExperimentSpec(
            task="Caco2_Wang", seed=1, model="random_forest",
            hyperparameters={"criterion": "gini"},
        ))
    with pytest.raises(ProtocolViolation):
        validate_experiment(ExperimentSpec(
            task="Caco2_Wang", seed=1, model="random_forest",
            hyperparameters={"bootstrap": False, "max_samples": 0.8},
        ))


def test_rule_brain_selects_run_id_not_model_name():
    profile = TaskProfile(
        task="Caco2_Wang", task_type="regression", metric="mae",
        n_train=10, n_valid=2, positive_rate_train=None, target_mean=0.0,
        target_std=1.0, invalid_smiles_train=0, unique_scaffolds_train=8,
        unseen_valid_scaffold_rate=0.5,
    )
    runs = [
        RunResult("run-a", profile.task, 1, "random_forest", "mae", 0.3, 1.0),
        RunResult("run-b", profile.task, 1, "random_forest", "mae", 0.2, 1.0),
    ]
    assert RuleBasedBrain().select_finalist(profile, runs) == "run-b"
