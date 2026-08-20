import pytest

from admet_agent.brain import RuleBasedBrain
from admet_agent.guard import ProtocolViolation, validate_experiment
from admet_agent.memory import ExperimentMemory
from admet_agent.registry import planner_catalog
from admet_agent.schemas import ExperimentSpec, TaskProfile


def profile(task_type="classification", positive_rate=0.1):
    return TaskProfile(
        task="HIA_Hou" if task_type == "classification" else "Caco2_Wang",
        task_type=task_type,
        metric="auroc" if task_type == "classification" else "mae",
        n_train=100,
        n_valid=20,
        positive_rate_train=positive_rate if task_type == "classification" else None,
        target_mean=0.1,
        target_std=0.2,
        invalid_smiles_train=0,
        unique_scaffolds_train=50,
        unseen_valid_scaffold_rate=0.8,
    )


def test_experiment_round_trip():
    spec = ExperimentSpec(task="Caco2_Wang", seed=1)
    assert ExperimentSpec.from_dict(spec.to_dict()) == spec


def test_guard_rejects_test_fields():
    spec = ExperimentSpec(task="Caco2_Wang", seed=1, hyperparameters={"y_test": [1]})
    with pytest.raises(ProtocolViolation, match="forbidden"):
        validate_experiment(spec)


def test_guard_rejects_classification_strategy_for_regression():
    spec = ExperimentSpec(task="Caco2_Wang", seed=1, imbalance="class_weight")
    with pytest.raises(ProtocolViolation, match="classification-only"):
        validate_experiment(spec)


def test_rule_brain_uses_imbalance_diagnosis():
    proposals = RuleBasedBrain(max_candidates=2).propose(profile(), [])
    assert len(proposals) == 2
    assert all(item.imbalance == "class_weight" for item in proposals)


def test_memory_is_append_only_jsonl(tmp_path):
    memory = ExperimentMemory(tmp_path / "memory.jsonl")
    memory.append({"task": "HIA_Hou", "score": 0.7})
    memory.append({"task": "Caco2_Wang", "score": 0.2})
    assert list(memory.records("hia_hou")) == [{"score": 0.7, "task": "HIA_Hou"}]
    assert len((tmp_path / "memory.jsonl").read_text().splitlines()) == 2


def test_unimol2_is_exposed_with_dependency_availability():
    representations = {
        item["name"]: item for item in planner_catalog()["representations"]
    }
    assert representations["unimol2"]["family"] == "pretrained_3d"
    assert isinstance(representations["unimol2"]["available"], bool)
