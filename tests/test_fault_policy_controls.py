import json
from pathlib import Path

from admet_agent.fault_policy_controls import compare_fault_controls
from admet_agent.hidden_fault_benchmark import FaultChallengeResult

ROOT = Path(__file__).parents[1]


def _rows():
    payload = json.loads(
        (ROOT / "reports/v3_hidden_fault_suite.json").read_text(encoding="utf-8")
    )
    return [FaultChallengeResult.from_dict(row) for row in payload["results"]]


def test_control_replay_reveals_automl_advantage_but_rules_tie():
    report = compare_fault_controls(_rows())
    agent = report["arms"]["agent_contract_guard"]["score"]
    rules = report["arms"]["complete_rules_guard"]
    automl = report["arms"]["conventional_automl_no_guard"]
    schema = report["arms"]["expanded_automl_schema_guard"]
    no_hardness = report["arms"]["agent_no_hardness_diagnosis"]
    assert agent["micro_f1"] == 1.0
    assert rules["score"]["micro_f1"] == 1.0
    assert rules["agent_comparison"]["bootstrap_95_ci"] == [0.0, 0.0]
    assert automl["score"]["micro_recall"] == 0.0
    assert automl["score"]["safety_gate_passed"] is False
    assert schema["score"]["micro_recall"] < 1.0
    assert schema["score"]["critical_recall"] < 1.0
    assert no_hardness["score"]["micro_recall"] == 0.0
    assert no_hardness["score"]["safety_gate_passed"] is False
    assert report["test_loaded"] is False


def test_control_replay_is_deterministic_for_fixed_policy_seed():
    assert compare_fault_controls(_rows(), policy_seed=7) == compare_fault_controls(
        _rows(), policy_seed=7
    )
