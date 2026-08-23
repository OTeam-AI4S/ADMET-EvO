from admet_agent.schemas import TaskProfile
from scripts.run_minimal_predictive_control import (
    build_arm_specs,
    choose_screen_winner,
    scientific_fingerprint,
)
from scripts.build_minimal_predictive_control_report import normalized_utility


def _profile(task="HIA_Hou", task_type="classification", metric="auroc", n_train=403, rate=0.2):
    return TaskProfile(task, task_type, metric, n_train, 100, rate, 0.2, 0.4, 0, 100, 0.8)


def test_agent_and_automl_have_equal_two_config_screen_budgets():
    profile = _profile()
    agent = build_arm_specs(profile, "streamlined_agent", 11, 4)
    automl = build_arm_specs(profile, "expanded_automl", 11, 4)
    fixed = build_arm_specs(profile, "strong_fixed", 12, 4)
    assert len(agent) == len(automl) == 2
    assert len(fixed) == 1
    assert {spec.seed for spec in (*agent.values(), *automl.values())} == {11}
    assert all("unimol2" not in spec.representations for spec in (*agent.values(), *automl.values(), *fixed.values()))


def test_agent_pair_is_single_axis_and_automl_is_model_search():
    agent = build_arm_specs(_profile(), "streamlined_agent", 11, 4)
    assert len({spec.model for spec in agent.values()}) == 1
    assert len({spec.representations for spec in agent.values()}) == 2
    automl = build_arm_specs(_profile(), "expanded_automl", 11, 4)
    assert len({spec.model for spec in automl.values()}) == 2
    assert len({spec.representations for spec in automl.values()}) == 1


def test_fingerprint_reuses_identical_scientific_fit_despite_arm_hypothesis():
    profile = _profile("LD50_Zhu", "regression", "mae", 5168, None)
    agent = build_arm_specs(profile, "streamlined_agent", 11, 4)
    automl = build_arm_specs(profile, "expanded_automl", 11, 4)
    assert scientific_fingerprint(agent["control"]) == scientific_fingerprint(automl["lightgbm"])
    assert scientific_fingerprint(agent["treatment"]) == scientific_fingerprint(automl["extra_trees"])


def test_screen_selection_respects_metric_direction():
    assignments = {
        "a": {"task": "LD50_Zhu", "arm": "streamlined_agent", "phase": "screen", "role": "control", "status": "completed", "result": {"validation_score": 0.5}},
        "b": {"task": "LD50_Zhu", "arm": "streamlined_agent", "phase": "screen", "role": "treatment", "status": "completed", "result": {"validation_score": 0.6}},
    }
    assert choose_screen_winner(assignments, "streamlined_agent", "LD50_Zhu") == "control"


def test_normalization_uses_train_target_scale_only_for_mae():
    assert normalized_utility("LD50_Zhu", 0.6, 2.0) == -0.3
    assert normalized_utility("HIA_Hou", 0.8, 0.4) == 0.8
