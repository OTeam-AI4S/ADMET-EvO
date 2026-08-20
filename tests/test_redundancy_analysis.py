import pytest

from admet_agent.redundancy_analysis import analyze_search_summary


def _result(run_id, score, *, seed=1, model="lightgbm", members=None):
    hyperparameters = {}
    if members is not None:
        hyperparameters = {"members": members, "aggregation": "probability"}
    return {
        "run_id": run_id,
        "task": "HIA_Hou",
        "seed": seed,
        "model": model,
        "metric": "auroc",
        "validation_score": score,
        "validation_std": 0.01,
        "elapsed_seconds": 1.0,
        "status": "completed",
        "representations": ["morgan"],
        "hyperparameters": hyperparameters,
        "hypothesis": "test",
        "data_policy": "official",
        "error": None,
    }


def test_redundancy_analysis_tracks_action_and_promotion_width():
    primary = [_result(f"r{i}", score) for i, score in enumerate([0.7, 0.8, 0.75, 0.77], 1)]
    robust = [_result("r2", 0.78), _result("r4", 0.76), _result("r3", 0.74)]
    ensemble = _result("e1", 0.79, model="ensemble", members=["r2", "r4"])
    summary = {
        "profile": {"task": "HIA_Hou"},
        "validation_seeds": [1, 2, 3, 4, 5],
        "results": primary,
        "robust_results": robust,
        "ensemble_results": [ensemble],
        "finalist_run_id": "e1",
    }
    report = analyze_search_summary(summary)
    assert report["minimum_execution_prefix_for_finalist"] == 4
    assert report["worst_primary_score_rank_for_finalist"] == 2
    assert report["minimum_current_promotion_width_for_finalist"] == 2
    assert report["best_action_position_in_first_8"] == 2
    assert report["oriented_gain_actions_5_to_8"] == 0.0
    assert report["ensemble_oriented_gain_over_best_single"] == pytest.approx(0.01)
    assert report["ensemble_material"] is True
    assert report["promotion_replay"][0]["practically_equivalent"] is False
    assert report["promotion_replay"][1]["practically_equivalent"] is True
    assert report["action_prefix_replay"][0]["width"] == 2
    assert report["action_prefix_replay"][-1]["practically_equivalent"] is True


def test_redundancy_analysis_orients_mae_gain():
    primary = [
        {**_result("r1", 0.5), "task": "Caco2_Wang", "metric": "mae"},
        {**_result("r2", 0.4), "task": "Caco2_Wang", "metric": "mae"},
    ]
    robust = [
        {**_result("r1", 0.48), "task": "Caco2_Wang", "metric": "mae"},
        {**_result("r2", 0.42), "task": "Caco2_Wang", "metric": "mae"},
    ]
    ensemble = {
        **_result("e1", 0.41, model="ensemble", members=["r1", "r2"]),
        "task": "Caco2_Wang",
        "metric": "mae",
    }
    report = analyze_search_summary(
        {
            "profile": {"task": "Caco2_Wang"},
            "validation_seeds": [1, 2],
            "results": primary,
            "robust_results": robust,
            "ensemble_results": [ensemble],
            "finalist_run_id": "e1",
        }
    )
    assert report["ensemble_oriented_gain_over_best_single"] > 0
