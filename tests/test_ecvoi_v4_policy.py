from __future__ import annotations

from admet_agent.ecvoi_v4_policy import evaluate_leave_one_endpoint_out
from scripts.analyze_ecvoi_v4_matched_states import analyze


def _audit(state: str) -> dict:
    if state == "domain_shift":
        after, rate = 0.35, 0.3
    elif state == "imbalance":
        after, rate = 0.6, 0.05
    else:
        after, rate = 0.6, 0.3
    return {
        "median_max_validation_similarity_before": 0.6,
        "median_max_validation_similarity_after": after,
        "input_positive_rate": 0.3,
        "output_positive_rate": rate,
    }


def _effect_rows(seeds=(17,)):
    rows = []
    for task in ("A", "B", "C"):
        for seed in seeds:
            for state in ("native", "domain_shift", "imbalance"):
                effects = {
                    "baseline": 0.0,
                    "feature_rich": 0.03 if state == "domain_shift" else -0.01,
                    "imbalance_aware": 0.03 if state == "imbalance" else -0.01,
                }
                for action, effect in effects.items():
                    rows.append({
                        "task": task,
                        "seed": seed,
                        "state": state,
                        "action": action,
                        "oriented_effect": effect,
                        "elapsed_seconds": 1.0,
                        "state_audit": _audit(state),
                    })
    return rows


def test_loeo_policy_excludes_target_and_uses_diagnosis():
    result = evaluate_leave_one_endpoint_out(
        _effect_rows(), harm_probability_limit=1.0, cost_penalty=0.0
    )

    assert all(row["target_endpoint_training_rows"] == 0 for row in result)
    by_state = {row["state"]: row["selected_action"] for row in result if row["task"] == "A"}
    assert by_state["domain_shift"] == "feature_rich"
    assert by_state["imbalance"] == "imbalance_aware"
    assert all(not row["harmful_intervention"] for row in result)


def _registry(seeds=(17,)):
    records = {}
    for row in _effect_rows(seeds):
        key = f"{row['task']}|{row['seed']}|{row['state']}|{row['action']}"
        records[key] = {
            "task": row["task"],
            "seed": row["seed"],
            "state": row["state"],
            "action": row["action"],
            "state_audit": {
                **row["state_audit"],
                "validation_labels_read_for_state": False,
            },
            "profile": {},
            "spec": {"data_policy": "identity_safe"},
            "result": {
                "status": "completed",
                "validation_score": 0.7 + row["oriented_effect"],
                "elapsed_seconds": 1.0,
                "run_id": key,
            },
            "test_loaded": False,
            "leaderboard_loaded": False,
        }
    return {
        "protocol_sha256": "matched",
        "records": records,
        "test_loaded": False,
        "leaderboard_loaded": False,
    }


def _protocol():
    return {
        "sha256": "matched",
        "actions": ["baseline", "feature_rich", "imbalance_aware"],
        "screen_seeds": [17],
        "confirmation_seeds": [29, 43],
    }


def _analysis_protocol():
    return {
        "sha256": "analysis",
        "shuffle_seed": 20260823,
        "random_seed": 20260823,
        "policy_options": {
            "harm_margin": 0.005,
            "harm_probability_limit": 1.0,
            "cost_penalty": 0.0,
        },
    }


def test_screen_analysis_cannot_unlock_confirmation():
    result = analyze(_registry(), _protocol(), _analysis_protocol())

    assert result["evidence_stage"] == "screen_only"
    assert result["confirmation_19_unlocked"] is False


def test_complete_three_seed_analysis_can_pass_mechanism_gate():
    result = analyze(
        _registry((17, 29, 43)), _protocol(), _analysis_protocol()
    )

    assert result["evidence_stage"] == "full_development"
    assert result["policy_summary"]["full_v4"]["mean_selection_regret"] == 0.0
    assert result["checks"][0]["passed"] is True
