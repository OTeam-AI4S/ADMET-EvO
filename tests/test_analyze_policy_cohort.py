import pytest

from scripts.analyze_policy_cohort import analyze, render_markdown


def _row(task, arm, score, configs=8):
    return {
        "task": task,
        "arm": arm,
        "validation_score": score,
        "validation_std": 0.01,
        "accounted_fit_seconds": 10.0,
        "accounted_unique_configurations": configs,
        "protocol_sha256": "a" * 64,
    }


def test_partial_cohort_ranks_direction_and_forbids_inferential_claim():
    payload = {
        "results": [
            _row("BBB_Martins", "agent", 0.90),
            _row("BBB_Martins", "control", 0.80),
            _row("Caco2_Wang", "agent", 0.40),
            _row("Caco2_Wang", "control", 0.50),
        ]
    }
    result = analyze(payload, reference_arm="agent", bootstrap_replicates=1000)
    assert result["evidence_scope"] == "discovery_development_only"
    assert result["per_task"]["BBB_Martins"]["ranks"]["agent"] == 1
    # Caco2 is MAE and lower is better.
    assert result["per_task"]["Caco2_Wang"]["ranks"]["agent"] == 1
    comparison = result["comparisons"]["control"]
    assert comparison["wins"] == 2
    assert comparison["losses"] == 0
    assert comparison["inferential_claim_allowed"] is False
    markdown = render_markdown(result)
    assert "Test loaded: **no**" in markdown
    assert "Caco2_Wang" in markdown


def test_cohort_requires_identical_coverage_and_budget_audit_is_explicit():
    missing = {
        "results": [
            _row("BBB_Martins", "agent", 0.9),
            _row("BBB_Martins", "control", 0.8),
            _row("Caco2_Wang", "agent", 0.4),
        ]
    }
    with pytest.raises(ValueError, match="identical cohort"):
        analyze(missing, reference_arm="agent", bootstrap_replicates=1000)

    unequal = {
        "results": [
            _row("BBB_Martins", "agent", 0.9, configs=8),
            _row("BBB_Martins", "control", 0.8, configs=7),
        ]
    }
    result = analyze(unequal, reference_arm="agent", bootstrap_replicates=1000)
    assert result["comparisons"]["control"]["per_task_action_budget_parity"] is False


def test_accepts_current_collector_accounting_field_names():
    current = {
        "results": [
            {
                "task": "BBB_Martins",
                "arm": arm,
                "validation_score": score,
                "validation_std": 0.01,
                "fit_seconds": seconds,
                "compute_seconds": seconds,
                "unique_configurations": 8,
                "actions_attempted": 8,
                "protocol_sha256": "a" * 64,
            }
            for arm, score, seconds in (
                ("agent", 0.9, 12.0),
                ("control", 0.8, 15.0),
            )
        ]
    }
    result = analyze(current, reference_arm="agent", bootstrap_replicates=1000)
    assert result["arm_summary"]["agent"]["total_fit_seconds"] == 12.0
    assert result["arm_summary"]["agent"]["total_unique_configurations"] == 8
