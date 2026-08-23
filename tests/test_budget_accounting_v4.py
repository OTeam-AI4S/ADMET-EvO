"""Regression tests for formal benchmark budget and fit accounting."""

from __future__ import annotations

import json
import sys

import pytest

from admet_agent.benchmark_protocol import (
    PolicyTaskResult,
    validate_search_budget_request,
    write_default_protocol,
)
from admet_agent.cli import main
from admet_agent.policy_results import collect_policy_results


def _agent_arm(protocol: dict) -> dict:
    return next(item for item in protocol["arms"] if item["name"] == "agent")


def test_v5_protocol_preregisters_committee_and_expanded_automl_at_equal_budget(
    tmp_path,
):
    protocol = write_default_protocol(tmp_path / "protocol.json")
    assert protocol["protocol_version"] == "tdc22-policy-benchmark-v5"
    sequential = {
        item["name"]: (
            item["max_actions_per_task"],
            item["max_candidates_per_task"],
            item["max_compute_minutes_per_task"],
        )
        for item in protocol["arms"]
        if item["comparison_class"] == "same_budget_sequential"
    }
    assert sequential["committee"] == sequential["agent"]
    assert sequential["random"] == sequential["agent"]
    assert sequential["rules"] == sequential["agent"]
    assert sequential["expanded_automl"] == sequential["agent"]


def _summary(protocol_hash: str, results: list[dict]) -> dict:
    return {
        "benchmark_arm": "agent",
        "benchmark_protocol_sha256": protocol_hash,
        "validation_seeds": [1, 2, 3, 4, 5],
        "results": results,
        "robust_results": [
            {
                "run_id": "winner",
                "task": "HIA_Hou",
                "validation_score": 0.9,
                "validation_std": 0.01,
                # Aggregate time is derived and must never be counted again.
                "elapsed_seconds": 9999.0,
            }
        ],
        "ensemble_results": [
            {
                "run_id": "derived-ensemble",
                "task": "HIA_Hou",
                "validation_score": 0.89,
                "elapsed_seconds": 8888.0,
            }
        ],
        "finalist_run_id": "winner",
    }


def test_preflight_rejects_two_rounds_of_six_against_eight_action_limit(tmp_path):
    protocol = write_default_protocol(tmp_path / "protocol.json")
    arm = _agent_arm(protocol)
    with pytest.raises(
        ValueError, match=r"12 unique configurations.*max_actions_per_task=8"
    ):
        validate_search_budget_request(
            arm,
            rounds=2,
            candidates_per_round=6,
        )

    accepted = validate_search_budget_request(
        arm,
        rounds=2,
        candidates_per_round=4,
    )
    assert accepted["requested_unique_configurations"] == 8


def test_cli_rejects_overbudget_search_before_agent_construction(tmp_path, monkeypatch):
    protocol_path = tmp_path / "protocol.json"
    write_default_protocol(protocol_path)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "admet-agent",
            "search",
            "--task",
            "HIA_Hou",
            "--brain",
            "agent",
            "--benchmark-protocol",
            str(protocol_path),
            "--validation-seeds",
            "1,2,3,4,5",
            "--rounds",
            "2",
            "--candidates-per-round",
            "6",
            "--run-dir",
            str(tmp_path / "must-not-be-created"),
        ],
    )
    with pytest.raises(SystemExit, match="12 unique configurations"):
        main()
    assert not (tmp_path / "must-not-be-created").exists()


def test_collector_separates_actions_replicates_failures_and_ensembles(tmp_path):
    protocol_path = tmp_path / "protocol.json"
    protocol = write_default_protocol(protocol_path)
    run_dir = tmp_path / "agent"
    task_dir = run_dir / "tasks" / "HIA_Hou"
    task_dir.mkdir(parents=True)

    results = [
        {
            "run_id": "primary-a",
            "seed": 1,
            "status": "completed",
            "elapsed_seconds": 2.0,
        },
        {
            "run_id": "primary-b",
            "seed": 1,
            "status": "failed",
            "elapsed_seconds": 3.0,
        },
    ]
    results.extend(
        {
            "run_id": f"repeat-a-{seed}",
            "seed": seed,
            "status": "completed",
            "elapsed_seconds": 1.0,
        }
        for seed in (2, 3, 4, 5)
    )
    results.extend(
        {
            "run_id": f"repeat-b-{seed}",
            "seed": seed,
            "status": "completed",
            "elapsed_seconds": 0.5,
        }
        for seed in (2, 3, 4, 5)
    )
    for run_id, model in (("primary-a", "lightgbm"), ("primary-b", "extra_trees")):
        artifact = run_dir / "artifacts" / run_id
        artifact.mkdir(parents=True)
        (artifact / "experiment.json").write_text(
            json.dumps(
                {
                    "task": "HIA_Hou",
                    "seed": 1,
                    "model": model,
                    "representations": ["morgan"],
                    "imbalance": "none",
                    "data_policy": "official",
                    "hyperparameters": {},
                }
            ),
            encoding="utf-8",
        )
    (task_dir / "search_seed_1.json").write_text(
        json.dumps(_summary(protocol["sha256"], results)),
        encoding="utf-8",
    )

    payload = collect_policy_results(protocol_path, {"agent": run_dir})
    row = payload["results"][0]
    assert row["unique_configurations"] == 2
    assert row["actions_attempted"] == 2
    assert row["replicate_fits"] == 8
    assert row["total_fits"] == 10
    assert row["completed_fits"] == 9
    assert row["failed_fits"] == 1
    assert row["completed_fit_seconds"] == pytest.approx(8.0)
    assert row["failed_fit_seconds"] == pytest.approx(3.0)
    assert row["fit_seconds"] == pytest.approx(11.0)
    assert row["compute_seconds"] == pytest.approx(11.0)
    assert row["ensemble_evaluations"] == 1
    assert payload["arms"]["agent"]["total_fits"] == 10
    assert (
        "excluded from total_fits"
        in payload["accounting_semantics"]["ensemble_evaluations"]
    )


def test_collector_resolves_protocol_shared_anchor_finalist(tmp_path):
    protocol_path = tmp_path / "protocol.json"
    protocol = write_default_protocol(protocol_path)
    run_dir = tmp_path / "agent"
    task_dir = run_dir / "tasks" / "HIA_Hou"
    task_dir.mkdir(parents=True)
    summary = _summary(protocol["sha256"], [])
    summary["robust_results"] = []
    summary["ensemble_results"] = []
    summary["finalist_run_id"] = "hia_hou-shared-anchor-1"
    summary["shared_strong_anchor"] = {
        "aggregate_result": {
            "run_id": "hia_hou-shared-anchor-1",
            "task": "HIA_Hou",
            "validation_score": 0.91,
            "validation_std": 0.02,
            "status": "completed",
        }
    }
    (task_dir / "search_seed_1.json").write_text(
        json.dumps(summary), encoding="utf-8"
    )

    payload = collect_policy_results(protocol_path, {"agent": run_dir})

    assert payload["arms"]["agent"]["tasks"] == 1
    assert payload["results"][0]["validation_score"] == pytest.approx(0.91)


def test_collector_enforces_unique_configuration_action_budget(tmp_path):
    protocol_path = tmp_path / "protocol.json"
    protocol = write_default_protocol(protocol_path)
    run_dir = tmp_path / "agent"
    task_dir = run_dir / "tasks" / "HIA_Hou"
    task_dir.mkdir(parents=True)
    results = [
        {
            "run_id": f"primary-{index}",
            "seed": 1,
            "status": "completed",
            "elapsed_seconds": 1.0,
        }
        for index in range(9)
    ]
    (task_dir / "search_seed_1.json").write_text(
        json.dumps(_summary(protocol["sha256"], results)),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="unique-configuration action budget"):
        collect_policy_results(protocol_path, {"agent": run_dir})


def test_collector_accounts_failed_search_even_without_finalist(tmp_path):
    protocol_path = tmp_path / "protocol.json"
    protocol = write_default_protocol(protocol_path)
    run_dir = tmp_path / "agent"
    task_dir = run_dir / "tasks" / "HIA_Hou"
    task_dir.mkdir(parents=True)
    summary = _summary(
        protocol["sha256"],
        [
            {
                "run_id": "failed-primary",
                "seed": 1,
                "status": "failed",
                "elapsed_seconds": 7.5,
            }
        ],
    )
    summary["robust_results"] = []
    summary["finalist_run_id"] = None
    (task_dir / "search_seed_1.json").write_text(json.dumps(summary), encoding="utf-8")

    payload = collect_policy_results(protocol_path, {"agent": run_dir})
    arm = payload["arms"]["agent"]
    assert arm["tasks"] == 0
    assert arm["accounted_searches"] == 1
    assert arm["total_fits"] == 1
    assert arm["failed_fits"] == 1
    assert arm["failed_fit_seconds"] == pytest.approx(7.5)
    assert arm["compute_seconds"] == pytest.approx(7.5)


def test_old_policy_rows_keep_legacy_action_and_compute_semantics():
    row = PolicyTaskResult.from_dict(
        {
            "task": "HIA_Hou",
            "arm": "agent",
            "validation_score": 0.9,
            "validation_std": 0.01,
            "compute_seconds": 12.5,
            "actions_attempted": 8,
            "protocol_sha256": "legacy",
        }
    )
    assert row.accounted_unique_configurations == 8
    assert row.accounted_fit_seconds == pytest.approx(12.5)
