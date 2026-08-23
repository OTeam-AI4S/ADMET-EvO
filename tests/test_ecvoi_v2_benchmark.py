import hashlib
import json

import pytest

from admet_agent.cli import parser
from admet_agent.ecvoi_v2_benchmark import (
    DEVELOPMENT_TASKS,
    protocol_payload,
    validate_protocol,
)
from admet_agent.policy_brains import benchmark_brain
from admet_agent.policy_results import collect_policy_results


def test_v2_protocol_freezes_disjoint_three_and_nineteen_task_cohorts():
    development = protocol_payload("development")
    confirmation = protocol_payload("confirmation")
    assert tuple(development["tasks"]) == DEVELOPMENT_TASKS
    assert len(confirmation["tasks"]) == 19
    assert set(development["tasks"]).isdisjoint(confirmation["tasks"])
    assert len(set(development["tasks"]) | set(confirmation["tasks"])) == 22
    for payload in (development, confirmation):
        claimed = payload.pop("sha256")
        actual = hashlib.sha256(
            json.dumps(payload, sort_keys=True).encode("utf-8")
        ).hexdigest()
        assert claimed == actual
        assert payload["test_feedback_to_discovery"] is False
        assert payload["leaderboard_feedback_to_discovery"] is False
        assert payload["pair_contract"]["incomplete_pair_updates_posterior"] is False
        assert payload["frozen_invariants"]["ecvoi_v2_memory_snapshot_sha256"]


def test_v2_protocol_rejects_cohort_contamination():
    payload = protocol_payload("confirmation")
    payload["tasks"].append(DEVELOPMENT_TASKS[0])
    with pytest.raises(ValueError, match="3/19"):
        validate_protocol(payload)


def test_v2_cli_and_brain_names_are_explicit():
    parsed = parser().parse_args(
        [
            "search",
            "--task",
            "BBB_Martins",
            "--brain",
            "ecvoi-v2-shuffled-memory",
            "--ecvoi-v2-memory-snapshot",
            "memory.jsonl",
        ]
    )
    assert parsed.brain == "ecvoi-v2-shuffled-memory"
    assert parsed.ecvoi_v2_memory_snapshot.name == "memory.jsonl"
    assert (
        benchmark_brain("ecvoi_v2_shuffled_memory").benchmark_arm
        == "ecvoi_v2_shuffled_memory"
    )


def test_v2_collector_requires_memory_binding_and_cohort_boundary(tmp_path):
    protocol = protocol_payload("development")
    protocol_path = tmp_path / "protocol.json"
    protocol_path.write_text(json.dumps(protocol), encoding="utf-8")
    run_dir = tmp_path / "run"
    task_dir = run_dir / "tasks" / "BBB_Martins"
    task_dir.mkdir(parents=True)
    invariants = protocol["frozen_invariants"]
    summary = {
        "benchmark_arm": "ecvoi_v2",
        "benchmark_protocol_sha256": protocol["sha256"],
        "source_sha256": protocol["implementation_source_sha256"],
        "research_routing_policy": "ecvoi-v2",
        "action_memory_snapshot_sha256": invariants[
            "action_memory_snapshot_sha256"
        ],
        "action_context_registry_sha256": invariants[
            "action_context_registry_sha256"
        ],
        "ecvoi_v2_memory_snapshot_sha256": "wrong",
        "profile": {"task": "BBB_Martins"},
        "results": [],
        "test_loaded": False,
        "leaderboard_loaded": False,
    }
    path = task_dir / "search_seed_1.json"
    path.write_text(json.dumps(summary), encoding="utf-8")
    with pytest.raises(ValueError, match="v2 memory"):
        collect_policy_results(protocol_path, {"ecvoi_v2": run_dir})

    summary["ecvoi_v2_memory_snapshot_sha256"] = invariants[
        "ecvoi_v2_memory_snapshot_sha256"
    ]
    summary["profile"]["task"] = "AMES"
    path.write_text(json.dumps(summary), encoding="utf-8")
    with pytest.raises(ValueError, match="outside the protocol"):
        collect_policy_results(protocol_path, {"ecvoi_v2": run_dir})
