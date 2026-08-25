from admet_agent.task_data_registry import build_task_data_registry
from admet_agent.tasks import TASKS


def test_registry_covers_all_tasks_and_has_references():
    registry = build_task_data_registry()
    rows = registry["tasks"]
    assert {row["task"] for row in rows} == {spec.name for spec in TASKS.values()}
    assert len(rows) == 22
    assert all(row["benchmark_reference"] for row in rows)
    assert all(row["sources"] for row in rows)


def test_firewall_is_strictly_lower_than_point_eight():
    registry = build_task_data_registry()
    policy = registry["firewall_policy"]
    assert policy["threshold"] == 0.80
    assert "< 0.80" in policy["similarity_rule"]
    assert policy["test_labels_accessible"] is False


def test_only_endpoint_matched_local_sources_are_pilot_runnable():
    registry = build_task_data_registry()
    by_task = {row["task"]: row for row in registry["tasks"]}
    assert set(by_task["BBB_Martins"]["local_direct_ready"]) == {"b3db"}
    assert set(by_task["hERG"]["local_direct_ready"]) == {"hergcentral"}
    assert by_task["Caco2_Wang"]["local_direct_ready"] == []
    assert any(source["role"] == "bridge" for source in by_task["Caco2_Wang"]["sources"])
