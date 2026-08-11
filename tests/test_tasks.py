from admet_agent.tasks import TASKS, resolve_task


def test_registry_has_exactly_22_tasks():
    assert len(TASKS) == 22


def test_task_resolution_is_case_insensitive():
    assert resolve_task("caco2_wang").name == "Caco2_Wang"


def test_metric_directions():
    assert not resolve_task("PPBR_AZ").maximize
    assert resolve_task("VDss_Lombardo").maximize
