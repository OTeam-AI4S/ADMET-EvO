import importlib.util
from pathlib import Path

from admet_agent.evidence_packet_loop import CANONICAL_TASKS


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "plan_evidence_packet_acquisition_v4",
    ROOT / "scripts/plan_evidence_packet_acquisition_v4.py",
)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_acquisition_plan_targets_weak_tasks_and_forbids_automatic_training():
    scores = {task: 0.1 for task in CANONICAL_TASKS}
    scores["hERG"] = -0.2
    scores["DILI"] = -0.1
    scores["HIA_Hou"] = -0.05
    plan = MODULE.build_plan(scores, count=3)
    assert plan["target_tasks"] == ["hERG", "DILI", "HIA_Hou"]
    assert all(
        ticket["automatic_download_or_training_allowed"] is False
        for ticket in plan["tickets"]
    )
    assert all(ticket["minimum_rows"] == 1 for ticket in plan["tickets"])
    assert plan["legacy_bulk_pool_used"] is False
    assert "Hansen" in MODULE.SOURCE_GUARDS["AMES"]["benchmark_origin"]
    assert "PubChem AID 1851" in MODULE.SOURCE_GUARDS["CYP3A4_Veith"][
        "forbidden_reaggregations"
    ]
    assert MODULE.SOURCE_GUARDS["CYP2C9_Veith"]["candidate_leads"][0][
        "status"
    ] == "lineage_review_required_before_packet_materialization"
