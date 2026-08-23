import pytest

from admet_agent.strong_anchor_registry import registry_manifest, route_strong_anchors


def test_small_auroc_classification_gets_one_pretrained_treatment():
    decision = route_strong_anchors("DILI", 325)
    assert decision["control_anchor_ids"] == [
        "rich2d_lightgbm",
        "rich2d_extra_trees",
    ]
    assert decision["treatment_anchor_id"] == "maplight_gnn_classification"
    assert len(decision["admitted_anchor_ids"]) == 3
    assert decision["unimol2_admitted"] is False


def test_auprc_uses_same_preregistered_maplight_priority_not_task_rank():
    small = route_strong_anchors("CYP2C9_Substrate_CarbonMangels", 467)
    large = route_strong_anchors("CYP2C9_Veith", 8463)
    assert small["treatment_anchor_id"] == "maplight_gnn_classification"
    assert large["treatment_anchor_id"] == "maplight_gnn_classification"


def test_regression_only_admits_executable_low_cost_controls():
    decision = route_strong_anchors("Caco2_Wang", 637)
    assert decision["admitted_anchor_ids"] == [
        "rich2d_lightgbm",
        "rich2d_extra_trees",
    ]
    assert decision["treatment_anchor_id"] is None


@pytest.mark.parametrize(
    "signals",
    [
        {"leaderboard_rank": 1},
        {"test_score": 0.9},
        {"nested": {"heldout_metric": 0.9}},
    ],
)
def test_test_or_leaderboard_derived_routing_inputs_are_rejected(signals):
    with pytest.raises(ValueError, match="forbidden"):
        route_strong_anchors("DILI", 325, signals=signals)


def test_runtime_unavailability_blocks_pretrained_anchor():
    decision = route_strong_anchors(
        "DILI", 325, signals={"maplight_runtime_available": False}
    )
    assert decision["treatment_anchor_id"] == "minimol_morgan_kernel_small_auroc"


def test_registry_and_decisions_are_deterministic_and_test_free():
    first = route_strong_anchors("hERG", 457)
    second = route_strong_anchors("hERG", 457)
    assert first["decision_sha256"] == second["decision_sha256"]
    manifest = registry_manifest()
    assert manifest["test_loaded"] is False
    assert manifest["unimol2_default_admitted"] is False
