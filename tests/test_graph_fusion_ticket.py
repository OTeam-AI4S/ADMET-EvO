from pathlib import Path

from scripts.build_graph_fusion_ticket import build, render


def test_current_workspace_fails_closed_without_distinct_mpnn_provider():
    root = Path(__file__).resolve().parents[1]
    payload = build(root)
    assert payload["readiness"]["gin"] is True
    assert payload["readiness"]["oof_gate"] is True
    assert payload["readiness"]["mpnn"] is False
    assert payload["status"] == "blocked_missing_distinct_mpnn_provider"
    assert payload["experiment"]["stage_a_representation"]["maximum_fits"] == 10
    assert payload["experiment"]["stage_b_fusion"]["maximum_fits"] == 10
    assert payload["scope"]["test_labels_loaded"] is False


def test_render_preserves_compute_and_claim_boundaries():
    root = Path(__file__).resolve().parents[1]
    text = render(build(root))
    assert "总硬上限 20 fits" in text
    assert "不能把现有 GIN 改名替代" in text
    assert "未过门即保留负结果并停止" in text
