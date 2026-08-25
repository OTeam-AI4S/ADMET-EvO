from pathlib import Path

from scripts.build_external_data_agent_report import build, render


def test_external_data_report_keeps_test_and_claim_boundaries():
    root = Path(__file__).resolve().parents[1]
    payload = build(root)
    assert payload["discovery"]["candidate_count"] == 3
    assert payload["discovery"]["decision_counts"] == {"bridge_required": 3}
    assert payload["sealed_structure_firewall"]["test_labels_loaded"] is False
    assert payload["sealed_structure_firewall"]["thresholds_frozen"] == {
        "test_scaffold_coverage": 0.05,
        "ecfp4_tanimoto": 0.8,
    }
    assert payload["fair_comparison_contract"]["same_structure_firewall"] is True
    assert "improved" in payload["claim_boundary"]["forbidden"]


def test_report_renders_three_exclusions():
    root = Path(__file__).resolve().parents[1]
    text = render(build(root))
    assert "超过 5%" in text
    assert "Tanimoto ≥0.90" in text
    assert "完全相同" in text
