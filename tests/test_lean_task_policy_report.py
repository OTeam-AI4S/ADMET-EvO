from pathlib import Path

from scripts.build_lean_task_policy_report import build


ROOT = Path(__file__).resolve().parents[1]


def test_qualified_tdc22_defaults_to_zero_research_fits():
    payload = build(
        ROOT / "reports/v3_tdc22_hardness_summary.json",
        ROOT / "reports/performance_qualification_v1.json",
        ROOT / "runs/v27-maplight-gnn-freeze/full-22-task.mapping.json",
    )
    assert payload["task_count"] == 22
    assert payload["data_regime_counts"] == {"higher_data": 9, "low_data": 13}
    assert payload["task_family_counts"] == {
        "binary_auroc": 8,
        "continuous_mae": 5,
        "imbalanced_binary_auprc": 5,
        "rank_pk_spearman": 4,
    }
    assert payload["default_route_counts"] == {"none": 22}
    assert payload["default_research_fit_ceiling"] == 0
    assert payload["frozen_artifact_coverage"] == 22
    assert payload["test_loaded"] is False
