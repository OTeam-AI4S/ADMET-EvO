import importlib.util
import json
from pathlib import Path


ROOT = Path(__file__).parents[1]
SCRIPT = ROOT / "scripts/evaluate_router_metric_risk_guard_v27.py"


def _module():
    spec = importlib.util.spec_from_file_location("router_metric_risk_guard_v27", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_guard_is_metric_only_and_fail_closed():
    module = _module()
    assert module.guard_policy("spearman") == "score"
    for metric in ("auroc", "auprc", "mae"):
        assert module.guard_policy(metric) == "unguarded_cost_aware"
    try:
        module.guard_policy("rmse")
    except ValueError:
        pass
    else:
        raise AssertionError("Unknown metrics must fail closed")


def test_persisted_fresh_guard_report_obeys_firewall_and_preregistration():
    path = ROOT / "runs/v27-router-risk-guard/report.json"
    if not path.exists():
        return
    report = json.loads(path.read_text(encoding="utf-8"))
    assert report["task_count"] == 22
    assert report["fresh_validation_seeds"] == [4, 5]
    assert report["identity_overlap_total"] == 0
    assert report["official_test_loaded"] is False
    assert report["official_results_used"] is False
    assert report["leaderboard_used"] is False
    assert report["selection_or_official_freeze_changed"] is False
    assert len(report["preregistration"]["sha256"]) == 64
    assert sum(row["guard_policy"] == "score" for row in report["tasks"]) == 4
    assert sum(row["guard_policy"] == "unguarded_cost_aware" for row in report["tasks"]) == 18
    assert report["primary_guard_minus_unguarded"]["mean"] == 0
    assert report["primary_guard_minus_unguarded"]["tied"] == 22
    assert report["claim_gate"]["passed"] is False
    assert report["significance_gate_passed"] is False
    assert report["claim_gate"]["aggregate_guard_cost_ratio_vs_score"] < 0.35
