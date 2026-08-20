import json
from pathlib import Path

from admet_agent.efficiency_certification import (
    PracticalEfficiencyPolicy,
    certify_router_efficiency,
)

ROOT = Path(__file__).parents[1]


def _report():
    return json.loads(
        (ROOT / "runs/v27-router-risk-guard/report.json").read_text(encoding="utf-8")
    )


def test_fresh_seed_router_passes_bounded_practical_efficiency_gate():
    result = certify_router_efficiency(_report())
    assert result.passed is True
    assert result.task_count == 22
    assert result.aggregate_cost_ratio == 0.2780324339961261
    assert result.estimated_cost_reduction > 0.72
    assert result.material_regressions == 2
    assert result.nonmaterial_task_rate == 20 / 22
    assert result.bootstrap_95_ci[0] > -0.005
    assert result.test_loaded is False


def test_stricter_noninferiority_or_tail_gate_blocks_claim():
    result = certify_router_efficiency(
        _report(),
        PracticalEfficiencyPolicy(
            noninferiority_margin=0.001,
            minimum_nonmaterial_task_rate=0.95,
        ),
    )
    assert result.passed is False
    assert "bootstrap_lower_bound_below_noninferiority_margin" in result.blockers
    assert "nonmaterial_task_rate_below_threshold" in result.blockers
