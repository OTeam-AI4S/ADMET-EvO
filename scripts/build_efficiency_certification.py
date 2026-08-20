"""Build the bounded fresh-validation practical-efficiency certificate."""

from __future__ import annotations

import json
from pathlib import Path

from admet_agent.efficiency_certification import certify_router_efficiency


def main() -> None:
    root = Path(__file__).parents[1]
    source = json.loads(
        (root / "runs/v27-router-risk-guard/report.json").read_text(encoding="utf-8")
    )
    result = certify_router_efficiency(source).to_dict()
    (root / "reports/practical_efficiency_certification_v1.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    lines = [
        "# Fresh-validation practical-efficiency certification",
        "",
        f"Status: **{'PASS' if result['passed'] else 'BLOCKED'}**",
        "",
        f"- Tasks: {result['task_count']}/22",
        f"- Mean oriented gain vs score-only selection: {result['mean_oriented_gain']:.6f}",
        (
            "- Task-bootstrap 95% CI: "
            f"[{result['bootstrap_95_ci'][0]:.6f}, {result['bootstrap_95_ci'][1]:.6f}]"
        ),
        f"- Preregistered practical noninferiority margin: -{result['policy']['noninferiority_margin']:.3f}",
        f"- Non-material-regression tasks: {result['nonmaterial_task_rate']:.3f}",
        f"- Material regressions: {result['material_regressions']}/22",
        (
            f"- Worst endpoint: {result['worst_task']} "
            f"({result['worst_oriented_gain']:.6f})"
        ),
        f"- Aggregate selected-recipe cost ratio: {result['aggregate_cost_ratio']:.3f}",
        f"- Estimated selected-recipe cost reduction: {result['estimated_cost_reduction']:.1%}",
        "",
        "This supports a bounded Pareto-efficiency claim on fresh validation seeds. It does",
        "not show routing superiority over random or AutoML, does not erase the two endpoint",
        "tail regressions, and does not use or alter the frozen TDC test result.",
        "",
    ]
    (root / "reports/practical_efficiency_certification_v1.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
