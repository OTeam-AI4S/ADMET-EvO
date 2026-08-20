"""Build the validation-only cross-task memory readiness evidence package."""

from __future__ import annotations

import json
from pathlib import Path

from admet_agent.cross_task_memory_readiness import (
    audit_cross_task_memory_readiness,
)


def main() -> None:
    root = Path(__file__).parents[1]
    report = audit_cross_task_memory_readiness(
        root / "runs/v3-agent-tdc22-v5",
        root / "runs/v3-agent-tdc22-v5/action_outcomes.jsonl",
    )
    (root / "reports/cross_task_memory_readiness_v1.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    actions = report["historical_action_counts"]
    statuses = report["historical_status_counts"]
    next_evidence = report["recommended_next_evidence"]
    lines = [
        "# Cross-task action-memory readiness audit",
        "",
        "Status: **NOT YET ELIGIBLE to influence routing**",
        "",
        f"- TDC endpoints audited: {report['task_count']}/22",
        f"- Single-axis evidence complete: {report['single_axis_verified_tasks']}/22",
        f"- Selected actions: `{actions}`",
        f"- Outcomes: `{statuses}`",
        (
            "- Largest-action share / normalized action entropy: "
            f"{report['largest_action_share']:.3f} / "
            f"{report['selected_action_normalized_entropy']:.3f}"
        ),
        (
            "- Transport-eligible records: "
            f"{report['transport_eligible_record_count']}/{report['memory_record_count']}"
        ),
        "- Test loaded: false",
        "",
        "## What this means",
        "",
        "The historical results are valid evidence that the Agent can execute matched,",
        "single-axis experiments and retain supported/refuted/inconclusive outcomes. They",
        "are not a causal training set for cross-task routing: the existing policy selected",
        "the action, 18/22 endpoints used the same feature action, and counterfactual actions",
        "are missing. Treating those rows as priors would create selection/positivity bias.",
        "",
        "The runtime now enforces this boundary. Observational memory remains auditable but",
        "uses a neutral effect multiplier, cannot reorder actions, and cannot enable UniMol2.",
        "Therefore `agent_no_memory` parity is expected until eligible evidence exists; it",
        "must not be marketed as a demonstrated memory advantage.",
        "",
        "## Minimum next experiment",
        "",
        (
            f"Run a preregistered complete action grid on at least "
            f"{next_evidence['minimum_sentinel_endpoints']} sentinel endpoints across "
            f"{len(next_evidence['actions'])} currently observed action families "
            f"({next_evidence['minimum_matched_action_episodes']} matched episodes minimum)."
        ),
        "Use identical budgets, one shared protocol hash, train/validation only, and retain",
        "all negative outcomes. This experiment estimates transport; it does not reopen or",
        "optimize the frozen application test models.",
        "",
    ]
    (root / "reports/cross_task_memory_readiness_v1.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
