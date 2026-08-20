"""Build the train-normalized legacy Agent-vs-AutoML negative-evidence audit."""

from __future__ import annotations

import json
from pathlib import Path

from tdc.benchmark_group import admet_group

from admet_agent.legacy_policy_audit import audit_legacy_agent_vs_automl
from admet_agent.metric_normalization import build_metric_normalization_profile
from admet_agent.tasks import TASKS


def main() -> None:
    root = Path(__file__).parents[1]
    group = admet_group(path=str(root / "data"))
    profiles = {}
    for task in TASKS.values():
        train, _ = group.get_train_valid_split(
            benchmark=task.name,
            split_type="default",
            seed=1,
        )
        profiles[task.name] = build_metric_normalization_profile(
            task, train["Y"].to_numpy()
        )
    report = audit_legacy_agent_vs_automl(
        agent_dir=root / "runs/v3-agent-tdc22-v5",
        automl_dir=root / "runs/v3-automl-tdc22-v1",
        protocol_path=root / "runs/v3-policy-benchmark-v3/protocol.json",
        profiles=profiles,
    )
    (root / "reports/legacy_agent_automl_audit_v1.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    pred = report["predictive_comparison"]
    cost = report["cost_comparison"]
    budget = report["budget_audit"]
    source = report["source_audit"]
    lines = [
        "# Legacy Agent-vs-AutoML audit",
        "",
        "Status: **NOT ELIGIBLE for a formal same-budget claim**",
        "",
        f"- TDC endpoints: {report['task_count']}/22",
        (
            f"- Observed action parity: {budget['actual_action_parity_tasks']}/22; "
            f"Agent same-or-lower actions: {budget['agent_same_or_lower_action_tasks']}/22"
        ),
        (
            f"- Declared max actions: Agent={budget['declared_agent_max_actions']}, "
            f"AutoML={budget['declared_automl_max_actions']}; both compliance flags=false"
        ),
        (
            "- Mean train-normalized oriented Agent−AutoML delta: "
            f"{pred['mean_normalized_oriented_delta']:.6f}"
        ),
        (
            "- Task-bootstrap 95% CI: "
            f"[{pred['task_bootstrap_mean_95_ci'][0]:.6f}, "
            f"{pred['task_bootstrap_mean_95_ci'][1]:.6f}]"
        ),
        f"- Wins/ties/losses: {pred['wins']}/{pred['ties']}/{pred['losses']}",
        f"- Tasks within −0.01: {pred['tasks_within_minus_0_01']}/22",
        (
            f"- Worst endpoint: {pred['worst_task']} "
            f"({pred['worst_normalized_delta']:.6f})"
        ),
        f"- Aggregate Agent/AutoML fit-time ratio: {cost['aggregate_agent_cost_ratio']:.3f}",
        f"- Execution-kernel hashes identical: {source['execution_kernel_identical']}",
        f"- Full source snapshot identical: {source['same_source_snapshot']}",
        "- Test loaded: false",
        "",
        "The exploratory result is negative: the legacy Agent was worse on normalized",
        "macro validation performance and consumed more fit time. This does not challenge",
        "the separate application-performance qualification; it blocks predictive or compute",
        "superiority claims against AutoML. Protocol violations and source mismatch make the",
        "comparison ineligible for formal noninferiority certification in either direction.",
        "",
    ]
    (root / "reports/legacy_agent_automl_audit_v1.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
