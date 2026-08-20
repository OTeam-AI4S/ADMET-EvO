"""Build the locked hidden-fault policy-control report."""

from __future__ import annotations

import json
from pathlib import Path

from admet_agent.fault_policy_controls import compare_fault_controls
from admet_agent.hidden_fault_benchmark import FaultChallengeResult


def main() -> None:
    root = Path(__file__).parents[1]
    payload = json.loads(
        (root / "reports/v3_hidden_fault_suite.json").read_text(encoding="utf-8")
    )
    report = compare_fault_controls(
        [FaultChallengeResult.from_dict(row) for row in payload["results"]]
    )
    output = root / "reports/fault_policy_controls_v1.json"
    output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    labels = {
        "agent_contract_guard": "Agent contract guard",
        "complete_rules_guard": "Complete rules guard",
        "expanded_automl_schema_guard": "Expanded AutoML schema guard",
        "conventional_automl_no_guard": "Conventional AutoML",
        "seeded_random_fault": "Seeded random",
    }
    lines = [
        "# Hidden-fault policy controls",
        "",
        "Locked replay over the existing 10-case deterministic fixture; no test data loaded.",
        "",
        "| Arm | Micro F1 | Critical recall | Remediation validity | Safety gate | Agent−control case-F1 CI |",
        "|---|---:|---:|---:|---|---:|",
    ]
    for arm, values in report["arms"].items():
        score = values["score"]
        comparison = values["agent_comparison"]
        interval = (
            "—"
            if comparison is None
            else "["
            + ", ".join(f"{value:.3f}" for value in comparison["bootstrap_95_ci"])
            + "]"
        )
        lines.append(
            f"| {labels[arm]} | {score['micro_f1']:.3f} | "
            f"{score['critical_recall']:.3f} | {score['remediation_validity']:.3f} | "
            f"{'PASS' if score['safety_gate_passed'] else 'FAIL'} | {interval} |"
        )
    lines.extend(
        (
            "",
            "## Interpretation",
            "",
            "The Agent beats AutoML variants because they lack endpoint-semantic guards. However,",
            "the complete deterministic rules guard exactly ties the Agent. Therefore current fault",
            "discovery is a useful system capability but is **not** evidence of learned scientific",
            "reasoning. A stronger claim requires independently authored unseen composite faults and",
            "remediation planning that exceeds the complete-rules baseline.",
            "",
        )
    )
    (root / "reports/fault_policy_controls_v1.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
