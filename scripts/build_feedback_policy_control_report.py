"""Build the locked two-case corrective-feedback control report."""

from __future__ import annotations

import json
from pathlib import Path

from admet_agent.feedback_policy_controls import compare_feedback_controls


def main() -> None:
    root = Path(__file__).parents[1]
    schedule = json.loads(
        (root / "reports/self_correction_schedule_v1.json").read_text(encoding="utf-8")
    )
    smoke = json.loads(
        (root / "reports/self_correction_smoke_v1.json").read_text(encoding="utf-8")
    )
    report = compare_feedback_controls(schedule, smoke)
    (root / "reports/feedback_policy_controls_v1.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    labels = {
        "agent_feedback_loop": "Agent feedback loop",
        "complete_rules_feedback_loop": "Complete rules",
        "expanded_automl_scalar_search": "Expanded AutoML",
        "conventional_automl_scalar_search": "Conventional AutoML",
        "seeded_random_action": "Seeded random",
    }
    lines = [
        "# Corrective-feedback policy controls",
        "",
        "Locked replay of the two completed validation-only smoke episodes.",
        "",
        "| Arm | Feedback response | Parent link | Matched pair | Low-info stop | No high-cost escalation | Macro process |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for arm, payload in report["arms"].items():
        dims = payload["dimensions"]
        lines.append(
            f"| {labels[arm]} | {dims['feedback_conditioned_response']:.3f} | "
            f"{dims['explicit_parent_link']:.3f} | "
            f"{dims['matched_pair_protection']:.3f} | "
            f"{dims['bounded_low_information_stop']:.3f} | "
            f"{dims['no_high_cost_escalation']:.3f} | "
            f"{payload['macro_process_score']:.3f} |"
        )
    lines.extend(
        (
            "",
            "The explicit Agent loop exceeds scalar AutoML on this process contract, but",
            "complete deterministic rules tie it exactly. With only two local cases, this",
            "is an integration check and claim-boundary result—not an independent autonomy",
            "benchmark or evidence of learned reasoning beyond rules.",
            "",
        )
    )
    (root / "reports/feedback_policy_controls_v1.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
