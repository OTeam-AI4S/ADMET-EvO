"""Build the sealed local external-data stewardship control report."""

from __future__ import annotations

import json
from pathlib import Path

from admet_agent.data_stewardship_challenge import (
    compare_data_stewardship_controls,
)


def main() -> None:
    root = Path(__file__).parents[1]
    report = compare_data_stewardship_controls()
    (root / "reports/data_stewardship_controls_v1.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    labels = {
        "agent_contract_steward": "Agent contract steward",
        "complete_rules_steward": "Complete rules steward",
        "expanded_automl_schema_steward": "Expanded AutoML schema steward",
        "conventional_automl_direct_fit": "Conventional AutoML direct fit",
        "seeded_random_admission": "Seeded random",
    }
    lines = [
        "# External-data stewardship controls",
        "",
        "Sealed local 11-dossier fixture; no TDC test data loaded.",
        "",
        "| Arm | Decision accuracy | Issue F1 | Manifest accuracy | Unsafe merges | Safety gate |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for arm, score in report["arms"].items():
        lines.append(
            f"| {labels[arm]} | {score['decision_accuracy']:.3f} | "
            f"{score['issue_f1']:.3f} | {score['manifest_assessment_accuracy']:.3f} | "
            f"{score['unsafe_merge_count']} | "
            f"{'PASS' if score['safety_gate_passed'] else 'FAIL'} |"
        )
    lines.extend(
        (
            "",
            "The Agent prevents unsafe merges that schema-only and direct-fit AutoML miss, but",
            "it exactly ties the complete deterministic rules steward. This establishes a useful",
            "data-governance safety capability, not learned autonomous reasoning. Independent",
            "composite dossiers and evidence-dependent remediation are required for a stronger claim.",
            "",
        )
    )
    (root / "reports/data_stewardship_controls_v1.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
