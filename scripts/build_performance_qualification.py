"""Build the immutable application-performance qualification record."""

from __future__ import annotations

import json
from pathlib import Path

from admet_agent.performance_qualification import qualify_frozen_scorecard_file


def main() -> None:
    root = Path(__file__).parents[1]
    qualification = qualify_frozen_scorecard_file(
        root / "reports/goal_scorecard_v27.json"
    )
    payload = qualification.to_dict()
    output = root / "reports/performance_qualification_v1.json"
    output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    exceptions = ", ".join(payload["exception_tasks"]) or "none"
    markdown = "\n".join(
        (
            "# TDC-22 application performance qualification",
            "",
            f"Status: **{'PASS' if payload['passed'] else 'BLOCKED'}**",
            "",
            f"- Frozen outcomes: {payload['tasks_observed']}/22 endpoints",
            f"- Top-5: {payload['top5_tasks']}/22 (floor: {payload['policy']['minimum_top5_tasks']})",
            f"- Top-10: {payload['top10_tasks']}/22 (floor: {payload['policy']['minimum_top10_tasks']})",
            f"- Explicit exception(s): {exceptions}",
            f"- Full freeze SHA-256: `{payload['full_freeze_sha256']}`",
            f"- Qualification seal SHA-256: `{payload['seal_sha256']}`",
            "- Test labels loaded by this qualification step: no",
            "- Selection feedback to discovery: forbidden",
            "",
            "This is an application-performance gate only. It does not prove predictive",
            "superiority over AutoML or a research-process advantage. Those claims require",
            "the preregistered validation-only controls and capability scorecard.",
            "",
        )
    )
    (root / "reports/performance_qualification_v1.md").write_text(
        markdown, encoding="utf-8"
    )


if __name__ == "__main__":
    main()
