"""Extract a validation-only research-process evidence record from one run."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from admet_agent.research_process_metrics import extract_research_process_metrics


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build(run_dir: Path) -> dict:
    summary_paths = sorted(run_dir.glob("tasks/*/search_seed_*.json"))
    if not summary_paths:
        raise ValueError(f"No search summaries found under {run_dir}")
    summaries = [json.loads(path.read_text(encoding="utf-8")) for path in summary_paths]
    ledger_path = run_dir / "evidence_ledger.jsonl"
    ledger = (
        [
            json.loads(line)
            for line in ledger_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        if ledger_path.exists()
        else []
    )
    metrics = extract_research_process_metrics(summaries, ledger).to_dict()
    return {
        "protocol": "validation-only-research-process-baseline-v1",
        "run_dir": str(run_dir),
        "metrics": metrics,
        "evidence_hashes": {
            "search_summaries": {str(path): _sha256(path) for path in summary_paths},
            "evidence_ledger": _sha256(ledger_path) if ledger_path.exists() else None,
        },
        "test_loaded": False,
        "selection_feedback_to_discovery": False,
        "evidence_scope": "retrospective_validation_process_audit",
        "limitations": [
            "This is not a same-budget prospective control comparison.",
            "Missing self-correction or action ordering evidence is not counted as success.",
            "Historical runs without robust-racing events cannot claim compute saved.",
        ],
    }


def render(payload: dict) -> str:
    metrics = payload["metrics"]
    resources = metrics["resources"]
    outcomes = metrics["outcome_counts"]
    values = (
        ("Single-axis validity", metrics["single_axis_contrast_validity"]["rate"]),
        (
            "Treatment/control completeness",
            metrics["treatment_control_completeness"]["rate"],
        ),
        ("Falsification resolution", metrics["falsification_resolution"]["rate"]),
        ("Negative-result retention", metrics["negative_result_retention"]["rate"]),
        ("Self-correction", metrics["self_correction_rate"]["rate"]),
        (
            "Preregistered action discipline",
            metrics["preregistered_action_discipline"]["rate"],
        ),
    )
    rows = [
        "# Retrospective validation-only research-process baseline",
        "",
        f"Run: `{payload['run_dir']}`",
        "",
        "| Metric | Conservative rate | Evidence coverage |",
        "|---|---:|---:|",
    ]
    for label, value in values:
        key = {
            "Single-axis validity": "single_axis_contrast_validity",
            "Treatment/control completeness": "treatment_control_completeness",
            "Falsification resolution": "falsification_resolution",
            "Negative-result retention": "negative_result_retention",
            "Self-correction": "self_correction_rate",
            "Preregistered action discipline": "preregistered_action_discipline",
        }[label]
        coverage = metrics[key]["coverage"]
        rendered_value = "n/a" if value is None else f"{value:.3f}"
        rendered_coverage = "n/a" if coverage is None else f"{coverage:.3f}"
        rows.append(f"| {label} | {rendered_value} | {rendered_coverage} |")
    rows.extend(
        (
            "",
            (
                f"Outcomes: {outcomes['supported']} supported, "
                f"{outcomes['refuted']} refuted, "
                f"{outcomes['inconclusive']} inconclusive."
            ),
            "",
            (
                f"Resources: {resources['total_fits']} fits, "
                f"{resources['fit_seconds']:.1f} fit-seconds; planned fits avoided "
                f"by recorded racing: {resources['planned_fits_avoided']}."
            ),
            "",
            (
                "Current blockers: self-correction evidence has zero coverage; action "
                "ordering has zero coverage; this historical run has no robust-racing "
                "record. These are explicit missing capabilities/evidence, not successes."
            ),
            "",
            (
                "This report is retrospective validation evidence only. It does not "
                "establish an Agent-vs-AutoML advantage until the preregistered "
                "same-budget controls are executed."
            ),
            "",
        )
    )
    return "\n".join(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output-prefix", type=Path, required=True)
    args = parser.parse_args()
    payload = build(args.run_dir)
    args.output_prefix.parent.mkdir(parents=True, exist_ok=True)
    args.output_prefix.with_suffix(".json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    args.output_prefix.with_suffix(".md").write_text(render(payload), encoding="utf-8")


if __name__ == "__main__":
    main()
