"""Detect suspicious cross-task duplication in a frozen leaderboard snapshot."""

from __future__ import annotations

import argparse
import json
from itertools import combinations
from pathlib import Path


def entry_key(row: dict) -> tuple[str, float, float]:
    return str(row["model"]), float(row["mean"]), float(row["std"])


def audit(snapshot: dict, minimum_jaccard: float) -> dict:
    pages = snapshot["pages"]
    entries = {
        task: {entry_key(row) for row in payload["entries"]}
        for task, payload in pages.items()
    }
    findings = []
    for left, right in combinations(sorted(entries), 2):
        shared = entries[left].intersection(entries[right])
        union = entries[left].union(entries[right])
        jaccard = len(shared) / len(union) if union else 0.0
        if jaccard < minimum_jaccard:
            continue
        findings.append(
            {
                "left_task": left,
                "right_task": right,
                "left_metric": pages[left]["metric"],
                "right_metric": pages[right]["metric"],
                "shared_entries": len(shared),
                "union_entries": len(union),
                "jaccard": jaccard,
                "identical_model_score_std": [
                    {"model": model, "mean": mean, "std": std}
                    for model, mean, std in sorted(shared)
                ],
            }
        )
    return {
        "snapshot_captured_at": snapshot.get("captured_at"),
        "snapshot_source": snapshot.get("source"),
        "minimum_jaccard": minimum_jaccard,
        "task_count": len(pages),
        "findings": findings,
        "integrity_status": "manual_verification_required" if findings else "no_duplicate_page_signal",
        "interpretation": (
            "High cross-task overlap of exact (model, mean, std) triples is a "
            "leaderboard-integrity warning. It does not prove which page is wrong "
            "and must not be used to silently replace a threshold."
        ),
    }


def markdown(report: dict) -> str:
    lines = [
        "# Leaderboard consistency audit",
        "",
        f"Snapshot: `{report['snapshot_captured_at']}`",
        "",
        f"Status: **{report['integrity_status']}**",
        "",
        report["interpretation"],
        "",
        "| Left task | Right task | Exact shared rows | Union rows | Jaccard |",
        "|---|---|---:|---:|---:|",
    ]
    for finding in report["findings"]:
        lines.append(
            "| {left_task} | {right_task} | {shared_entries} | {union_entries} | {jaccard:.3f} |".format(
                **finding
            )
        )
    if not report["findings"]:
        lines.append("| none | none | 0 | 0 | 0.000 |")
    lines.extend(
        [
            "",
            "## Exact duplicated rows",
            "",
        ]
    )
    for finding in report["findings"]:
        lines.append(f"### {finding['left_task']} / {finding['right_task']}")
        lines.append("")
        lines.append("| Model | Mean | Std |")
        lines.append("|---|---:|---:|")
        for row in finding["identical_model_score_std"]:
            lines.append(f"| {row['model']} | {row['mean']:.6g} | {row['std']:.6g} |")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--snapshot",
        type=Path,
        default=Path("reports/leaderboard_snapshot_2026-08-13.json"),
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path("reports/leaderboard_integrity_audit_v24.json"),
    )
    parser.add_argument(
        "--output-md",
        type=Path,
        default=Path("reports/leaderboard_integrity_audit_v24.md"),
    )
    parser.add_argument("--minimum-jaccard", type=float, default=0.75)
    args = parser.parse_args()
    report = audit(
        json.loads(args.snapshot.read_text(encoding="utf-8")),
        args.minimum_jaccard,
    )
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(report, indent=2, sort_keys=True), encoding="utf-8"
    )
    args.output_md.write_text(markdown(report), encoding="utf-8")
    print(
        json.dumps(
            {
                "integrity_status": report["integrity_status"],
                "finding_count": len(report["findings"]),
                "pairs": [
                    [row["left_task"], row["right_task"], row["jaccard"]]
                    for row in report["findings"]
                ],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
