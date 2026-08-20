"""Assemble measured compute and explicit peak-memory missingness for v2.7."""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
from pathlib import Path
from typing import Any


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _pilot_expert_rows(root: Path, task: str) -> list[dict[str, Any]]:
    path = root / "runs/unimol2-pilot-v22" / task / "results.json"
    report = _load(path)
    if report["protocol"]["test_labels_used"] is not False:
        raise ValueError(f"{task} pilot is not validation-only")
    expert_names = ("2d_lightgbm", "2d_extra_trees", "finetuned_unimol2")
    rows = []
    for expert in expert_names:
        values = [
            float(record["elapsed_seconds"][expert])
            for _, record in sorted(report["seeds"].items(), key=lambda item: int(item[0]))
        ]
        rows.append(
            {
                "scope": task,
                "expert": expert,
                "replicates": len(values),
                "mean_wall_seconds": statistics.mean(values),
                "total_wall_seconds": sum(values),
                "peak_memory_mib": None,
                "peak_memory_status": "not_recorded",
                "source": path.relative_to(root).as_posix(),
                "source_sha256": sha256(path),
            }
        )
    return rows


def _scale_rows(root: Path) -> list[dict[str, Any]]:
    rows = []
    for label, relative in (
        (
            "finetuned_unimol2_84m",
            "runs/unimol2-scale-ablation-84m-b8/84m/"
            "Lipophilicity_AstraZeneca/lr_0.0001",
        ),
        (
            "finetuned_unimol2_164m",
            "runs/unimol2-scale-ablation-b8/164m/"
            "Lipophilicity_AstraZeneca/lr_0.0001",
        ),
    ):
        directory = root / relative
        paths = sorted(directory.glob("seed_*/results.json"))
        records = [_load(path) for path in paths]
        if [int(record["seed"]) for record in records] != [1, 2, 3, 4, 5]:
            raise ValueError(f"Incomplete scale records for {label}")
        if any(record["test_labels_used"] is not False for record in records):
            raise ValueError(f"{label} scale records are not validation-only")
        values = [float(record["elapsed_seconds"]) for record in records]
        rows.append(
            {
                "scope": "Lipophilicity_AstraZeneca",
                "expert": label,
                "replicates": len(values),
                "mean_wall_seconds": statistics.mean(values),
                "total_wall_seconds": sum(values),
                "peak_memory_mib": None,
                "peak_memory_status": "not_recorded",
                "source": directory.relative_to(root).as_posix(),
                "source_files_sha256": [sha256(path) for path in paths],
            }
        )
    return rows


def build_report(root: Path) -> dict[str, Any]:
    root = root.resolve()
    router_path = root / "runs/v24-router-replay/report.json"
    router = _load(router_path)
    if router["official_test_loaded"] is not False or router["task_count"] != 22:
        raise ValueError("Router replay is not a complete validation-only report")
    task_rows = []
    for row in router["tasks"]:
        selected = row["selections"]["cost_aware_lcb"]
        score = row["selections"]["score"]
        task_rows.append(
            {
                "task": row["task"],
                "validation_seeds": row["validation_seeds"],
                "selected_action": selected["action"],
                "selected_recipe_seconds": selected["recipe_seconds"],
                "score_policy_recipe_seconds": score["recipe_seconds"],
                "selected_to_score_cost_ratio": selected["recipe_seconds"]
                / score["recipe_seconds"],
                "peak_memory_mib": None,
                "peak_memory_status": "not_recorded",
            }
        )

    expert_rows = [
        *_pilot_expert_rows(root, "DILI"),
        *_pilot_expert_rows(root, "hERG"),
        *_scale_rows(root),
    ]
    confirmation_path = root / "runs/v27-maplight-gnn-confirmation/result.json"
    confirmation = _load(confirmation_path)
    if confirmation["protocol"]["official_test_loaded"] is not False:
        raise ValueError("MapLight+GNN confirmation is not validation-only")
    expert_rows.append(
        {
            "scope": "Bioavailability_Ma + CYP2C9_Substrate_CarbonMangels",
            "expert": "MapLight and MapLight+GNN confirmation campaign",
            "replicates": 20,
            "mean_wall_seconds": confirmation["elapsed_seconds"] / 20.0,
            "total_wall_seconds": confirmation["elapsed_seconds"],
            "timing_scope": (
                "campaign wall time divided by 20 task/seed/model fits; shared feature "
                "generation means this is not an isolated per-fit benchmark"
            ),
            "peak_memory_mib": None,
            "peak_memory_status": "not_recorded",
            "source": confirmation_path.relative_to(root).as_posix(),
            "source_sha256": sha256(confirmation_path),
        }
    )
    peak_records = [*task_rows, *expert_rows]
    return {
        "version": "v27-compute-memory-appendix-1",
        "analysis_role": "retrospective_accounting_of_recorded_validation_compute",
        "official_test_loaded": False,
        "official_results_used": False,
        "selection_or_freeze_changed": False,
        "comparability_warning": (
            "Wall times come from different experiment phases and environments and are not "
            "hardware-normalized. Only matched within-protocol ratios are interpreted."
        ),
        "peak_memory_warning": (
            "Historical runs did not record process or CUDA peak memory. Missing values are "
            "reported explicitly; absolute peak-memory claims require a fresh instrumented rerun."
        ),
        "router_replay": {
            "source": router_path.relative_to(root).as_posix(),
            "source_sha256": sha256(router_path),
            "task_count": len(task_rows),
            "validation_seed_count": 3,
            "selected_total_hours": sum(row["selected_recipe_seconds"] for row in task_rows)
            / 3600.0,
            "score_policy_total_hours": sum(
                row["score_policy_recipe_seconds"] for row in task_rows
            )
            / 3600.0,
            "selected_to_score_total_cost_ratio": sum(
                row["selected_recipe_seconds"] for row in task_rows
            )
            / sum(row["score_policy_recipe_seconds"] for row in task_rows),
            "tasks": task_rows,
        },
        "expert_measurements": expert_rows,
        "peak_memory_coverage": {
            "records_total": len(peak_records),
            "records_measured": sum(row["peak_memory_mib"] is not None for row in peak_records),
            "records_missing": sum(row["peak_memory_mib"] is None for row in peak_records),
        },
    }


def markdown(report: dict[str, Any]) -> str:
    replay = report["router_replay"]
    lines = [
        "# v2.7 compute and peak-memory appendix",
        "",
        "## Router candidate-stage compute",
        "",
        (
            f"Across 22 tasks and three saved validation seeds, the cost-aware policy used "
            f"{replay['selected_total_hours']:.3f} recorded recipe-hours versus "
            f"{replay['score_policy_total_hours']:.3f} for score selection "
            f"(ratio {replay['selected_to_score_total_cost_ratio']:.3f})."
        ),
        "",
        "| Task | Action | Cost-aware seconds | Score-policy seconds | Ratio | Peak memory |",
        "|---|---|---:|---:|---:|---|",
    ]
    for row in replay["tasks"]:
        lines.append(
            f"| {row['task']} | {row['selected_action']} | "
            f"{row['selected_recipe_seconds']:.1f} | {row['score_policy_recipe_seconds']:.1f} | "
            f"{row['selected_to_score_cost_ratio']:.3f} | not recorded |"
        )
    lines.extend(
        [
            "",
            "## Recorded expert-family wall time",
            "",
            "| Scope | Expert | Replicates | Mean seconds | Total hours | Peak memory |",
            "|---|---|---:|---:|---:|---|",
        ]
    )
    for row in report["expert_measurements"]:
        lines.append(
            f"| {row['scope']} | {row['expert']} | {row['replicates']} | "
            f"{row['mean_wall_seconds']:.1f} | {row['total_wall_seconds'] / 3600:.3f} | "
            "not recorded |"
        )
    coverage = report["peak_memory_coverage"]
    lines.extend(
        [
            "",
            "## Measurement boundary",
            "",
            report["comparability_warning"],
            "",
            report["peak_memory_warning"],
            "",
            (
                f"Peak-memory coverage: {coverage['records_measured']}/{coverage['records_total']} "
                "records measured."
            ),
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument(
        "--output", type=Path, default=Path("reports/compute_memory_appendix_v27")
    )
    args = parser.parse_args()
    report = build_report(args.root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.with_suffix(".json").write_text(
        json.dumps(report, indent=2, sort_keys=True), encoding="utf-8"
    )
    args.output.with_suffix(".md").write_text(markdown(report), encoding="utf-8")
    print(
        json.dumps(
            {
                "router_tasks": report["router_replay"]["task_count"],
                "selected_to_score_ratio": report["router_replay"][
                    "selected_to_score_total_cost_ratio"
                ],
                "peak_memory_coverage": report["peak_memory_coverage"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
