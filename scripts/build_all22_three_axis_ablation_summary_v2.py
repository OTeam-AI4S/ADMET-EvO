#!/usr/bin/env python3
"""Build endpoint, aggregate, and reviewer-facing summaries for ablation v2."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pandas as pd
from openpyxl import load_workbook
from openpyxl.styles import Font, PatternFill

import build_all22_three_axis_ablation_summary as core


ROOT = Path(__file__).resolve().parents[1]
VERSION = os.environ.get("THREE_AXIS_VERSION", "v2")
if VERSION not in {"v2", "v3"}:
    raise ValueError("THREE_AXIS_VERSION must be v2 or v3")
REPORT = ROOT / f"reports/all22_three_axis_ablation_{VERSION}"
CSV_PATH = REPORT / f"all22_three_axis_ablation_{VERSION}_results.csv"
XLSX_PATH = REPORT / f"all22_three_axis_ablation_{VERSION}_results.xlsx"
AUDIT_PATH = REPORT / f"all22_three_axis_ablation_{VERSION}_results.audit.json"


def _win_tie_loss(values: pd.Series, tolerance: float = 1e-12) -> dict[str, int]:
    numeric = pd.to_numeric(values, errors="raise")
    return {
        "win": int((numeric > tolerance).sum()),
        "tie": int((numeric.abs() <= tolerance).sum()),
        "loss": int((numeric < -tolerance).sum()),
    }


def main() -> None:
    core.REPORT = REPORT
    core.CSV_PATH = CSV_PATH
    core.XLSX_PATH = XLSX_PATH
    core.AUDIT_PATH = AUDIT_PATH
    core.main()

    frame = pd.read_csv(CSV_PATH)
    audit = json.loads(AUDIT_PATH.read_text(encoding="utf-8"))
    data_increment = frame["Cell 111 Normalized Delta"] - frame["Cell 110 Normalized Delta"]
    admitted = frame["Data Source Admitted"].astype(bool)
    route_counts = {
        str(key).zfill(3): int(value)
        for key, value in frame["Routed Cell"].astype(str).value_counts().items()
    }
    selected_data = []
    for task in frame["Task"]:
        discovery = json.loads(
            (REPORT / "discovery" / f"{task}.json").read_text(encoding="utf-8")
        )
        selected_data.append(discovery["selected"]["data"])
    audit.update(
        {
            "protocol": f"all22-three-axis-ablation-summary-{VERSION}",
            "routing_selection_data": "independent confirmation validation only",
            "axis_ladder_win_tie_loss": {
                axis: _win_tie_loss(frame[f"{axis} Normalized Delta"])
                for axis in ("M1", "M2", "M3", "F1", "F2", "F3", "D1", "D2", "D3")
            },
            "route_cell_counts": route_counts,
            "data_policy": {
                "source_admitted_tasks": int(admitted.sum()),
                "external_acquisition_selected_tasks": int(
                    sum(bool(value["D3"]["config"].get("enabled")) for value in selected_data)
                ),
                "data_increment_under_model_feature_all_tasks": {
                    "mean": float(data_increment.mean()),
                    **_win_tie_loss(data_increment),
                },
                "data_increment_under_model_feature_admitted_tasks": {
                    "mean": float(data_increment[admitted].mean()) if admitted.any() else 0.0,
                    **_win_tie_loss(data_increment[admitted]),
                },
                "interpretation": "data-centric curation, valuation, selective acquisition, and abstention",
            },
        }
    )
    AUDIT_PATH.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n")

    workbook = load_workbook(XLSX_PATH)
    sheet = workbook.create_sheet("Reviewer Summary", 0)
    sheet.append(["Section", "Metric", "Value"])
    rows = [
        ("Protocol", "Endpoints", len(frame)),
        ("Protocol", "Routing selection", "Independent confirmation validation only"),
        ("Protocol", "Hidden-test selection", "Forbidden"),
        ("Data", "Admitted external sources", int(admitted.sum())),
        ("Data", "Selected external acquisition", audit["data_policy"]["external_acquisition_selected_tasks"]),
        ("Data", "M+F conditional mean increment", float(data_increment.mean())),
        ("Data", "M+F conditional admitted-task increment", audit["data_policy"]["data_increment_under_model_feature_admitted_tasks"]["mean"]),
    ]
    for axis in ("M3", "F3", "D3"):
        values = audit["aggregate_axis_ladders"][axis]
        outcome = audit["axis_ladder_win_tie_loss"][axis]
        rows.append((axis[0] + " axis", "Mean normalized delta", values["mean"]))
        rows.append((axis[0] + " axis", "Win/tie/loss", f"{outcome['win']}/{outcome['tie']}/{outcome['loss']}"))
    for row in rows:
        sheet.append(row)
    for cell in sheet[1]:
        cell.font = Font(bold=True, color="FFFFFF")
        cell.fill = PatternFill("solid", fgColor="1F4E78")
    sheet.freeze_panes = "A2"
    sheet.column_dimensions["A"].width = 18
    sheet.column_dimensions["B"].width = 42
    sheet.column_dimensions["C"].width = 28
    workbook.save(XLSX_PATH)
    print(json.dumps(audit, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
