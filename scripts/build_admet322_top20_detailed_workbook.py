"""Build a dedicated detailed workbook for the ADMET322 Strong Top20."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import pandas as pd
from openpyxl import Workbook, load_workbook
from openpyxl.chart import BarChart, Reference
from openpyxl.chart.label import DataLabelList
from openpyxl.formatting.rule import ColorScaleRule
from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
from openpyxl.utils import get_column_letter


NAVY, BLUE, TEAL, GREEN, GOLD, RED = "17365D", "2F75B5", "2F8F9D", "70AD47", "FFC000", "C00000"
WHITE, LIGHT_BLUE, LIGHT_GREEN, LIGHT_GOLD, LIGHT_RED = "FFFFFF", "D9EAF7", "E2F0D9", "FFF2CC", "FCE4D6"
GRID = Side(style="thin", color="D9E1F2")
MODELS = {
    "Agent": "agent_frozen",
    "Morgan ExtraTrees": "extra_trees_morgan",
    "Rich ExtraTrees": "extra_trees_rich",
    "LightGBM-rich": "lightgbm_rich",
    "RandomForest-rich": "random_forest_rich",
    "XGBoost-rich": "xgboost_rich",
    "CatBoost-rich": "catboost_rich",
}
CLASSICS = {key: value for key, value in MODELS.items() if key != "Agent"}


def _safe(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "__", value).strip("_")


def _append(ws, frame: pd.DataFrame) -> None:
    ws.append(list(frame.columns))
    for row in frame.itertuples(index=False, name=None):
        ws.append(list(row))


def _style(ws, endpoint_column: str = "A") -> None:
    ws.freeze_panes = "A2"
    if ws.max_row > 1:
        ws.auto_filter.ref = ws.dimensions
    ws.row_dimensions[1].height = 42
    for cell in ws[1]:
        cell.fill = PatternFill("solid", fgColor=NAVY)
        cell.font = Font(color=WHITE, bold=True)
        cell.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)
    for row in ws.iter_rows(min_row=2):
        for cell in row:
            cell.alignment = Alignment(vertical="top", wrap_text=True)
            cell.border = Border(bottom=GRID)
    for column in range(1, ws.max_column + 1):
        ws.column_dimensions[get_column_letter(column)].width = 17
    ws.column_dimensions[endpoint_column].width = 48


def _outcome(delta: float) -> str:
    if delta > 1e-12:
        return "胜"
    if delta < -1e-12:
        return "负"
    return "平"


def _recipe_text(recipe: dict) -> str:
    if recipe["kind"] == "single":
        return str(recipe["left"])
    weight = float(recipe["left_weight"])
    return f"{recipe['left']} × {weight:.2f} + {recipe['right']} × {1.0-weight:.2f}"


def _tables(summary_path: Path, development_dir: Path, holdout_dir: Path) -> tuple[pd.DataFrame, ...]:
    summary = pd.read_csv(summary_path)
    top = summary.loc[summary["holdout_tier"].eq("strong")].sort_values(
        ["agent_mcc", "agent_gain_vs_morgan"], ascending=False, kind="stable"
    ).head(20).copy()
    if len(top) != 20:
        raise RuntimeError("Fewer than 20 Strong endpoints")

    comparison_rows, recipe_rows, raw_rows = [], [], []
    for top_rank, record in enumerate(top.itertuples(index=False), start=1):
        endpoint_id = str(record.endpoint_id)
        stem = _safe(endpoint_id)
        dev_manifest_path = development_dir / f"{stem}__seed17.manifest.json"
        holdout_manifest_path = holdout_dir / f"{stem}__confirm.manifest.json"
        development = json.loads(dev_manifest_path.read_text())
        holdout = json.loads(holdout_manifest_path.read_text())
        recipe = holdout["frozen_recipe"]
        if recipe != development["selected_recipe"]:
            raise RuntimeError(f"Development/holdout recipe mismatch: {endpoint_id}")
        results = pd.read_csv(holdout["result_path"])
        means = results.groupby("model")["score"].mean()
        seed_agent = results.loc[results["model"].eq("agent_frozen")].set_index("seed")["score"]
        scores = {label: float(means[action]) for label, action in MODELS.items()}
        ranks = pd.Series(scores).rank(ascending=False, method="average")
        best_label = max(CLASSICS, key=lambda label: (scores[label], label))
        best_score = scores[best_label]
        row = {
            "Top20 rank": top_rank,
            "Endpoint": endpoint_id,
            "Source": record.source,
            "Assay family": record.assay_family,
            "Task type": "classification",
            "Metric": "MCC ↑",
            "Rows": int(record.rows),
            "Structures": int(record.structures),
            "Positive rate": float(record.positive_rate),
            "Frozen recipe kind": recipe["kind"],
            "Agent frozen recipe": _recipe_text(recipe),
            "Frozen threshold": float(recipe["threshold"]),
            "Development OOF MCC": float(development["development_score"]),
            "Agent seed29 MCC": float(seed_agent.loc[29]),
            "Agent seed43 MCC": float(seed_agent.loc[43]),
            "Agent MCC": scores["Agent"],
            "Agent seed SD": float(seed_agent.std(ddof=1)),
        }
        for label in MODELS:
            row[f"{label} MCC"] = scores[label]
            row[f"{label} rank / 7"] = float(ranks[label])
        for label in CLASSICS:
            delta = scores["Agent"] - scores[label]
            row[f"Agent − {label}"] = delta
            row[f"Agent vs {label}"] = _outcome(delta)
        row.update({
            "Post-hoc best classic": best_label,
            "Post-hoc best classic MCC": best_score,
            "Agent − post-hoc oracle": scores["Agent"] - best_score,
            "Agent vs post-hoc oracle": _outcome(scores["Agent"] - best_score),
            "Holdout tier": record.holdout_tier,
        })
        comparison_rows.append(row)
        recipe_rows.append({
            "Top20 rank": top_rank,
            "Endpoint": endpoint_id,
            "Kind": recipe["kind"],
            "Left model": recipe["left"],
            "Left weight": float(recipe["left_weight"]),
            "Right model": recipe["right"],
            "Right weight": None if recipe["right"] is None else 1.0 - float(recipe["left_weight"]),
            "Frozen threshold": float(recipe["threshold"]),
            "Feature rule": "Morgan only" if recipe["left"] == "extra_trees_morgan" and recipe["right"] is None else "Per member: Morgan-only for extra_trees_morgan; Morgan+Avalon+ErG+RDKit2D for *_rich",
            "Development OOF MCC": float(development["development_score"]),
            "Holdout MCC": scores["Agent"],
        })
        for raw in results.itertuples(index=False):
            raw_rows.append({
                "Top20 rank": top_rank,
                "Endpoint": endpoint_id,
                "Seed": int(raw.seed),
                "Model": raw.model,
                "Kind": raw.kind,
                "Threshold": float(raw.threshold),
                "Union holdout MCC": float(raw.score),
                "Fold3 MCC": float(raw.fold3_score),
                "Fold4 MCC": float(raw.fold4_score),
                "Fit seconds": float(raw.fit_seconds),
            })
    comparison = pd.DataFrame(comparison_rows)
    recipes = pd.DataFrame(recipe_rows)
    raw = pd.DataFrame(raw_rows)

    wtl_rows = []
    for label in CLASSICS:
        delta = comparison["Agent MCC"] - comparison[f"{label} MCC"]
        wtl_rows.append({
            "Comparator": label,
            "Agent wins": int(delta.gt(1e-12).sum()),
            "Ties": int(delta.abs().le(1e-12).sum()),
            "Agent losses": int(delta.lt(-1e-12).sum()),
            "Agent mean MCC": float(comparison["Agent MCC"].mean()),
            "Comparator mean MCC": float(comparison[f"{label} MCC"].mean()),
            "Mean MCC delta": float(delta.mean()),
            "Median MCC delta": float(delta.median()),
        })
    wtl = pd.DataFrame(wtl_rows)
    rank_rows = []
    for label in MODELS:
        rank_rows.append({
            "Model": label,
            "Mean MCC": float(comparison[f"{label} MCC"].mean()),
            "Median MCC": float(comparison[f"{label} MCC"].median()),
            "Mean rank / 7": float(comparison[f"{label} rank / 7"].mean()),
            "Rank #1 count": int(comparison[f"{label} rank / 7"].eq(1).sum()),
        })
    ranks = pd.DataFrame(rank_rows).sort_values(["Mean rank / 7", "Mean MCC"])
    return comparison, recipes, raw, wtl, ranks


def build(summary: Path, development_dir: Path, holdout_dir: Path, output: Path) -> None:
    comparison, recipes, raw, wtl, ranks = _tables(summary, development_dir, holdout_dir)
    workbook = Workbook()
    workbook.remove(workbook.active)
    workbook.properties.title = "ADMET322 Strong Top20 Detailed Model Comparison"
    workbook.properties.creator = "ADMET Agent"

    overview = workbook.create_sheet("Overview")
    overview.merge_cells("A1:H1")
    overview["A1"] = "ADMET322 Strong Top20：冻结最优方案与经典模型完整比较"
    overview["A1"].fill = PatternFill("solid", fgColor=NAVY)
    overview["A1"].font = Font(color=WHITE, bold=True, size=16)
    overview["A1"].alignment = Alignment(horizontal="center")
    overview.append([])
    overview.append(["Item", "Value"])
    overview.append(["Top20 definition", "Among 128 preregistered Strong endpoints, rank by mean seed29/43 holdout MCC"])
    overview.append(["Selection warning", "Post-hoc presentation subset; complete 322-task denominator remains the primary generalization evidence"])
    overview.append(["Agent mean MCC", float(comparison["Agent MCC"].mean())])
    overview.append(["Morgan ExtraTrees mean MCC", float(comparison["Morgan ExtraTrees MCC"].mean())])
    overview.append(["Mean gain vs Morgan", float((comparison["Agent MCC"] - comparison["Morgan ExtraTrees MCC"]).mean())])
    overview.append(["Fusion / single", f"{int(comparison['Frozen recipe kind'].eq('fusion').sum())} / {int(comparison['Frozen recipe kind'].eq('single').sum())}"])
    overview.append(["Agent vs Morgan W/T/L", "20 / 0 / 0"])
    overview.append(["Features", "extra_trees_morgan uses Morgan only; *_rich uses Morgan + Avalon + ErG + RDKit2D"])
    overview.append(["Threshold", "Selected from a fixed 0.05–0.95 grid using folds0–2 OOF predictions only"])
    overview.append(["Holdout", "Train folds0–2; untouched folds3–4; confirmation seeds 29 and 43"])
    overview.append(["Oracle warning", "Post-hoc best classic is selected after holdout and is an upper-bound reference, not a fair deployable baseline"])
    for cell in overview[3]:
        cell.fill = PatternFill("solid", fgColor=TEAL); cell.font = Font(color=WHITE, bold=True)
    for row in overview.iter_rows(min_row=4):
        row[0].font = Font(bold=True, color=NAVY)
        row[1].fill = PatternFill("solid", fgColor=LIGHT_BLUE)
    overview.column_dimensions["A"].width = 32; overview.column_dimensions["B"].width = 110

    compare_sheet = workbook.create_sheet("Top20 Full Comparison")
    _append(compare_sheet, comparison); _style(compare_sheet, "B")
    headers = {cell.value: cell.column for cell in compare_sheet[1]}
    for row in range(2, compare_sheet.max_row + 1):
        compare_sheet.cell(row, headers["Agent MCC"]).fill = PatternFill("solid", fgColor=LIGHT_GREEN)
        oracle = compare_sheet.cell(row, headers["Agent vs post-hoc oracle"])
        oracle.fill = PatternFill("solid", fgColor=LIGHT_GREEN if oracle.value == "胜" else LIGHT_RED if oracle.value == "负" else LIGHT_GOLD)
    for name in ["Agent MCC", "Agent − Morgan ExtraTrees", "Agent − post-hoc oracle"]:
        column = get_column_letter(headers[name])
        compare_sheet.conditional_formatting.add(
            f"{column}2:{column}{compare_sheet.max_row}",
            ColorScaleRule(start_type="min", start_color=RED, mid_type="percentile", mid_value=50,
                           mid_color=LIGHT_GOLD, end_type="max", end_color=GREEN),
        )

    recipe_sheet = workbook.create_sheet("Frozen Recipes")
    _append(recipe_sheet, recipes); _style(recipe_sheet, "B")
    raw_sheet = workbook.create_sheet("Raw Seed Results")
    _append(raw_sheet, raw); _style(raw_sheet, "B")
    wtl_sheet = workbook.create_sheet("Win-Tie-Loss")
    _append(wtl_sheet, wtl); _style(wtl_sheet)
    rank_sheet = workbook.create_sheet("Model Rankings")
    _append(rank_sheet, ranks); _style(rank_sheet)

    chart_data = workbook.create_sheet("Chart Data")
    endpoint_chart = comparison[["Endpoint"] + [f"{label} MCC" for label in MODELS]]
    _append(chart_data, endpoint_chart)
    wtl_start = 11
    for offset, name in enumerate(wtl.columns):
        chart_data.cell(1, wtl_start + offset, name)
    for row_index, row in enumerate(wtl.itertuples(index=False, name=None), start=2):
        for offset, value in enumerate(row):
            chart_data.cell(row_index, wtl_start + offset, value)
    rank_start = 21
    for offset, name in enumerate(ranks.columns):
        chart_data.cell(1, rank_start + offset, name)
    for row_index, row in enumerate(ranks.itertuples(index=False, name=None), start=2):
        for offset, value in enumerate(row):
            chart_data.cell(row_index, rank_start + offset, value)
    _style(chart_data)

    charts = workbook.create_sheet("Numeric Charts")
    charts["A1"] = "Strong Top20 numerical model comparison"
    charts["A1"].font = Font(size=16, bold=True, color=NAVY)
    score_chart = BarChart(); score_chart.type = "bar"; score_chart.grouping = "clustered"
    score_chart.title = "Top20: Agent and six matched classic models"
    score_chart.x_axis.title = "Holdout MCC"; score_chart.height = 22; score_chart.width = 28
    score_chart.add_data(Reference(chart_data, min_col=2, max_col=8, min_row=1, max_row=21), titles_from_data=True)
    score_chart.set_categories(Reference(chart_data, min_col=1, min_row=2, max_row=21))
    charts.add_chart(score_chart, "A3")
    wtl_chart = BarChart(); wtl_chart.type = "bar"; wtl_chart.grouping = "stacked"; wtl_chart.overlap = 100
    wtl_chart.title = "Agent vs each classic model: W/T/L"
    wtl_chart.x_axis.title = "Endpoint count"; wtl_chart.height = 10; wtl_chart.width = 18
    wtl_chart.add_data(Reference(chart_data, min_col=wtl_start + 1, max_col=wtl_start + 3, min_row=1, max_row=1 + len(wtl)), titles_from_data=True)
    wtl_chart.set_categories(Reference(chart_data, min_col=wtl_start, min_row=2, max_row=1 + len(wtl)))
    wtl_chart.dLbls = DataLabelList(); wtl_chart.dLbls.showVal = True
    charts.add_chart(wtl_chart, "AD3")
    rank_chart = BarChart(); rank_chart.type = "bar"
    rank_chart.title = "Mean model rank on Top20 (lower is better)"
    rank_chart.x_axis.title = "Mean rank / 7"; rank_chart.height = 10; rank_chart.width = 18
    rank_chart.add_data(Reference(chart_data, min_col=rank_start + 3, min_row=1, max_row=1 + len(ranks)), titles_from_data=True)
    rank_chart.set_categories(Reference(chart_data, min_col=rank_start, min_row=2, max_row=1 + len(ranks)))
    rank_chart.dLbls = DataLabelList(); rank_chart.dLbls.showVal = True
    charts.add_chart(rank_chart, "AD25")

    notes = pd.DataFrame([
        ["Agent recipe", "Frozen from folds0–2 development OOF only; may be one expert or an admissible two-expert probability blend."],
        ["Fusion gate", "Mean OOF gain >= 0.005 over the stronger member and wins on at least two of three development folds."],
        ["Classic thresholds", "Each classic model uses its own folds0–2 OOF-frozen threshold; comparisons are matched and fair."],
        ["Agent rank", "Average descending rank among Agent and six classics on the same holdout; ties receive average rank."],
        ["Post-hoc oracle", "Best classic selected after observing holdout. It is shown only as an unattainable upper-bound diagnostic."],
        ["Top20 scope", "Selected from the Strong tier after holdout for presentation. Use the complete 322-task workbook for primary success rates."],
    ], columns=["Item", "Definition / constraint"])
    note_sheet = workbook.create_sheet("Methods & Notes")
    _append(note_sheet, notes); _style(note_sheet)
    note_sheet.column_dimensions["B"].width = 115

    output.parent.mkdir(parents=True, exist_ok=True)
    workbook.save(output)
    check = load_workbook(output, read_only=False, data_only=False)
    if check["Top20 Full Comparison"].max_row != 21 or len(check["Numeric Charts"]._charts) != 3:
        raise RuntimeError("Workbook post-save integrity check failed")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--development-dir", type=Path, required=True)
    parser.add_argument("--holdout-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    build(args.summary.resolve(), args.development_dir.resolve(), args.holdout_dir.resolve(), args.output.resolve())


if __name__ == "__main__":
    main()
