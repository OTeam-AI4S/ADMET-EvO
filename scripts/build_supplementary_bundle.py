#!/usr/bin/env python3
"""Build the compact Supplementary Figure S1 and Tables S1-S3 bundle.

The script intentionally keeps the rendered Markdown compact while exporting
the complete numeric and curation fields as machine-readable CSV files.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from statistics import mean


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "paper" / "supplementary"

FIGURE_S1 = (
    ROOT
    / "paper"
    / "methods"
    / "figures"
    / "figure1_nature_autonomous_research_architecture.png"
)
FIGURE_S1_PDF = FIGURE_S1.with_suffix(".pdf")
FIGURE_S1_SVG = FIGURE_S1.with_suffix(".svg")

S1_INPUT = ROOT / "figures" / "three_axis_ablation_v2" / "source_data.csv"
S1_DETAIL_INPUT = (
    ROOT / "figures" / "five_axis_observed_gain_v2" / "data_axis_import_audit.csv"
)
S1_ADMISSION_DIR = (
    ROOT / "reports" / "all22_adaptive_agent_data_v1" / "development"
)
S2_INPUT = (
    ROOT
    / "figures"
    / "figure4_task_discovery"
    / "tox43_task_taxonomy"
    / "tox43_classified_endpoints.csv"
)
S3_INPUT = (
    ROOT / "reports" / "admet322_all_tasks_best_model_detailed_2026-08-31.csv"
)

S1_OUTPUT = OUTPUT_DIR / "table_s1_three_axis_ablation.csv"
S2_OUTPUT = OUTPUT_DIR / "table_s2_tox43_task_dictionary.csv"
S3_OUTPUT = OUTPUT_DIR / "table_s3_tox43_model_comparison.csv"
MD_OUTPUT = OUTPUT_DIR / "figure_s1_tables_s1_s3.md"

CLASS_ORDER = [
    "Cytotoxicity",
    "Cellular dysfunction",
    "Endocrine disruption",
    "Stress response",
    "Immune dysregulation",
    "Vascular dysfunction",
    "Tissue remodelling",
    "Developmental toxicity",
    "Assay interference",
]

FIXED_PIPELINES = [
    ("ExtraTrees-Morgan", "ExtraTrees-Morgan MCC", "extratrees_morgan_mcc"),
    ("ExtraTrees-Rich", "ExtraTrees-Rich MCC", "extratrees_rich_mcc"),
    ("LightGBM-Rich", "LightGBM-Rich MCC", "lightgbm_rich_mcc"),
    ("RandomForest-Rich", "RandomForest-Rich MCC", "random_forest_rich_mcc"),
    ("XGBoost-Rich", "XGBoost-Rich MCC", "xgboost_rich_mcc"),
    ("CatBoost-Rich", "CatBoost-Rich MCC", "catboost_rich_mcc"),
]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, str]], fields: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def f6(value: str | float) -> str:
    return f"{float(value):.6f}"


def md(value: str) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ").strip()


def rel(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def build_s1() -> tuple[list[dict[str, str]], list[str]]:
    rows = read_csv(S1_INPUT)
    detail_rows = read_csv(S1_DETAIL_INPUT)
    assert len(rows) == 22
    assert len({row["task"] for row in rows}) == 22
    assert len(detail_rows) == 22
    detail_by_task = {row["task"]: row for row in detail_rows}
    assert set(detail_by_task) == {row["task"] for row in rows}

    out: list[dict[str, str]] = []
    for row in rows:
        detail = detail_by_task[row["task"]]
        metric = row["metric"]
        metric_key = metric.upper()
        baseline = float(row["baseline_score_mean"])
        feature = float(row["feature_score_mean"])
        model = float(row["model_score_mean"])
        data_original = float(row["data_original_score"])
        data_final = float(row["data_final_score"])
        direction = -1.0 if metric_key == "MAE" else 1.0

        assert abs(direction * (feature - baseline) - float(row["feature_delta"])) < 1e-12
        assert abs(direction * (model - baseline) - float(row["model_delta"])) < 1e-12
        assert abs(direction * (data_final - data_original) - float(row["data_delta"])) < 1e-12

        admission_decision = ""
        failed_checks = ""
        review_rationale = ""
        policy_status = ""
        admission_record = ""
        if row["data_status"] == "positive":
            outcome_reason = (
                f"Selected Broad-DB candidate (ratio={float(row['data_best_ratio']):g}; "
                f"seed={float(row['data_best_seed']):g})"
            )
        elif row["data_status"] == "baseline_retained":
            outcome_reason = (
                "Baseline retained; best explored candidate was non-improving "
                f"(oriented delta={float(detail['reported_best_delta']):.6f})"
            )
        else:
            admission_path = S1_ADMISSION_DIR / f"{row['task']}.json"
            with admission_path.open("r", encoding="utf-8") as handle:
                admission = json.load(handle)
            prior = admission["prior_admission_decision"]
            assert prior["decision"] == "fail_closed"
            assert prior["model_scores_consulted"] is False
            assert prior["permitted_for_direct_augmentation"] is False
            observed = prior["observed"]
            minimum = (
                prior["thresholds"]["min_classification_identities"]
                if admission["task_type"] == "classification"
                else prior["thresholds"]["min_regression_identities"]
            )
            if row["task"] in {
                "CYP2C9_Substrate_CarbonMangels",
                "CYP2D6_Substrate_CarbonMangels",
            }:
                outcome_reason = (
                    "Rejected before training: "
                    f"{observed['output_identities']} identities (<{minimum}), "
                    "within-source label conflicts, shared/unresolved lineage and "
                    "exact-overlap target conflicts; absence-as-negative risk"
                )
            else:
                assert row["task"] == "Clearance_Hepatocyte_AZ"
                outcome_reason = (
                    "Rejected before training: "
                    f"{observed['output_identities']} identities (<{minimum}), no "
                    "qualifying chemical-space coverage and two non-equivalent "
                    "normalization units"
                )
            admission_decision = prior["decision"]
            failed_checks = ";".join(prior["failed_checks"])
            review_rationale = prior["review_rationale"]
            policy_status = admission["status"]
            admission_record = rel(admission_path)

        out.append(
            {
                "task_id": row["task"],
                "display_name": row["display_name"],
                "category": row["category"],
                "metric": metric,
                "baseline_score_mean": row["baseline_score_mean"],
                "baseline_score_sd": row["baseline_score_sd"],
                "feature_score_mean": row["feature_score_mean"],
                "feature_score_sd": row["feature_score_sd"],
                "feature_oriented_delta": row["feature_delta"],
                "model_score_mean": row["model_score_mean"],
                "model_score_sd": row["model_score_sd"],
                "model_oriented_delta": row["model_delta"],
                "data_original_score": row["data_original_score"],
                "data_final_score": row["data_final_score"],
                "data_plotted_oriented_delta": row["data_delta"],
                "data_observed_oriented_delta": (
                    "" if row["data_status"] == "no_external_data" else row["data_delta"]
                ),
                "data_status": row["data_status"],
                "data_outcome_reason": outcome_reason,
                "data_candidate_oriented_delta": detail["reported_best_delta"],
                "data_admission_decision": admission_decision,
                "data_failed_checks": failed_checks,
                "data_review_rationale": review_rationale,
                "data_policy_status": policy_status,
                "data_admission_record": admission_record,
                "data_best_ratio": row["data_best_ratio"],
                "data_best_seed": row["data_best_seed"],
                "data_seed2_4_variance": row["data_seed2_4_variance"],
                "feature_evidence": row["feature_evidence"],
                "model_evidence": row["model_evidence"],
                "data_evidence": row["data_evidence"],
            }
        )

    fields = list(out[0])
    write_csv(S1_OUTPUT, out, fields)
    return out, fields


def build_s2() -> tuple[list[dict[str, str]], dict[str, str]]:
    rows = read_csv(S2_INPUT)
    assert len(rows) == 43
    assert len({row["task_id"] for row in rows}) == 43
    class_rank = {name: index for index, name in enumerate(CLASS_ORDER)}
    original_rank = {row["task_id"]: index for index, row in enumerate(rows)}
    rows.sort(
        key=lambda row: (
            class_rank[row["liability_class"]],
            original_rank[row["task_id"]],
        )
    )

    out: list[dict[str, str]] = []
    task_number: dict[str, str] = {}
    keep_fields = [
        "task_id",
        "display_label",
        "liability_class",
        "liability_class_zh",
        "specific_readout",
        "biological_level",
        "evidence_role",
        "curation_confidence",
        "metadata_source",
        "classification_note",
        "source_assay_family",
        "sample_count",
        "positive_rate",
    ]
    for index, row in enumerate(rows, start=1):
        number = f"T{index:02d}"
        task_number[row["task_id"]] = number
        out.append({"task_no": number, **{field: row[field] for field in keep_fields}})

    fields = list(out[0])
    write_csv(S2_OUTPUT, out, fields)
    return out, task_number


def build_s3(
    s2_rows: list[dict[str, str]], task_number: dict[str, str]
) -> list[dict[str, str]]:
    report = read_csv(S3_INPUT)
    positive_toxcast = [
        row
        for row in report
        if row["数据源"].strip().lower() == "toxcast"
        and float(row["Agent相对最佳经典 ΔMCC"]) > 0
    ]
    curated_ids = {row["task_id"] for row in s2_rows}
    assert len(positive_toxcast) == 43
    assert {row["Task"] for row in positive_toxcast} == curated_ids
    report_by_task = {row["Task"]: row for row in positive_toxcast}

    out: list[dict[str, str]] = []
    for task in s2_rows:
        task_id = task["task_id"]
        row = report_by_task[task_id]
        fixed_scores = {
            display_name: float(row[source_field])
            for display_name, source_field, _ in FIXED_PIPELINES
        }
        best_name = max(fixed_scores, key=fixed_scores.get)
        best_score = fixed_scores[best_name]
        agent_score = float(row["Agent MCC"])
        delta = agent_score - best_score

        assert best_name == row["最佳经典模型"]
        assert abs(best_score - float(row["最佳经典MCC"])) < 1e-6
        assert abs(delta - float(row["Agent相对最佳经典 ΔMCC"])) < 1.1e-6
        assert delta > 0

        item = {
            "task_no": task_number[task_id],
            "task_id": task_id,
            "display_label": task["display_label"],
            "liability_class": task["liability_class"],
            "admet_evo_recipe": row["冻结Agent模型"],
            "admet_evo_mcc": f6(agent_score),
        }
        for display_name, _, output_field in FIXED_PIPELINES:
            item[output_field] = f6(fixed_scores[display_name])
        item.update(
            {
                "best_fixed_pipeline": best_name,
                "best_fixed_mcc": f6(row["最佳经典MCC"]),
                # Retain the workbook-reported six-decimal delta rather than
                # subtracting two already-rounded display values.
                "delta_mcc": f6(row["Agent相对最佳经典 ΔMCC"]),
                "sample_count": row["样本数"],
                "positive_rate": row["阳性率"],
                "local_holdout_grade": row["原始Holdout等级"],
            }
        )
        out.append(item)

    assert len(out) == 43
    fields = list(out[0])
    write_csv(S3_OUTPUT, out, fields)
    return out


def render_markdown(
    s1_rows: list[dict[str, str]],
    s2_rows: list[dict[str, str]],
    s3_rows: list[dict[str, str]],
) -> str:
    lines: list[str] = [
        "# Supplementary Figure and Tables",
        "",
        "## Figure S1 | Operational architecture of ADMET-EvO",
        "",
        f"![Figure S1. Operational architecture of ADMET-EvO]({rel(FIGURE_S1)})",
        "",
        (
            "**Figure S1 | Operational architecture of ADMET-EvO.** "
            "**a,** The research loop updates an explicit state through contract, "
            "diagnosis, hypothesis, intervention, evaluation and revision. Evidence "
            "is retained as supported, refuted or inconclusive, and the frozen "
            "predictor is separated from blind certification. **b,** Data expansion "
            "is treated as evidence admission: candidate records pass endpoint "
            "definition, provenance preservation, assay harmonization, exact-analogue "
            "decontamination and role assignment before use. Endpoint-name agreement "
            "alone does not authorize merging."
        ),
        "",
        f"Vector files: [PDF]({rel(FIGURE_S1_PDF)}) · [SVG]({rel(FIGURE_S1_SVG)})",
        "",
        "## Table S1 | Raw values underlying the three-axis ablation heatmap",
        "",
        "| Task | Metric | Feature Δ | Model Δ | Data Δ | Data outcome / reason |",
        "|---|---:|---:|---:|---:|---|",
    ]

    for row in s1_rows:
        if row["data_status"] == "no_external_data":
            data_value = "NA†"
        elif row["data_status"] == "baseline_retained":
            data_value = f6(row["data_observed_oriented_delta"])
        else:
            data_value = f6(row["data_observed_oriented_delta"])
        lines.append(
            "| "
            + " | ".join(
                [
                    md(row["display_name"]),
                    md(row["metric"]),
                    f6(row["feature_oriented_delta"]),
                    f6(row["model_oriented_delta"]),
                    data_value,
                    md(row["data_outcome_reason"]),
                ]
            )
            + " |"
        )

    lines.extend(
        [
            "",
            (
                "Values are native-metric oriented differences: Δ = candidate − "
                "baseline for AUROC, AUPRC and Spearman, and Δ = baseline − candidate "
                "for MAE; positive values therefore indicate improvement. Feature and "
                "Model are four-seed certification means (seeds 41–44). Data is the "
                "Broad DB best-seed exploratory result and is not at the same evidence "
                "tier as Feature or Model. †For three tasks, the Broad-DB candidate "
                "source failed the admission gate before model fitting; the heatmap "
                "encoded the realised policy effect as zero, whereas Table S1 marks the "
                "unobserved augmentation effect as NA and reports the recorded rejection "
                "reason. This Broad-DB-specific label does not imply that no later "
                "evidence-packet experiment existed. Values are shown to six decimals; "
                "full stored precision and audit fields are retained in the CSV."
            ),
            "",
            f"Complete baseline, candidate, SD and Data-selection fields: [{S1_OUTPUT.name}]({S1_OUTPUT.name})",
            "",
            (
                "Source trace: "
                "[plotted three-axis values](figures/three_axis_ablation_v2/source_data.csv); "
                "[Broad-DB import audit](figures/five_axis_observed_gain_v2/data_axis_import_audit.csv); "
                "[upstream admission decisions](reports/all22_adaptive_agent_data_v1/development/)."
            ),
            "",
            "## Table S2 | Definitions and classifications of the 43 ToxCast tasks (landscape layout)",
            "",
            "| Liability class | Task 1 | Task 2 | Task 3 | Task 4 |",
            "|---|---|---|---|---|",
        ]
    )

    for liability_class in CLASS_ORDER:
        class_rows = [
            row for row in s2_rows if row["liability_class"] == liability_class
        ]
        for offset in range(0, len(class_rows), 4):
            chunk = class_rows[offset : offset + 4]
            class_cell = (
                f"**{md(liability_class)} (n={len(class_rows)})**"
                if offset == 0
                else ""
            )
            task_cells: list[str] = []
            for row in chunk:
                task_id = row["task_id"].removeprefix("toxcast::")
                role = row["evidence_role"]
                task_cells.append(
                    f"**{row['task_no']} [{md(row['curation_confidence'][0])}]** "
                    f"`{md(task_id)}`<br>{md(row['specific_readout'])}<br>"
                    f"*{md(row['biological_level'])}; {md(role)}*"
                )
            task_cells.extend([""] * (4 - len(task_cells)))
            lines.append("| " + " | ".join([class_cell, *task_cells]) + " |")

    lines.extend(
        [
            "",
            (
                "All 43 tasks are binary ToxCast classifications evaluated by MCC. "
                "The landscape matrix is read left to right within each liability "
                "class; all identifiers carry the `toxcast::` namespace. H, M and L "
                "denote high, medium and low curation confidence, respectively. "
                "Thirty-five rows are biologically interpretable readouts; the final "
                "eight assay-interference rows are technical control or "
                "interference-sensitive channels and should not be interpreted as "
                "standalone biological toxicity endpoints."
            ),
            "",
            f"Complete classification notes and metadata fields: [{S2_OUTPUT.name}]({S2_OUTPUT.name})",
            "",
            "## Table S3 | ADMET-EvO versus the strongest fixed pipeline on the 43 tasks",
            "",
            "| No. | Endpoint | ADMET-EvO MCC | Best fixed pipeline | Best fixed MCC | ΔMCC |",
            "|---:|---|---:|---|---:|---:|",
        ]
    )

    for row in s3_rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    row["task_no"],
                    md(row["display_label"]),
                    row["admet_evo_mcc"],
                    md(row["best_fixed_pipeline"]),
                    row["best_fixed_mcc"],
                    row["delta_mcc"],
                ]
            )
            + " |"
        )

    agent_mean = mean(float(row["admet_evo_mcc"]) for row in s3_rows)
    fixed_mean = mean(float(row["best_fixed_mcc"]) for row in s3_rows)
    delta_mean = mean(float(row["delta_mcc"]) for row in s3_rows)
    lines.extend(
        [
            "",
            (
                "MCC is higher-is-better. The fixed comparison set comprised "
                "ExtraTrees-Morgan, ExtraTrees-Rich, LightGBM-Rich, "
                "RandomForest-Rich, XGBoost-Rich and CatBoost-Rich. Across these "
                f"preselected rows, mean MCC was {agent_mean:.6f} for the "
                f"workbook-labelled ADMET-EvO models and {fixed_mean:.6f} for the "
                f"per-task best fixed pipeline (mean paired ΔMCC = {delta_mean:.6f})."
            ),
            "",
            (
                "ADMET-EvO MCC, best-fixed MCC and ΔMCC are independently reported "
                "six-decimal source fields; subtraction of the two displayed MCCs can "
                "therefore differ from the displayed ΔMCC by 0.000001."
            ),
            "",
            (
                "These 43 rows are a post-hoc positive subset of 309 retained ToxCast "
                "endpoints, selected because the workbook-labelled ADMET-EvO point "
                "estimate exceeded all six fixed-pipeline point estimates on the same "
                "endpoint-wise local holdout. The source does not bind repeated-seed "
                "uncertainty, confidence intervals, retained predictions or a public "
                "leaderboard result; the table is therefore descriptive and does not "
                "establish prospective or statistical superiority."
            ),
            "",
            f"Complete ADMET-EvO recipes and six-pipeline MCC matrix: [{S3_OUTPUT.name}]({S3_OUTPUT.name})",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in [
        FIGURE_S1,
        FIGURE_S1_PDF,
        FIGURE_S1_SVG,
        S1_INPUT,
        S1_DETAIL_INPUT,
        S2_INPUT,
        S3_INPUT,
    ]:
        if not path.is_file():
            raise FileNotFoundError(path)

    s1_rows, _ = build_s1()
    s2_rows, task_number = build_s2()
    s3_rows = build_s3(s2_rows, task_number)
    MD_OUTPUT.write_text(render_markdown(s1_rows, s2_rows, s3_rows), encoding="utf-8")

    print(f"Wrote {MD_OUTPUT}")
    print(f"Wrote {S1_OUTPUT} ({len(s1_rows)} rows)")
    print(f"Wrote {S2_OUTPUT} ({len(s2_rows)} rows)")
    print(f"Wrote {S3_OUTPUT} ({len(s3_rows)} rows)")


if __name__ == "__main__":
    main()
