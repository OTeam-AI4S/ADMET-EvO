"""Build the frozen 22-task safe-expansion tables and machine-readable report."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

CLASSIC_MODELS = ("random_forest", "linear_svm", "morgan_mlp")
MODEL_LABELS = {
    "random_forest": "RF",
    "linear_svm": "Linear SVM",
    "morgan_mlp": "Morgan MLP",
    "lightgbm": "LightGBM",
}


def _load(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text())


def _agent_candidate(task: str, root: Path, herg: dict[str, Any]) -> dict[str, Any]:
    if task == "hERG":
        summary = herg["model_summaries"]["lightgbm"]
        return {
            "model": "LightGBM",
            "baseline": summary["baseline_mean"],
            "augmented": summary["augmented_mean"],
            "delta": summary["oriented_delta_mean"],
            "positive_fraction": summary["positive_seed_fraction"],
            "seed_count": len(herg["seeds"]),
            "evidence": "independent_confirmation_seeds_4_7",
            "source": "hERG_controlled_bridge_seed4_7.json",
        }
    special = {
        "BBB_Martins": root / "reports/bbb_b3db_data_scaling_pilot_v1.json",
        "DILI": root / "reports/dili_dilirank2_data_scaling_pilot_v1.json",
    }
    path = special.get(
        task,
        root / "reports/chembl37_tdc22_scaling_screen_v1" / f"{task}.json",
    )
    payload = _load(path)
    if "summary" not in payload:
        return {
            "model": "LightGBM",
            "baseline": None,
            "augmented": None,
            "delta": None,
            "positive_fraction": None,
            "seed_count": 0,
            "evidence": f"candidate_not_run_{payload.get('status', 'failed')}",
            "source": str(path.relative_to(root)),
        }
    summary = payload["summary"]
    if "baseline_mean" in summary:
        baseline = summary["baseline_mean"]
        augmented = summary["augmented_mean"]
        delta = summary["oriented_delta_mean"]
        positive = summary["positive_seed_fraction"]
    else:
        metric_key = next(key for key in summary if key.startswith("baseline_"))
        suffix = metric_key.removeprefix("baseline_")
        augmented_key = f"augmented_{suffix}"
        if augmented_key not in summary:
            augmented_key = "firewalled_auroc"
        delta_key = "delta" if "delta" in summary else "firewalled_delta"
        baseline = summary[metric_key]["mean"]
        augmented = summary[augmented_key]["mean"]
        delta = summary[delta_key]["mean"]
        positive = summary[delta_key]["positive_seed_fraction"]
    return {
        "model": "LightGBM",
        "baseline": baseline,
        "augmented": augmented,
        "delta": delta,
        "positive_fraction": positive,
        "seed_count": len(payload.get("per_seed", [])),
        "evidence": "pre_audit_candidate_only",
        "source": str(path.relative_to(root)),
    }


def _rejection_reasons(decision: dict[str, Any]) -> list[str]:
    reasons = list(decision.get("failed_checks", []))
    if decision.get("endpoint_review_status") != "approved_direct":
        reasons.append(f"endpoint_review={decision.get('endpoint_review_status')}")
    if decision.get("lineage_status") != "cleared_independent":
        reasons.append(f"lineage={decision.get('lineage_status')}")
    return reasons


def _fmt(value: Any) -> str:
    return "—" if value is None else f"{float(value):.4f}"


def _transition(before: float | None, after: float | None, delta: float | None) -> str:
    if before is None:
        return "未运行"
    if after is None:
        return f"{_fmt(before)} → 未扩充"
    return f"{_fmt(before)} → {_fmt(after)} ({float(delta):+.4f})"


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def build(args: argparse.Namespace) -> dict[str, Any]:
    root = args.root.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    scorecard = _load(root / "reports/goal_scorecard_v27.json")
    admission = _load(
        root / "reports/classic_ml_external_data_all22_audit_v2/admission_report_v2.json"
    )
    herg = _load(output_dir / "hERG_controlled_bridge_seed4_7.json")
    if herg.get("seeds") != [4, 5, 6, 7]:
        raise RuntimeError("hERG confirmation must use frozen independent seeds 4-7")
    if herg.get("safety", {}).get("tdc_test_loaded") is not False:
        raise RuntimeError("hERG confirmation did not certify test isolation")

    historical = {row["task"]: row for row in scorecard["tasks"]}
    provenance = scorecard["provenance"]
    agent_rows: list[dict[str, Any]] = []
    classic_long_rows: list[dict[str, Any]] = []
    classic_wide_rows: list[dict[str, Any]] = []

    for task_row in scorecard["tasks"]:
        task = task_row["task"]
        source_record = provenance[task]
        source_payload = _load(root / source_record["source"])
        if source_payload.get("freeze_sha256") != source_record["freeze_sha256"]:
            raise RuntimeError(f"Historical freeze mismatch for {task}")
        source_results = list(source_payload.get("results", {}).values())
        if len(source_results) != 1:
            raise RuntimeError(f"Historical source is not single-task for {task}")
        if not (
            math.isclose(float(source_results[0][0]), float(task_row["score"]))
            and math.isclose(float(source_results[0][1]), float(task_row["std"]))
        ):
            raise RuntimeError(f"Historical score mismatch for {task}")
        decision = admission["decisions"][task]
        admitted = decision.get("decision") == "admit_controlled_bridge"
        candidate = _agent_candidate(task, root, herg)
        agent_model_promotion = bool(
            admitted
            and herg["model_summaries"]["lightgbm"]["promotion_passed"]
        )
        cross_model_confirmation = bool(
            admitted and herg["task_robustness"]["robust_across_models"]
        )
        promoted_model_names = (
            herg["task_robustness"]["promoted_models"]
            if cross_model_confirmation
            else []
        )
        promoted_deltas = [
            herg["model_summaries"][name]["oriented_delta_mean"]
            for name in promoted_model_names
            if name in CLASSIC_MODELS
        ]
        agent_rows.append(
            {
                "task": task,
                "metric": task_row["metric"],
                "direction": task_row["direction"],
                "historical_official_best": task_row["score"],
                "historical_std": task_row["std"],
                "historical_recipe": task_row["finalist"],
                "historical_source": provenance[task]["source"],
                "data_decision": decision["decision"],
                "rejection_reasons": ";".join(_rejection_reasons(decision)) if not admitted else "",
                "candidate_model": candidate["model"],
                "candidate_validation_baseline": candidate["baseline"],
                "candidate_validation_augmented": candidate["augmented"],
                "candidate_oriented_delta": candidate["delta"],
                "candidate_positive_seed_fraction": candidate["positive_fraction"],
                "candidate_seed_count": candidate["seed_count"],
                "candidate_evidence": candidate["evidence"],
                "candidate_source": candidate["source"],
                "safe_augmentation_executed": admitted,
                "agent_model_promotion_passed": agent_model_promotion,
                "cross_model_expansion_confirmed": cross_model_confirmation,
                "promoted_classic_models": ";".join(promoted_model_names),
                "safe_confirmed_delta_min": min(promoted_deltas) if promoted_deltas else None,
                "safe_confirmed_delta_max": max(promoted_deltas) if promoted_deltas else None,
                "agent_model_safe_retained_delta": (
                    candidate["delta"] if agent_model_promotion else 0.0
                ),
                "retained_historical_official_best": task_row["score"],
                "official_test_gain_claimed": False,
            }
        )

        baseline_path = output_dir / "baselines" / f"{task}.json"
        baseline_payload = _load(baseline_path) if baseline_path.is_file() else None
        if task != "hERG" and baseline_payload is None:
            raise FileNotFoundError(baseline_path)
        if baseline_payload is not None:
            if len(baseline_payload.get("per_seed_model", [])) != 12:
                raise RuntimeError(f"Incomplete classic baseline result for {task}")
            if baseline_payload.get("safety", {}).get("tdc_test_loaded") is not False:
                raise RuntimeError(f"Classic baseline test isolation missing for {task}")

        old_classic_path = (
            root / "reports/classic_ml_external_data_confirmation_v1" / f"{task}.json"
        )
        old_classic = _load(old_classic_path) if old_classic_path.is_file() else None
        wide = {
            "task": task,
            "metric": task_row["metric"],
            "data_decision": decision["decision"],
        }
        for model in CLASSIC_MODELS:
            if task == "hERG":
                summary = herg["model_summaries"][model]
                baseline_mean = summary["baseline_mean"]
                augmented = summary["augmented_mean"]
                delta = summary["oriented_delta_mean"]
                model_promotion = bool(summary["promotion_passed"])
                evidence = "independent_confirmation_seeds_4_7"
            else:
                summary = baseline_payload["model_summaries"][model]
                baseline_mean = summary["baseline_mean"]
                augmented = None
                delta = None
                model_promotion = False
                evidence = "not_run_quality_gate_rejected"
            preaudit = (
                old_classic.get("model_summaries", {}).get(model)
                if old_classic is not None
                else None
            )
            safe_final = augmented if model_promotion else baseline_mean
            safe_delta = delta if model_promotion else 0.0
            classic_long_rows.append(
                {
                    "task": task,
                    "metric": task_row["metric"],
                    "model": MODEL_LABELS[model],
                    "data_decision": decision["decision"],
                    "validation_baseline": baseline_mean,
                    "validation_augmented": augmented,
                    "oriented_delta": delta,
                    "positive_seed_fraction": summary.get("positive_seed_fraction"),
                    "promotion_passed": model_promotion,
                    "safe_final_validation_score": safe_final,
                    "safe_retained_delta": safe_delta,
                    "evidence": evidence,
                    "preaudit_candidate_augmented": (
                        preaudit.get("augmented_mean") if preaudit else None
                    ),
                    "preaudit_candidate_delta": (
                        preaudit.get("oriented_delta_mean") if preaudit else None
                    ),
                }
            )
            prefix = model
            wide[f"{prefix}_baseline"] = baseline_mean
            wide[f"{prefix}_augmented"] = augmented
            wide[f"{prefix}_oriented_delta"] = delta
            wide[f"{prefix}_safe_final"] = safe_final
            wide[f"{prefix}_safe_retained_delta"] = safe_delta
        classic_wide_rows.append(wide)

    _write_csv(output_dir / "agent_all22_table.csv", agent_rows)
    _write_csv(output_dir / "classic_models_long_table.csv", classic_long_rows)
    _write_csv(output_dir / "classic_models_all22_table.csv", classic_wide_rows)

    promoted_agent = [
        row for row in agent_rows if row["cross_model_expansion_confirmed"]
    ]
    promoted_classic = [row for row in classic_long_rows if row["promotion_passed"]]
    report = {
        "protocol": "classic-ml-all22-safe-expansion-report-v1",
        "task_count": len(agent_rows),
        "historical_official_result_count": len(historical),
        "historical_source_freezes_verified": len(historical),
        "quality_gate": {
            "direct_admitted": admission["admitted_tasks"],
            "controlled_bridge_admitted": admission["controlled_bridge_admitted_tasks"],
            "failed_closed_count": admission["fail_closed_task_count"],
            "model_scores_consulted": admission["model_scores_consulted"],
            "test_labels_loaded": admission["test_labels_loaded"],
        },
        "agent": {
            "candidate_task_count": len(agent_rows),
            "safe_augmentation_task_count": sum(
                row["safe_augmentation_executed"] for row in agent_rows
            ),
            "promoted_task_count": len(promoted_agent),
            "promoted_tasks": [row["task"] for row in promoted_agent],
        },
        "classic_models": {
            "model_count": len(CLASSIC_MODELS),
            "baseline_fit_count": 22 * 4 * len(CLASSIC_MODELS),
            "paired_augmented_fit_count": 4 * len(CLASSIC_MODELS),
            "promoted": [
                {"task": row["task"], "model": row["model"], "delta": row["oriented_delta"]}
                for row in promoted_classic
            ],
        },
        "official_result_policy": (
            "All 22 historical official best results are retained. No official test "
            "score was used for augmentation selection or rerun in this experiment."
        ),
        "statistical_scope": (
            "Promotion uses a predeclared practical delta and seed-consistency rule. "
            "With four confirmation seeds, exact sign-flip tests are low-powered; "
            "no formal significance claim is made."
        ),
        "future_data_budget_policy_not_evaluated_in_this_round": {
            "principle": "use the smallest high-quality external subset that is statistically indistinguishable from the best validation result",
            "candidate_external_to_tdc_ratios": [0.5, 1, 2, 4, 8],
            "selection_rule": "one-standard-error rule on train/validation only",
            "external_weight_rule": "initially cap total external sample weight at total TDC training weight",
            "test_labels_used": False,
        },
        "agent_rows": agent_rows,
        "classic_model_rows": classic_long_rows,
    }
    (output_dir / "experiment_summary.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n"
    )

    agent_lines = [
        "| Task | 指标 | 历史官方最优 | LightGBM候选 B→A | 数据审核 | 跨模型确认 | LightGBM安全保留增益 |",
        "|---|---:|---:|---:|---|---|---:|",
    ]
    for row in agent_rows:
        agent_lines.append(
            "| {task} | {metric} | {official} | {transition} | {decision} | {cross_model} | {delta:+.4f} |".format(
                task=row["task"],
                metric=row["metric"],
                official=_fmt(row["historical_official_best"]),
                transition=_transition(
                    row["candidate_validation_baseline"],
                    row["candidate_validation_augmented"],
                    row["candidate_oriented_delta"],
                ),
                decision=row["data_decision"],
                cross_model=(
                    f"通过 ({row['promoted_classic_models']})"
                    if row["cross_model_expansion_confirmed"]
                    else "未通过/未执行"
                ),
                delta=row["agent_model_safe_retained_delta"],
            )
        )
    classic_lines = [
        "| Task | 指标 | 审核决定 | RF B→安全A | Linear SVM B→安全A | Morgan MLP B→安全A |",
        "|---|---:|---|---:|---:|---:|",
    ]
    by_task = {
        task: {row["model"]: row for row in classic_long_rows if row["task"] == task}
        for task in historical
    }
    for task, task_row in historical.items():
        models = by_task[task]
        cells = []
        for label in ("RF", "Linear SVM", "Morgan MLP"):
            row = models[label]
            cells.append(
                _transition(
                    row["validation_baseline"],
                    row["safe_final_validation_score"],
                    row["safe_retained_delta"],
                )
            )
        classic_lines.append(
            f"| {task} | {task_row['metric']} | {admission['decisions'][task]['decision']} | "
            + " | ".join(cells)
            + " |"
        )
    readme = "\n".join(
        [
            "# 22-task safe data-expansion experiment",
            "",
            "方向统一增益：AUROC/AUPRC/Spearman 为 `after-before`；MAE 为 `before-after`，所以正数恒为提升。",
            "候选验证结果用于展示 agent 曾找到的扩充信号；只有预先冻结的数据审核通过且独立 seeds 确认提升的结果才允许进入‘安全保留增益’。历史官方测试最优全部原样保留，本实验未读取测试标签。",
            "本轮使用的是冻结数据规模，没有追加规模扫描。hERG 外部行约为单折 TDC 训练行的 60 倍，因此这里的结果不能证明海量数据优于适量数据。后续默认只评估 0.5×、1×、2×、4×、8× 的高质量子集，并用 one-standard-error rule 选择最小规模；初始时外部样本总权重不超过 TDC 训练样本总权重。该建议尚未在本轮验证。",
            "",
            "## Agent / LightGBM 候选与安全决策",
            "",
            *agent_lines,
            "",
            "## 经典模型在安全策略下的结果",
            "",
            *classic_lines,
            "",
            "完整逐模型值、未晋级的预审核候选值和来源见 CSV/JSON 文件。",
            "",
        ]
    )
    (output_dir / "README.md").write_text(readme)
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("reports/classic_ml_all22_safe_policy_v1"),
    )
    return parser.parse_args()


if __name__ == "__main__":
    result = build(parse_args())
    print(json.dumps({key: result[key] for key in ("task_count", "agent", "classic_models")}, indent=2))
