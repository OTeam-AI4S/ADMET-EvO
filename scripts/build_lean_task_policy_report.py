"""Build a zero-training replay of the two-lane TDC-22 policy."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

from admet_agent.lean_task_policy import route_task


def build(hardness_path: Path, qualification_path: Path, freeze_path: Path) -> dict:
    hardness = json.loads(hardness_path.read_text(encoding="utf-8"))
    qualification = json.loads(qualification_path.read_text(encoding="utf-8"))
    frozen = json.loads(freeze_path.read_text(encoding="utf-8"))
    if not qualification.get("passed") or qualification.get("tasks_observed") != 22:
        raise ValueError("The lean default requires the sealed 22-task qualification to pass")
    expected_tasks = {row["task"] for row in hardness["tasks"]}
    if set(frozen) != expected_tasks:
        raise ValueError("The frozen finalist mapping must cover exactly the canonical 22 tasks")
    rows = []
    matrix: dict[str, Counter] = defaultdict(Counter)
    for source in sorted(hardness["tasks"], key=lambda item: item["task"]):
        route = route_task(source["task"], int(source["n_train"]))
        row = route.to_dict()
        rows.append(row)
        matrix[route.task_family][route.data_regime] += 1
    default_research_fits = sum(row["maximum_research_fits"] for row in rows)
    return {
        "protocol": "tdc22-two-lane-category-size-policy-v2",
        "status": "default_scheduler_candidate",
        "task_count": len(rows),
        "qualification_seal": qualification["seal_sha256"],
        "frozen_artifact_coverage": len(frozen),
        "frozen_mapping_sha256": hashlib.sha256(freeze_path.read_bytes()).hexdigest(),
        "task_family_counts": dict(sorted(Counter(row["task_family"] for row in rows).items())),
        "data_regime_counts": dict(sorted(Counter(row["data_regime"] for row in rows).items())),
        "family_by_data_regime": {
            family: {regime: matrix[family].get(regime, 0) for regime in ("low_data", "higher_data")}
            for family in (
                "binary_auroc",
                "imbalanced_binary_auprc",
                "continuous_mae",
                "rank_pk_spearman",
            )
        },
        "default_route_counts": dict(sorted(Counter(row["research_action"] for row in rows).items())),
        "default_research_fit_ceiling": default_research_fits,
        "exception_fit_ceiling": {"low_data": 10, "higher_data": 6},
        "removed_from_default_scheduler": [
            "per_task_matched_pair",
            "per_task_strong_anchor",
            "screen_then_five_seed_confirmation",
            "automatic_cross_axis_correction",
            "medium_data_branch",
            "scalar_hardness_routing",
            "committee",
            "hpo",
            "ensemble_search",
            "unimol2",
            "external_data_search",
        ],
        "two_lanes": {
            "production": "contract check -> reuse qualified frozen recipe -> stop",
            "research_exception": "specific validation-side trigger -> one hypothesis -> one matched pair -> retain result -> stop",
        },
        "tasks": rows,
        "training_started": False,
        "test_loaded": False,
        "replay_assumption": "No new validation-side alert was supplied; each task follows the healthy default route.",
    }


def render(payload: dict) -> str:
    labels = {
        "binary_auroc": "普通二分类（AUROC）",
        "imbalanced_binary_auprc": "不平衡二分类（AUPRC）",
        "continuous_mae": "连续值回归（MAE）",
        "rank_pk_spearman": "排序型 PK 回归（Spearman）",
    }
    lines = [
        "# TDC ADMET-22 两线极简策略 v2",
        "",
        "> 核心变化：预测已合格，因此默认不再让每个任务重复做科研实验。生产和科研彻底分线；test 继续冻结。",
        "",
        "## 默认流程",
        "",
        "```mermaid",
        "flowchart LR",
        "    A[\"Task\"] --> B{\"Endpoint/data contract\"}",
        "    B -->|\"失败\"| Q[\"Quarantine / data repair<br/>0 model fits\"]",
        "    B -->|\"通过\"| F{\"Frozen artifact available?\"}",
        "    F -->|\"否\"| R[\"Restore artifact<br/>不重新搜索\"]",
        "    F -->|\"是\"| G{\"明确的 validation-side problem?\"}",
        "    G -->|\"否\"| S[\"Reuse qualified recipe and stop<br/>0 research fits\"]",
        "    G -->|\"数据/漂移\"| D[\"One data-stewardship ticket<br/>先不训练模型\"]",
        "    G -->|\"性能/不确定性\"| E[\"One hypothesis + one matched pair\"]",
        "    E --> M[\"Record supported/refuted/inconclusive<br/>stop; no automatic second axis\"]",
        "```",
        "",
        "当前 22/22 已通过冻结资格门且默认没有新异常，因此：",
        f"**22 个任务全部复用并停止；默认科研拟合上限 = {payload['default_research_fit_ceiling']}。**",
        "这不等于模型推理免费；它表示不再为已经合格的任务追加搜索训练。",
        "",
        "## 四类任务 × 两档数据量",
        "",
        "A/D/M/E/T 只负责 endpoint、单位、物种和 assay contract；真正决定研究问题的是下面四类。",
        "",
        "| 任务家族 | 低数据 <1000 | 较高数据 ≥1000 | 研究重点 |",
        "|---|---:|---:|---|",
    ]
    questions = {
        "binary_auroc": "表示与 scaffold 泛化",
        "imbalanced_binary_auprc": "少数类召回且不牺牲 precision",
        "continuous_mae": "连续误差与异常值稳健性",
        "rank_pk_spearman": "跨 scaffold 排序一致性",
    }
    for family, counts in payload["family_by_data_regime"].items():
        lines.append(f"| {labels[family]} | {counts['low_data']} | {counts['higher_data']} | {questions[family]} |")
    lines.extend([
        "",
        "## 数据量只改变证据强度，不扩张搜索空间",
        "",
        "| 数据档 | 22-task 数量 | 若被触发时的实验 | 最大研究 fits | 默认禁止 |",
        "|---|---:|---|---:|---|",
        "| 低数据 <1000 | 13 | 同一 matched pair × 5 seeds | 10 | HPO、从头训练大模型、按单次分数决策 |",
        "| 较高数据 ≥1000 | 9 | 同一 matched pair × 3 seeds | 6 | 多轮 screen、自动 ensemble、默认 UniMol2 |",
        "",
        "低数据需要更多重复来控制方差，但候选仍只有 treatment/control 两个；较高数据用更少 seeds，避免样本规模导致计算膨胀。原 medium 分支被删除，因为它没有独立动作。",
        "",
        "## 什么情况下才启动研究线",
        "",
        "仅接受四种 validation-side 信号：contract 失败、冻结 artifact 缺失、资格退化/高不确定性、数据漂移或来源缺口。hardness 仍可用于解释和报告，但不再生成多条默认分支。",
        "",
        "研究线一旦启动，也只有一张票：一个可证伪假设、一个单轴 matched pair，然后无论 supported、refuted 或 inconclusive 都停止并写入 memory。若要开第二张票，必须作为新的、预注册的问题，而不是自动 correction。",
        "",
        "## 从默认路径删除的内容",
        "",
        *[f"- `{item}`" for item in payload["removed_from_default_scheduler"]],
        "",
        "多 Agent、外部数据与 UniMol2 并未从项目删除，但不属于任务默认调度：只有数据准入/高风险审计可请求 targeted reviewer；UniMol2 必须有表示失败证据、独立预算与可复用缓存；外部数据必须先有来源、许可和 source-heldout protocol。",
        "",
        "## 22 个任务的默认路由",
        "",
        "| Task | ADMET | 任务家族 | N train | 数据档 | 默认动作 | 研究 fits |",
        "|---|---|---|---:|---|---|---:|",
    ])
    for row in sorted(payload["tasks"], key=lambda item: (item["task_family"], item["n_train"])):
        lines.append(
            f"| {row['task']} | {row['admet_category']} | {labels[row['task_family']]} | {row['n_train']} | "
            f"{row['data_regime']} | `{row['production_action']}` | {row['maximum_research_fits']} |"
        )
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--hardness", type=Path, default=Path("reports/v3_tdc22_hardness_summary.json"))
    parser.add_argument("--qualification", type=Path, default=Path("reports/performance_qualification_v1.json"))
    parser.add_argument("--freeze", type=Path, default=Path("runs/v27-maplight-gnn-freeze/full-22-task.mapping.json"))
    parser.add_argument("--output-prefix", type=Path, default=Path("reports/lean_category_size_policy_v2"))
    args = parser.parse_args()
    payload = build(args.hardness, args.qualification, args.freeze)
    args.output_prefix.with_suffix(".json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    args.output_prefix.with_suffix(".md").write_text(render(payload), encoding="utf-8")


if __name__ == "__main__":
    main()
