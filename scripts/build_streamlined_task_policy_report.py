"""Build a zero-training TDC-22 replay of the minimal category × size policy."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

from admet_agent.streamlined_task_policy import build_minimal_task_policy


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build(hardness_path: Path, redundancy_path: Path) -> dict:
    hardness = json.loads(hardness_path.read_text(encoding="utf-8"))
    redundancy = json.loads(redundancy_path.read_text(encoding="utf-8"))
    rows = []
    matrix: dict[str, Counter] = defaultdict(Counter)
    for row in sorted(hardness["tasks"], key=lambda item: item["task"]):
        policy = build_minimal_task_policy(row["task"], int(row["n_train"]))
        payload = policy.to_dict()
        rows.append(payload)
        matrix[policy.category][policy.size_bucket] += 1

    old_primary = sum(row["primary_actions"] for row in redundancy["tasks"])
    old_fits = sum(row["total_recorded_fits"] for row in redundancy["tasks"])
    new_primary = sum(row["maximum_screen_configs"] for row in rows)
    typical_fits = sum(row["typical_fit_ceiling"] for row in rows)
    all_corrected_fits = sum(row["corrected_fit_ceiling"] for row in rows)
    observed_scheduled_corrections = 2
    correction_adjusted_fits = typical_fits + observed_scheduled_corrections * 6
    return {
        "protocol": "tdc22-category-size-minimal-policy-shadow-replay-v1",
        "activation_status": "shadow_only_pending_validation_noninferiority",
        "task_count": len(rows),
        "size_counts": dict(sorted(Counter(row["size_bucket"] for row in rows).items())),
        "category_size_matrix": {
            category: {
                bucket: matrix[category].get(bucket, 0)
                for bucket in ("small", "medium", "large")
            }
            for category in ("absorption", "distribution", "metabolism", "excretion", "toxicity")
        },
        "old_v5": {
            "primary_configurations": old_primary,
            "recorded_fits": old_fits,
        },
        "minimal_policy": {
            "screen_configurations": new_primary,
            "typical_fit_ceiling": typical_fits,
            "all_tasks_use_one_correction_fit_ceiling": all_corrected_fits,
            "historical_two_correction_adjusted_fit_ceiling": correction_adjusted_fits,
            "primary_configuration_reduction_rate": 1 - new_primary / old_primary,
            "typical_fit_reduction_rate": 1 - typical_fits / old_fits,
            "all_corrected_fit_reduction_rate": 1 - all_corrected_fits / old_fits,
            "historical_correction_adjusted_reduction_rate": 1
            - correction_adjusted_fits / old_fits,
        },
        "removed_from_default_path": [
            "full_committee",
            "all_axis_exhaustive_action_generation",
            "second_four_slot_search_round",
            "more_than_one_hyperparameter_refinement",
            "automatic_ensemble_search",
            "automatic_unimol2",
            "automatic_external_data_search",
            "more_than_one_corrective_followup",
        ],
        "retained_in_default_path": [
            "endpoint_contract",
            "one_falsifiable_hypothesis",
            "one_single_axis_matched_pair",
            "at_most_one_strong_anchor",
            "paired_seed_confirmation",
            "negative_result_retention",
            "bounded_stop_or_one_correction",
            "test_firewall_and_hash_audit",
        ],
        "activation_gate": {
            "representative_categories_required": 5,
            "maximum_allowed_oriented_regression": 0.005,
            "matched_pair_must_complete": True,
            "unsafe_data_actions_allowed": 0,
            "test_feedback_allowed": 0,
            "note": "Fit counts are ceilings, not measured savings; activate only after validation-only noninferiority pilot.",
        },
        "tasks": rows,
        "training_started": False,
        "test_loaded": False,
        "source_hashes": {
            str(hardness_path): _sha256(hardness_path),
            str(redundancy_path): _sha256(redundancy_path),
        },
    }


def render(payload: dict) -> str:
    m = payload["minimal_policy"]
    lines = [
        "# TDC ADMET-22 极简 category × data-size 策略（shadow replay）",
        "",
        "> 状态：只读 shadow policy，尚未替换生产策略。必须先通过 validation-only",
        "> 非劣 pilot；本报告没有训练模型，也没有读取 test。",
        "",
        "## 一眼结论",
        "",
        f"- 数据量：small 13/22，medium 3/22，large 6/22。",
        "- 每任务默认只做：endpoint contract → 1 个假设/1 个 matched pair + 最多 1 个",
        "  strong anchor → 5-seed 确认 → stop 或最多 1 次纠错。",
        f"- 旧 v5：{payload['old_v5']['primary_configurations']} primary configs、"
        f"{payload['old_v5']['recorded_fits']} recorded fits。",
        f"- 新 shadow ceiling：{m['screen_configurations']} screen configs、"
        f"typical {m['typical_fit_ceiling']} fits；按历史仅 2 个纠错为 "
        f"{m['historical_two_correction_adjusted_fit_ceiling']} fits。",
        f"- 对应账面缩减：primary configs {m['primary_configuration_reduction_rate']:.1%}；"
        f"typical fits {m['typical_fit_reduction_rate']:.1%}；历史纠错调整后 "
        f"{m['historical_correction_adjusted_reduction_rate']:.1%}。",
        "",
        "## 五大类 × 数据量",
        "",
        "| ADMET 类别 | Small <1000 | Medium 1000–4999 | Large ≥5000 |",
        "|---|---:|---:|---:|",
    ]
    labels = {
        "absorption": "Absorption",
        "distribution": "Distribution",
        "metabolism": "Metabolism",
        "excretion": "Excretion",
        "toxicity": "Toxicity",
    }
    for category, values in payload["category_size_matrix"].items():
        lines.append(
            f"| {labels[category]} | {values['small']} | {values['medium']} | {values['large']} |"
        )
    lines.extend(
        [
            "",
            "## 三阶段默认路径",
            "",
            "1. **Contract（0 fits）**：只运行该 ADMET 类别的三项 endpoint/data checks；",
            "   失败则 quarantine/修复，不进入模型搜索。",
            "2. **Minimal experiment**：一个单轴 treatment/control，加最多一个 strong",
            "   anchor。small/medium 用 seeds 1–2 筛选，large 只用 seed 1；正式 pair",
            "   补齐至 5 seeds。",
            "3. **Decision**：supported 且性能已合格就 stop；refuted/inconclusive 只允许",
            "   一次跨轴纠错或证据修复，再停止。",
            "",
            "## 数据量差异",
            "",
            "| 数据桶 | 默认策略 | 明确禁止 |",
            "|---|---|---|",
            "| Small（13 tasks） | 不做 HPO；优先固定预训练表示/低方差模型；2-seed screen + 5-seed pair | 从头训练大模型、广泛调参 |",
            "| Medium（3 tasks） | 最多 1 个局部 refinement；允许一个可复现 anchor | 多模型多轮搜索 |",
            "| Large（6 tasks） | 1-seed 低成本 screen；rich-2D boosting/tree 为主；pair 仍补齐 5 seeds | 默认 pretrained/UniMol2、重复表示提取 |",
            "",
            "## 从默认路径删除",
            "",
            *[f"- `{item}`" for item in payload["removed_from_default_path"]],
            "",
            "委员会、外部数据、UniMol2、ensemble 和跨任务 memory 没有被删除；它们变成",
            "exception handlers，只有结构化风险或前瞻证据触发，不能每个 task 都走一遍。",
            "",
            "## 激活边界",
            "",
            "这些 fit 数是协议上限的历史回放，不是已实现的时间节省，也没有证明新策略",
            "保持预测效果。正式替换前必须在 A/D/M/E/T 各一个代表任务做 validation-only",
            "pilot，要求 oriented regression 不超过 0.005、matched pair 完整、test feedback=0。",
            "",
            "## 22-task 任务策略",
            "",
            "| Task | 类别 | N train | Size | Axis | Matched pair | Optional anchor | Typical fits |",
            "|---|---|---:|---|---|---|---|---:|",
        ]
    )
    for row in sorted(payload["tasks"], key=lambda item: (item["category"], item["n_train"])):
        lines.append(
            f"| {row['task']} | {row['category']} | {row['n_train']} | {row['size_bucket']} | "
            f"{row['scientific_axis']} | `{row['matched_control']}` → "
            f"`{row['matched_treatment']}` | `{row['optional_strong_anchor'] or 'none'}` | "
            f"{row['typical_fit_ceiling']} |"
        )
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--hardness",
        type=Path,
        default=Path("reports/v3_tdc22_hardness_summary.json"),
    )
    parser.add_argument(
        "--redundancy",
        type=Path,
        default=Path("reports/v4_agent_v5_redundancy_replay.json"),
    )
    parser.add_argument(
        "--output-prefix",
        type=Path,
        default=Path("reports/streamlined_category_size_policy_v1"),
    )
    args = parser.parse_args()
    payload = build(args.hardness, args.redundancy)
    args.output_prefix.with_suffix(".json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    args.output_prefix.with_suffix(".md").write_text(render(payload), encoding="utf-8")


if __name__ == "__main__":
    main()
