"""Replay the risk-triggered committee policy on the frozen TDC-22 history."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

from admet_agent.committee_activation import ConditionalCommitteeActivationPolicy


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build(
    run_dir: Path,
    correction_path: Path,
    *,
    observed_divergence_tasks: tuple[str, ...] = ("hERG",),
) -> dict:
    summary_paths = sorted(run_dir.glob("tasks/*/search_seed_1.json"))
    if len(summary_paths) != 22:
        raise ValueError(f"Expected 22 frozen summaries, found {len(summary_paths)}")
    correction_payload = json.loads(correction_path.read_text(encoding="utf-8"))
    corrections = {row["task"]: row for row in correction_payload["decisions"]}
    policy = ConditionalCommitteeActivationPolicy()
    rows = []

    for path in summary_paths:
        summary = json.loads(path.read_text(encoding="utf-8"))
        task = summary["profile"]["task"]
        plan = summary["research_plan"]
        correction = corrections[task]
        pre = policy.decide(plan)
        post = policy.decide(plan, correction=correction)
        active_stages = []
        combined_roles = set()
        if pre.activate:
            active_stages.append("pre_action")
            combined_roles.update(pre.roles)
        if post.activate and correction["execution_status"] == "scheduled":
            active_stages.append("post_validation")
            combined_roles.update(post.roles)
        ordered_roles = tuple(
            role
            for role in ("data_steward", "model_researcher", "skeptic")
            if role in combined_roles
        )
        if len(ordered_roles) == 3:
            mode = "full_committee"
        elif ordered_roles:
            mode = "targeted_review"
        else:
            mode = "single_agent"
        rows.append(
            {
                "task": task,
                "selected_action_id": plan["selected_action_id"],
                "outcome_status": summary["action_evidence"]["status"],
                "correction_kind": correction["decision_kind"],
                "correction_execution_status": correction["execution_status"],
                "mode": mode,
                "roles": ordered_roles,
                "active_stages": active_stages,
                "pre_action_decision": pre.to_dict(),
                "post_validation_decision": post.to_dict(),
                "observed_historical_candidate_set_divergence": (
                    task in observed_divergence_tasks
                ),
                "test_loaded": False,
            }
        )

    mode_counts = Counter(row["mode"] for row in rows)
    escalated = [row for row in rows if row["mode"] != "single_agent"]
    scheduled = [
        row for row in rows if row["correction_execution_status"] == "scheduled"
    ]
    divergence = [
        row for row in rows if row["observed_historical_candidate_set_divergence"]
    ]
    covered_scheduled = [row for row in scheduled if row["mode"] != "single_agent"]
    covered_divergence = [row for row in divergence if row["mode"] != "single_agent"]
    conditional_role_assignments = sum(len(row["roles"]) for row in rows)
    default_role_assignments = len(rows) * 3

    return {
        "protocol": "tdc22-conditional-committee-activation-replay-v1",
        "run_dir": str(run_dir),
        "task_count": len(rows),
        "mode_counts": dict(sorted(mode_counts.items())),
        "escalated_task_count": len(escalated),
        "escalated_tasks": [row["task"] for row in escalated],
        "default_single_agent_task_count": mode_counts["single_agent"],
        "avoided_default_committee_invocations": mode_counts["single_agent"],
        "avoided_default_committee_rate": mode_counts["single_agent"] / len(rows),
        "conditional_role_assignments": conditional_role_assignments,
        "default_full_committee_role_assignments": default_role_assignments,
        "avoided_role_assignments": (
            default_role_assignments - conditional_role_assignments
        ),
        "avoided_role_assignment_rate": 1.0
        - conditional_role_assignments / default_role_assignments,
        "scheduled_correction_coverage": {
            "numerator": len(covered_scheduled),
            "denominator": len(scheduled),
            "rate": len(covered_scheduled) / len(scheduled) if scheduled else None,
        },
        "observed_candidate_divergence_coverage": {
            "numerator": len(covered_divergence),
            "denominator": len(divergence),
            "rate": len(covered_divergence) / len(divergence) if divergence else None,
            "source_scope": (
                "v4 first-round planning replay; 21/22 candidate sets identical, "
                "hERG differed"
            ),
        },
        "stage_activation_counts": {
            "pre_action": sum("pre_action" in row["active_stages"] for row in rows),
            "post_validation": sum(
                "post_validation" in row["active_stages"] for row in rows
            ),
        },
        "fit_budget_multiplier": 1.0,
        "new_model_fits": 0,
        "training_started": False,
        "test_loaded": False,
        "claim_boundary": (
            "This replay supports selective process coverage and lower planning "
            "redundancy; it does not establish predictive superiority of multi-agent "
            "planning over the single-agent policy."
        ),
        "tasks": rows,
        "source_hashes": {
            **{str(path): _sha256(path) for path in summary_paths},
            str(correction_path): _sha256(correction_path),
        },
    }


def render(payload: dict) -> str:
    coverage = payload["scheduled_correction_coverage"]
    divergence = payload["observed_candidate_divergence_coverage"]
    lines = [
        "# TDC ADMET 22：条件式多 Agent 激活回放",
        "",
        "## 结论",
        "",
        "生产默认保持单 Agent；只有会改变数据治理、成本风险或验证后纠错的决策，",
        "才调用专门角色。该门控没有启动训练，也没有读取冻结 test。",
        "",
        f"- 默认单 Agent：{payload['default_single_agent_task_count']}/22 tasks",
        f"- 条件升级：{payload['escalated_task_count']}/22 tasks（{', '.join(payload['escalated_tasks'])}）",
        f"- 避免默认委员会调用：{payload['avoided_default_committee_invocations']}/22 "
        f"（{payload['avoided_default_committee_rate']:.1%}）",
        f"- 避免逻辑角色分配：{payload['avoided_role_assignments']}/"
        f"{payload['default_full_committee_role_assignments']} "
        f"（{payload['avoided_role_assignment_rate']:.1%}）",
        f"- 已安排纠错覆盖：{coverage['numerator']}/{coverage['denominator']}",
        f"- 已观察候选分歧覆盖：{divergence['numerator']}/{divergence['denominator']}",
        "- Fit budget multiplier：1.0；新增模型 fits：0",
        "",
        "| Endpoint | 原动作 | 验证结论 | 后续决策 | 运行模式 | 激活角色 |",
        "|---|---|---|---|---|---|",
    ]
    for row in payload["tasks"]:
        if row["mode"] == "single_agent" and row["correction_kind"] == "stop":
            continue
        roles = ", ".join(row["roles"]) or "—"
        lines.append(
            f"| {row['task']} | `{row['selected_action_id']}` | "
            f"{row['outcome_status']} | {row['correction_kind']} | "
            f"{row['mode']} | {roles} |"
        )
    lines.extend(
        [
            "",
            "## 怎么解释“有效”",
            "",
            "- 机制有效：它覆盖了历史上唯一真正改变候选集合的 hERG，也覆盖了全部",
            "  两个已安排的验证后纠错任务。",
            "- 冗余受控：20/22 tasks 不升级；委员会仍经过同一确定性仲裁器和同一",
            "  candidate budget，因此不会把多角色数量直接乘到训练 fits 上。",
            "- 效果优势尚未证明：hERG 的委员会数据方案验证为 inconclusive，随后",
            "  被安全停止。因此当前证据支持多 Agent 作为审计、反证与异常处理层，",
            "  不支持把它宣传成普遍提高预测分数的原因。",
            "",
            "## 正式上线规则",
            "",
            "1. 普通 feature/model action 与无后续价值的 stop：单 Agent。",
            "2. 选中的 data action 或外部数据准入：Data Steward + Skeptic。",
            "3. 高成本表示（包括 Uni-Mol2）或 model 纠错：Model Researcher + Skeptic。",
            "4. 跨轴真实分歧：三个角色进入 full committee；确定性仲裁后仍只进入",
            "   一个共享执行队列。",
            "5. hardness 本身、轻微数据告警、inconclusive 标签本身均不能触发委员会；",
            "   必须存在结构化风险或已安排的可执行纠错。",
            "",
            "## 证据边界",
            "",
            payload["claim_boundary"],
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, default=Path("runs/v3-agent-tdc22-v5"))
    parser.add_argument(
        "--correction-path",
        type=Path,
        default=Path("reports/self_correction_schedule_v1.json"),
    )
    parser.add_argument(
        "--output-prefix",
        type=Path,
        default=Path("reports/conditional_committee_activation_v1"),
    )
    args = parser.parse_args()
    payload = build(args.run_dir, args.correction_path)
    args.output_prefix.parent.mkdir(parents=True, exist_ok=True)
    args.output_prefix.with_suffix(".json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    args.output_prefix.with_suffix(".md").write_text(render(payload), encoding="utf-8")


if __name__ == "__main__":
    main()
