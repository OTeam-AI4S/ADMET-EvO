"""Build the claim-bounded TDC-22 AI Scientist evidence scorecard."""

from __future__ import annotations

import json
from pathlib import Path

from admet_agent.ai_scientist_scorecard import build_evidence_scorecard


def render(payload: dict) -> str:
    v = payload["evidence_vector"]
    gates = payload["claim_gates"]
    lines = [
        "# TDC ADMET-22：AI Scientist 证据向量评分卡",
        "",
        f"- Overall status: `{payload['overall_status']}`",
        "- Scorecard build loaded TDC test: false",
        f"- Disclosed post-qualification test-access incidents: {v['audit_and_test_firewall']['post_qualification_test_access_incidents']}",
        f"- 原则：{payload['why_no_single_average']}",
        "",
        "## 一眼结论",
        "",
        "| 能力 | Agent | 对照/边界 | 证据状态 |",
        "|---|---|---|---|",
        f"| 预测合格 | {v['prediction_qualification']['top5_tasks']}/22 Top-5；{v['prediction_qualification']['top10_tasks']}/22 Top-10 | 只作应用资格，不宣称 AutoML 优越 | certified |",
        f"| 22-task 同预算预测对照 | Agent−expanded AutoML {v['same_budget_predictive_control']['agent_minus_expanded_automl_mean']:+.4f} | 95% CI [{v['same_budget_predictive_control']['agent_minus_expanded_automl_bootstrap_95_ci'][0]:+.4f}, {v['same_budget_predictive_control']['agent_minus_expanded_automl_bootstrap_95_ci'][1]:+.4f}]；W/T/L={v['same_budget_predictive_control']['agent_vs_expanded_automl_wins_ties_losses']} | complete; noninferiority FAIL |",
        f"| 复合研究决策 | macro {v['integrated_research_decision']['agent_macro']:.2f}；unsafe={v['integrated_research_decision']['agent_unsafe_actions']} | rules {v['integrated_research_decision']['rules_macro']:.2f}；expanded AutoML {v['integrated_research_decision']['expanded_automl_macro']:.2f} | local sealed |",
        f"| Agent−rules | {v['integrated_research_decision']['agent_minus_rules_case_score']:+.2f} | bootstrap 95% CI [{v['integrated_research_decision']['agent_minus_rules_bootstrap_95_ci'][0]:.2f}, {v['integrated_research_decision']['agent_minus_rules_bootstrap_95_ci'][1]:.2f}] | local positive |",
        f"| 数据治理 | accuracy {v['data_stewardship']['agent_decision_accuracy']:.2f}；unsafe merge={v['data_stewardship']['agent_unsafe_merges']} | expanded AutoML accuracy {v['data_stewardship']['expanded_automl_decision_accuracy']:.2f}；unsafe={v['data_stewardship']['expanded_automl_unsafe_merges']}；complete rules 与 Agent 持平 | AutoML advantage only |",
        f"| 真实外部数据票据 | frozen attempts={v['real_external_data_ticket']['frozen_protocol_attempts']}；fits={v['real_external_data_ticket']['fits']}；unsafe merge=0 | ChEMBL API HTTP 500，未接受部分数据 | failed closed / gain pending |",
        f"| 自动数据准入栈 | 发现 {v['autonomous_data_admission_stack']['candidate_sources_discovered']} 个候选；bridge={v['autonomous_data_admission_stack']['bridge_required']}；direct merge={v['autonomous_data_admission_stack']['direct_merges_allowed']} | endpoint judge + 密封三重结构防火墙；真实增益未证明 | mechanism ready |",
        f"| 图表征门控 | 2D/GIN/gate ready={v['bounded_graph_fusion']['two_d_ready']}/{v['bounded_graph_fusion']['gin_ready']}/{v['bounded_graph_fusion']['oof_decision_tree_gate_ready']} | distinct MPNN ready={v['bounded_graph_fusion']['distinct_mpnn_ready']}；票据上限 {v['bounded_graph_fusion']['maximum_ticket_fits']} fits | blocked before training |",
        f"| 反馈纠错 | macro {v['feedback_self_correction']['agent_macro']:.2f} | expanded AutoML {v['feedback_self_correction']['expanded_automl_macro']:.2f}；complete rules 持平 | 2 observed cases + composite |",
        f"| 成本停止 | pilot {v['cost_aware_stopping']['minimal_pilot_actual_fits']}/{v['cost_aware_stopping']['minimal_pilot_maximum_fits']} fits，节省 {v['cost_aware_stopping']['minimal_pilot_saving_fraction']:.0%} | composite 估算节省 {v['cost_aware_stopping']['composite_estimated_saving_fraction']:.1%} | pilot measured / composite estimated |",
        f"| 跨任务学习 | {v['cross_task_learning']['completed_fits']} fits，{len(v['cross_task_learning']['routing_eligible_actions'])} actions eligible | routing outcome advantage 尚未证明 | causal priors ready |",
        f"| no-memory 消融 | selection changed {v['key_process_ablations']['no_memory']['selection_changed_tasks']}/22 | legacy mean gain {v['key_process_ablations']['no_memory']['legacy_full_minus_winner_mean_oriented_gain']:+.5f} | negative result retained |",
        f"| no-hardness 消融 | critical diagnosis recall {v['key_process_ablations']['no_hardness']['ablation_micro_recall']:.2f} | full Agent {v['key_process_ablations']['no_hardness']['full_agent_micro_recall']:.2f} | mechanism ablation |",
        f"| no-data 消融 | accuracy {v['key_process_ablations']['no_data']['ablation_decision_accuracy']:.2f}；unsafe=0 | full Agent {v['key_process_ablations']['no_data']['full_agent_decision_accuracy']:.2f} | safe but over-conservative |",
        "",
        "## Claim gates",
        "",
        "| Gate | Status |",
        "|---|---|",
    ]
    for name, passed in gates.items():
        lines.append(f"| `{name}` | {'PASS' if passed else 'PENDING'} |")
    lines += [
        "",
        "## 当前允许的技术叙事",
        "",
        payload["allowed_claim"],
        "",
        "## 仍然禁止的表述",
        "",
        *[f"- {claim}" for claim in payload["forbidden_claims"]],
        "",
        "## 最小剩余证据",
        "",
        *[f"- {item}" for item in payload["remaining_blockers"]],
        "",
        "## 为什么正式对照分成两条 track",
        "",
        "- 22-task predictive track：streamlined Agent、expanded AutoML、strong-fixed。",
        "- same-action-budget process track：rules、random、no-memory、no-hardness、no-data。",
        "",
        payload["control_scope"]["reason_for_two_tracks"],
        "",
        "这里不计算单一总分。预测、安全、科研决策和外部数据是不可互相补偿的门控：",
        "例如更高 process score 不能抵消预测不合格，更多 Top-5 也不能抵消 unsafe merge。",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    payload = build_evidence_scorecard(Path("reports"))
    prefix = Path("reports/ai_scientist_evidence_scorecard_v1")
    prefix.with_suffix(".json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    prefix.with_suffix(".md").write_text(render(payload), encoding="utf-8")


if __name__ == "__main__":
    main()
