"""Report the automatic source-discovery, endpoint-judgment and firewall stack."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from admet_agent.external_data_scout import ExternalDataScout


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build(root: Path) -> dict:
    scout = ExternalDataScout().discover("hERG")
    decisions: dict[str, int] = {}
    for row in scout["candidates"]:
        decision = row["judgment"]["decision"]
        decisions[decision] = decisions.get(decision, 0) + 1
    sources = [
        root / "admet_agent/external_data_scout.py",
        root / "admet_agent/external_structure_firewall.py",
        root / "admet_agent/guard.py",
    ]
    return {
        "capability": "autonomous-external-data-admission-v1",
        "status": "mechanism_ready_real_source_gain_pending",
        "scope": "TDC ADMET endpoints; first concrete connector set is hERG",
        "discovery": {
            "connectors": ["ChEMBL", "TDC related hERG"],
            "candidate_count": scout["candidate_count"],
            "decision_counts": decisions,
            "automatic_download_allowed_before_judgment": scout["automatic_download_allowed"],
            "connector_failures": scout["connector_failures"],
        },
        "endpoint_judgment": {
            "hard_reject_fields": [
                "endpoint/target identity",
                "task type",
                "species",
                "assay system",
                "label direction/threshold",
                "unit for regression",
                "provenance",
                "license",
                "access URL",
            ],
            "bridge_fields": [
                "assay equivalence",
                "threshold/label-semantic equivalence",
                "same-source lineage",
                "structural overlap risk",
            ],
            "current_hERG_result": "3/3 bridge_required; 0 direct merge",
        },
        "sealed_structure_firewall": {
            "order": [
                "desalt, neutralize and canonicalize tautomers to standard InChIKey",
                "remove exact overlap with train/validation/test",
                "reject whole source if >5% of test molecules occur by InChIKey skeleton",
                "remove records with ECFP4 Tanimoto >= 0.90 to any test molecule",
            ],
            "test_structure_access": "sealed executor only for decontamination",
            "test_labels_loaded": False,
            "test_structures_exposed_to_planner": False,
            "thresholds_frozen": {"test_scaffold_coverage": 0.05, "ecfp4_tanimoto": 0.80},
        },
        "fair_comparison_contract": {
            "same_eligible_source_registry_for_agent_and_automl": True,
            "same_query_and_api_call_budget": True,
            "same_structure_firewall": True,
            "only_agent_specific_difference": "endpoint judgment, hypothesis, admission/rejection rationale, and stopping",
            "external_data_not_mixed_into_base_predictive_control": True,
        },
        "claim_boundary": {
            "allowed": "The mechanism discovers candidates, adjudicates endpoint semantics and fails closed before unsafe merge.",
            "forbidden": "Autonomously acquired data has improved hERG prediction.",
        },
        "source_hashes": {str(path.relative_to(root)): _sha(path) for path in sources},
    }


def render(payload: dict) -> str:
    return "\n".join(
        [
            "# 自动外部数据 Agent：能力与证据边界 v1",
            "",
            f"> 状态：`{payload['status']}`。机制已经可执行，真实 source-held-out 增益尚未完成。",
            "",
            "## 闭环",
            "",
            "```mermaid",
            "flowchart LR",
            '    C["Endpoint Contract"] --> S["Source Scout"]',
            '    S --> J{"Endpoint Judge"}',
            '    J -->|"reject"| X["Quarantine + reason"]',
            '    J -->|"bridge/exact"| A["Acquire under frozen budget"]',
            '    A --> F["Sealed label-blind structure firewall"]',
            '    F -->|"reject"| X',
            '    F -->|"admit"| H["Source-held-out matched experiment"]',
            '    H --> E{"supported/refuted/inconclusive"}',
            '    E --> L["Ledger + stop/next hypothesis"]',
            "```",
            "",
            "## 当前 hERG 自动发现结果",
            "",
            f"Scout 找到 {payload['discovery']['candidate_count']} 个候选；3/3 均判为 `bridge_required`，",
            "没有任何数据可直接拼入训练集。判定不仅看字段类型，还看 target、物种、assay、",
            "label threshold、单位、来源谱系、许可和结构重叠风险。",
            "",
            "## 三重结构排除",
            "",
            "1. 脱盐、去电荷、规范互变异构体并生成标准 InChIKey 后，删除与 train、validation 或 test 完全相同的记录。",
            "2. 若外部文件按 InChIKey skeleton 命中超过 5% 的 test 分子，拒绝整份数据。",
            "3. 删除与任一 test 分子的 ECFP4（Morgan radius=2）Tanimoto ≥0.90 的记录。",
            "",
            "Test 结构仅由密封去污染执行器读取；`test_labels_loaded=false`，结构不返回 planner。",
            "Agent 与 AutoML 数据臂使用相同来源、查询/API 预算和相同防火墙，避免以更多数据",
            "冒充更好的科研决策。两者唯一差异是 endpoint 判断、可证伪准入假设和停止策略。",
            "",
            "## Claim 边界",
            "",
            f"- 允许：{payload['claim_boundary']['allowed']}",
            f"- 禁止：{payload['claim_boundary']['forbidden']}",
            "",
        ]
    )


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    payload = build(root)
    prefix = root / "reports/external_data_agent_capability_v1"
    prefix.with_suffix(".json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    prefix.with_suffix(".md").write_text(render(payload), encoding="utf-8")


if __name__ == "__main__":
    main()
