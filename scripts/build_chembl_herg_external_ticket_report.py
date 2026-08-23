"""Build the audit report for the real ChEMBL hERG acquisition ticket."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build(root: Path) -> dict:
    paths = {
        "v1_protocol": root / "runs/chembl-herg-source-heldout-v1/protocol.json",
        "v1_failure": root / "runs/chembl-herg-source-heldout-v1/acquisition_failure.json",
        "v2_protocol": root / "runs/chembl-herg-source-heldout-v2/protocol.json",
        "v2_failure": root / "runs/chembl-herg-source-heldout-v2/acquisition_failure.json",
        "incident": root / "reports/test_firewall_incident_2026-08-20.json",
    }
    payloads = {name: _load(path) for name, path in paths.items()}
    if payloads["v1_failure"]["raw_artifact_created"] or payloads["v2_failure"]["raw_artifact_created"]:
        raise ValueError("Failure report cannot claim a raw artifact was accepted")
    return {
        "ticket": "real-external-data-chembl-herg-v1",
        "status": "failed_closed_external_service_unavailable",
        "endpoint_alignment": payloads["v2_protocol"]["endpoint"],
        "source": payloads["v2_protocol"]["source"],
        "frozen_protocol_attempts": 2,
        "acquisition_failures": [payloads["v1_failure"], payloads["v2_failure"]],
        "raw_artifacts_accepted": 0,
        "curation_runs": 0,
        "model_fits": 0,
        "unsafe_merges": 0,
        "external_gain_demonstrated": False,
        "self_correction": {
            "observed": True,
            "parent_failure_retained": True,
            "corrective_hypothesis": payloads["v1_failure"]["corrective_hypothesis"],
            "correction_result": "refuted_as_sufficient; API failed even on minimal control requests",
            "automatic_third_retry_allowed": False,
        },
        "firewall": {
            "incident": payloads["incident"]["incident_id"],
            "test_accessed_in_session": True,
            "test_feedback_used": False,
            "frozen_recipe_mutated": False,
            "strict_no_test_access_claim_allowed": False,
        },
        "resume_gate": "A documented ChEMBL service recovery or a separately frozen authoritative source is required; no default-scheduler retry.",
        "claim_boundary": {
            "allowed": "The agent independently specified provenance, license, endpoint alignment, source holdout and admission gates, then failed closed on an unavailable real service and retained the failed correction.",
            "forbidden": "Autonomously acquired external data improved prediction.",
        },
        "source_hashes": {str(path.relative_to(root)): _sha(path) for path in paths.values()},
    }


def render(payload: dict) -> str:
    return "\n".join(
        [
            "# ChEMBL hERG 真实外部数据票据：审计结果",
            "",
            f"> 状态：`{payload['status']}`。没有把失败包装成数据增益。",
            "",
            "## 一眼结论",
            "",
            "- 数据源：ChEMBL 官方 current data API，目标 `CHEMBL240`（human KCNH2/hERG）。",
            "- 许可：CC BY-SA 3.0；协议包含 attribution 与派生数据同许可要求。",
            "- endpoint：TDC hERG 是 heterogeneous blocker label；外部票据预注册为",
            "  `IC50 < 10,000 nM` 的 auxiliary hERG liability，不宣称标签完全等价。",
            f"- 冻结协议尝试：{payload['frozen_protocol_attempts']}；接受 raw artifacts：0；",
            f"  curation runs：{payload['curation_runs']}；model fits：{payload['model_fits']}；",
            f"  unsafe merges：{payload['unsafe_merges']}。",
            "- 结论：外部数据预测增益仍为 **未证明**。",
            "",
            "## 自主纠错链",
            "",
            "1. v1 在分页中遇到持续 HTTP 500，重试耗尽后 failed closed；没有保存半截数据。",
            "2. Agent 保留 parent failure，提出“减少字段和分页宽度可规避服务端失败”的假设。",
            "3. 在修改获取代码后冻结 v2；科学过滤、source split、模型与 gain gate 均未改变。",
            "4. v2 在第一页仍失败；5-row control request 也返回 HTTP 500，因此该修正不足，",
            "   作为 refuted operational hypothesis 保留。",
            "5. 按成本停止规则，不进行自动第三次重试，也不换一个更容易产生正结果的来源。",
            "",
            "## 已冻结但尚未执行的数据协议",
            "",
            "- 仅接受 human CHEMBL240、IC50 nM、relation `=`、standard flag、无 validity flag、",
            "  非 duplicate、functional/toxicity assay、有效 canonical SMILES。",
            "- 按 `document_chembl_id` 哈希划分 80/20 source-heldout；跨 split molecule 全部丢弃。",
            "- 去除所有与 TDC train 重叠的结构和 split 内冲突标签。",
            "- control 为 TDC train_val only；treatment 唯一变化是加入 external source-train。",
            "- 5 seeds、10 fits；gain gate 为 mean AUROC delta ≥0.01 且 paired bootstrap LCB>0。",
            "- 永不修改当前冻结 TDC recipe，也不读取 TDC test。",
            "",
            "## Test-firewall 边界",
            "",
            "本票据开始阶段发生过已披露的 test-access incident。后续脚本没有使用 test，且",
            "`test_feedback_used=false`，但当前 goal session 不能再声称 strict no-test-access。",
            "路径级 guard 已加入，事件永久保留，详见",
            "`reports/test_firewall_incident_2026-08-20.md`。",
            "",
            "## 恢复条件",
            "",
            payload["resume_gate"],
            "",
            "## 官方依据",
            "",
            "- [TDC Toxicity / hERG task](https://tdcommons.ai/single_pred_tasks/tox/)",
            "- [ChEMBL Data Web Services](https://chembl.gitbook.io/chembl-interface-documentation/web-services/chembl-data-web-services)",
            "- [ChEMBL data licensing](https://chembl.gitbook.io/chembl-interface-documentation/about)",
            "",
        ]
    )


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    payload = build(root)
    prefix = root / "reports/chembl_herg_external_ticket_v1"
    prefix.with_suffix(".json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    prefix.with_suffix(".md").write_text(render(payload), encoding="utf-8")


if __name__ == "__main__":
    main()
