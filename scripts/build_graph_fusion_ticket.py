"""Freeze the bounded GNN/MPNN + gated-tree representation ticket."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


def _sha(path: Path) -> str | None:
    if not path.is_file():
        return None
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build(root: Path) -> dict:
    gin_python = root / ".venvs/maplight-gnn/bin/python"
    gin_source = root / "vendor/tdc-admet-bench-audit/models/maplight_gnn/upstream/maplight_gnn.py"
    gate_source = root / "admet_agent/graph_gated_tree.py"
    # A distinct MPNN provider is deliberately explicit.  GIN cannot be
    # relabelled as MPNN, and a heavyweight environment is never installed by
    # this report builder.
    mpnn_cli = root / ".venvs/chemprop/bin/chemprop"
    mpnn_model = root / "models/chemprop_mpnn/model.pt"
    readiness = {
        "two_d": True,
        "gin": gin_python.exists() and gin_source.is_file(),
        "mpnn": mpnn_cli.exists() and mpnn_model.is_file(),
        "oof_gate": gate_source.is_file(),
    }
    missing = [name for name, ready in readiness.items() if not ready]
    return {
        "ticket": "bounded-graph-representation-fusion-v1",
        "status": "ready" if not missing else "blocked_missing_distinct_mpnn_provider",
        "purpose": "Test representation complementarity, not reopen leaderboard search.",
        "scope": {
            "task": "hERG",
            "selection_data": "train/validation only",
            "test_labels_loaded": False,
            "test_used_for_selection": False,
            "frozen_production_recipe_mutated": False,
        },
        "readiness": readiness,
        "missing_components": missing,
        "representation_contract": {
            "two_d": "existing hashed molecular descriptors/fingerprints",
            "gin": "pinned MapLight supervised-masking GIN embedding; existing cache reused",
            "mpnn": "independently versioned Chemprop message-passing fingerprint; no surrogate substitution",
            "manifest_required": [
                "canonical_smiles_sha256",
                "model_sha256",
                "software_version",
                "feature_shape",
                "feature_sha256",
            ],
        },
        "experiment": {
            "stage_a_representation": {
                "control": "2D only + ExtraTrees",
                "treatment": "2D + GIN + MPNN static concatenation + same ExtraTrees",
                "single_changed_axis": "representation",
                "paired_seeds": [1, 2, 3, 4, 5],
                "maximum_fits": 10,
            },
            "stage_b_fusion": {
                "run_only_if": "stage A paired mean delta >= 0.005 and no material calibration regression",
                "control": "static concatenation",
                "treatment": "OOF-safe shallow decision-tree gate over 2D/GIN/MPNN experts",
                "single_changed_axis": "fusion",
                "paired_seeds": [1, 2, 3, 4, 5],
                "maximum_fits": 10,
            },
            "promotion_gate": {
                "primary": "paired validation AUROC delta",
                "minimum_effect": 0.005,
                "uncertainty": "paired bootstrap 95% CI",
                "safety": [
                    "Brier score must not worsen by more than 0.01",
                    "seed standard deviation must not increase by more than 50%",
                    "wall-time and peak-disk deltas must be reported",
                ],
            },
            "hard_stop": "At most 20 fits total; no HPO, UniMol2, committee, ensemble expansion, or test feedback.",
        },
        "current_action": (
            "Do not train. Supply one content-addressed MPNN model/provider, then regenerate this ticket."
            if missing
            else "Run stage A only."
        ),
        "resource_policy": {
            "reuse_existing_gin_environment": True,
            "install_new_environment_automatically": False,
            "duplicate_embedding_cache_allowed": False,
        },
        "source_hashes": {
            str(path.relative_to(root)): _sha(path)
            for path in (gin_source, gate_source, mpnn_model)
        },
    }


def render(payload: dict) -> str:
    ready = payload["readiness"]
    stage_a = payload["experiment"]["stage_a_representation"]
    stage_b = payload["experiment"]["stage_b_fusion"]
    return "\n".join(
        [
            "# 有界图表征融合票据 v1",
            "",
            f"> 状态：`{payload['status']}`。当前动作：{payload['current_action']}",
            "",
            "## 不是新一轮刷榜",
            "",
            "该票据只回答一个科研问题：2D、GIN 与独立 MPNN 是否提供互补信息，以及",
            "OOF-safe 的浅层决策树门控是否优于静态拼接。生产 recipe 和 test 均不参与选择。",
            "",
            "## 运行时就绪度",
            "",
            f"- 2D：{ready['two_d']}；GIN：{ready['gin']}；独立 MPNN：{ready['mpnn']}；OOF gate：{ready['oof_gate']}。",
            "- GIN 使用已有 `.venvs/maplight-gnn` 和内容寻址缓存；禁止复制同一 embedding cache。",
            "- MPNN 必须有独立模型哈希与 Chemprop 运行时，不能把现有 GIN 改名替代。",
            "- 当前不会自动安装重型环境，也不会因为缺组件退化为不等价实验。",
            "",
            "## 两阶段单轴实验",
            "",
            f"1. Stage A：`{stage_a['control']}` 对 `{stage_a['treatment']}`；5 paired seeds，最多 10 fits。",
            f"2. 仅当 Stage A 的 paired mean delta ≥0.005 且校准无实质退化，才运行 Stage B。",
            f"3. Stage B：`{stage_b['control']}` 对 `{stage_b['treatment']}`；最多 10 fits。",
            "4. 总硬上限 20 fits；禁止 HPO、UniMol2、委员会、扩张集成和 test feedback。",
            "",
            "## 为什么采用 OOF gate",
            "",
            "Gate 的训练输入来自训练集的 out-of-fold expert predictions；随后各 expert 才在完整训练集",
            "重拟合。这样决策树不能用 in-sample 置信度学习一个虚假的专家选择规则，也不读取 test 标签。",
            "",
            "## 晋级标准",
            "",
            "主指标是 validation AUROC 的 paired delta，最小有意义效应为 0.005，并报告 paired",
            "bootstrap 95% CI。Brier 不得恶化超过 0.01，seed 标准差不得增加超过 50%，同时报告",
            "wall-time 与 peak-disk 增量。未过门即保留负结果并停止。",
            "",
        ]
    )


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    payload = build(root)
    prefix = root / "reports/bounded_graph_fusion_ticket_v1"
    prefix.with_suffix(".json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    prefix.with_suffix(".md").write_text(render(payload), encoding="utf-8")


if __name__ == "__main__":
    main()
