"""Audit whether existing prospective evidence can activate the minimal policy."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from admet_agent.streamlined_task_policy import build_minimal_task_policy


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build(transport_path: Path, confirmation_path: Path) -> dict:
    transport = json.loads(transport_path.read_text(encoding="utf-8"))
    confirmation = json.loads(confirmation_path.read_text(encoding="utf-8"))
    sentinels = []
    for episode in transport["episodes"]:
        if episode["action_id"] not in {
            "feature-domain-shift",
            "model-estimator-family",
        }:
            continue
        # One row per category using the minimal policy's selected axis.
        policy = build_minimal_task_policy(
            episode["task"],
            next(
                row["n_train"]
                for row in json.loads(
                    Path("reports/v3_tdc22_hardness_summary.json").read_text(
                        encoding="utf-8"
                    )
                )["tasks"]
                if row["task"] == episode["task"]
            ),
        )
        if episode["axis"] != policy.scientific_axis:
            continue
        exact_pair = (
            episode["action_id"] == "feature-domain-shift"
            and policy.matched_control == "basic2d_extra_trees"
            and policy.matched_treatment == "rich2d_extra_trees"
        )
        sentinels.append(
            {
                "category": episode["category"],
                "task": episode["task"],
                "minimal_axis": policy.scientific_axis,
                "minimal_pair": [policy.matched_control, policy.matched_treatment],
                "existing_action_id": episode["action_id"],
                "existing_status": episode["status"],
                "existing_mean_oriented_delta": episode["mean_oriented_delta"],
                "axis_covered": True,
                "exact_pair_covered": exact_pair,
                "deployment_action": (
                    "reuse_supported_pair_without_retraining"
                    if exact_pair and episode["status"] == "supported"
                    else "block_same_task_pair_and_use_anchor_or_stop"
                    if exact_pair and episode["status"] == "refuted"
                    else "pair_specific_validation_required"
                ),
                "reason": (
                    "existing prospective matched action is the exact proposed pair"
                    if exact_pair
                    else "existing matched action uses a different model/representation contrast"
                ),
            }
        )
    if len(sentinels) != 5:
        raise ValueError(f"Expected one axis-matched sentinel per ADMET category, found {len(sentinels)}")

    independent_model_rows = [
        {
            "category": row["category"],
            "task": row["task"],
            "status": row["status"],
            "mean_oriented_delta": row["mean_oriented_delta"],
            "positive_delta_rate": row["positive_delta_rate"],
        }
        for row in confirmation["episodes"]
    ]
    missing_exact = sum(not row["exact_pair_covered"] for row in sentinels)
    return {
        "protocol": "minimal-category-size-policy-existing-evidence-audit-v1",
        "category_count": 5,
        "axis_coverage": {
            "numerator": sum(row["axis_covered"] for row in sentinels),
            "denominator": 5,
        },
        "exact_pair_coverage": {
            "numerator": sum(row["exact_pair_covered"] for row in sentinels),
            "denominator": 5,
        },
        "sentinel_axis_evidence": sentinels,
        "independent_model_family_evidence": independent_model_rows,
        "independent_model_outcome_counts": confirmation["outcome_counts"],
        "activation_decision": "blocked_pending_two_exact_pair_pilots",
        "minimum_new_compute": {
            "missing_category_pairs": missing_exact,
            "initial_screen_fits": missing_exact * 2 * 2,
            "maximum_confirmation_fits": missing_exact * 2 * 5,
            "rule": "two configs x seeds 1-2 first; only protocol-qualified pairs continue to five seeds",
            "unimol2_fits": 0,
            "committee_training_multiplier": 1.0,
        },
        "claim_boundary": (
            "Existing prospective evidence covers the selected scientific axis in all five "
            "ADMET categories and three exact proposed pairs. Exact evidence is task-bound: "
            "supported pairs may be reused on that task and refuted pairs must not be rerun. "
            "Axis agreement cannot "
            "substitute for pair-specific validation, especially because independent model-family "
            "outcomes vary by category."
        ),
        "training_started": False,
        "test_loaded": False,
        "source_hashes": {
            str(transport_path): _sha256(transport_path),
            str(confirmation_path): _sha256(confirmation_path),
        },
    }


def render(payload: dict) -> str:
    lines = [
        "# Minimal policy：已有证据覆盖审计",
        "",
        f"- Scientific-axis coverage: {payload['axis_coverage']['numerator']}/5",
        f"- Exact proposed-pair coverage: {payload['exact_pair_coverage']['numerator']}/5",
        f"- Activation: `{payload['activation_decision']}`",
        "- Test loaded: false; training started: false",
        "",
        "| Category | Sentinel | Minimal axis/pair | Existing action | Outcome | Delta | Exact? |",
        "|---|---|---|---|---|---:|---|",
    ]
    for row in payload["sentinel_axis_evidence"]:
        pair = " → ".join(f"`{item}`" for item in row["minimal_pair"])
        lines.append(
            f"| {row['category']} | {row['task']} | {row['minimal_axis']}: {pair} | "
            f"`{row['existing_action_id']}` | {row['existing_status']} | "
            f"{row['existing_mean_oriented_delta']:+.4f} | "
            f"{'yes' if row['exact_pair_covered'] else 'no'} |"
        )
    lines.extend(
        [
            "",
            "## 为什么不能直接宣布精简策略有效",
            "",
            "现有 complete grid 在五大类都测试过相关科学轴，其中 Absorption、Distribution、",
            "Excretion 的 rich-vs-basic 2D ExtraTrees 正好是新 minimal pair。Caco2 与",
            "Clearance_Hepatocyte supported，可在同一任务复用；BBB refuted，必须阻止",
            "同一 pair 重跑并改用 strong anchor 或停止。",
            "",
            "Metabolism 和 Toxicity 的 contrast 仍不同。",
            "例如 model axis 的 RandomForest-vs-ExtraTrees 不能证明 class weighting 或",
            "LightGBM-vs-ExtraTrees pair 有效。把 axis 证据当作具体 pair 证据会再次",
            "造成过度泛化。",
            "",
            "独立 fresh-seed model-family confirmation 又显示明显异质性：Absorption/",
            "Distribution supported，Metabolism/Excretion refuted，Toxicity inconclusive。",
            "因此不能建立一个跨五类通用的 model anchor 或负向剪枝规则。",
            "",
            "## 最小新增计算",
            "",
            f"- 首轮：2 missing categories × 2 configs × 2 seeds = "
            f"{payload['minimum_new_compute']['initial_screen_fits']} fits。",
            f"- 只有通过预注册门槛的 pair 才补齐 5 seeds；两类都晋级时上限 "
            f"{payload['minimum_new_compute']['maximum_confirmation_fits']} fits。",
            "- UniMol2 fits=0；委员会不复制训练，fit multiplier=1.0。",
            "",
            payload["claim_boundary"],
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--transport",
        type=Path,
        default=Path("reports/transport_grid_results_v1.json"),
    )
    parser.add_argument(
        "--confirmation",
        type=Path,
        default=Path("reports/post_memory_confirmation_v1.json"),
    )
    parser.add_argument(
        "--output-prefix",
        type=Path,
        default=Path("reports/minimal_policy_evidence_coverage_v1"),
    )
    args = parser.parse_args()
    payload = build(args.transport, args.confirmation)
    args.output_prefix.with_suffix(".json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    args.output_prefix.with_suffix(".md").write_text(render(payload), encoding="utf-8")


if __name__ == "__main__":
    main()
