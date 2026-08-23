"""Render the sealed minimal exact-pair pilot without loading test data."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from admet_agent.tasks import resolve_task


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build(run_dir: Path) -> dict:
    protocol_path = run_dir / "protocol.json"
    registry_path = run_dir / "registry.json"
    decision_path = run_dir / "promotion_decision.json"
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    decision = json.loads(decision_path.read_text(encoding="utf-8"))
    episodes = []
    for task in protocol["tasks"]:
        rows = [row for row in registry["records"].values() if row["task"] == task]
        seeds = sorted({int(row["seed"]) for row in rows})
        direction = 1.0 if resolve_task(task).maximize else -1.0
        by_key = {(int(row["seed"]), row["role"]): row for row in rows}
        deltas = [
            direction * (
                float(by_key[(seed, "treatment")]["result"]["validation_score"])
                - float(by_key[(seed, "control")]["result"]["validation_score"])
            )
            for seed in seeds
        ]
        mean_delta = sum(deltas) / len(deltas)
        agreement = sum(delta > 0 for delta in deltas) / len(deltas)
        if len(seeds) == 5:
            if mean_delta >= 0.005 and agreement >= 0.8:
                status = "supported"
            elif mean_delta <= -0.005 and agreement <= 0.2:
                status = "refuted"
            else:
                status = "inconclusive"
        else:
            status = "refuted_at_screen" if mean_delta <= -0.005 and agreement == 0 else "stopped_at_screen"
        episodes.append(
            {
                "task": task,
                "category": protocol["tasks"][task]["category"],
                "axis": protocol["tasks"][task]["axis"],
                "seeds": seeds,
                "oriented_deltas": deltas,
                "mean_oriented_delta": mean_delta,
                "positive_delta_rate": agreement,
                "status": status,
                "screen_promoted": decision["tasks"][task]["promoted"],
                "fits": len(rows),
                "elapsed_seconds": sum(float(row["result"]["elapsed_seconds"]) for row in rows),
            }
        )
    fits = sum(row["fits"] for row in episodes)
    return {
        "protocol": "minimal-category-size-exact-pair-pilot-report-v1",
        "episodes": episodes,
        "actual_fits": fits,
        "maximum_fits": int(protocol["maximum_total_fits"]),
        "fits_avoided_by_gate": int(protocol["maximum_total_fits"]) - fits,
        "unimol2_fits": 0,
        "committee_multiplier": 1.0,
        "production_decision": {
            "CYP2C9_Veith": "keep unweighted default; class weighting only as a fault-triggered correction",
            "LD50_Zhu": "keep LightGBM control; block the refuted ExtraTrees treatment",
        },
        "claim_boundary": "Task-local validation evidence only; no category-wide transport and no test feedback.",
        "test_loaded": False,
        "source_hashes": {
            str(protocol_path): _sha(protocol_path),
            str(registry_path): _sha(registry_path),
            str(decision_path): _sha(decision_path),
        },
    }


def render(payload: dict) -> str:
    lines = [
        "# Minimal exact-pair pilot：简化策略的最后两类验证",
        "",
        f"- 实际拟合：{payload['actual_fits']}/{payload['maximum_fits']}（门控节省 {payload['fits_avoided_by_gate']} fits）",
        "- UniMol2：0 fits；committee multiplier：1.0",
        "- Test loaded：false",
        "",
        "| Task | Category / axis | Seeds | Oriented delta | Positive rate | Outcome | Decision |",
        "|---|---|---:|---:|---:|---|---|",
    ]
    for row in payload["episodes"]:
        lines.append(
            f"| {row['task']} | {row['category']} / {row['axis']} | {len(row['seeds'])} | "
            f"{row['mean_oriented_delta']:+.5f} | {row['positive_delta_rate']:.0%} | "
            f"{row['status']} | {payload['production_decision'][row['task']]} |"
        )
    lines += [
        "",
        "## 结论",
        "",
        "两种子只承担低成本筛选，不承担生产结论。CYP2C9 的初筛通过，但补齐五种子后",
        "效应缩小且方向一致率下降，所以 class weighting 不能成为代谢类默认动作。LD50 在",
        "两个种子都显著变差，因此无需消耗额外六次拟合即可否证 ExtraTrees treatment。",
        "",
        "这两项结果说明 category × size 应当用于缩小候选空间，而不能直接决定赢家；",
        "最终动作仍由同任务、同切分规则下的单轴证据决定。",
        "",
        payload["claim_boundary"],
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    run_dir = Path("runs/minimal-exact-pair-pilot-v1")
    payload = build(run_dir)
    prefix = Path("reports/minimal_exact_pair_pilot_v1")
    prefix.with_suffix(".json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    prefix.with_suffix(".md").write_text(render(payload), encoding="utf-8")


if __name__ == "__main__":
    main()
