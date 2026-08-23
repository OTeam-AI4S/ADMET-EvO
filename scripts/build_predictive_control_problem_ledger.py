"""Convert the frozen predictive-control failures into future, testable hypotheses."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build(source: Path, *, exploratory_threshold: float = 0.02) -> dict:
    report = json.loads(source.read_text(encoding="utf-8"))
    if report.get("test_loaded") is not False:
        raise ValueError("Problem discovery must not load test data")
    losses, gains = [], []
    for row in report["tasks"]:
        delta = row["utilities"]["streamlined_agent"] - row["utilities"]["expanded_automl"]
        record = {
            "task": row["task"],
            "category": row["category"],
            "metric": row["metric"],
            "normalized_delta": delta,
            "agent_selection": row["selected_roles"]["streamlined_agent"],
            "automl_selection": row["selected_roles"]["expanded_automl"],
        }
        if delta <= -exploratory_threshold:
            losses.append(record)
        elif delta >= exploratory_threshold:
            gains.append(record)
    loss_categories = Counter(row["category"] for row in losses)
    automl_winners = Counter(row["automl_selection"] for row in losses)
    hypotheses = [
        {
            "hypothesis_id": "separate-research-pair-from-predictive-anchor",
            "observation": "All material losses selected LightGBM in the AutoML arm while the minimal Agent candidate budget was consumed by its scientific pair.",
            "falsifiable_change": "Give both adaptive arms three equal screen slots: Agent matched pair plus frozen LightGBM anchor; AutoML three scalar candidates.",
            "future_protocol": "Freeze before use; select on seed 13 and evaluate once on seed 14.",
            "success_gate": "Agent-vs-AutoML task-bootstrap lower bound >= -0.01 with no extra logical fit budget for Agent.",
        },
        {
            "hypothesis_id": "large-metabolism-needs-model-anchor-not-only-imbalance-axis",
            "observation": "All three large Veith endpoints favored the AutoML LightGBM selection on fresh evaluation.",
            "falsifiable_change": "Retain imbalance matched evidence as research output but prevent it from displacing the frozen model anchor in finalist selection.",
            "future_protocol": "Evaluate only in a newly frozen large-metabolism cohort; no seed-12 reuse.",
            "success_gate": "No material regression >0.02 on any Veith endpoint and mean oriented delta >= -0.005.",
        },
        {
            "hypothesis_id": "rank-regression-feature-axis-can-miss-estimator-effect",
            "observation": "Clearance_Microsome showed the largest loss when the feature-axis control was selected instead of LightGBM.",
            "falsifiable_change": "Trigger one model-anchor comparison when rank-metric feature evidence is unresolved or selects the basic feature control.",
            "future_protocol": "Matched future-seed comparison on excretion rank endpoints only.",
            "success_gate": "Correction improves mean normalized utility >=0.02 without more than one added configuration.",
        },
    ]
    return {
        "protocol": "predictive-control-autonomous-problem-ledger-v1",
        "source_report": str(source),
        "source_sha256": _sha(source),
        "exploratory_not_confirmatory": True,
        "exploratory_materiality_threshold": exploratory_threshold,
        "material_loss_count": len(losses),
        "material_gain_count": len(gains),
        "material_losses": sorted(losses, key=lambda row: row["normalized_delta"]),
        "material_gains": sorted(gains, key=lambda row: -row["normalized_delta"]),
        "loss_categories": dict(sorted(loss_categories.items())),
        "automl_winner_roles_on_losses": dict(sorted(automl_winners.items())),
        "falsifiable_future_hypotheses": hypotheses,
        "policy_mutated_after_evaluation": False,
        "evaluation_seed_reuse_for_future_confirmation": False,
        "test_loaded": False,
        "claim_boundary": "Post-evaluation diagnosis only. These hypotheses are not validated fixes and may only be tested on a newly frozen future split.",
    }


def render(payload: dict) -> str:
    lines = [
        "# Predictive-control autonomous problem ledger",
        "",
        f"- Material losses: {payload['material_loss_count']}；material gains: {payload['material_gain_count']}",
        "- Policy mutated after evaluation: false",
        "- Evaluation seed reused for confirmation: false",
        "- Test loaded: false",
        "",
        "| Task | Category | Delta | Agent selection | AutoML selection |",
        "|---|---|---:|---|---|",
    ]
    for row in payload["material_losses"]:
        lines.append(
            f"| {row['task']} | {row['category']} | {row['normalized_delta']:+.4f} | "
            f"{row['agent_selection']} | {row['automl_selection']} |"
        )
    lines += ["", "## 可证伪的下一版本假设", ""]
    for index, hypothesis in enumerate(payload["falsifiable_future_hypotheses"], start=1):
        lines += [
            f"### {index}. `{hypothesis['hypothesis_id']}`",
            "",
            f"- Observation: {hypothesis['observation']}",
            f"- Change: {hypothesis['falsifiable_change']}",
            f"- Protocol: {hypothesis['future_protocol']}",
            f"- Gate: {hypothesis['success_gate']}",
            "",
        ]
    lines += [payload["claim_boundary"], ""]
    return "\n".join(lines)


def main() -> None:
    source = Path("reports/minimal_predictive_control_v1.json")
    payload = build(source)
    prefix = Path("reports/predictive_control_problem_ledger_v1")
    prefix.with_suffix(".json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    prefix.with_suffix(".md").write_text(render(payload), encoding="utf-8")


if __name__ == "__main__":
    main()
