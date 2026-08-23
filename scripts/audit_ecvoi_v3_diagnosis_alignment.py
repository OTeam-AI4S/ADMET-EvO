"""Audit whether EC-VOI v3 routing uses its endpoint diagnosis.

This is a development-only mechanism diagnostic.  It reads frozen discovery
artifacts and never loads certification/test/leaderboard data.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_protocol(path: Path) -> dict[str, Any]:
    protocol = _load(path)
    claimed = protocol.get("sha256")
    body = {key: value for key, value in protocol.items() if key != "sha256"}
    actual = hashlib.sha256(json.dumps(body, sort_keys=True).encode()).hexdigest()
    if claimed != actual:
        raise RuntimeError("EC-VOI v3 protocol SHA-256 is invalid")
    if protocol.get("cohort") != "development":
        raise RuntimeError("This diagnostic is restricted to the development cohort")
    return protocol


def audit(protocol: dict[str, Any], run_dir: Path) -> dict[str, Any]:
    tasks: list[dict[str, Any]] = []
    forbidden_accesses: list[str] = []
    for task in protocol["tasks"]:
        summary = _load(
            run_dir / "tasks" / task / f"search_seed_{protocol['discovery_seeds'][0]}.json"
        )
        if summary.get("benchmark_protocol_sha256") != protocol["sha256"]:
            raise RuntimeError(f"Protocol mismatch for {task}")
        if summary.get("source_sha256") != protocol["implementation_source_sha256"]:
            raise RuntimeError(f"Source mismatch for {task}")
        if summary.get("test_loaded") is not False:
            forbidden_accesses.append(f"{task}/test")
        if summary.get("leaderboard_loaded") is not False:
            forbidden_accesses.append(f"{task}/leaderboard")

        decisions = [
            item["decision"]
            for item in summary["ecvoi_v2_episode"]["ledger"]
            if item.get("event_type") == "decision_emitted"
            and item.get("decision", {}).get("decision") == "execute_pair"
        ]
        if len(decisions) != 1:
            raise RuntimeError(f"Expected one executed routing decision for {task}")
        decision = decisions[0]
        selected_id = decision["selected_action_id"]
        actions = {
            item["action_id"]: item for item in summary["research_plan"]["actions"]
        }
        if selected_id not in actions:
            raise RuntimeError(f"Selected action is missing from the research plan for {task}")
        gain_actions = [item for item in actions.values() if item["axis"] != "stop"]
        diagnosis_top = max(
            gain_actions,
            key=lambda item: (
                float(item["expected_effect"]) * float(item["evidence_confidence"]),
                item["action_id"],
            ),
        )
        selected_row = next(
            item
            for item in decision["ranked_pairs"]
            if item["action_id"] == selected_id
        )
        local_effects = [
            item
            for item in summary["action_evidence"]
            if item.get("endpoint_local") is True
            and item.get("complete_pair") is True
            and item.get("action_id") == selected_id
        ]
        if len(local_effects) != 1:
            raise RuntimeError(f"Expected one complete selected-action effect for {task}")
        tasks.append(
            {
                "task": task,
                "selected_action_id": selected_id,
                "diagnosis_top_action_id": diagnosis_top["action_id"],
                "selected_matches_diagnosis_top": selected_id
                == diagnosis_top["action_id"],
                "selected_diagnosis_score": float(
                    actions[selected_id]["expected_effect"]
                    * actions[selected_id]["evidence_confidence"]
                ),
                "top_diagnosis_score": float(
                    diagnosis_top["expected_effect"]
                    * diagnosis_top["evidence_confidence"]
                ),
                "selected_acquisition_utility": float(
                    selected_row["acquisition_utility"]
                ),
                "selected_posterior_evidence_source": selected_row["posterior"][
                    "evidence_source"
                ],
                "selected_posterior_admitted_endpoints": int(
                    selected_row["posterior"]["admitted_endpoints"]
                ),
                "historical_target_endpoint_evidence_available": bool(
                    decision["historical_endpoint_local_evidence_available"]
                ),
                "decision_basis": decision["decision_basis"],
                "mean_oriented_matched_effect": float(
                    local_effects[0]["native_oriented_delta"]
                ),
            }
        )

    if forbidden_accesses:
        raise RuntimeError(f"Forbidden evaluation access: {forbidden_accesses}")
    selected = [item["selected_action_id"] for item in tasks]
    return {
        "version": "ecvoi-v3-diagnosis-alignment-development-audit-v1",
        "scope": "three_endpoint_development_mechanism_diagnostic_no_inferential_claim",
        "protocol_sha256": protocol["sha256"],
        "test_loaded": False,
        "leaderboard_loaded": False,
        "forbidden_accesses": [],
        "tasks": tasks,
        "aggregate": {
            "task_count": len(tasks),
            "unique_selected_actions": len(set(selected)),
            "diagnosis_top_match_count": sum(
                item["selected_matches_diagnosis_top"] for item in tasks
            ),
            "prior_only_selected_count": sum(
                item["selected_posterior_evidence_source"] == "prior_only"
                for item in tasks
            ),
            "historical_target_endpoint_evidence_available_count": sum(
                item["historical_target_endpoint_evidence_available"]
                for item in tasks
            ),
            "negative_selected_effect_count": sum(
                item["mean_oriented_matched_effect"] < 0 for item in tasks
            ),
        },
        "design_implications": [
            "Bind the action acquisition prior to calibrated diagnosis evidence rather than rewarding prior-only uncertainty alone.",
            "Exclude all target-endpoint action outcomes for leave-one-endpoint-out policy evaluation.",
            "Add an explicit posterior harmful-action probability and a preregistered stop threshold.",
            "Evaluate the revised router with matched-state interventions before any 19-task confirmation run.",
        ],
        "interpretation_guard": (
            "This audit diagnoses a routing mechanism on three development endpoints. "
            "It neither estimates confirmation performance nor proves that a proposed "
            "replacement policy will generalize."
        ),
    }


def render_markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# EC-VOI v3 diagnosis–routing alignment audit",
        "",
        f"Protocol: `{payload['protocol_sha256']}`  ",
        "Scope: **three-endpoint development mechanism diagnostic only**.  ",
        "Test/leaderboard access: **none**.",
        "",
        "| Task | Diagnosis top | Router selected | Match | Posterior source | Historical target evidence | Matched effect |",
        "|---|---|---|---:|---|---:|---:|",
    ]
    for row in payload["tasks"]:
        lines.append(
            f"| {row['task']} | `{row['diagnosis_top_action_id']}` | "
            f"`{row['selected_action_id']}` | "
            f"{'yes' if row['selected_matches_diagnosis_top'] else 'no'} | "
            f"{row['selected_posterior_evidence_source']} | "
            f"{'yes' if row['historical_target_endpoint_evidence_available'] else 'no'} | "
            f"{row['mean_oriented_matched_effect']:+.6f} |"
        )
    aggregate = payload["aggregate"]
    lines.extend(
        [
            "",
            "## Aggregate diagnosis",
            "",
            f"The router selected `{aggregate['unique_selected_actions']}` unique action(s) "
            f"across `{aggregate['task_count']}` tasks and matched the diagnosis-top action "
            f"on `{aggregate['diagnosis_top_match_count']}` tasks. The selected action used a "
            f"prior-only posterior on `{aggregate['prior_only_selected_count']}` tasks; target-"
            f"endpoint historical evidence was available on "
            f"`{aggregate['historical_target_endpoint_evidence_available_count']}` tasks.",
            "",
            "## Frozen design implications",
            "",
        ]
    )
    lines.extend(f"- {item}" for item in payload["design_implications"])
    lines.extend(["", "## Interpretation guard", "", payload["interpretation_guard"], ""])
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--markdown", type=Path, required=True)
    args = parser.parse_args()
    payload = audit(_load_protocol(args.protocol), args.run_dir)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    args.markdown.write_text(render_markdown(payload), encoding="utf-8")
    print(json.dumps(payload["aggregate"], sort_keys=True))


if __name__ == "__main__":
    main()
