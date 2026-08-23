"""Evaluate the preregistered EC-VOI v3 development promotion gate."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from admet_agent.tasks import resolve_task


TOLERANCE = 1e-12


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _protocol(path: Path) -> dict[str, Any]:
    value = _load(path)
    claimed = value.get("sha256")
    payload = {key: item for key, item in value.items() if key != "sha256"}
    actual = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
    if claimed != actual:
        raise ValueError("EC-VOI v3 protocol SHA-256 is invalid")
    return value


def _oriented_delta(task_name: str, treatment: float, control: float) -> float:
    direction = 1.0 if resolve_task(task_name).maximize else -1.0
    return direction * (treatment - control)


def _check(name: str, observed: Any, requirement: str, passed: bool) -> dict[str, Any]:
    return {
        "name": name,
        "observed": observed,
        "requirement": requirement,
        "passed": bool(passed),
    }


def evaluate(
    *,
    protocol_path: Path,
    collector_path: Path,
    analysis_path: Path,
    primary_run_dir: Path,
) -> dict[str, Any]:
    protocol = _protocol(protocol_path)
    collector = _load(collector_path)
    analysis = _load(analysis_path)
    protocol_hash = protocol["sha256"]
    primary_arm = protocol["primary_arm"]
    tasks = list(protocol["tasks"])
    gate = protocol["development_promotion_gate"]
    invariants = protocol["frozen_invariants"]

    expected_arms = {item["name"] for item in protocol["arms"]}
    collector_arms = set(collector.get("arms", {}))
    complete_arm_coverage = collector_arms == expected_arms and all(
        set(collector["arms"][arm].get("task_names", [])) == set(tasks)
        for arm in expected_arms
    )
    hash_match = (
        collector.get("protocol_sha256") == protocol_hash
        and analysis.get("protocol_sha256") == protocol_hash
        and analysis.get("reference_arm") == primary_arm
    )

    primary_summaries: dict[str, dict[str, Any]] = {}
    anchor_deltas: dict[str, float] = {}
    local_pairs: set[tuple[str, str]] = set()
    unsafe_actions: list[dict[str, Any]] = []
    identity_safe_executor_audits: list[dict[str, Any]] = []
    forbidden_access: list[str] = []
    frozen_task_hashes_match = True
    for task_name in tasks:
        task_dir = primary_run_dir / "tasks" / task_name
        summary = _load(task_dir / f"search_seed_{protocol['discovery_seeds'][0]}.json")
        anchor = _load(task_dir / "shared_strong_anchor.json")
        primary_summaries[task_name] = summary
        frozen_task_hashes_match &= all(
            (
                summary.get("benchmark_protocol_sha256") == protocol_hash,
                summary.get("source_sha256")
                == protocol.get("implementation_source_sha256"),
                summary.get("action_memory_snapshot_sha256")
                == invariants.get("action_memory_snapshot_sha256"),
                summary.get("action_context_registry_sha256")
                == invariants.get("action_context_registry_sha256"),
                summary.get("ecvoi_v2_memory_snapshot_sha256")
                == invariants.get("ecvoi_v2_memory_snapshot_sha256"),
                summary.get("research_routing_policy")
                == next(
                    item["research_routing_policy"]
                    for item in protocol["arms"]
                    if item["name"] == primary_arm
                ),
                anchor.get("mapping_sha256")
                == invariants.get("shared_anchor_mapping_sha256"),
                anchor.get("complete") is True,
            )
        )
        for label, payload in (("summary", summary), ("anchor", anchor)):
            if payload.get("test_loaded") is not False:
                forbidden_access.append(f"{task_name}/{label}/test")
            if payload.get("leaderboard_loaded") is not False:
                forbidden_access.append(f"{task_name}/{label}/leaderboard")

        finalist_id = summary["finalist_run_id"]
        finalists = summary.get("robust_results", []) + summary.get(
            "ensemble_results", []
        )
        finalist = next(
            (item for item in finalists if item.get("run_id") == finalist_id), None
        )
        if finalist is None:
            raise ValueError(f"Missing finalist result for {task_name}")
        anchor_score = float(anchor["aggregate_result"]["validation_score"])
        anchor_deltas[task_name] = _oriented_delta(
            task_name, float(finalist["validation_score"]), anchor_score
        )

        if summary.get("mandatory_identity_safety_floor") is not True:
            unsafe_actions.append(
                {"task": task_name, "reason": "identity_safety_floor_not_recorded"}
            )
        for pair in summary.get("resolved_action_pairs", []):
            if any(
                side.get("data_policy") != invariants["mandatory_discovery_data_policy"]
                for side in (pair.get("treatment", {}), pair.get("control", {}))
            ):
                unsafe_actions.append(
                    {"task": task_name, "pair_id": pair.get("pair_id"), "reason": "unsafe_data_policy"}
                )
        for evidence in summary.get("action_evidence", []):
            evidence_safe = (
                evidence.get("test_derived") is False
                and evidence.get("leaderboard_derived") is False
                and evidence.get("single_axis_verified") is True
            )
            if not evidence_safe:
                unsafe_actions.append(
                    {"task": task_name, "pair_id": evidence.get("pair_id"), "reason": "unsafe_evidence"}
                )
            if (
                evidence_safe
                and evidence.get("complete_pair") is True
                and evidence.get("endpoint_local") is True
            ):
                local_pairs.add((task_name, str(evidence["pair_id"])))
                execution = next(
                    (
                        item
                        for item in summary.get("ecvoi_v2_episode", {}).get(
                            "ledger", []
                        )
                        if item.get("event_type") == "pair_execution_complete"
                        and item.get("pair_id") == evidence.get("pair_id")
                    ),
                    None,
                )
                if execution is None:
                    unsafe_actions.append(
                        {
                            "task": task_name,
                            "pair_id": evidence.get("pair_id"),
                            "reason": "missing_pair_execution_event",
                        }
                    )
                    continue
                for role, key in (
                    ("treatment", "treatment_run_ids"),
                    ("control", "control_run_ids"),
                ):
                    for run_id in execution.get(key, []):
                        audit_path = (
                            primary_run_dir
                            / "artifacts"
                            / str(run_id)
                            / "data_audit.json"
                        )
                        if not audit_path.exists():
                            unsafe_actions.append(
                                {
                                    "task": task_name,
                                    "pair_id": evidence.get("pair_id"),
                                    "run_id": run_id,
                                    "reason": "missing_executor_data_audit",
                                }
                            )
                            continue
                        data_audit = _load(audit_path)
                        audit_safe = (
                            data_audit.get("policy")
                            == invariants["mandatory_discovery_data_policy"]
                            and int(data_audit.get("output_rows", -1)) > 0
                            and int(data_audit.get("output_rows", -1))
                            <= int(data_audit.get("input_rows", -1))
                            and int(data_audit.get("removed_validation_overlap", -1))
                            >= 0
                        )
                        identity_safe_executor_audits.append(
                            {
                                "task": task_name,
                                "pair_id": evidence.get("pair_id"),
                                "role": role,
                                "run_id": run_id,
                                "policy": data_audit.get("policy"),
                                "input_rows": data_audit.get("input_rows"),
                                "output_rows": data_audit.get("output_rows"),
                                "removed_validation_overlap": data_audit.get(
                                    "removed_validation_overlap"
                                ),
                                "safe": audit_safe,
                            }
                        )
                        if not audit_safe:
                            unsafe_actions.append(
                                {
                                    "task": task_name,
                                    "pair_id": evidence.get("pair_id"),
                                    "run_id": run_id,
                                    "reason": "unsafe_executor_data_audit",
                                }
                            )

    per_task = analysis["per_task"]
    nonloss_tasks = [
        task_name
        for task_name in tasks
        if _oriented_delta(
            task_name,
            float(per_task[task_name]["scores"][primary_arm]),
            float(per_task[task_name]["scores"]["automl"]),
        )
        >= -TOLERANCE
    ]
    mean_vs_automl = float(
        analysis["comparisons"]["automl"]["mean_oriented_delta"]
    )
    gain_tasks = [task for task, delta in anchor_deltas.items() if delta > TOLERANCE]
    harm_tasks = [task for task, delta in anchor_deltas.items() if delta < -TOLERANCE]

    if collector.get("test_loaded") is not False:
        forbidden_access.append("collector/test")
    if analysis.get("test_loaded") is not False:
        forbidden_access.append("analysis/test")
    if analysis.get("leaderboard_loaded") is not False:
        forbidden_access.append("analysis/leaderboard")

    checks = [
        _check(
            "complete_arm_task_coverage",
            {"arms": sorted(collector_arms), "tasks": tasks},
            f"all {len(expected_arms)} arms cover all {len(tasks)} tasks",
            complete_arm_coverage,
        ),
        _check(
            "all_frozen_hashes_match",
            hash_match and frozen_task_hashes_match,
            "protocol, source, memory, context and anchor hashes all match",
            hash_match and frozen_task_hashes_match,
        ),
        _check(
            "test_or_leaderboard_access",
            forbidden_access,
            f"at most {gate['test_or_leaderboard_access_max']} forbidden accesses",
            len(forbidden_access) <= gate["test_or_leaderboard_access_max"],
        ),
        _check(
            "unsafe_actions",
            {
                "unsafe_actions": unsafe_actions,
                "executor_identity_safe_audits": len(
                    identity_safe_executor_audits
                ),
                "removed_validation_overlap_total": sum(
                    int(item["removed_validation_overlap"])
                    for item in identity_safe_executor_audits
                    if item["safe"]
                ),
            },
            f"at most {protocol['claim_gates']['unsafe_actions_max']} unsafe actions",
            len(unsafe_actions) <= protocol["claim_gates"]["unsafe_actions_max"],
        ),
        _check(
            "complete_endpoint_local_pairs",
            len(local_pairs),
            f"at least {gate['minimum_complete_local_pairs_across_tasks']}",
            len(local_pairs) >= gate["minimum_complete_local_pairs_across_tasks"],
        ),
        _check(
            "vs_automl_nonloss_tasks",
            nonloss_tasks,
            f"at least {gate['vs_automl_min_nonloss_tasks']} tasks",
            len(nonloss_tasks) >= gate["vs_automl_min_nonloss_tasks"],
        ),
        _check(
            "vs_automl_mean_oriented_delta",
            mean_vs_automl,
            f">= {gate['vs_automl_mean_oriented_delta_min']}",
            mean_vs_automl >= gate["vs_automl_mean_oriented_delta_min"] - TOLERANCE,
        ),
        _check(
            "vs_shared_anchor_gain_tasks",
            {"tasks": gain_tasks, "oriented_deltas": anchor_deltas},
            f"at least {gate['vs_shared_anchor_min_gain_tasks']} strictly positive tasks",
            len(gain_tasks) >= gate["vs_shared_anchor_min_gain_tasks"],
        ),
        _check(
            "vs_shared_anchor_harm_tasks",
            harm_tasks,
            f"at most {gate['vs_shared_anchor_max_harm_tasks']} strictly negative tasks",
            len(harm_tasks) <= gate["vs_shared_anchor_max_harm_tasks"],
        ),
    ]
    promoted = all(item["passed"] for item in checks)
    return {
        "version": "ecvoi-v3-development-promotion-gate-v1",
        "protocol_sha256": protocol_hash,
        "scope": gate["scope"],
        "primary_arm": primary_arm,
        "development_tasks": tasks,
        "checks": checks,
        "promotion_passed": promoted,
        "confirmation_unlocked": promoted,
        "confirmation_policy_modification_allowed": protocol[
            "confirmation_can_modify_policy"
        ],
        "test_loaded": False,
        "leaderboard_loaded": False,
        "interpretation": (
            "The preregistered development gate passed; the frozen policy may be evaluated on the untouched confirmation cohort."
            if promoted
            else "The preregistered development gate failed; the untouched 19-task superiority cohort remains sealed."
        ),
    }


def render_markdown(result: dict[str, Any]) -> str:
    lines = [
        "# EC-VOI v3 development promotion gate",
        "",
        f"Protocol: `{result['protocol_sha256']}`  ",
        f"Primary arm: `{result['primary_arm']}`  ",
        f"Promotion passed: **{'yes' if result['promotion_passed'] else 'no'}**  ",
        f"19-task confirmation unlocked: **{'yes' if result['confirmation_unlocked'] else 'no'}**.",
        "",
        "| Check | Observed | Requirement | Passed |",
        "|---|---|---|---:|",
    ]
    for item in result["checks"]:
        observed = json.dumps(item["observed"], sort_keys=True)
        lines.append(
            f"| `{item['name']}` | `{observed}` | {item['requirement']} | "
            f"{'yes' if item['passed'] else 'no'} |"
        )
    lines.extend(["", "## Decision", "", result["interpretation"], ""])
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--collector", type=Path, required=True)
    parser.add_argument("--analysis", type=Path, required=True)
    parser.add_argument("--primary-run-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--markdown", type=Path)
    args = parser.parse_args()
    result = evaluate(
        protocol_path=args.protocol,
        collector_path=args.collector,
        analysis_path=args.analysis,
        primary_run_dir=args.primary_run_dir,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    if args.markdown is not None:
        args.markdown.write_text(render_markdown(result), encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
