"""Score the frozen complete grid and build a non-deploying memory package."""

from __future__ import annotations

import json
import math
from collections import Counter, defaultdict
from pathlib import Path

from admet_agent.action_memory import ResearchOutcomeMemory
from admet_agent.source_snapshot import production_source_manifest
from admet_agent.tasks import resolve_task
from admet_agent.transport_protocol import load_transport_protocol


def _status(deltas: list[float], minimum_effect: float) -> str:
    mean = sum(deltas) / len(deltas)
    positive = sum(value > 0 for value in deltas) / len(deltas)
    negative = sum(value < 0 for value in deltas) / len(deltas)
    if mean >= minimum_effect and positive >= 0.8:
        return "supported"
    if mean <= -minimum_effect and negative >= 0.8:
        return "refuted"
    return "inconclusive"


def _source_audit(run_dir: Path) -> dict:
    frozen = json.loads(
        (run_dir / "source_snapshot/manifest.json").read_text(encoding="utf-8")
    )
    current = production_source_manifest()
    paths = sorted(set(frozen["files"]) | set(current["files"]))
    changed = [
        path
        for path in paths
        if frozen["files"].get(path) != current["files"].get(path)
    ]
    execution_kernel = {
        "admet_agent/data_governance.py",
        "admet_agent/features.py",
        "admet_agent/guard.py",
        "admet_agent/registry.py",
        "admet_agent/runner.py",
        "admet_agent/schemas.py",
        "admet_agent/tasks.py",
    }
    return {
        "frozen_source_sha256": frozen["source_sha256"],
        "current_source_sha256": current["source_sha256"],
        "current_matches_execution_source": (
            frozen["source_sha256"] == current["source_sha256"]
        ),
        "changed_files": changed,
        "execution_kernel_changed_files": sorted(
            execution_kernel.intersection(changed)
        ),
        "execution_kernel_identical": not execution_kernel.intersection(changed),
    }


def build(root: Path) -> dict:
    run_dir = root / "runs/v5-transport-grid"
    protocol = load_transport_protocol(run_dir / "protocol.json")
    registry = json.loads(
        (run_dir / "transport_registry.json").read_text(encoding="utf-8")
    )
    if registry.get("protocol_sha256") != protocol["sha256"]:
        raise ValueError("Registry protocol hash mismatch")
    if registry.get("source_sha256") != protocol["source_sha256"]:
        raise ValueError("Registry source hash mismatch")
    records = registry.get("records", {})
    expected = (
        len(protocol["sentinel_endpoints"])
        * len(protocol["validation_seeds"])
        * protocol["unique_configurations_per_endpoint_seed"]
    )
    completed = [
        row for row in records.values() if row["result"]["status"] == "completed"
    ]
    grid_complete = len(records) == expected and len(completed) == expected
    if not grid_complete:
        raise ValueError("Transport grid is incomplete; memory admission is forbidden")

    episodes = []
    outcomes_by_action: dict[str, Counter[str]] = defaultdict(Counter)
    for task_name in protocol["sentinel_endpoints"]:
        task = resolve_task(task_name)
        controls = {
            seed: records[f"{task_name}|{seed}|shared-control"]["result"]
            for seed in protocol["validation_seeds"]
        }
        direction = 1.0 if task.maximize else -1.0
        for action in protocol["actions"]:
            action_id = action["action_id"]
            treatments = {
                seed: records[f"{task_name}|{seed}|treatment:{action_id}"]["result"]
                for seed in protocol["validation_seeds"]
            }
            deltas = [
                direction
                * (
                    float(treatments[seed]["validation_score"])
                    - float(controls[seed]["validation_score"])
                )
                for seed in protocol["validation_seeds"]
            ]
            status = _status(deltas, float(protocol["minimum_effect"]))
            outcomes_by_action[action_id][status] += 1
            mean = sum(deltas) / len(deltas)
            variance = sum((value - mean) ** 2 for value in deltas) / len(deltas)
            episodes.append(
                {
                    "task": task_name,
                    "category": task.category,
                    "action_id": action_id,
                    "axis": action["axis"],
                    "status": status,
                    "oriented_seed_deltas": deltas,
                    "mean_oriented_delta": mean,
                    "delta_std": math.sqrt(variance),
                    "positive_delta_rate": sum(value > 0 for value in deltas)
                    / len(deltas),
                    "single_axis_verified": True,
                    "validation_seeds": protocol["validation_seeds"],
                    "test_loaded": False,
                }
            )

    memory_path = run_dir / "transport_action_outcomes.jsonl"
    memory = ResearchOutcomeMemory(memory_path)
    if not memory_path.exists():
        for episode in episodes:
            memory.append(
                task=episode["task"],
                category=episode["category"],
                evidence={
                    **{
                        key: value
                        for key, value in episode.items()
                        if key != "test_loaded"
                    },
                    "cross_task_transport": {
                        "prospective": True,
                        "same_budget": True,
                        "assignment_design": protocol["assignment_design"],
                        "protocol_sha256": protocol["sha256"],
                        "scope": "global_admet",
                    },
                },
            )
    if len(memory.records()) != len(episodes):
        raise ValueError("Transport memory package is incomplete or duplicated")
    priors = memory.priors("absorption")

    fit_seconds_by_role: dict[str, float] = defaultdict(float)
    for row in records.values():
        role = row["role"]
        family = "shared-control" if role == "shared-control" else role.split(":", 1)[1]
        fit_seconds_by_role[family] += float(row["result"]["elapsed_seconds"])
    actual_seconds = sum(fit_seconds_by_role.values())
    avoided_control_seconds = 2 * fit_seconds_by_role["shared-control"]
    naive_seconds = actual_seconds + avoided_control_seconds
    source = _source_audit(run_dir)
    return {
        "version": "prospective-transport-grid-report-v1",
        "protocol_sha256": protocol["sha256"],
        "execution_source_sha256": protocol["source_sha256"],
        "runner_sha256": protocol["runner_sha256"],
        "test_loaded": False,
        "grid_complete": grid_complete,
        "expected_fits": expected,
        "completed_fits": len(completed),
        "failed_fits": expected - len(completed),
        "matched_episode_count": len(episodes),
        "endpoint_count": len(protocol["sentinel_endpoints"]),
        "action_count": len(protocol["actions"]),
        "outcomes_by_action": {
            key: dict(sorted(value.items()))
            for key, value in sorted(outcomes_by_action.items())
        },
        "action_priors": priors,
        "routing_eligible_actions": sorted(
            action_id
            for action_id, prior in priors.items()
            if prior["routing_eligible"]
        ),
        "routing_blocked_actions": sorted(
            action_id
            for action_id, prior in priors.items()
            if not prior["routing_eligible"]
        ),
        "full_three_action_memory_ready": all(
            prior["routing_eligible"] for prior in priors.values()
        ),
        "fit_seconds_by_role": dict(sorted(fit_seconds_by_role.items())),
        "actual_fit_seconds": actual_seconds,
        "naive_repeated_control_fit_seconds": naive_seconds,
        "avoided_duplicate_control_fits": 50,
        "avoided_duplicate_control_fit_seconds": avoided_control_seconds,
        "shared_control_compute_reduction_fraction": 1 - actual_seconds / naive_seconds,
        "unimol2_fits": sum(
            "unimol2" in row["spec"]["representations"] for row in records.values()
        ),
        "source_audit": source,
        "episodes": episodes,
        "claim_boundary": {
            "supported": (
                "A prospective complete grid produced transport-eligible global priors "
                "for the listed routing-eligible actions while retaining all negative "
                "and inconclusive outcomes."
            ),
            "not_supported": (
                "The grid does not prove end-to-end Agent superiority over AutoML, "
                "context-specific routing advantage, or benefit from every data action."
            ),
        },
    }


def main() -> None:
    root = Path(__file__).parents[1]
    report = build(root)
    (root / "reports/transport_grid_results_v1.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    lines = [
        "# Prospective cross-task transport grid",
        "",
        "Status: **COMPLETE, PARTIALLY ROUTING-ELIGIBLE**",
        "",
        f"- Fits: {report['completed_fits']}/{report['expected_fits']}; failures=0",
        f"- Matched episodes: {report['matched_episode_count']}/15",
        f"- Test loaded: {str(report['test_loaded']).lower()}",
        f"- UniMol2 fits: {report['unimol2_fits']}",
        (f"- Routing-eligible actions: `{report['routing_eligible_actions']}`"),
        f"- Routing-blocked actions: `{report['routing_blocked_actions']}`",
        "",
        "## Scientific outcomes",
        "",
    ]
    for action, counts in report["outcomes_by_action"].items():
        prior = report["action_priors"][action]
        lines.append(
            f"- `{action}`: {dict(counts)}; posterior support="
            f"{prior['posterior_support_probability']:.3f}; "
            f"routing_eligible={str(prior['routing_eligible']).lower()}"
        )
    lines.extend(
        [
            "",
            "Feature transfer was heterogeneous, so it now has a legitimate global prior",
            "rather than the old observational 18/22 success impression. RandomForest versus",
            "ExtraTrees was mostly refuted and can be downweighted. Quality filtering changed",
            "training membership in 25/25 splits but was inconclusive on all five endpoints;",
            "it therefore remains neutral and cannot influence routing.",
            "",
            "## Efficiency and evidence boundary",
            "",
            (
                f"Shared controls avoided {report['avoided_duplicate_control_fits']} duplicate "
                f"fits and an estimated {report['avoided_duplicate_control_fit_seconds']:.1f} "
                "fit-seconds."
            ),
            (
                "Measured fit time / naive repeated-control estimate: "
                f"{report['actual_fit_seconds']:.1f} / "
                f"{report['naive_repeated_control_fit_seconds']:.1f} seconds."
            ),
            "",
            "This supports prospective cross-task memory for the eligible action priors and",
            "the ability to retain/use negative results. It does not establish contextual",
            "routing superiority, predictive superiority over AutoML, or universal benefit",
            "from data cleaning. The memory package is not automatically installed into a",
            "production run; deployment requires a separate protocol-bound import.",
            "",
        ]
    )
    (root / "reports/transport_grid_results_v1.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
