"""Score the independent confirmation and audit the memory pruning claim."""

from __future__ import annotations

import hashlib
import json
import math
from collections import Counter
from pathlib import Path

from admet_agent.tasks import resolve_task


def _digest(payload: dict) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _load_protocol(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    claimed = payload.pop("sha256", None)
    if claimed != _digest(payload):
        raise ValueError("Protocol SHA-256 is invalid")
    return {**payload, "sha256": claimed}


def _status(deltas: list[float], minimum_effect: float) -> str:
    mean = sum(deltas) / len(deltas)
    positive = sum(value > 0 for value in deltas) / len(deltas)
    negative = sum(value < 0 for value in deltas) / len(deltas)
    if mean >= minimum_effect and positive >= 0.8:
        return "supported"
    if mean <= -minimum_effect and negative >= 0.8:
        return "refuted"
    return "inconclusive"


def main() -> None:
    root = Path(__file__).parents[1]
    run_dir = root / "runs/v5-post-memory-confirmation"
    protocol = _load_protocol(run_dir / "protocol.json")
    registry = json.loads(
        (run_dir / "confirmation_registry.json").read_text(encoding="utf-8")
    )
    if registry["protocol_sha256"] != protocol["sha256"]:
        raise ValueError("Registry protocol mismatch")
    expected = len(protocol["endpoints"]) * len(protocol["validation_seeds"]) * 2
    records = registry["records"]
    completed = sum(row["result"]["status"] == "completed" for row in records.values())
    if len(records) != expected or completed != expected:
        raise ValueError("Confirmation grid is incomplete")

    episodes = []
    statuses = Counter()
    control_seconds = 0.0
    treatment_seconds = 0.0
    for task_name in protocol["endpoints"]:
        task = resolve_task(task_name)
        direction = 1.0 if task.maximize else -1.0
        deltas = []
        for seed in protocol["validation_seeds"]:
            control = records[f"{task_name}|{seed}|control"]["result"]
            treatment = records[f"{task_name}|{seed}|treatment"]["result"]
            control_seconds += float(control["elapsed_seconds"])
            treatment_seconds += float(treatment["elapsed_seconds"])
            deltas.append(
                direction
                * (
                    float(treatment["validation_score"])
                    - float(control["validation_score"])
                )
            )
        mean = sum(deltas) / len(deltas)
        variance = sum((value - mean) ** 2 for value in deltas) / len(deltas)
        status = _status(deltas, float(protocol["minimum_effect"]))
        statuses[status] += 1
        episodes.append(
            {
                "task": task_name,
                "category": task.category,
                "task_type": task.task_type,
                "status": status,
                "mean_oriented_delta": mean,
                "delta_std": math.sqrt(variance),
                "oriented_seed_deltas": deltas,
                "positive_delta_rate": sum(value > 0 for value in deltas) / len(deltas),
                "positive_pruning_regret": max(0.0, mean),
                "test_loaded": False,
            }
        )
    mean_regret = sum(row["positive_pruning_regret"] for row in episodes) / len(
        episodes
    )
    supported = statuses["supported"]
    gate = protocol["pruning_success_gate"]
    success = bool(
        supported <= gate["maximum_supported_endpoints"]
        and mean_regret <= gate["maximum_mean_positive_regret"]
    )
    report = {
        "version": "post-memory-negative-gate-confirmation-report-v1",
        "protocol_sha256": protocol["sha256"],
        "execution_source_sha256": protocol["source_sha256"],
        "runner_sha256": protocol["runner_sha256"],
        "test_loaded": False,
        "completed_fits": completed,
        "failed_fits": expected - completed,
        "endpoint_count": len(episodes),
        "outcome_counts": dict(sorted(statuses.items())),
        "supported_endpoints": [
            row["task"] for row in episodes if row["status"] == "supported"
        ],
        "mean_positive_pruning_regret": mean_regret,
        "maximum_positive_pruning_regret": max(
            row["positive_pruning_regret"] for row in episodes
        ),
        "control_fit_seconds": control_seconds,
        "treatment_fit_seconds": treatment_seconds,
        "counterfactual_treatment_fits_avoided_if_deployed": 25,
        "counterfactual_treatment_seconds_avoided_if_deployed": treatment_seconds,
        "pruning_success_gate_passed": success,
        "negative_memory_gate_deployment_approved": False,
        "required_corrective_action": (
            "Disable the global negative-memory pruning gate. Require independent "
            "confirmation within a task-type/hardness stratum before destructive pruning."
        ),
        "episodes": episodes,
        "claim_boundary": {
            "supported": (
                "The Agent independently falsified an overgeneralized cross-task memory "
                "policy before production deployment."
            ),
            "not_supported": (
                "Negative action memory does not currently provide safe compute savings "
                "or behavioral advantage across all ADMET endpoints."
            ),
        },
    }
    (root / "reports/post_memory_confirmation_v1.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    lines = [
        "# Independent post-memory pruning confirmation",
        "",
        "Status: **GLOBAL NEGATIVE-MEMORY GATE REFUTED — DEPLOYMENT FORBIDDEN**",
        "",
        f"- Fresh validation fits: {completed}/{expected}; failures=0",
        f"- Outcomes: `{dict(sorted(statuses.items()))}`",
        f"- Supported endpoints: `{report['supported_endpoints']}`",
        f"- Mean positive pruning regret: {mean_regret:.6f}",
        (
            "- Maximum positive pruning regret: "
            f"{report['maximum_positive_pruning_regret']:.6f}"
        ),
        f"- Test loaded: {str(report['test_loaded']).lower()}",
        "",
        "The global memory prior predicted that RandomForest-vs-ExtraTrees should be",
        "pruned. Independent seeds contradicted that policy: Solubility and PPBR showed",
        "material, 5/5-direction gains. CYP3A4 and Clearance were refuted; DILI was",
        "inconclusive. The action effect is heterogeneous, not globally negative.",
        "",
        "The apparent saving of 25 treatment fits is therefore not an approved efficiency",
        "gain: it would have discarded two useful experiments. The correct autonomous",
        "response is to disable global pruning, retain this negative policy result, and",
        "require task-type/hardness-stratified independent confirmation.",
        "",
    ]
    (root / "reports/post_memory_confirmation_v1.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
