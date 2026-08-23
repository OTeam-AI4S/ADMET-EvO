"""Audit the endpoint-local matched interventions executed by EC-VOI v3."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from admet_agent.tasks import resolve_task


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _protocol(path: Path) -> dict[str, Any]:
    value = _load(path)
    claimed = value.get("sha256")
    body = {key: item for key, item in value.items() if key != "sha256"}
    actual = hashlib.sha256(json.dumps(body, sort_keys=True).encode()).hexdigest()
    if claimed != actual:
        raise RuntimeError("EC-VOI v3 protocol SHA-256 is invalid")
    return value


def analyze(
    *,
    protocol: dict[str, Any],
    run_dir: Path,
    bootstrap_seed: int = 20260821,
    bootstrap_replicates: int = 20000,
) -> dict[str, Any]:
    if bootstrap_replicates < 1000:
        raise ValueError("At least 1000 bootstrap replicates are required")
    rows = []
    forbidden_access = []
    for task_name in protocol["tasks"]:
        task_dir = run_dir / "tasks" / task_name
        summary = _load(task_dir / f"search_seed_{protocol['discovery_seeds'][0]}.json")
        anchor = _load(task_dir / "shared_strong_anchor.json")
        if summary.get("benchmark_protocol_sha256") != protocol["sha256"]:
            raise RuntimeError(f"Protocol mismatch for {task_name}")
        if summary.get("source_sha256") != protocol["implementation_source_sha256"]:
            raise RuntimeError(f"Source mismatch for {task_name}")
        for label, payload in (("summary", summary), ("anchor", anchor)):
            if payload.get("test_loaded") is not False:
                forbidden_access.append(f"{task_name}/{label}/test")
            if payload.get("leaderboard_loaded") is not False:
                forbidden_access.append(f"{task_name}/{label}/leaderboard")
        evidence = [
            item
            for item in summary.get("action_evidence", [])
            if item.get("complete_pair") is True
            and item.get("endpoint_local") is True
        ]
        if len(evidence) != 1:
            raise RuntimeError(
                f"Expected exactly one complete local pair for {task_name}, got {len(evidence)}"
            )
        observation = evidence[0]
        if observation.get("test_derived") is not False or observation.get(
            "leaderboard_derived"
        ) is not False:
            raise RuntimeError(f"Forbidden evidence provenance for {task_name}")
        if observation.get("single_axis_verified") is not True:
            raise RuntimeError(f"Pair is not single-axis verified for {task_name}")
        pair_id = observation["pair_id"]
        pair_spec = next(
            item
            for item in summary["resolved_action_pairs"]
            if item["pair_id"] == pair_id
        )
        execution = next(
            item
            for item in summary["ecvoi_v2_episode"]["ledger"]
            if item["event_type"] == "pair_execution_complete"
            and item["pair_id"] == pair_id
        )
        decision = next(
            item["decision"]
            for item in summary["ecvoi_v2_episode"]["ledger"]
            if item["event_type"] == "decision_emitted"
            and item["decision"].get("selected_pair_id") == pair_id
        )
        results = {item["run_id"]: item for item in summary["results"]}
        treatment_ids = execution["treatment_run_ids"]
        control_ids = execution["control_run_ids"]
        if len({len(treatment_ids), len(control_ids), len(protocol["discovery_seeds"])}) != 1:
            raise RuntimeError(f"Matched seed count mismatch for {task_name}")
        direction = 1.0 if resolve_task(task_name).maximize else -1.0
        seed_rows = []
        for seed, treatment_id, control_id in zip(
            protocol["discovery_seeds"], treatment_ids, control_ids, strict=True
        ):
            treatment = results[treatment_id]
            control = results[control_id]
            if treatment["seed"] != seed or control["seed"] != seed:
                raise RuntimeError(f"Same-seed pair contract failed for {task_name}")
            delta = direction * (
                float(treatment["validation_score"])
                - float(control["validation_score"])
            )
            seed_rows.append(
                {
                    "seed": seed,
                    "treatment_run_id": treatment_id,
                    "control_run_id": control_id,
                    "treatment_score": float(treatment["validation_score"]),
                    "control_score": float(control["validation_score"]),
                    "oriented_delta": delta,
                }
            )
        deltas = np.asarray([item["oriented_delta"] for item in seed_rows])
        if not np.isclose(
            float(np.mean(deltas)),
            float(observation["native_oriented_delta"]),
            atol=1e-12,
            rtol=0,
        ):
            raise RuntimeError(f"Recorded paired effect mismatch for {task_name}")
        finalist = next(
            item
            for item in summary["robust_results"] + summary["ensemble_results"]
            if item["run_id"] == summary["finalist_run_id"]
        )
        anchor_score = float(anchor["aggregate_result"]["validation_score"])
        rows.append(
            {
                "task": task_name,
                "metric": resolve_task(task_name).metric,
                "action_id": observation["action_id"],
                "axis": pair_spec["axis"],
                "changed_fields": pair_spec["changed_fields"],
                "pair_id": pair_id,
                "pair_sha256": pair_spec["pair_sha256"],
                "decision_basis": decision["decision_basis"],
                "emitted_before_execution": decision["emitted_before_execution"],
                "ordered_seeds": execution["ordered_seeds"],
                "seed_results": seed_rows,
                "mean_oriented_delta": float(np.mean(deltas)),
                "sample_standard_deviation": float(np.std(deltas, ddof=1)),
                "native_standard_error": float(observation["native_standard_error"]),
                "actual_pair_fit_seconds": float(execution["actual_fit_seconds"]),
                "finalist_score": float(finalist["validation_score"]),
                "shared_anchor_score": anchor_score,
                "finalist_minus_anchor_oriented": direction
                * (float(finalist["validation_score"]) - anchor_score),
                "finalist_is_shared_anchor": summary["finalist_run_id"]
                == anchor["aggregate_result"]["run_id"],
            }
        )

    if forbidden_access:
        raise RuntimeError(f"Forbidden evaluation access detected: {forbidden_access}")
    endpoint_effects = np.asarray([item["mean_oriented_delta"] for item in rows])
    rng = np.random.default_rng(bootstrap_seed)
    indices = rng.integers(
        0, len(endpoint_effects), size=(bootstrap_replicates, len(endpoint_effects))
    )
    bootstrap = endpoint_effects[indices].mean(axis=1)
    return {
        "version": "ecvoi-v3-matched-pair-development-audit-v1",
        "protocol_sha256": protocol["sha256"],
        "scope": "three_endpoint_development_pilot_no_inferential_claim",
        "test_loaded": False,
        "leaderboard_loaded": False,
        "forbidden_accesses": forbidden_access,
        "complete_endpoint_local_pairs": len(rows),
        "tasks": rows,
        "aggregate": {
            "mean_endpoint_oriented_delta": float(np.mean(endpoint_effects)),
            "median_endpoint_oriented_delta": float(np.median(endpoint_effects)),
            "bootstrap_95_ci": [
                float(np.quantile(bootstrap, 0.025)),
                float(np.quantile(bootstrap, 0.975)),
            ],
            "positive_pairs": int(np.sum(endpoint_effects > 0)),
            "negative_pairs": int(np.sum(endpoint_effects < 0)),
            "zero_pairs": int(np.sum(endpoint_effects == 0)),
            "shared_anchor_retained_tasks": sum(
                item["finalist_is_shared_anchor"] for item in rows
            ),
            "total_pair_fit_seconds": float(
                sum(item["actual_pair_fit_seconds"] for item in rows)
            ),
        },
        "interpretation_guard": (
            "These same-seed, single-axis pairs identify local action effects on the "
            "three development endpoints. They do not establish cross-endpoint "
            "generalization or policy superiority."
        ),
    }


def render_markdown(payload: dict[str, Any]) -> str:
    aggregate = payload["aggregate"]
    lines = [
        "# EC-VOI v3 matched-pair development audit",
        "",
        f"Protocol: `{payload['protocol_sha256']}`  ",
        f"Complete endpoint-local pairs: **{payload['complete_endpoint_local_pairs']}**  ",
        "Test/leaderboard access: **none**.",
        "",
        "| Task | Action | Axis | Seed deltas | Mean delta | Finalist − anchor | Anchor retained |",
        "|---|---|---|---|---:|---:|---:|",
    ]
    for row in payload["tasks"]:
        deltas = ", ".join(
            f"{item['oriented_delta']:+.6f}" for item in row["seed_results"]
        )
        lines.append(
            f"| {row['task']} | `{row['action_id']}` | {row['axis']} | {deltas} | "
            f"{row['mean_oriented_delta']:+.6f} | "
            f"{row['finalist_minus_anchor_oriented']:+.6f} | "
            f"{'yes' if row['finalist_is_shared_anchor'] else 'no'} |"
        )
    low, high = aggregate["bootstrap_95_ci"]
    lines.extend(
        [
            "",
            "## Aggregate",
            "",
            f"Mean endpoint effect: `{aggregate['mean_endpoint_oriented_delta']:+.6f}`; "
            f"development bootstrap 95% interval: `[{low:+.6f}, {high:+.6f}]`. "
            f"Positive/negative pairs: `{aggregate['positive_pairs']}/{aggregate['negative_pairs']}`. "
            f"The shared anchor was retained on `{aggregate['shared_anchor_retained_tasks']}` tasks.",
            "",
            "## Interpretation guard",
            "",
            payload["interpretation_guard"],
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--markdown", type=Path)
    parser.add_argument("--bootstrap-seed", type=int, default=20260821)
    parser.add_argument("--bootstrap-replicates", type=int, default=20000)
    args = parser.parse_args()
    payload = analyze(
        protocol=_protocol(args.protocol),
        run_dir=args.run_dir,
        bootstrap_seed=args.bootstrap_seed,
        bootstrap_replicates=args.bootstrap_replicates,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    if args.markdown is not None:
        args.markdown.write_text(render_markdown(payload), encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
