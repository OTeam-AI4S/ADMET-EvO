"""Score the completed three-arm predictive control without test access."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from admet_agent.tasks import resolve_task


def _digest(payload: Any) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode()
    ).hexdigest()


def normalized_utility(task: str, score: float, target_std: float) -> float:
    spec = resolve_task(task)
    if spec.metric == "mae":
        return -float(score) / max(abs(float(target_std)), 1e-12)
    return float(score)


def _bootstrap(values: list[float], *, seed: int, replicates: int) -> list[float]:
    array = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(array), size=(replicates, len(array)))
    means = array[indices].mean(axis=1)
    return [float(value) for value in np.quantile(means, (0.025, 0.975))]


def _comparison(rows: list[dict], left: str, right: str, *, replicates: int) -> dict:
    deltas = [row["utilities"][left] - row["utilities"][right] for row in rows]
    return {
        "left": left,
        "right": right,
        "tasks": len(deltas),
        "mean_normalized_utility_delta": float(np.mean(deltas)),
        "median_normalized_utility_delta": float(np.median(deltas)),
        "task_bootstrap_95_ci": _bootstrap(deltas, seed=20260820, replicates=replicates),
        "wins": sum(delta > 1e-12 for delta in deltas),
        "ties": sum(abs(delta) <= 1e-12 for delta in deltas),
        "losses": sum(delta < -1e-12 for delta in deltas),
    }


def build(run_dir: Path) -> dict:
    protocol_path = run_dir / "protocol.json"
    registry_path = run_dir / "registry.json"
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    if protocol.get("test_loaded") is not False or registry.get("test_loaded") is not False:
        raise ValueError("Predictive-control report requires test_loaded=false")
    protocol_base = {key: value for key, value in protocol.items() if key != "sha256"}
    if protocol.get("sha256") != _digest(protocol_base):
        raise ValueError("Protocol digest mismatch")
    if registry.get("protocol_sha256") != protocol["sha256"]:
        raise ValueError("Registry protocol binding mismatch")
    for key, selection in registry["selections"].items():
        core = {name: value for name, value in selection.items() if name != "decision_sha256"}
        if selection.get("decision_sha256") != _digest(core):
            raise ValueError(f"Selection digest mismatch: {key}")
        screen_rows = sorted(
            (
                row for row in registry["assignments"].values()
                if row["task"] == selection["task"]
                and row["arm"] == selection["arm"]
                and row["phase"] == "screen"
            ),
            key=lambda row: row["role"],
        )
        if selection["screen_assignments_sha256"] != _digest(screen_rows):
            raise ValueError(f"Selection screen evidence changed: {key}")
    for key, assignment in registry["assignments"].items():
        physical = registry["physical_records"].get(assignment["fingerprint"])
        if physical is None or assignment["result"] != physical["result"]:
            raise ValueError(f"Assignment physical-result binding mismatch: {key}")

    rows = []
    for task in protocol["tasks"]:
        profile = registry["task_profiles"][task]
        scores = {}
        selections = {}
        for arm in ("streamlined_agent", "expanded_automl"):
            selected = registry["selections"][f"{task}|{arm}"]["selected_role"]
            assignment = registry["assignments"][f"{task}|{arm}|evaluation|{selected}"]
            if assignment["status"] != "completed":
                raise ValueError(f"Incomplete evaluation: {task}|{arm}")
            scores[arm] = float(assignment["result"]["validation_score"])
            selections[arm] = selected
        fixed = registry["assignments"][f"{task}|strong_fixed|evaluation|fixed"]
        if fixed["status"] != "completed":
            raise ValueError(f"Incomplete fixed evaluation: {task}")
        scores["strong_fixed"] = float(fixed["result"]["validation_score"])
        utilities = {
            arm: normalized_utility(task, score, profile["target_std"])
            for arm, score in scores.items()
        }
        rows.append(
            {
                "task": task,
                "category": resolve_task(task).category,
                "metric": resolve_task(task).metric,
                "target_std": profile["target_std"],
                "selected_roles": selections,
                "scores": scores,
                "utilities": utilities,
            }
        )
    comparisons = {
        "agent_vs_expanded_automl": _comparison(
            rows, "streamlined_agent", "expanded_automl",
            replicates=protocol["bootstrap_replicates"],
        ),
        "agent_vs_strong_fixed": _comparison(
            rows, "streamlined_agent", "strong_fixed",
            replicates=protocol["bootstrap_replicates"],
        ),
        "expanded_automl_vs_strong_fixed": _comparison(
            rows, "expanded_automl", "strong_fixed",
            replicates=protocol["bootstrap_replicates"],
        ),
    }
    main = comparisons["agent_vs_expanded_automl"]
    margin = float(protocol["agent_vs_expanded_automl_noninferiority_margin"])
    logical = len(registry["assignments"])
    physical = len(registry["physical_records"])
    return {
        "protocol": "tdc22-minimal-three-arm-predictive-control-report-v1",
        "status": "complete",
        "tasks": rows,
        "comparisons": comparisons,
        "agent_vs_expanded_automl_noninferiority_margin": margin,
        "agent_vs_expanded_automl_noninferiority_passed": main["task_bootstrap_95_ci"][0] >= margin,
        "logical_charged_fits": logical,
        "maximum_logical_fits": protocol["maximum_logical_fits"],
        "physical_unique_fits": physical,
        "physical_fit_reuse_fraction": 1.0 - physical / logical,
        "actual_unique_fit_seconds": sum(
            float(row["result"]["elapsed_seconds"])
            for row in registry["physical_records"].values()
        ),
        "test_loaded": False,
        "claim_boundary": (
            "Fresh validation seeds 11->12 under a frozen protocol. This is predictive-control "
            "evidence, not frozen TDC test superiority or an application leaderboard update."
        ),
        "source_hashes": {
            str(protocol_path): hashlib.sha256(protocol_path.read_bytes()).hexdigest(),
            str(registry_path): hashlib.sha256(registry_path.read_bytes()).hexdigest(),
        },
    }


def render(payload: dict) -> str:
    lines = [
        "# TDC ADMET-22 minimal three-arm predictive control",
        "",
        f"- Status: `{payload['status']}`",
        f"- Logical charged fits: {payload['logical_charged_fits']}/{payload['maximum_logical_fits']}",
        f"- Physical unique fits: {payload['physical_unique_fits']}（reuse {payload['physical_fit_reuse_fraction']:.1%}）",
        "- Discovery seed: 11；held-out evaluation seed: 12；test loaded: false",
        "",
        "| Comparison | Mean normalized delta | 95% task-bootstrap CI | W/T/L |",
        "|---|---:|---:|---:|",
    ]
    for comparison in payload["comparisons"].values():
        ci = comparison["task_bootstrap_95_ci"]
        lines.append(
            f"| {comparison['left']} − {comparison['right']} | "
            f"{comparison['mean_normalized_utility_delta']:+.4f} | "
            f"[{ci[0]:+.4f}, {ci[1]:+.4f}] | "
            f"{comparison['wins']}/{comparison['ties']}/{comparison['losses']} |"
        )
    lines += [
        "",
        f"Agent vs expanded AutoML noninferiority margin = {payload['agent_vs_expanded_automl_noninferiority_margin']:+.3f}; "
        f"gate = {'PASS' if payload['agent_vs_expanded_automl_noninferiority_passed'] else 'FAIL'}.",
        "",
        "## Per-task selections and evaluation",
        "",
        "| Task | Metric | Agent selection | AutoML selection | Agent | AutoML | Strong fixed |",
        "|---|---|---|---|---:|---:|---:|",
    ]
    for row in payload["tasks"]:
        lines.append(
            f"| {row['task']} | {row['metric']} | {row['selected_roles']['streamlined_agent']} | "
            f"{row['selected_roles']['expanded_automl']} | {row['scores']['streamlined_agent']:.5f} | "
            f"{row['scores']['expanded_automl']:.5f} | {row['scores']['strong_fixed']:.5f} |"
        )
    lines += ["", payload["claim_boundary"], ""]
    return "\n".join(lines)


def main() -> None:
    run_dir = Path("runs/minimal-predictive-control-v1")
    payload = build(run_dir)
    prefix = Path("reports/minimal_predictive_control_v1")
    prefix.with_suffix(".json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    prefix.with_suffix(".md").write_text(render(payload), encoding="utf-8")


if __name__ == "__main__":
    main()
