"""Freeze a development-derived task-type route before blind evaluation."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def freeze(protocol_path: Path, result_path: Path) -> dict[str, Any]:
    protocol = _load(protocol_path)
    result = _load(result_path)
    if result.get("protocol_sha256") != protocol.get("sha256"):
        raise RuntimeError("External propensity result/protocol mismatch")
    if result.get("test_labels_loaded") is not False or result.get(
        "leaderboard_feedback_used"
    ) is not False:
        raise RuntimeError("Blind feedback is forbidden for task-type routing")
    endpoint_values: dict[str, dict[str, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    endpoint_types: dict[str, str] = {}
    for fold_seed in result["fold_seed_results"]:
        for arm, endpoint_metrics in fold_seed["endpoint_metrics"].items():
            for endpoint, metrics in endpoint_metrics.items():
                if metrics.get("st_rae") is not None:
                    endpoint_types[endpoint] = "regression"
                    endpoint_values[endpoint][arm].append(float(metrics["st_rae"]))
                elif metrics.get("mcc") is not None:
                    endpoint_types[endpoint] = "classification"
                    endpoint_values[endpoint][arm].append(float(metrics["mcc"]))
                else:
                    raise RuntimeError(f"Endpoint {endpoint} has no primary metric")
    rows = []
    for endpoint in sorted(endpoint_values):
        task_type = endpoint_types[endpoint]
        control = float(np.mean(endpoint_values[endpoint]["b1_control"]))
        treatment = float(
            np.mean(endpoint_values[endpoint]["b1_external_propensity"])
        )
        oriented_gain = treatment - control if task_type == "classification" else control - treatment
        rows.append(
            {
                "endpoint": endpoint,
                "task_type": task_type,
                "control_mean": control,
                "treatment_mean": treatment,
                "external_propensity_oriented_gain": oriented_gain,
            }
        )
    regression = [row for row in rows if row["task_type"] == "regression"]
    classification = [row for row in rows if row["task_type"] == "classification"]
    checks = {
        "all_regression_endpoints_positive": bool(regression)
        and all(row["external_propensity_oriented_gain"] > 0 for row in regression),
        "all_classification_endpoints_nonpositive": bool(classification)
        and all(row["external_propensity_oriented_gain"] <= 0 for row in classification),
        "blind_feedback_absent": True,
        "complete_endpoint_coverage": len(rows) == 6,
    }
    frozen = all(checks.values())
    payload = {
        "version": "openadmet-propensity-task-type-route-v1",
        "scope": "development_derived_recipe_freeze_not_blind_confirmation",
        "source_protocol_sha256": protocol["sha256"],
        "source_result_sha256": _sha256(result_path),
        "route": {
            "primary_direct_regression": "b1_external_propensity",
            "primary_classification": "b1_control",
            "fallback_unseen_endpoint_type": "b1_control",
        },
        "endpoint_development_effects": rows,
        "checks": checks,
        "route_frozen": frozen,
        "blind_test_can_modify_route": False,
        "leaderboard_can_modify_route": False,
        "test_labels_loaded": False,
        "leaderboard_feedback_used": False,
        "interpretation_guard": (
            "The route was selected from training-only cross-validation and is a "
            "candidate recipe, not independent evidence of blind improvement."
        ),
    }
    payload["sha256"] = hashlib.sha256(
        json.dumps(payload, sort_keys=True).encode()
    ).hexdigest()
    return payload


def render_markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# OpenADMET propensity task-type route freeze",
        "",
        f"Route frozen: **{'yes' if payload['route_frozen'] else 'no'}**.  ",
        "Blind/test/leaderboard access: **none**.",
        "",
        "| Endpoint | Type | Control | Propensity | Oriented gain |",
        "|---|---|---:|---:|---:|",
    ]
    for row in payload["endpoint_development_effects"]:
        lines.append(
            f"| `{row['endpoint']}` | {row['task_type']} | "
            f"{row['control_mean']:.6f} | {row['treatment_mean']:.6f} | "
            f"{row['external_propensity_oriented_gain']:+.6f} |"
        )
    lines.extend(
        [
            "",
            "## Frozen route",
            "",
            "- `primary_direct` regression: B1 + external propensity.",
            "- primary classification/TDI: B1 control.",
            "- unseen endpoint type: fail-safe B1 control.",
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
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--markdown", type=Path, required=True)
    args = parser.parse_args()
    payload = freeze(args.protocol, args.result)
    args.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    args.markdown.write_text(render_markdown(payload), encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
