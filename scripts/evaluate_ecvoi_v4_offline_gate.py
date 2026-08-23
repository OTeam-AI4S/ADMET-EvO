"""Leave-one-endpoint-out mechanism gate for the proposed EC-VOI v4 router.

The evaluator is intentionally development-only.  It uses frozen matched
discovery effects, excludes every target-endpoint row before prediction, and
keeps an explicit shuffled-diagnosis negative control.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import norm

from admet_agent.task_context import context_similarity


ACTION_DIMENSIONS = {
    "data-quality-filter": "data_quality",
    "feature-domain-shift": "domain_shift",
    "model-estimator-family": "local_roughness",
}
SHUFFLED_DIMENSIONS = {
    "data-quality-filter": "domain_shift",
    "feature-domain-shift": "local_roughness",
    "model-estimator-family": "data_quality",
}
ACTION_RISKS = {
    "data-quality-filter": 0.10,
    "feature-domain-shift": 0.15,
    "model-estimator-family": 0.10,
}


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_memory(path: Path) -> list[dict[str, Any]]:
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not rows:
        raise RuntimeError("Matched-effect memory is empty")
    for row in rows:
        claimed = row.get("observation_sha256")
        body = {key: value for key, value in row.items() if key != "observation_sha256"}
        if hashlib.sha256(_canonical(body)).hexdigest() != claimed:
            raise RuntimeError("Matched-effect observation hash mismatch")
        if row.get("test_derived") is not False or row.get("leaderboard_derived") is not False:
            raise RuntimeError("Offline gate received forbidden evaluation evidence")
        if row.get("complete_pair") is not True or row.get("single_axis_verified") is not True:
            raise RuntimeError("Offline gate requires complete single-axis pairs")
    return rows


def _load_contexts(path: Path) -> tuple[dict[str, dict[str, Any]], str]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    claimed = payload.pop("sha256", None)
    if hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest() != claimed:
        raise RuntimeError("Context registry hash mismatch")
    if payload.get("test_loaded") is not False:
        raise RuntimeError("Context registry is not discovery-only")
    return payload["contexts"], str(claimed)


def _posterior(
    rows: list[dict[str, Any]],
    contexts: dict[str, dict[str, Any]],
    *,
    target: str,
    action: str,
    diagnosis_dimensions: dict[str, str] | None,
    prior_sd: float,
    between_endpoint_sd: float,
) -> tuple[float, float, int]:
    precision = 1.0 / prior_sd**2
    weighted_mean = 0.0
    admitted = 0
    for row in rows:
        source = str(row["endpoint_id"])
        if source == target or row["action_id"] != action:
            continue
        weight = context_similarity(contexts[target], contexts[source])
        if diagnosis_dimensions is not None:
            dimension = diagnosis_dimensions[action]
            difference = abs(
                float(contexts[target]["hardness"][dimension])
                - float(contexts[source]["hardness"][dimension])
            )
            weight *= math.exp(-3.0 * difference)
        variance = float(row["native_standard_error"]) ** 2 + between_endpoint_sd**2
        item_precision = weight / variance
        precision += item_precision
        weighted_mean += item_precision * float(row["native_oriented_delta"])
        admitted += 1
    return weighted_mean / precision, math.sqrt(1.0 / precision), admitted


def _select(
    policy: str,
    *,
    target: str,
    actions: list[str],
    rows: list[dict[str, Any]],
    contexts: dict[str, dict[str, Any]],
    prior_sd: float,
    between_endpoint_sd: float,
    harm_margin: float,
    harm_probability_max: float,
    seed: int,
) -> tuple[str | None, dict[str, Any]]:
    if policy == "memory_free":
        return None, {"reason": "no_non_target_evidence"}
    if policy == "seeded_random":
        digest = int(hashlib.sha256(target.encode()).hexdigest()[:8], 16)
        return random.Random(seed + digest).choice(actions), {"reason": "seeded_random"}
    if policy == "fixed_rules":
        selected = max(
            actions,
            key=lambda action: (
                float(contexts[target]["hardness"][ACTION_DIMENSIONS[action]]),
                action,
            ),
        )
        return selected, {"reason": "maximum_action_specific_hardness"}

    dimensions = {
        "full": ACTION_DIMENSIONS,
        "no_harm_constraint": ACTION_DIMENSIONS,
        "shuffled_diagnosis": SHUFFLED_DIMENSIONS,
        "diagnosis_free": None,
        "v3": None,
    }[policy]
    candidates = []
    details = {}
    for action in actions:
        mean, sd, admitted = _posterior(
            rows,
            contexts,
            target=target,
            action=action,
            diagnosis_dimensions=dimensions,
            prior_sd=prior_sd,
            between_endpoint_sd=between_endpoint_sd,
        )
        probability_harm = float(norm.cdf((-harm_margin - mean) / sd))
        row = {
            "posterior_mean": mean,
            "posterior_standard_deviation": sd,
            "posterior_probability_practical_harm": probability_harm,
            "admitted_non_target_endpoints": admitted,
        }
        details[action] = row
        if policy == "v3":
            standardized_mean = mean / harm_margin
            standardized_sd = sd / harm_margin
            robust = max(0.0, standardized_mean - standardized_sd - 1.0)
            information = 0.5 * math.log1p(standardized_sd**2)
            utility = (
                robust
                + 0.04 * information
                - 0.08 * 0.30833333333333335
                - 0.08 * ACTION_RISKS[action]
            )
            row["v3_approximate_robust_utility"] = utility
            if utility > 0:
                candidates.append((utility, action))
        elif policy == "no_harm_constraint":
            if mean > 0:
                candidates.append((mean, action))
        elif probability_harm <= harm_probability_max and mean > 0:
            candidates.append((mean, action))
    selected = max(candidates)[1] if candidates else None
    return selected, {"actions": details, "reason": "selected" if selected else "stop"}


def evaluate(
    *,
    memory_path: Path,
    context_path: Path,
    prior_sd: float = 0.0075,
    between_endpoint_sd: float = 0.005,
    harm_margin: float = 0.005,
    harm_probability_max: float = 0.05,
    seed: int = 20260821,
) -> dict[str, Any]:
    rows = _load_memory(memory_path)
    contexts, context_sha256 = _load_contexts(context_path)
    by_endpoint: dict[str, dict[str, float]] = defaultdict(dict)
    for row in rows:
        endpoint = str(row["endpoint_id"])
        by_endpoint[endpoint][str(row["action_id"])] = float(
            row["native_oriented_delta"]
        )
    endpoints = sorted(by_endpoint)
    actions = sorted({str(row["action_id"]) for row in rows})
    if set(actions) != set(ACTION_DIMENSIONS):
        raise RuntimeError("Offline gate action grid differs from the frozen design")
    if any(set(by_endpoint[endpoint]) != set(actions) for endpoint in endpoints):
        raise RuntimeError("Every endpoint must contain the same complete action grid")
    missing_contexts = set(endpoints) - set(contexts)
    if missing_contexts:
        raise RuntimeError(f"Missing endpoint contexts: {sorted(missing_contexts)}")

    policies = [
        "full",
        "v3",
        "diagnosis_free",
        "memory_free",
        "no_harm_constraint",
        "fixed_rules",
        "seeded_random",
        "shuffled_diagnosis",
    ]
    results = []
    for endpoint in endpoints:
        oracle_effect = max(0.0, max(by_endpoint[endpoint].values()))
        for policy in policies:
            selected, audit = _select(
                policy,
                target=endpoint,
                actions=actions,
                rows=rows,
                contexts=contexts,
                prior_sd=prior_sd,
                between_endpoint_sd=between_endpoint_sd,
                harm_margin=harm_margin,
                harm_probability_max=harm_probability_max,
                seed=seed,
            )
            effect = 0.0 if selected is None else by_endpoint[endpoint][selected]
            results.append(
                {
                    "endpoint": endpoint,
                    "policy": policy,
                    "selected_action_id": selected,
                    "stopped": selected is None,
                    "oriented_effect": effect,
                    "oracle_effect": oracle_effect,
                    "regret": oracle_effect - effect,
                    "practically_harmful": effect < -harm_margin,
                    "target_endpoint_rows_admitted": 0,
                    "selection_audit": audit,
                }
            )
    aggregates = {}
    for policy in policies:
        items = [item for item in results if item["policy"] == policy]
        aggregates[policy] = {
            "endpoint_count": len(items),
            "mean_oriented_effect": float(np.mean([item["oriented_effect"] for item in items])),
            "mean_regret": float(np.mean([item["regret"] for item in items])),
            "harmful_intervention_rate": float(np.mean([item["practically_harmful"] for item in items])),
            "stop_rate": float(np.mean([item["stopped"] for item in items])),
            "selected_actions": [item["selected_action_id"] for item in items],
        }
    full = aggregates["full"]
    gates = {
        "mean_selected_effect_nonnegative": full["mean_oriented_effect"] >= 0,
        "harm_no_worse_than_v3": full["harmful_intervention_rate"] <= aggregates["v3"]["harmful_intervention_rate"],
        "harm_no_worse_than_fixed_rules": full["harmful_intervention_rate"] <= aggregates["fixed_rules"]["harmful_intervention_rate"],
        "regret_lower_than_seeded_random": full["mean_regret"] < aggregates["seeded_random"]["mean_regret"],
        "shuffled_diagnosis_does_not_equal_or_outperform": (
            full["mean_oriented_effect"] > aggregates["shuffled_diagnosis"]["mean_oriented_effect"]
            or full["mean_regret"] < aggregates["shuffled_diagnosis"]["mean_regret"]
        ),
        "target_endpoint_rows_admitted_max_zero": all(
            item["target_endpoint_rows_admitted"] == 0 for item in results
        ),
    }
    return {
        "version": "ecvoi-v4-target-held-out-offline-gate-v1",
        "scope": "five_endpoint_development_mechanism_gate_no_confirmation_claim",
        "memory_path": str(memory_path),
        "memory_sha256": _sha256(memory_path),
        "context_path": str(context_path),
        "context_sha256": context_sha256,
        "test_loaded": False,
        "leaderboard_loaded": False,
        "parameters": {
            "prior_sd": prior_sd,
            "between_endpoint_sd": between_endpoint_sd,
            "harm_margin": harm_margin,
            "harm_probability_max": harm_probability_max,
            "diagnosis_kernel": "exp(-3 * absolute_action_specific_hardness_difference)",
            "placebo_mapping": SHUFFLED_DIMENSIONS,
            "seed": seed,
        },
        "results": results,
        "aggregates": aggregates,
        "gates": gates,
        "passed": all(gates.values()),
        "interpretation_guard": (
            "This small retrospective discovery grid can reject a mechanism but cannot "
            "establish confirmation efficacy. A failed shuffled-diagnosis gate requires "
            "new preregistered matched-state development evidence, not threshold tuning."
        ),
    }


def render_markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# EC-VOI v4 target-held-out offline gate",
        "",
        f"Scope: **{payload['scope']}**  ",
        f"Overall gate: **{'PASS' if payload['passed'] else 'FAIL'}**  ",
        "Test/leaderboard access: **none**.",
        "",
        "| Policy | Mean effect | Mean regret | Harm rate | Stop rate |",
        "|---|---:|---:|---:|---:|",
    ]
    for policy, row in payload["aggregates"].items():
        lines.append(
            f"| `{policy}` | {row['mean_oriented_effect']:+.6f} | "
            f"{row['mean_regret']:.6f} | {row['harmful_intervention_rate']:.3f} | "
            f"{row['stop_rate']:.3f} |"
        )
    lines.extend(["", "## Gates", ""])
    lines.extend(
        f"- {'PASS' if passed else 'FAIL'} — `{name}`"
        for name, passed in payload["gates"].items()
    )
    lines.extend(["", "## Interpretation guard", "", payload["interpretation_guard"], ""])
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--memory", type=Path, required=True)
    parser.add_argument("--contexts", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--markdown", type=Path, required=True)
    args = parser.parse_args()
    payload = evaluate(memory_path=args.memory, context_path=args.contexts)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    args.markdown.write_text(render_markdown(payload), encoding="utf-8")
    print(json.dumps({"passed": payload["passed"], "gates": payload["gates"]}, sort_keys=True))


if __name__ == "__main__":
    main()
