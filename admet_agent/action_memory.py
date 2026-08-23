"""Append-only discovery evidence used to calibrate future research actions."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .task_context import context_similarity

_TRANSPORT_DESIGNS = frozenset({"randomized_action_assignment", "complete_action_grid"})
_TRANSPORT_SCOPES = frozenset({"category", "global_admet"})
_MINIMUM_TRANSPORT_ENDPOINTS = 5
_MINIMUM_CONTEXT_EFFECTIVE_ENDPOINTS = 3.0
_MINIMUM_CONTEXT_MAX_SIMILARITY = 0.35


def _valid_sha256(value: Any) -> bool:
    return bool(
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _summarize_transport_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Summarize only prospectively transport-eligible endpoint outcomes."""

    supported = sum(row["status"] == "supported" for row in rows)
    refuted = sum(row["status"] == "refuted" for row in rows)
    inconclusive = len(rows) - supported - refuted
    blockers = []
    if len(rows) < _MINIMUM_TRANSPORT_ENDPOINTS:
        blockers.append("insufficient_transport_eligible_endpoints")
    observed_statuses = {row["status"] for row in rows}
    if len(observed_statuses) < 2:
        blockers.append("transport_outcome_variation_missing")
    protocols = {row.get("transport_protocol_sha256") for row in rows}
    if len(protocols) > 1:
        blockers.append("transport_protocol_mismatch")
    routing_eligible = not blockers
    posterior = (
        (2.0 + supported + 0.5 * inconclusive) / (4.0 + len(rows))
        if routing_eligible
        else 0.5
    )
    deltas = [
        float(row["mean_oriented_delta"])
        for row in rows
        if row.get("mean_oriented_delta") is not None
        and math.isfinite(float(row["mean_oriented_delta"]))
    ]
    return {
        "transport_eligible_observations": len(rows),
        "supported": supported,
        "refuted": refuted,
        "inconclusive": inconclusive,
        "posterior_support_probability": posterior,
        "mean_oriented_delta": (
            sum(deltas) / len(deltas) if routing_eligible and deltas else None
        ),
        "routing_eligible": routing_eligible,
        "negative_gate_confirmed": False,
        "routing_eligibility_blockers": tuple(blockers),
        "effect_multiplier": 0.5 + posterior if routing_eligible else 1.0,
    }


def _summarize_contextual_rows(
    rows: list[dict[str, Any]],
    target_context: dict[str, Any],
    context_registry: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    weighted = [
        (row, context_similarity(target_context, context_registry[row["task"]]))
        for row in rows
        if row.get("task") in context_registry
    ]
    weights = [weight for _, weight in weighted]
    weight_sum = sum(weights)
    weight_square_sum = sum(weight * weight for weight in weights)
    effective_endpoints = (
        weight_sum * weight_sum / weight_square_sum
        if weight_square_sum > 0
        else 0.0
    )
    supported = sum(
        weight for row, weight in weighted if row["status"] == "supported"
    )
    refuted = sum(
        weight for row, weight in weighted if row["status"] == "refuted"
    )
    inconclusive = weight_sum - supported - refuted
    blockers = []
    if len(weighted) < _MINIMUM_TRANSPORT_ENDPOINTS:
        blockers.append("insufficient_transport_eligible_endpoints")
    if effective_endpoints < _MINIMUM_CONTEXT_EFFECTIVE_ENDPOINTS:
        blockers.append("insufficient_context_effective_endpoints")
    max_similarity = max(weights, default=0.0)
    if max_similarity < _MINIMUM_CONTEXT_MAX_SIMILARITY:
        blockers.append("context_similarity_coverage_missing")
    statuses = {row["status"] for row, _ in weighted}
    if len(statuses) < 2:
        blockers.append("transport_outcome_variation_missing")
    protocols = {row.get("transport_protocol_sha256") for row, _ in weighted}
    if len(protocols) > 1:
        blockers.append("transport_protocol_mismatch")
    routing_eligible = not blockers
    posterior = (
        (2.0 + supported + 0.5 * inconclusive) / (4.0 + weight_sum)
        if routing_eligible
        else 0.5
    )
    deltas = [
        (float(row["mean_oriented_delta"]), weight)
        for row, weight in weighted
        if row.get("mean_oriented_delta") is not None
        and math.isfinite(float(row["mean_oriented_delta"]))
    ]
    delta_weight = sum(weight for _, weight in deltas)
    return {
        "transport_eligible_observations": len(weighted),
        "effective_transport_endpoints": effective_endpoints,
        "max_context_similarity": max_similarity,
        "supported": supported,
        "refuted": refuted,
        "inconclusive": inconclusive,
        "posterior_support_probability": posterior,
        "mean_oriented_delta": (
            sum(delta * weight for delta, weight in deltas) / delta_weight
            if routing_eligible and delta_weight > 0
            else None
        ),
        "routing_eligible": routing_eligible,
        "negative_gate_confirmed": False,
        "routing_eligibility_blockers": tuple(blockers),
        "context_weights": {
            row["task"]: weight for row, weight in sorted(weighted, key=lambda x: x[0]["task"])
        },
    }


def _transport_record(evidence: dict[str, Any]) -> dict[str, Any]:
    metadata = evidence.get("cross_task_transport")
    if metadata is None:
        return {
            "transport_eligible": False,
            "transport_assignment_design": None,
            "transport_protocol_sha256": None,
        }
    if not isinstance(metadata, Mapping):
        raise TypeError("cross_task_transport must be a mapping when present")
    design = metadata.get("assignment_design")
    protocol = metadata.get("protocol_sha256")
    scope = metadata.get("scope", "category")
    eligible = bool(
        metadata.get("prospective") is True
        and metadata.get("same_budget") is True
        and design in _TRANSPORT_DESIGNS
        and scope in _TRANSPORT_SCOPES
        and _valid_sha256(protocol)
    )
    return {
        "transport_eligible": eligible,
        "transport_assignment_design": design if design in _TRANSPORT_DESIGNS else None,
        "transport_protocol_sha256": protocol if _valid_sha256(protocol) else None,
        "transport_scope": scope if scope in _TRANSPORT_SCOPES else None,
    }


def _forbidden_test_paths(value: Any, path: str = "evidence") -> set[str]:
    """Find test-like keys at any depth in JSON-shaped action evidence."""

    forbidden: set[str] = set()
    if isinstance(value, Mapping):
        for key, child in value.items():
            key_text = str(key)
            child_path = f"{path}.{key_text}"
            if "test" in key_text.lower():
                forbidden.add(child_path)
            forbidden.update(_forbidden_test_paths(child, child_path))
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for index, child in enumerate(value):
            forbidden.update(_forbidden_test_paths(child, f"{path}[{index}]"))
    return forbidden


class ResearchOutcomeMemory:
    """Store only validation-side action outcomes; test fields are forbidden."""

    def __init__(self, path: Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def append(
        self,
        *,
        task: str,
        category: str,
        evidence: dict[str, Any],
    ) -> dict[str, Any]:
        forbidden = _forbidden_test_paths(evidence)
        if forbidden:
            raise ValueError(
                f"Test-derived action evidence is forbidden: {sorted(forbidden)}"
            )
        base = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "task": task,
            "category": category,
            "action_id": evidence["action_id"],
            "axis": evidence["axis"],
            "status": evidence["status"],
            "mean_oriented_delta": evidence.get("mean_oriented_delta"),
            "delta_std": evidence.get("delta_std"),
            "paired_seeds": len(evidence.get("oriented_seed_deltas", [])),
            "single_axis_verified": bool(evidence.get("single_axis_verified", False)),
            "evidence_protocol": "discovery_only_matched_action_v1",
            **_transport_record(evidence),
        }
        digest = hashlib.sha256(
            json.dumps(base, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        record = {**base, "record_sha256": digest}
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, sort_keys=True) + "\n")
        return record

    def records(self) -> list[dict[str, Any]]:
        if not self.path.exists():
            return []
        return [
            json.loads(line)
            for line in self.path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]

    def priors(
        self,
        category: str,
        *,
        context_signature: dict[str, Any] | None = None,
        context_registry: dict[str, dict[str, Any]] | None = None,
    ) -> dict[str, dict[str, Any]]:
        """Return audited priors; only transport-eligible evidence may route.

        Ordinary selected-action outcomes remain useful observational records,
        but cannot change future action utilities.  Routing requires a
        prospective randomized assignment or complete action grid, a shared
        budget/protocol, at least five unique endpoints, and outcome variation.
        """
        if (context_signature is None) != (context_registry is None):
            raise ValueError(
                "context_signature and context_registry must be supplied together"
            )
        rows = [
            row
            for row in self.records()
            if (
                row.get("category") == category
                or row.get("transport_scope") == "global_admet"
            )
            and row.get("single_axis_verified")
            and row.get("status") in {"supported", "refuted", "inconclusive"}
        ]
        # Re-running the same task must not create pseudoreplication.  Keep the
        # latest outcome per task/action for calibration while retaining the
        # complete append-only audit trail on disk.
        latest: dict[tuple[str, str], dict[str, Any]] = {}
        for row in rows:
            latest[(row["task"], row["action_id"])] = row
        rows = list(latest.values())
        action_ids = sorted({row["action_id"] for row in rows})
        result: dict[str, dict[str, Any]] = {}
        for action_id in action_ids:
            selected = [row for row in rows if row["action_id"] == action_id]
            observational_supported = sum(
                row["status"] == "supported" for row in selected
            )
            observational_refuted = sum(row["status"] == "refuted" for row in selected)
            observational_inconclusive = (
                len(selected) - observational_supported - observational_refuted
            )
            observational_posterior = (
                2.0 + observational_supported + 0.5 * observational_inconclusive
            ) / (4.0 + len(selected))
            observational_deltas = [
                float(row["mean_oriented_delta"])
                for row in selected
                if row.get("mean_oriented_delta") is not None
                and math.isfinite(float(row["mean_oriented_delta"]))
            ]
            eligible_rows = [
                row for row in selected if row.get("transport_eligible") is True
            ]
            transport = _summarize_transport_rows(eligible_rows)
            if context_signature is not None and context_registry is not None:
                contextual = _summarize_contextual_rows(
                    eligible_rows, context_signature, context_registry
                )
                contextual_mode = "kernel_context_matched_transport"
            else:
                contextual_rows = [
                    row
                    for row in eligible_rows
                    if row.get("category") == category
                    and row.get("transport_scope") == "category"
                ]
                contextual = _summarize_transport_rows(contextual_rows)
                contextual_mode = "exact_category_transport"
            result[action_id] = {
                "category": category,
                "observations": len(selected),
                "observational_observations": len(selected),
                "observational_posterior_support_probability": observational_posterior,
                "observational_mean_oriented_delta": (
                    sum(observational_deltas) / len(observational_deltas)
                    if observational_deltas
                    else None
                ),
                **transport,
                "contextual_transport_eligible_observations": contextual[
                    "transport_eligible_observations"
                ],
                "contextual_supported": contextual["supported"],
                "contextual_refuted": contextual["refuted"],
                "contextual_inconclusive": contextual["inconclusive"],
                "contextual_posterior_support_probability": contextual[
                    "posterior_support_probability"
                ],
                "contextual_mean_oriented_delta": contextual[
                    "mean_oriented_delta"
                ],
                "contextual_routing_eligible": contextual["routing_eligible"],
                "contextual_negative_gate_confirmed": contextual[
                    "negative_gate_confirmed"
                ],
                "contextual_routing_eligibility_blockers": contextual[
                    "routing_eligibility_blockers"
                ],
                "contextual_evidence_mode": contextual_mode,
                "contextual_effective_transport_endpoints": contextual.get(
                    "effective_transport_endpoints"
                ),
                "contextual_max_similarity": contextual.get(
                    "max_context_similarity"
                ),
                "contextual_weights": contextual.get("context_weights", {}),
            }
        return result
