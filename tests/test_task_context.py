import hashlib
import json

import pytest

from admet_agent.action_memory import ResearchOutcomeMemory
from admet_agent.task_context import (
    CONTEXT_VERSION,
    context_distance,
    context_similarity,
    load_context_registry,
)


def _context(
    task: str,
    *,
    task_type: str = "classification",
    metric: str = "auroc",
    category: str = "absorption",
    hardness: float = 0.4,
):
    return {
        "context_version": CONTEXT_VERSION,
        "task": task,
        "task_type": task_type,
        "metric": metric,
        "category": category,
        "hardness": {
            "scarcity": hardness,
            "imbalance": hardness,
            "domain_shift": hardness,
            "local_roughness": hardness,
            "data_quality": hardness,
            "split_leakage_risk": hardness,
        },
        "hardness_evidence_sha256": "a" * 64,
        "discovery_only": True,
    }


def test_context_distance_is_symmetric_bounded_and_identity_preserving():
    left = _context("left")
    right = _context("right", hardness=0.6)
    assert context_distance(left, left) == pytest.approx(0.0)
    assert context_similarity(left, left) == pytest.approx(1.0)
    assert context_distance(left, right) == pytest.approx(
        context_distance(right, left)
    )
    assert 0.0 < context_similarity(left, right) < 1.0


def test_contextual_memory_uses_effective_sample_and_similarity_gates(tmp_path):
    memory = ResearchOutcomeMemory(tmp_path / "memory.jsonl")
    protocol = "b" * 64
    statuses = ["supported", "supported", "supported", "refuted", "refuted"]
    registry = {}
    for index, status in enumerate(statuses):
        task = f"sentinel-{index}"
        registry[task] = _context(task, hardness=0.35 + 0.025 * index)
        memory.append(
            task=task,
            category="absorption",
            evidence={
                "action_id": "feature-domain-shift",
                "axis": "feature",
                "status": status,
                "single_axis_verified": True,
                "mean_oriented_delta": 0.01 if status == "supported" else -0.01,
                "cross_task_transport": {
                    "prospective": True,
                    "same_budget": True,
                    "assignment_design": "complete_action_grid",
                    "protocol_sha256": protocol,
                    "scope": "global_admet",
                },
            },
        )
    near = memory.priors(
        "toxicity",
        context_signature=_context("target", hardness=0.4),
        context_registry=registry,
    )["feature-domain-shift"]
    assert near["contextual_routing_eligible"] is True
    assert near["contextual_evidence_mode"] == "kernel_context_matched_transport"
    assert near["contextual_effective_transport_endpoints"] >= 3.0
    assert near["contextual_max_similarity"] >= 0.35
    assert set(near["contextual_weights"]) == set(registry)

    far = memory.priors(
        "toxicity",
        context_signature=_context(
            "far",
            task_type="regression",
            metric="mae",
            category="toxicity",
            hardness=1.0,
        ),
        context_registry=registry,
    )["feature-domain-shift"]
    assert far["contextual_routing_eligible"] is False
    assert "context_similarity_coverage_missing" in far[
        "contextual_routing_eligibility_blockers"
    ]


def test_context_registry_loader_rejects_tampering(tmp_path):
    body = {"test_loaded": False, "contexts": {}}
    payload = {
        **body,
        "sha256": hashlib.sha256(
            json.dumps(body, sort_keys=True).encode("utf-8")
        ).hexdigest(),
    }
    path = tmp_path / "contexts.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    assert load_context_registry(path)["sha256"] == payload["sha256"]
    payload["contexts"]["tampered"] = {}
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="SHA-256"):
        load_context_registry(path)
