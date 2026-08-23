from __future__ import annotations

import hashlib
import json

import pytest

from admet_agent.ecvoi_v2_memory import load_matched_effect_memory
from admet_agent.task_context import CONTEXT_VERSION, HARDNESS_DIMENSIONS


def _context(task: str, offset: float) -> dict:
    return {
        "context_version": CONTEXT_VERSION,
        "task": task,
        "task_type": "classification",
        "metric": "auroc",
        "category": "distribution",
        "hardness": {name: offset for name in HARDNESS_DIMENSIONS},
        "hardness_evidence_sha256": "a" * 64,
        "discovery_only": True,
    }


def _write_memory(path, *, test_derived=False):
    rows = []
    for task, action, delta in (
        ("BBB_Martins", "feature-domain-shift", 0.01),
        ("HIA_Hou", "data-quality-filter", -0.01),
    ):
        base = {
            "endpoint_id": task,
            "pair_id": f"pair-{task}",
            "action_id": action,
            "native_oriented_delta": delta,
            "native_standard_error": 0.002,
            "effect_standardization_scale": 0.005,
            "single_axis_verified": True,
            "complete_pair": True,
            "transport_eligible": True,
            "test_derived": test_derived,
            "leaderboard_derived": False,
        }
        rendered = json.dumps(base, sort_keys=True, separators=(",", ":")).encode()
        base["observation_sha256"] = hashlib.sha256(rendered).hexdigest()
        rows.append(base)
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n")


def test_loads_contextual_memory_and_marks_endpoint_local(tmp_path):
    path = tmp_path / "memory.jsonl"
    _write_memory(path)
    contexts = {
        "BBB_Martins": _context("BBB_Martins", 0.1),
        "HIA_Hou": _context("HIA_Hou", 0.7),
    }
    observations, audit = load_matched_effect_memory(
        path, target_endpoint="BBB_Martins", contexts=contexts
    )
    assert len(observations) == 2
    assert observations[0].endpoint_local is True
    assert observations[1].context_similarity < 1
    assert audit["test_loaded"] is False


def test_memory_and_context_placebos_are_deterministic(tmp_path):
    path = tmp_path / "memory.jsonl"
    _write_memory(path)
    contexts = {
        "BBB_Martins": _context("BBB_Martins", 0.1),
        "HIA_Hou": _context("HIA_Hou", 0.7),
    }
    first, _ = load_matched_effect_memory(
        path,
        target_endpoint="BBB_Martins",
        contexts=contexts,
        memory_mode="shuffled",
        context_mode="shuffled",
    )
    second, _ = load_matched_effect_memory(
        path,
        target_endpoint="BBB_Martins",
        contexts=contexts,
        memory_mode="shuffled",
        context_mode="shuffled",
    )
    assert first == second


def test_rejects_test_derived_memory(tmp_path):
    path = tmp_path / "memory.jsonl"
    _write_memory(path, test_derived=True)
    with pytest.raises(RuntimeError, match="discovery-only"):
        load_matched_effect_memory(
            path,
            target_endpoint="BBB_Martins",
            contexts={
                "BBB_Martins": _context("BBB_Martins", 0.1),
                "HIA_Hou": _context("HIA_Hou", 0.7),
            },
        )
