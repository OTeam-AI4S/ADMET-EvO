"""Load and placebo-transform frozen EC-VOI v2 matched-effect memory."""

from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path
from typing import Any, Literal

from .research_policy_v2 import MatchedEffectObservation
from .task_context import context_similarity

MemoryMode = Literal["correct", "shuffled", "none"]
ContextMode = Literal["correct", "shuffled", "global"]


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_matched_effect_memory(
    path: Path,
    *,
    target_endpoint: str,
    contexts: dict[str, dict[str, Any]],
    expected_sha256: str | None = None,
    memory_mode: MemoryMode = "correct",
    context_mode: ContextMode = "correct",
    placebo_seed: int = 20260821,
) -> tuple[tuple[MatchedEffectObservation, ...], dict[str, Any]]:
    """Load discovery-only effects and apply only preregistered placebo modes."""

    if memory_mode not in {"correct", "shuffled", "none"}:
        raise ValueError("Unknown EC-VOI v2 memory mode")
    if context_mode not in {"correct", "shuffled", "global"}:
        raise ValueError("Unknown EC-VOI v2 context mode")
    path = path.resolve()
    actual_sha256 = _sha256(path)
    if expected_sha256 is not None and actual_sha256 != expected_sha256:
        raise RuntimeError("EC-VOI v2 matched-effect memory hash mismatch")
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    for row in rows:
        claimed = row.get("observation_sha256")
        base = {key: value for key, value in row.items() if key != "observation_sha256"}
        actual = hashlib.sha256(_canonical(base)).hexdigest()
        if claimed != actual:
            raise RuntimeError("EC-VOI v2 observation hash mismatch")
        if row.get("test_derived") is not False or row.get(
            "leaderboard_derived"
        ) is not False:
            raise RuntimeError("Matched-effect memory must be discovery-only")
    if target_endpoint not in contexts:
        raise ValueError(f"Missing target context for {target_endpoint}")
    if memory_mode == "none":
        rows = []

    rng = random.Random(placebo_seed)
    action_ids = [str(row["action_id"]) for row in rows]
    shuffled_actions = list(action_ids)
    rng.shuffle(shuffled_actions)
    endpoints = sorted({str(row["endpoint_id"]) for row in rows})
    shuffled_endpoints = list(endpoints)
    rng.shuffle(shuffled_endpoints)
    endpoint_placebo = dict(zip(endpoints, shuffled_endpoints, strict=True))

    observations = []
    similarities = []
    for index, row in enumerate(rows):
        source_endpoint = str(row["endpoint_id"])
        endpoint_local = source_endpoint == target_endpoint
        if endpoint_local or context_mode == "global":
            similarity = 1.0
        else:
            context_endpoint = (
                endpoint_placebo[source_endpoint]
                if context_mode == "shuffled"
                else source_endpoint
            )
            if context_endpoint not in contexts:
                raise ValueError(f"Missing source context for {context_endpoint}")
            similarity = context_similarity(
                contexts[target_endpoint], contexts[context_endpoint]
            )
        similarities.append(similarity)
        observations.append(
            MatchedEffectObservation(
                endpoint_id=source_endpoint,
                pair_id=str(row["pair_id"]),
                action_id=(
                    shuffled_actions[index]
                    if memory_mode == "shuffled"
                    else str(row["action_id"])
                ),
                native_oriented_delta=float(row["native_oriented_delta"]),
                native_standard_error=float(row["native_standard_error"]),
                effect_standardization_scale=float(
                    row["effect_standardization_scale"]
                ),
                context_similarity=float(similarity),
                endpoint_local=endpoint_local,
                transport_eligible=bool(row["transport_eligible"]),
                test_derived=False,
                leaderboard_derived=False,
                single_axis_verified=bool(row["single_axis_verified"]),
                complete_pair=bool(row["complete_pair"]),
                provenance_sha256=str(row["observation_sha256"]),
            )
        )
    audit = {
        "version": "ecvoi-v2-memory-load-v1",
        "path": str(path),
        "sha256": actual_sha256,
        "target_endpoint": target_endpoint,
        "records": len(observations),
        "memory_mode": memory_mode,
        "context_mode": context_mode,
        "placebo_seed": placebo_seed,
        "mean_context_similarity": (
            sum(similarities) / len(similarities) if similarities else None
        ),
        "test_loaded": False,
        "leaderboard_loaded": False,
    }
    audit["audit_sha256"] = hashlib.sha256(_canonical(audit)).hexdigest()
    return tuple(observations), audit


__all__ = ["load_matched_effect_memory"]
