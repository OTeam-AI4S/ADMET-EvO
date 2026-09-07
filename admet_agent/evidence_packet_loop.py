"""Closed-loop selection and snapshots for the Evidence Packet data axis."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .tasks import TASKS

LOOP_PROTOCOL = "admet-evidence-packet-closed-loop-v4.1"
CANONICAL_TASKS = tuple(spec.name for spec in TASKS.values())


def _canonical_sha256(payload: dict[str, Any]) -> str:
    encoded = json.dumps(
        payload, allow_nan=False, separators=(",", ":"), sort_keys=True
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _validated_scores(scores: Mapping[str, float]) -> dict[str, float]:
    if set(scores) != set(CANONICAL_TASKS):
        missing = sorted(set(CANONICAL_TASKS) - set(scores))
        extra = sorted(set(scores) - set(CANONICAL_TASKS))
        raise ValueError(
            f"scores must contain exactly all 22 canonical tasks; missing={missing}, extra={extra}"
        )
    output = {task: float(scores[task]) for task in CANONICAL_TASKS}
    if not all(math.isfinite(value) for value in output.values()):
        raise ValueError("all normalized task scores must be finite")
    return output


def weakest_tasks(scores: Mapping[str, float], *, count: int = 3) -> tuple[str, ...]:
    """Choose the 2-3 lowest normalized-improvement tasks deterministically."""

    if count not in {2, 3}:
        raise ValueError("closed-loop acquisition targets exactly 2 or 3 weak tasks")
    validated = _validated_scores(scores)
    return tuple(
        task
        for task, _ in sorted(validated.items(), key=lambda item: (item[1], item[0]))[
            :count
        ]
    )


def _validated_target_tasks(tasks: Sequence[str]) -> tuple[str, ...]:
    targets = tuple(str(task) for task in tasks)
    if len(targets) not in {2, 3} or len(set(targets)) != len(targets):
        raise ValueError("acquisition plan must contain exactly 2 or 3 unique tasks")
    unknown = sorted(set(targets) - set(CANONICAL_TASKS))
    if unknown:
        raise ValueError(f"acquisition plan contains unknown tasks: {unknown}")
    return targets


def _validate_single_packet_mutation(
    incumbent: Sequence[str], challenger: Sequence[str]
) -> dict[str, str | None]:
    old = set(incumbent)
    new = set(challenger)
    if len(old) != len(tuple(incumbent)) or len(new) != len(tuple(challenger)):
        raise ValueError("packet snapshots may not contain duplicate packet IDs")
    added = sorted(new - old)
    removed = sorted(old - new)
    if len(added) == 1 and not removed:
        return {"action": "add", "added": added[0], "removed": None}
    if len(removed) == 1 and not added:
        return {"action": "remove", "added": None, "removed": removed[0]}
    if len(added) == 1 and len(removed) == 1:
        return {"action": "replace", "added": added[0], "removed": removed[0]}
    raise ValueError("each closed-loop trial must add, remove, or replace one packet")


@dataclass(frozen=True)
class TrialDecision:
    keep: bool
    incumbent_mean: float
    challenger_mean: float
    aggregate_increment: float
    mutation: dict[str, str | None]
    targeted_weak_task: bool
    target_tasks: tuple[str, ...]
    reason: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def evaluate_packet_trial(
    *,
    incumbent_packet_ids: Sequence[str],
    challenger_packet_ids: Sequence[str],
    incumbent_scores: Mapping[str, float],
    challenger_scores: Mapping[str, float],
    packet_task_by_id: Mapping[str, str],
    weak_task_count: int = 3,
    acquisition_target_tasks: Sequence[str] | None = None,
) -> TrialDecision:
    """Keep only a strict all-22 mean normalized-improvement increase."""

    current = _validated_scores(incumbent_scores)
    challenger = _validated_scores(challenger_scores)
    mutation = _validate_single_packet_mutation(
        incumbent_packet_ids, challenger_packet_ids
    )
    targets = (
        weakest_tasks(current, count=weak_task_count)
        if acquisition_target_tasks is None
        else _validated_target_tasks(acquisition_target_tasks)
    )
    changed = mutation["added"] or mutation["removed"]
    if changed not in packet_task_by_id:
        raise ValueError("changed packet is missing a task binding")
    changed_task = packet_task_by_id[str(changed)]
    if changed_task not in CANONICAL_TASKS:
        raise ValueError("changed packet is bound to an unknown task")
    targeted = changed_task in targets
    if mutation["action"] in {"add", "replace"} and not targeted:
        raise ValueError(
            "new or replacement Evidence Packets must target a current bottom-2/3 task"
        )
    incumbent_mean = sum(current.values()) / len(current)
    challenger_mean = sum(challenger.values()) / len(challenger)
    increment = challenger_mean - incumbent_mean
    keep = increment > 0.0
    return TrialDecision(
        keep=keep,
        incumbent_mean=incumbent_mean,
        challenger_mean=challenger_mean,
        aggregate_increment=increment,
        mutation=mutation,
        targeted_weak_task=targeted,
        target_tasks=targets,
        reason=(
            "strict_all22_aggregate_improvement"
            if keep
            else "no_strict_all22_aggregate_improvement"
        ),
    )


class EvidencePacketLoopStore:
    """Atomic JSON snapshot store; rejected trials leave the incumbent unchanged."""

    def __init__(self, path: str | Path):
        self.path = Path(path)

    def initialize(
        self,
        *,
        packet_ids: Sequence[str],
        scores: Mapping[str, float],
        overwrite: bool = False,
    ) -> dict[str, Any]:
        if self.path.exists() and not overwrite:
            raise FileExistsError(f"closed-loop state already exists: {self.path}")
        state = {
            "protocol": LOOP_PROTOCOL,
            "iteration": 0,
            "incumbent": self._snapshot(packet_ids, _validated_scores(scores)),
            "history": [],
            "test_labels_loaded": False,
            "legacy_bulk_pool_used": False,
        }
        self._write(state)
        return state

    def load(self) -> dict[str, Any]:
        state = json.loads(self.path.read_text(encoding="utf-8"))
        if state.get("protocol") != LOOP_PROTOCOL:
            raise ValueError("closed-loop state protocol mismatch")
        _validated_scores(state["incumbent"]["scores"])
        expected = self._snapshot(
            state["incumbent"]["packet_ids"], state["incumbent"]["scores"]
        )["snapshot_sha256"]
        if state["incumbent"].get("snapshot_sha256") != expected:
            raise ValueError("closed-loop incumbent snapshot hash mismatch")
        return state

    def submit_trial(
        self,
        *,
        challenger_packet_ids: Sequence[str],
        challenger_scores: Mapping[str, float],
        packet_task_by_id: Mapping[str, str],
        weak_task_count: int = 3,
    ) -> TrialDecision:
        del challenger_packet_ids, challenger_scores, packet_task_by_id, weak_task_count
        raise RuntimeError(
            "manual score submission is disabled; use "
            "admet_agent.evidence_packet_closed_loop.EvidencePacketClosedLoop"
        )

    @staticmethod
    def _snapshot(
        packet_ids: Sequence[str], scores: Mapping[str, float]
    ) -> dict[str, Any]:
        core = {
            "packet_ids": sorted(packet_ids),
            "scores": _validated_scores(scores),
        }
        return {
            **core,
            "mean_normalized_improvement": sum(core["scores"].values())
            / len(core["scores"]),
            "snapshot_sha256": _canonical_sha256(core),
        }

    def _write(self, state: dict[str, Any]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_suffix(self.path.suffix + ".tmp")
        temporary.write_text(
            json.dumps(state, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        temporary.replace(self.path)


__all__ = [
    "CANONICAL_TASKS",
    "LOOP_PROTOCOL",
    "EvidencePacketLoopStore",
    "TrialDecision",
    "evaluate_packet_trial",
    "weakest_tasks",
]
