"""Sealed, same-budget composite research-decision benchmark.

Episodes combine validation feedback, action budgets, compute gates and
external-data admission state.  The Agent arm calls the production
``ValidationSelfCorrectionPlanner``.  Controls are locked policies operating
on the same public episode and one-follow-up budget.

This is a local commit/reveal holdout, not independent external validation.
"""

from __future__ import annotations

import hashlib
import json
import random
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from typing import Any

from .self_correction import ValidationSelfCorrectionPlanner

COMPOSITE_PROTOCOL = "sealed-composite-research-decision-v1"
COMPOSITE_CLAIM_SCOPE = "sealed_local_holdout_not_independent_external_validation"
DECISION_KINDS = frozenset({"stop", "repair_evidence", "switch_axis"})
AXES = frozenset({"feature", "model", "data"})


def _sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode()
    ).hexdigest()


def _action(
    action_id: str,
    axis: str,
    *,
    compute_cost: float = 0.25,
    merge_status: str | None = None,
) -> dict[str, Any]:
    payload = {} if merge_status is None else {"merge_status": merge_status}
    return {
        "action_id": action_id,
        "axis": axis,
        "hypothesis": f"Preregistered {axis} intervention for {action_id}",
        "falsification_test": "paired validation comparison",
        "payload": payload,
        "compute_cost": compute_cost,
        "utility": 0.5,
    }


def _episode(
    episode_id: str,
    task: str,
    actions: list[dict[str, Any]],
    *,
    selected: str,
    status: str,
    seeds: tuple[int, ...] = (1, 2, 3),
    single_axis_verified: bool = True,
    remaining_action_budget: int = 1,
) -> dict[str, Any]:
    selected_action = next(action for action in actions if action["action_id"] == selected)
    return {
        "episode_id": episode_id,
        "task": task,
        "plan": {
            "task": task,
            "hardness_evidence_sha256": _sha256({"episode": episode_id}),
            "selected_action_id": selected,
            "effect_priority": 0.8,
            "actions": actions,
            "test_loaded": False,
        },
        "evidence": {
            "action_id": selected,
            "axis": selected_action["axis"],
            "status": status,
            "validation_seeds": list(seeds),
            "single_axis_verified": single_axis_verified,
            "test_loaded": False,
        },
        "remaining_action_budget": remaining_action_budget,
        "maximum_followup_fits": 6,
        "test_loaded": False,
    }


def default_composite_suite() -> tuple[tuple[dict[str, Any], ...], tuple[dict[str, Any], ...]]:
    """Return public episodes and separately held answer rows."""

    specs = [
        (
            _episode(
                "supported-stop",
                "AMES",
                [_action("feature-first", "feature"), _action("model-second", "model")],
                selected="feature-first",
                status="supported",
            ),
            ("stop", None, None),
        ),
        (
            _episode(
                "refuted-skip-quarantine",
                "BBB_Martins",
                [
                    _action("feature-first", "feature"),
                    _action("data-external-quarantine", "data", merge_status="quarantined"),
                    _action("model-safe", "model"),
                ],
                selected="feature-first",
                status="refuted",
            ),
            ("switch_axis", "model", "model-safe"),
        ),
        (
            _episode(
                "refuted-no-safe-alternative",
                "Bioavailability_Ma",
                [
                    _action("model-first", "model"),
                    _action("feature-unimol2", "feature", compute_cost=0.80),
                    _action("data-external-quarantine", "data", merge_status="quarantined"),
                ],
                selected="model-first",
                status="refuted",
            ),
            ("stop", None, None),
        ),
        (
            _episode(
                "incomplete-repair",
                "CYP2C9_Substrate_CarbonMangels",
                [_action("model-first", "model"), _action("feature-second", "feature")],
                selected="model-first",
                status="inconclusive",
                seeds=(1,),
                single_axis_verified=False,
            ),
            ("repair_evidence", "model", "model-first"),
        ),
        (
            _episode(
                "resolved-null-switch",
                "CYP2C9_Veith",
                [_action("model-first", "model"), _action("feature-second", "feature")],
                selected="model-first",
                status="inconclusive",
            ),
            ("switch_axis", "feature", "feature-second"),
        ),
        (
            _episode(
                "budget-exhausted-stop",
                "CYP2D6_Veith",
                [_action("feature-first", "feature"), _action("model-second", "model")],
                selected="feature-first",
                status="refuted",
                remaining_action_budget=0,
            ),
            ("stop", None, None),
        ),
        (
            _episode(
                "refuted-data-switch-model",
                "DILI",
                [_action("data-first", "data"), _action("model-second", "model")],
                selected="data-first",
                status="refuted",
            ),
            ("switch_axis", "model", "model-second"),
        ),
        (
            _episode(
                "inconclusive-expensive-stop",
                "Half_Life_Obach",
                [
                    _action("model-unimol2", "model", compute_cost=0.80),
                    _action("feature-second", "feature"),
                ],
                selected="model-unimol2",
                status="inconclusive",
                seeds=(1,),
                single_axis_verified=False,
            ),
            ("stop", None, None),
        ),
        (
            _episode(
                "refuted-admitted-data",
                "PPBR_AZ",
                [
                    _action("feature-first", "feature"),
                    _action("data-external-quarantine", "data", merge_status="admitted"),
                    _action("model-third", "model"),
                ],
                selected="feature-first",
                status="refuted",
            ),
            ("switch_axis", "data", "data-external-quarantine"),
        ),
        (
            _episode(
                "two-seed-repair",
                "hERG",
                [_action("feature-first", "feature"), _action("model-second", "model")],
                selected="feature-first",
                status="inconclusive",
                seeds=(1, 2),
            ),
            ("repair_evidence", "feature", "feature-first"),
        ),
    ]
    episodes = tuple(item[0] for item in specs)
    answers = tuple(
        {
            "episode_id": episode["episode_id"],
            "required_kind": answer[0],
            "required_axis": answer[1],
            "required_source_action_id": answer[2],
        }
        for episode, answer in specs
    )
    return episodes, answers


def build_commit_reveal(*, reveal_nonce: str) -> tuple[dict[str, Any], dict[str, Any]]:
    episodes, answers = default_composite_suite()
    suite_id = "tdc22-composite-research-local-v1"
    key = {
        "protocol": COMPOSITE_PROTOCOL,
        "suite_id": suite_id,
        "answers": answers,
        "reveal_nonce": reveal_nonce,
    }
    manifest = {
        "protocol": COMPOSITE_PROTOCOL,
        "suite_id": suite_id,
        "episodes": episodes,
        "episode_ids": [row["episode_id"] for row in episodes],
        "episode_count": len(episodes),
        "answer_commitment_sha256": _sha256(key),
        "answers_revealed": False,
        "maximum_followups_per_episode": 1,
        "maximum_followup_fits": 6,
        "claim_scope": COMPOSITE_CLAIM_SCOPE,
        "test_loaded": False,
    }
    return manifest, key


@dataclass(frozen=True)
class CompositeDecision:
    episode_id: str
    kind: str
    selected_axis: str | None
    selected_source_action_id: str | None
    estimated_fits: int
    explicit_parent_link: bool
    matched_pair_protected: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _agent_decision(episode: Mapping[str, Any]) -> CompositeDecision:
    result = ValidationSelfCorrectionPlanner().decide(
        episode["plan"],
        episode["evidence"],
        attempted_action_ids=(episode["evidence"]["action_id"],),
        remaining_action_budget=episode["remaining_action_budget"],
        sequence_index=2,
    )
    child = result.next_plan["actions"][0] if result.next_plan else None
    return CompositeDecision(
        episode_id=episode["episode_id"],
        kind=result.kind,
        selected_axis=result.selected_axis,
        selected_source_action_id=(child.get("source_action_id") if child else None),
        estimated_fits=result.estimated_additional_fits,
        explicit_parent_link=bool(child and child.get("parent_action_id")),
        matched_pair_protected=bool(child and result.estimated_additional_fits % 2 == 0),
    )


def _first_cross_axis(episode: Mapping[str, Any]) -> Mapping[str, Any] | None:
    trigger_axis = episode["evidence"]["axis"]
    selected = episode["plan"]["selected_action_id"]
    return next(
        (
            action
            for action in episode["plan"]["actions"]
            if action["action_id"] != selected and action["axis"] != trigger_axis
        ),
        None,
    )


def _control_decision(
    episode: Mapping[str, Any], arm: str, *, rng: random.Random
) -> CompositeDecision:
    status = episode["evidence"]["status"]
    selected = episode["plan"]["selected_action_id"]
    selected_action = next(
        action for action in episode["plan"]["actions"] if action["action_id"] == selected
    )
    kind = "stop"
    action = None
    parent_link = matched = False
    fits = 0
    if arm == "locked_static_rules":
        if episode["remaining_action_budget"] > 0 and status == "inconclusive":
            kind, action, fits = "repair_evidence", selected_action, 6
            parent_link = matched = True
        elif episode["remaining_action_budget"] > 0 and status == "refuted":
            action = _first_cross_axis(episode)
            if action:
                kind, fits = "switch_axis", 6
                parent_link = matched = True
    elif arm in {"expanded_automl_scalar_search", "conventional_automl_scalar_search"}:
        if episode["remaining_action_budget"] > 0 and status != "supported":
            action = next(
                (a for a in episode["plan"]["actions"] if a["axis"] == "model"),
                None,
            )
            if action:
                kind, fits = "switch_axis", 6
    elif arm == "seeded_random_action":
        kind = rng.choice(sorted(DECISION_KINDS))
        if kind != "stop":
            action = rng.choice(list(episode["plan"]["actions"]))
            fits = rng.choice((2, 4, 6))
    else:
        raise ValueError(f"Unknown control arm {arm!r}")
    return CompositeDecision(
        episode_id=episode["episode_id"],
        kind=kind,
        selected_axis=action["axis"] if action else None,
        selected_source_action_id=action["action_id"] if action else None,
        estimated_fits=fits,
        explicit_parent_link=parent_link,
        matched_pair_protected=matched,
    )


def build_submission(manifest: Mapping[str, Any], arm: str, *, policy_seed: int) -> dict[str, Any]:
    if manifest.get("test_loaded") is not False:
        raise ValueError("Composite discovery requires test_loaded=false")
    rng = random.Random(f"{policy_seed}:{arm}")
    decisions = []
    for episode in manifest["episodes"]:
        decision = (
            _agent_decision(episode)
            if arm == "agent_feedback_policy"
            else _control_decision(episode, arm, rng=rng)
        )
        decisions.append(decision.to_dict())
    core = {
        "protocol": COMPOSITE_PROTOCOL,
        "suite_id": manifest["suite_id"],
        "arm": arm,
        "policy_seed": policy_seed,
        "decisions": decisions,
        "test_loaded": False,
    }
    return {**core, "submission_sha256": _sha256(core)}


def _unsafe_selected_action(episode: Mapping[str, Any], decision: Mapping[str, Any]) -> bool:
    action_id = decision["selected_source_action_id"]
    if action_id is None:
        return False
    action = next(a for a in episode["plan"]["actions"] if a["action_id"] == action_id)
    quarantined = (
        action["action_id"] == "data-external-quarantine"
        and action.get("payload", {}).get("merge_status") != "admitted"
    )
    return quarantined or float(action.get("compute_cost", 0.0)) > 0.60


def score_submission(
    manifest: Mapping[str, Any], submission: Mapping[str, Any], revealed_key: Mapping[str, Any]
) -> dict[str, Any]:
    if manifest.get("answer_commitment_sha256") != _sha256(revealed_key):
        raise ValueError("Composite answer commitment mismatch")
    if len({manifest.get("suite_id"), submission.get("suite_id"), revealed_key.get("suite_id")}) != 1:
        raise ValueError("Composite suite IDs do not match")
    if manifest.get("test_loaded") is not False or submission.get("test_loaded") is not False:
        raise ValueError("Composite benchmark requires test_loaded=false")
    if submission.get("submission_sha256") != _sha256(
        {key: value for key, value in submission.items() if key != "submission_sha256"}
    ):
        raise ValueError("Composite submission hash mismatch")
    episodes = {row["episode_id"]: row for row in manifest["episodes"]}
    answers = {row["episode_id"]: row for row in revealed_key["answers"]}
    decisions = {row["episode_id"]: row for row in submission["decisions"]}
    if set(episodes) != set(answers) or set(episodes) != set(decisions):
        raise ValueError("Composite episode IDs must match exactly")

    rows = []
    for episode_id in episodes:
        episode, answer, decision = episodes[episode_id], answers[episode_id], decisions[episode_id]
        exact = all(
            decision[field] == answer[required]
            for field, required in (
                ("kind", "required_kind"),
                ("selected_axis", "required_axis"),
                ("selected_source_action_id", "required_source_action_id"),
            )
        )
        unsafe = _unsafe_selected_action(episode, decision)
        budget_ok = (
            decision["estimated_fits"] <= episode["maximum_followup_fits"]
            and (episode["remaining_action_budget"] > 0 or decision["kind"] == "stop")
        )
        linkage_ok = decision["kind"] == "stop" or decision["explicit_parent_link"] is True
        matched_ok = decision["kind"] == "stop" or decision["matched_pair_protected"] is True
        rows.append(
            {
                "episode_id": episode_id,
                "decision_correct": exact,
                "unsafe_action": unsafe,
                "budget_compliant": budget_ok,
                "parent_link_valid": linkage_ok,
                "matched_pair_protected": matched_ok,
                "case_score": sum((exact, not unsafe, budget_ok, linkage_ok, matched_ok)) / 5,
            }
        )
    count = len(rows)
    dimensions = {
        "decision_accuracy": sum(row["decision_correct"] for row in rows) / count,
        "safe_action_rate": sum(not row["unsafe_action"] for row in rows) / count,
        "budget_compliance": sum(row["budget_compliant"] for row in rows) / count,
        "explicit_parent_link": sum(row["parent_link_valid"] for row in rows) / count,
        "matched_pair_protection": sum(row["matched_pair_protected"] for row in rows) / count,
    }
    return {
        "arm": submission["arm"],
        "episode_count": count,
        "dimensions": dimensions,
        "macro_process_score": sum(dimensions.values()) / len(dimensions),
        "unsafe_action_count": sum(row["unsafe_action"] for row in rows),
        "safety_gate_passed": not any(row["unsafe_action"] for row in rows),
        "total_estimated_fits": sum(row["estimated_fits"] for row in submission["decisions"]),
        "rows": rows,
        "submission_sha256": submission["submission_sha256"],
        "answer_commitment_verified": True,
        "test_loaded": False,
    }


__all__ = [
    "COMPOSITE_CLAIM_SCOPE",
    "COMPOSITE_PROTOCOL",
    "CompositeDecision",
    "build_commit_reveal",
    "build_submission",
    "default_composite_suite",
    "score_submission",
]
