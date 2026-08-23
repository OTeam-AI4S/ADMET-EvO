"""Label-blind routing replay over previously frozen HardnessCards."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from .action_memory import ResearchOutcomeMemory
from .endpoint_contracts import endpoint_contract
from .hardness import HardnessCard, HardnessDimension
from .research_loop import EffectFirstResearchPlanner
from .task_context import load_context_registry, task_context_signature
from .tasks import TASKS

REPLAY_POLICIES = (
    "legacy",
    "ecvoi",
    "ecvoi-no-context",
    "ecvoi-no-memory",
    "ecvoi-no-voi",
    "ecvoi-no-cost",
    "ecvoi-no-risk",
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _card(value: dict[str, Any]) -> HardnessCard:
    payload = dict(value)
    payload["dimensions"] = {
        name: HardnessDimension(**dimension)
        for name, dimension in payload["dimensions"].items()
    }
    payload["dominant_factors"] = tuple(payload["dominant_factors"])
    payload["recommended_actions"] = tuple(payload["recommended_actions"])
    return HardnessCard(**payload)


def replay_routing(
    diagnosis_run: Path,
    transport_report: Path,
    action_memory_snapshot: Path,
    action_context_registry: Path,
) -> dict[str, Any]:
    transport = json.loads(transport_report.read_text(encoding="utf-8"))
    if transport.get("test_loaded") is not False:
        raise ValueError("Routing replay requires a test-blind transport report")
    if not action_memory_snapshot.is_file():
        raise FileNotFoundError(
            f"Action-memory snapshot does not exist: {action_memory_snapshot}"
        )
    memory = ResearchOutcomeMemory(action_memory_snapshot)
    registry = load_context_registry(action_context_registry)
    paths = sorted(diagnosis_run.glob("tasks/*/diagnosis_seed_1.json"))
    diagnosis_manifest = {
        str(path.relative_to(diagnosis_run)): _sha256(path) for path in paths
    }
    diagnosis_sha256 = hashlib.sha256(
        json.dumps(diagnosis_manifest, sort_keys=True).encode("utf-8")
    ).hexdigest()
    packets = [json.loads(path.read_text(encoding="utf-8")) for path in paths]
    by_task = {packet["task"]: packet for packet in packets}
    expected = {item.name for item in TASKS.values()}
    if set(by_task) != expected:
        raise ValueError(
            "Routing replay requires exactly the canonical 22 diagnosis packets"
        )
    rows = []
    changes = {policy: 0 for policy in REPLAY_POLICIES}
    for task_name in sorted(by_task):
        card = _card(by_task[task_name]["hardness"])
        category = endpoint_contract(task_name).category
        priors = memory.priors(
            category,
            context_signature=task_context_signature(card.to_dict(), category),
            context_registry=registry["contexts"],
        )
        decisions = {}
        for policy in REPLAY_POLICIES:
            plan = EffectFirstResearchPlanner(policy).plan(
                card,
                effect_priority=0.80,
                action_priors=priors,
            )
            selected = next(
                action
                for action in plan["actions"]
                if action["action_id"] == plan["selected_action_id"]
            )
            decisions[policy] = {
                "selected_action_id": plan["selected_action_id"],
                "utility": selected["utility"],
                "evidence_source": selected.get("evidence_source", "legacy"),
            }
        legacy = decisions["legacy"]["selected_action_id"]
        for policy in REPLAY_POLICIES:
            changes[policy] += decisions[policy]["selected_action_id"] != legacy
        rows.append(
            {
                "task": task_name,
                "category": category,
                "legacy_action": legacy,
                "decisions": decisions,
            }
        )
    return {
        "version": "ecvoi-label-blind-routing-replay-v1",
        "diagnosis_run": str(diagnosis_run),
        "diagnosis_manifest_sha256": diagnosis_sha256,
        "diagnosis_files": diagnosis_manifest,
        "transport_report": str(transport_report),
        "transport_report_sha256": _sha256(transport_report),
        "action_memory_snapshot": str(action_memory_snapshot),
        "action_memory_snapshot_sha256": _sha256(action_memory_snapshot),
        "action_context_registry": str(action_context_registry),
        "action_context_registry_sha256": registry["sha256"],
        "tasks": len(rows),
        "policies": list(REPLAY_POLICIES),
        "selection_changes_vs_legacy": changes,
        "rows": rows,
        "test_loaded": False,
        "claim_boundary": {
            "supported": (
                "The EC-VOI implementation produces deterministic, auditable, "
                "scope-aware routing behavior. Global complete-grid records are "
                "projected through a frozen task-context kernel and admitted only "
                "when raw-count, effective-sample-size and similarity gates pass."
            ),
            "not_supported": (
                "Replay does not establish predictive benefit, selection "
                "superiority, prospective validity, or leaderboard gain."
            ),
        },
    }


def render_markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# EC-VOI label-blind routing replay v1",
        "",
        f"- Tasks: {payload['tasks']}/22",
        "- Test loaded: false",
        f"- Transport report SHA-256: `{payload['transport_report_sha256']}`",
        f"- Action-memory snapshot SHA-256: `{payload['action_memory_snapshot_sha256']}`",
        f"- Action-context registry SHA-256: `{payload['action_context_registry_sha256']}`",
        f"- Diagnosis manifest SHA-256: `{payload['diagnosis_manifest_sha256']}`",
        "",
        "## Selection changes relative to legacy",
        "",
        "| Policy | Changed tasks |",
        "|---|---:|",
    ]
    for policy in payload["policies"]:
        lines.append(
            f"| {policy} | {payload['selection_changes_vs_legacy'][policy]} |"
        )
    lines.extend(
        [
            "",
            "## Task-level decisions",
            "",
            "| Task | Category | Legacy | EC-VOI | No memory | No VOI | No cost | No risk |",
            "|---|---|---|---|---|---|---|---|",
        ]
    )
    for row in payload["rows"]:
        decision = row["decisions"]
        lines.append(
            "| {task} | {category} | {legacy} | {full} | {memory} | {voi} | "
            "{cost} | {risk} |".format(
                task=row["task"],
                category=row["category"],
                legacy=row["legacy_action"],
                full=decision["ecvoi"]["selected_action_id"],
                memory=decision["ecvoi-no-memory"]["selected_action_id"],
                voi=decision["ecvoi-no-voi"]["selected_action_id"],
                cost=decision["ecvoi-no-cost"]["selected_action_id"],
                risk=decision["ecvoi-no-risk"]["selected_action_id"],
            )
        )
    lines.extend(
        [
            "",
            "## Claim boundary",
            "",
            f"Supported: {payload['claim_boundary']['supported']}",
            "",
            f"Not supported: {payload['claim_boundary']['not_supported']}",
            "",
        ]
    )
    return "\n".join(lines)


def write_replay(
    diagnosis_run: Path,
    transport_report: Path,
    action_memory_snapshot: Path,
    action_context_registry: Path,
    output_json: Path,
    output_markdown: Path,
) -> dict[str, Any]:
    payload = replay_routing(
        diagnosis_run,
        transport_report,
        action_memory_snapshot,
        action_context_registry,
    )
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_markdown.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(
        json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
    )
    output_markdown.write_text(render_markdown(payload), encoding="utf-8")
    return payload


__all__ = ["render_markdown", "replay_routing", "write_replay"]
