import hashlib
import json
from pathlib import Path

import pytest

from admet_agent.action_memory import ResearchOutcomeMemory
from admet_agent.cli import parser
from admet_agent.ecvoi_benchmark import protocol_payload, write_protocol
from admet_agent.ecvoi_replay import replay_routing
from admet_agent.policy_brains import benchmark_brain
from admet_agent.policy_results import collect_policy_results
from admet_agent.research_loop import EffectFirstResearchPlanner, ResearchAction


def _action(
    action_id: str,
    *,
    effect: float,
    confidence: float,
    cost: float,
    risk: float,
) -> ResearchAction:
    return ResearchAction(
        action_id,
        "model",
        f"Hypothesis for {action_id}",
        "Use a paired validation comparison.",
        ("bounded unit-test action",),
        effect,
        confidence,
        cost,
        risk,
        {},
    )


def test_ecvoi_protocol_is_hashed_test_blind_and_same_budget(tmp_path):
    payload = protocol_payload()
    claimed = payload.pop("sha256")
    actual = hashlib.sha256(
        json.dumps(payload, sort_keys=True).encode("utf-8")
    ).hexdigest()
    assert claimed == actual
    assert payload["test_feedback_to_discovery"] is False
    assert payload["leaderboard_feedback_to_discovery"] is False
    assert payload["statistical_unit"] == "endpoint_task"
    assert len(payload["implementation_source_sha256"]) == 64
    assert len(payload["tasks"]) == 22
    mapping = Path(payload["frozen_invariants"]["tdc22_mapping_path"])
    assert hashlib.sha256(mapping.read_bytes()).hexdigest() == payload[
        "frozen_invariants"
    ]["tdc22_mapping_sha256"]
    memory = Path(
        payload["frozen_invariants"]["action_memory_snapshot_path"]
    )
    assert hashlib.sha256(memory.read_bytes()).hexdigest() == payload[
        "frozen_invariants"
    ]["action_memory_snapshot_sha256"]
    contexts = json.loads(
        Path(
            payload["frozen_invariants"]["action_context_registry_path"]
        ).read_text(encoding="utf-8")
    )
    assert contexts["sha256"] == payload["frozen_invariants"][
        "action_context_registry_sha256"
    ]
    budgets = {
        (
            arm["max_actions_per_task"],
            arm["max_candidates_per_task"],
            arm["max_compute_minutes_per_task"],
        )
        for arm in payload["arms"]
    }
    assert budgets == {(8, 24, 90)}
    written = write_protocol(tmp_path / "ecvoi.json")
    assert written["sha256"] == claimed


def test_ecvoi_cli_and_brain_arms_are_explicit():
    parsed = parser().parse_args(
        [
            "search",
            "--task",
            "AMES",
            "--brain",
            "ecvoi-no-memory",
            "--research-routing-policy",
            "ecvoi-no-memory",
        ]
    )
    assert parsed.brain == "ecvoi-no-memory"
    assert parsed.research_routing_policy == "ecvoi-no-memory"
    assert benchmark_brain("ecvoi").benchmark_arm == "ecvoi"
    assert benchmark_brain("ecvoi_no_memory").benchmark_arm == "ecvoi_no_memory"


def test_action_memory_separates_context_matched_from_global_transport(tmp_path):
    memory = ResearchOutcomeMemory(tmp_path / "actions.jsonl")
    protocol = "a" * 64
    for index, status in enumerate(
        ["supported", "supported", "supported", "supported", "refuted"]
    ):
        memory.append(
            task=f"absorption-{index}",
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
                    "scope": "category",
                },
            },
        )
    for index, status in enumerate(
        ["refuted", "refuted", "refuted", "refuted", "supported"]
    ):
        memory.append(
            task=f"global-{index}",
            category="toxicity",
            evidence={
                "action_id": "feature-domain-shift",
                "axis": "feature",
                "status": status,
                "single_axis_verified": True,
                "mean_oriented_delta": -0.01 if status == "refuted" else 0.01,
                "cross_task_transport": {
                    "prospective": True,
                    "same_budget": True,
                    "assignment_design": "complete_action_grid",
                    "protocol_sha256": protocol,
                    "scope": "global_admet",
                },
            },
        )
    prior = memory.priors("absorption")["feature-domain-shift"]
    assert prior["contextual_routing_eligible"] is True
    assert prior["contextual_transport_eligible_observations"] == 5
    assert prior["contextual_supported"] == 4
    assert prior["contextual_refuted"] == 1
    assert prior["transport_eligible_observations"] == 10
    assert prior["supported"] == 5
    assert prior["refuted"] == 5


def test_ecvoi_ablation_switches_are_behavioral_not_labels_only():
    expensive_safe = _action(
        "expensive-safe", effect=0.7, confidence=0.8, cost=1.0, risk=0.0
    )
    cheap_risky = _action(
        "cheap-risky", effect=0.7, confidence=0.8, cost=0.0, risk=1.0
    )
    no_cost = EffectFirstResearchPlanner("ecvoi-no-cost").route(
        [expensive_safe, cheap_risky]
    )
    no_risk = EffectFirstResearchPlanner("ecvoi-no-risk").route(
        [expensive_safe, cheap_risky]
    )
    assert no_cost.selected_action_id == "expensive-safe"
    assert no_risk.selected_action_id == "cheap-risky"

    evidence = {
        "expensive-safe": {
            "routing_eligible": True,
            "contextual_routing_eligible": True,
            "contextual_supported": 5,
            "contextual_refuted": 0,
            "contextual_inconclusive": 0,
        }
    }
    full = EffectFirstResearchPlanner("ecvoi").route(
        [expensive_safe], action_priors=evidence
    ).ranked_actions[0]
    no_memory = EffectFirstResearchPlanner("ecvoi-no-memory").route(
        [expensive_safe], action_priors=evidence
    ).ranked_actions[0]
    assert full["evidence_source"] == "context_matched_transport"
    assert no_memory["evidence_source"] == "endpoint_local_only"
    assert no_memory["prior_evidence"] is None
    assert no_memory["prior_evidence_ignored"] is True


def test_ecvoi_protocol_refuses_silent_overwrite(tmp_path):
    path = tmp_path / "ecvoi.json"
    path.write_text(json.dumps({"sha256": "different"}), encoding="utf-8")
    with pytest.raises(RuntimeError, match="Refusing to overwrite"):
        write_protocol(path)


def test_result_collection_rejects_router_arm_mismatch(tmp_path):
    protocol_path = tmp_path / "ecvoi.json"
    protocol = write_protocol(protocol_path)
    task_dir = tmp_path / "run" / "tasks" / "AMES"
    task_dir.mkdir(parents=True)
    (task_dir / "search_seed_1.json").write_text(
        json.dumps(
                {
                    "benchmark_arm": "ecvoi",
                    "benchmark_protocol_sha256": protocol["sha256"],
                    "source_sha256": protocol["implementation_source_sha256"],
                    "research_routing_policy": "legacy",
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="records research router legacy"):
        collect_policy_results(protocol_path, {"ecvoi": tmp_path / "run"})

    (task_dir / "search_seed_1.json").write_text(
        json.dumps(
            {
                "benchmark_arm": "ecvoi",
                "benchmark_protocol_sha256": protocol["sha256"],
                "source_sha256": protocol["implementation_source_sha256"],
                "research_routing_policy": "ecvoi",
                "action_memory_snapshot_sha256": "wrong",
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="frozen action-memory snapshot"):
        collect_policy_results(protocol_path, {"ecvoi": tmp_path / "run"})


def test_routing_replay_fails_closed_if_transport_loaded_test(tmp_path):
    transport = tmp_path / "transport.json"
    transport.write_text(json.dumps({"test_loaded": True}), encoding="utf-8")
    with pytest.raises(ValueError, match="test-blind"):
        replay_routing(
            tmp_path / "diagnosis",
            transport,
            tmp_path / "memory.jsonl",
            tmp_path / "contexts.json",
        )
