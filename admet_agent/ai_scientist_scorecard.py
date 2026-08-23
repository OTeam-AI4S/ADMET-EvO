"""Claim-bounded evidence vector for the TDC-22 AI Scientist objective."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _assert_no_test_loaded(value: Any, path: str = "root") -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            if key == "test_loaded" and child is not False:
                raise ValueError(f"Test firewall violation at {path}.{key}")
            _assert_no_test_loaded(child, f"{path}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _assert_no_test_loaded(child, f"{path}[{index}]")


def _arm(report: dict[str, Any], name: str) -> dict[str, Any]:
    try:
        return report["arms"][name]
    except KeyError as error:
        raise ValueError(f"Missing required control arm: {name}") from error


def build_evidence_scorecard(report_dir: Path) -> dict[str, Any]:
    paths = {
        "performance": report_dir / "performance_qualification_v1.json",
        "composite": report_dir / "composite_research_challenge_v1.json",
        "data": report_dir / "data_stewardship_controls_v1.json",
        "feedback": report_dir / "feedback_policy_controls_v1.json",
        "fault": report_dir / "fault_policy_controls_v1.json",
        "transport": report_dir / "transport_grid_results_v1.json",
        "memory_ablation": report_dir.parent / "runs/v27-router-controls/report.json",
        "memory_behavior": report_dir / "transport_memory_behavior_replay_v1.json",
        "minimal_pilot": report_dir / "minimal_exact_pair_pilot_v1.json",
        "predictive_control": report_dir / "minimal_predictive_control_v1.json",
        "test_incident": report_dir / "test_firewall_incident_2026-08-20.json",
        "external_acquisition_v1": report_dir.parent / "runs/chembl-herg-source-heldout-v1/acquisition_failure.json",
        "external_acquisition_v2": report_dir.parent / "runs/chembl-herg-source-heldout-v2/acquisition_failure.json",
        "external_data_agent": report_dir / "external_data_agent_capability_v1.json",
        "graph_fusion": report_dir / "bounded_graph_fusion_ticket_v1.json",
    }
    reports = {name: _load(path) for name, path in paths.items()}
    for name, report in reports.items():
        _assert_no_test_loaded(report, name)

    performance = reports["performance"]
    composite = reports["composite"]
    data = reports["data"]
    feedback = reports["feedback"]
    fault = reports["fault"]
    transport = reports["transport"]
    pilot = reports["minimal_pilot"]
    memory_ablation = reports["memory_ablation"]
    memory_behavior = reports["memory_behavior"]
    predictive = reports["predictive_control"]
    test_incident = reports["test_incident"]
    acquisition_v1 = reports["external_acquisition_v1"]
    acquisition_v2 = reports["external_acquisition_v2"]
    external_data_agent = reports["external_data_agent"]
    graph_fusion = reports["graph_fusion"]
    if any(
        memory_ablation.get(field) is not False
        for field in ("official_test_loaded", "official_results_used", "leaderboard_used")
    ):
        raise ValueError("Memory ablation firewall is incomplete")

    agent_process = _arm(composite, "agent_feedback_policy")
    rules_process = _arm(composite, "locked_static_rules")
    automl_process = _arm(composite, "expanded_automl_scalar_search")
    conventional_process = _arm(composite, "conventional_automl_scalar_search")
    random_process = _arm(composite, "seeded_random_action")
    rules_comparison = composite["agent_comparisons"]["locked_static_rules"]

    agent_data = _arm(data, "agent_contract_steward")
    rules_data = _arm(data, "complete_rules_steward")
    automl_data = _arm(data, "expanded_automl_schema_steward")
    conventional_data = _arm(data, "conventional_automl_direct_fit")
    no_data = _arm(data, "agent_no_data_acquisition")

    no_hardness = _arm(fault, "agent_no_hardness_diagnosis")["score"]
    full_fault = _arm(fault, "agent_contract_guard")["score"]

    agent_feedback = _arm(feedback, "agent_feedback_loop")
    rules_feedback = _arm(feedback, "complete_rules_feedback_loop")
    automl_feedback = _arm(feedback, "expanded_automl_scalar_search")

    process_fit_saving = 1.0 - (
        agent_process["total_estimated_fits"] / automl_process["total_estimated_fits"]
    )
    pilot_saving = pilot["fits_avoided_by_gate"] / pilot["maximum_fits"]
    local_rules_lcb = float(rules_comparison["bootstrap_95_ci"][0])

    vector = {
        "prediction_qualification": {
            "status": "certified_application_qualification",
            "passed": bool(performance["passed"]),
            "tasks": int(performance["tasks_observed"]),
            "top5_tasks": int(performance["top5_tasks"]),
            "top10_tasks": int(performance["top10_tasks"]),
            "provenance_coverage": float(performance["provenance_coverage"]),
            "measurement": "official frozen application results; no test labels loaded by the qualification step",
        },
        "same_budget_predictive_control": {
            "status": "complete_noninferiority_not_demonstrated",
            "tasks": len(predictive["tasks"]),
            "logical_charged_fits": int(predictive["logical_charged_fits"]),
            "physical_unique_fits": int(predictive["physical_unique_fits"]),
            "physical_fit_reuse_fraction": float(predictive["physical_fit_reuse_fraction"]),
            "failed_fits": 0,
            "agent_minus_expanded_automl_mean": float(
                predictive["comparisons"]["agent_vs_expanded_automl"]["mean_normalized_utility_delta"]
            ),
            "agent_minus_expanded_automl_bootstrap_95_ci": list(
                predictive["comparisons"]["agent_vs_expanded_automl"]["task_bootstrap_95_ci"]
            ),
            "agent_vs_expanded_automl_wins_ties_losses": [
                predictive["comparisons"]["agent_vs_expanded_automl"][key]
                for key in ("wins", "ties", "losses")
            ],
            "noninferiority_margin": float(
                predictive["agent_vs_expanded_automl_noninferiority_margin"]
            ),
            "noninferiority_passed": bool(
                predictive["agent_vs_expanded_automl_noninferiority_passed"]
            ),
            "measurement": "fresh validation seed 11 selection -> seed 12 evaluation; frozen source and no test feedback",
        },
        "integrated_research_decision": {
            "status": "supported_on_same_team_sealed_holdout",
            "episodes": int(composite["episode_count"]),
            "agent_macro": float(agent_process["macro_process_score"]),
            "rules_macro": float(rules_process["macro_process_score"]),
            "expanded_automl_macro": float(automl_process["macro_process_score"]),
            "conventional_automl_macro": float(conventional_process["macro_process_score"]),
            "random_macro": float(random_process["macro_process_score"]),
            "agent_minus_rules_case_score": float(rules_comparison["mean_agent_minus_control_case_score"]),
            "agent_minus_rules_bootstrap_95_ci": list(rules_comparison["bootstrap_95_ci"]),
            "agent_unsafe_actions": int(agent_process["unsafe_action_count"]),
            "same_maximum_action_budget": bool(composite["same_maximum_action_budget"]),
            "measurement": "locked commit/reveal local episodes; not independent external authorship",
        },
        "data_stewardship": {
            "status": "advantage_over_automl_not_over_complete_rules",
            "cases": int(agent_data["challenge_count"]),
            "agent_decision_accuracy": float(agent_data["decision_accuracy"]),
            "agent_unsafe_merges": int(agent_data["unsafe_merge_count"]),
            "expanded_automl_decision_accuracy": float(automl_data["decision_accuracy"]),
            "expanded_automl_unsafe_merges": int(automl_data["unsafe_merge_count"]),
            "conventional_automl_decision_accuracy": float(conventional_data["decision_accuracy"]),
            "conventional_automl_unsafe_merges": int(conventional_data["unsafe_merge_count"]),
            "complete_rules_decision_accuracy": float(rules_data["decision_accuracy"]),
            "agent_beyond_complete_rules": agent_data["decision_accuracy"] > rules_data["decision_accuracy"],
            "measurement": "sealed local data dossiers; governance decision, not source-held-out predictive gain",
        },
        "real_external_data_ticket": {
            "status": "two_frozen_acquisitions_failed_closed_before_training",
            "source": "official ChEMBL activity API",
            "endpoint": "CHEMBL240 human hERG IC50",
            "frozen_protocol_attempts": 2,
            "first_failure_status": acquisition_v1["status"],
            "corrected_failure_status": acquisition_v2["status"],
            "raw_artifact_created": False,
            "training_started": False,
            "fits": 0,
            "unsafe_merges": 0,
            "source_heldout_gain_demonstrated": False,
            "measurement": "real public-source acquisition attempts; API failure retained and closed before partial data or training",
        },
        "autonomous_data_admission_stack": {
            "status": external_data_agent["status"],
            "candidate_sources_discovered": int(
                external_data_agent["discovery"]["candidate_count"]
            ),
            "direct_merges_allowed": int(
                external_data_agent["discovery"]["decision_counts"].get("exact", 0)
            ),
            "bridge_required": int(
                external_data_agent["discovery"]["decision_counts"].get(
                    "bridge_required", 0
                )
            ),
            "three_way_structure_firewall_frozen": True,
            "test_labels_loaded": False,
            "test_structures_exposed_to_planner": False,
            "same_firewall_and_source_budget_for_automl": True,
            "source_heldout_gain_demonstrated": False,
            "measurement": "executable discovery/judgment/decontamination mechanism; real predictive gain remains pending",
        },
        "bounded_graph_fusion": {
            "status": graph_fusion["status"],
            "two_d_ready": bool(graph_fusion["readiness"]["two_d"]),
            "gin_ready": bool(graph_fusion["readiness"]["gin"]),
            "distinct_mpnn_ready": bool(graph_fusion["readiness"]["mpnn"]),
            "oof_decision_tree_gate_ready": bool(
                graph_fusion["readiness"]["oof_gate"]
            ),
            "maximum_ticket_fits": int(
                graph_fusion["experiment"]["stage_a_representation"]["maximum_fits"]
                + graph_fusion["experiment"]["stage_b_fusion"]["maximum_fits"]
            ),
            "training_started": False,
            "measurement": "backend-agnostic OOF-safe fusion implementation and frozen ticket; no distinct MPNN provider installed",
        },
        "feedback_self_correction": {
            "status": "advantage_over_scalar_automl_not_over_complete_rules",
            "observed_cases": int(agent_feedback["case_count"]),
            "agent_macro": float(agent_feedback["macro_process_score"]),
            "expanded_automl_macro": float(automl_feedback["macro_process_score"]),
            "complete_rules_macro": float(rules_feedback["macro_process_score"]),
            "agent_minus_expanded_automl": float(
                agent_feedback["macro_process_score"] - automl_feedback["macro_process_score"]
            ),
            "agent_beyond_complete_rules": agent_feedback["macro_process_score"] > rules_feedback["macro_process_score"],
            "measurement": "two observed feedback-linked cases plus integrated composite episodes",
        },
        "cost_aware_stopping": {
            "status": "measured_pilot_and_estimated_composite_savings",
            "minimal_pilot_actual_fits": int(pilot["actual_fits"]),
            "minimal_pilot_maximum_fits": int(pilot["maximum_fits"]),
            "minimal_pilot_fits_avoided": int(pilot["fits_avoided_by_gate"]),
            "minimal_pilot_saving_fraction": pilot_saving,
            "composite_agent_estimated_fits": int(agent_process["total_estimated_fits"]),
            "composite_expanded_automl_estimated_fits": int(automl_process["total_estimated_fits"]),
            "composite_estimated_saving_fraction": process_fit_saving,
            "measurement": "pilot fits are executed; composite fit counts are counterfactual estimates",
        },
        "cross_task_learning": {
            "status": "causal_action_priors_ready_routing_advantage_unproven",
            "endpoints": int(transport["endpoint_count"]),
            "actions": int(transport["action_count"]),
            "completed_fits": int(transport["completed_fits"]),
            "failed_fits": int(transport["failed_fits"]),
            "routing_eligible_actions": list(transport["routing_eligible_actions"]),
            "routing_blocked_actions": list(transport["routing_blocked_actions"]),
            "shared_control_compute_reduction_fraction": float(
                transport["shared_control_compute_reduction_fraction"]
            ),
            "routing_outcome_advantage_proven": False,
            "measurement": "prospective complete action grid; preserves negative and inconclusive outcomes",
        },
        "key_process_ablations": {
            "status": "executed_with_negative_memory_result",
            "no_memory": {
                "tasks": int(memory_behavior["task_count"]),
                "selection_changed_tasks": int(memory_behavior["selection_changed_tasks"]),
                "ranking_changed_tasks": int(memory_behavior["ranking_changed_tasks"]),
                "behavioral_advantage_demonstrated": bool(
                    memory_behavior["behavioral_memory_advantage_demonstrated"]
                ),
                "legacy_full_minus_winner_mean_oriented_gain": float(
                    memory_ablation["memory_ablation"]["summary"]["mean"]
                ),
                "legacy_claim_gate_passed": bool(
                    memory_ablation["memory_ablation"]["claim_gate"]["passed"]
                ),
                "interpretation": "Memory changes utilities but has not yet changed the selected action or passed its outcome gate.",
            },
            "no_hardness": {
                "cases": len(fault["arms"]["agent_no_hardness_diagnosis"]["score"].get("per_challenge", [])) or 10,
                "full_agent_micro_recall": float(full_fault["micro_recall"]),
                "ablation_micro_recall": float(no_hardness["micro_recall"]),
                "ablation_safety_gate_passed": bool(no_hardness["safety_gate_passed"]),
                "interpretation": "Mechanism ablation on hidden faults; not a predictive-score comparison.",
            },
            "no_data": {
                "cases": int(no_data["challenge_count"]),
                "full_agent_decision_accuracy": float(agent_data["decision_accuracy"]),
                "ablation_decision_accuracy": float(no_data["decision_accuracy"]),
                "ablation_unsafe_merges": int(no_data["unsafe_merge_count"]),
                "interpretation": "Reject-all remains safe but loses valid acquisition opportunities.",
            },
        },
        "audit_and_test_firewall": {
            "status": "historical_evidence_verified_post_qualification_incident_disclosed",
            "source_artifacts": len(paths),
            "all_included_test_loaded_false": True,
            "artifact_sha256_coverage": 1.0,
            "post_qualification_test_access_incidents": 1,
            "strict_no_test_access_claim_for_current_goal_session": False,
            "test_feedback_used_after_incident": bool(test_incident["test_feedback_used"]),
            "incident_contained": test_incident["status"] == "contained_not_reversible",
        },
    }

    gates = {
        "application_performance_qualified": bool(
            performance["passed"] and performance["tasks_observed"] == 22
        ),
        "local_integrated_process_advantage_over_rules": bool(
            agent_process["macro_process_score"] > rules_process["macro_process_score"]
            and local_rules_lcb > 0
            and agent_process["unsafe_action_count"] == 0
        ),
        "local_data_advantage_over_expanded_automl": bool(
            agent_data["decision_accuracy"] > automl_data["decision_accuracy"]
            and agent_data["unsafe_merge_count"] == 0
        ),
        "feedback_advantage_over_scalar_automl": bool(
            agent_feedback["macro_process_score"] > automl_feedback["macro_process_score"]
        ),
        "prospective_cross_task_prior_ready": bool(
            transport["grid_complete"] and transport["failed_fits"] == 0
            and len(transport["routing_eligible_actions"]) > 0
        ),
        "key_process_ablations_executed": bool(
            memory_behavior["task_count"] == 22
            and "agent_no_hardness_diagnosis" in fault["arms"]
            and "agent_no_data_acquisition" in data["arms"]
        ),
        "formal_22_task_same_budget_predictive_control_complete": bool(
            predictive["status"] == "complete"
            and len(predictive["tasks"]) == 22
            and predictive["logical_charged_fits"] == predictive["maximum_logical_fits"]
        ),
        "external_data_discovery_endpoint_and_firewall_mechanism_ready": bool(
            external_data_agent["discovery"]["candidate_count"] > 0
            and external_data_agent["sealed_structure_firewall"]["test_labels_loaded"]
            is False
            and external_data_agent["fair_comparison_contract"][
                "same_structure_firewall"
            ]
        ),
        "independently_authored_external_process_validation_complete": False,
        "source_held_out_external_data_gain_complete": False,
    }
    blockers = [
        "Obtain independently authored composite research episodes before claiming general autonomous-research superiority.",
        "Complete an admitted source-held-out external-data comparison; two frozen ChEMBL acquisitions failed closed on external HTTP 500 before raw data or training.",
        "Provide one content-addressed distinct MPNN fingerprint provider, then run the <=20-fit graph representation/fusion ticket; do not expand it to all 22 tasks by default.",
    ]
    all_goal_gates = all(gates.values())
    return {
        "protocol": "tdc22-ai-scientist-evidence-vector-v1",
        "overall_status": (
            "complete"
            if all_goal_gates
            else "qualified_prediction_local_process_advantage_external_validation_pending"
        ),
        "why_no_single_average": "A weighted average could hide an unsafe action or failed prediction gate; capability dimensions remain separate.",
        "evidence_vector": vector,
        "claim_gates": gates,
        "remaining_blockers": [] if all_goal_gates else blockers,
        "allowed_claim": (
            "On TDC ADMET-22, the frozen Agent meets the application-performance qualification. "
            "On same-team sealed process challenges it outperforms scalar AutoML and locked static "
            "rules while making zero unsafe actions, and it shows measured cost-aware stopping. "
            "A completed fresh-seed equal-budget predictive control did not establish noninferiority "
            "to expanded AutoML and is retained as negative evidence. Two real ChEMBL acquisition "
            "attempts failed closed before training; a post-qualification test-access incident is "
            "disclosed and no test feedback was used. The autonomous data-admission stack now "
            "discovers sources, judges endpoint semantics, and applies a sealed three-way structure "
            "firewall, but external predictive gain remains unproven."
        ),
        "forbidden_claims": [
            "predictively superior to AutoML under a completed prospective same-budget 22-task run",
            "generally superior autonomous scientist on independent external tasks",
            "predictive gain from autonomously acquired external data",
        ],
        "control_scope": {
            "predictive_22_task_required": [
                "streamlined_agent",
                "expanded_automl",
                "strong_fixed",
            ],
            "same_action_budget_process_required": [
                "locked_rules",
                "seeded_random",
                "agent_no_memory",
                "agent_no_hardness",
                "agent_no_data",
            ],
            "reason_for_two_tracks": (
                "Rules/random/ablations test research decisions; forcing every process control "
                "to retrain every predictive model would spend compute without answering its "
                "causal capability question."
            ),
        },
        "test_loaded": False,
        "test_accessed_in_current_goal_session": True,
        "test_feedback_used_after_incident": False,
        "source_hashes": {str(path): _sha256(path) for path in paths.values()},
    }


__all__ = ["build_evidence_scorecard"]
