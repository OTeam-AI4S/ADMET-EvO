"""Audit the complete evidence-to-rollback policy self-correction episode."""

from __future__ import annotations

import json
from pathlib import Path

from admet_agent.action_memory import ResearchOutcomeMemory
from admet_agent.research_loop import EffectFirstResearchPlanner, ResearchAction


def main() -> None:
    root = Path(__file__).parents[1]
    confirmation = json.loads(
        (root / "reports/post_memory_confirmation_v1.json").read_text(encoding="utf-8")
    )
    if confirmation["pruning_success_gate_passed"] is not False:
        raise ValueError("Expected an independently refuted pruning policy")
    memory = ResearchOutcomeMemory(
        root / "runs/v5-transport-grid/transport_action_outcomes.jsonl"
    )
    prior = memory.priors("absorption")["model-estimator-family"]
    action = ResearchAction(
        "model-estimator-family",
        "model",
        "RandomForest may help.",
        "Compare against ExtraTrees.",
        ("local roughness",),
        0.4,
        0.7,
        0.3,
        0.1,
        {"candidate_model": "random_forest"},
    )
    current = (
        EffectFirstResearchPlanner()
        .route([action], action_priors={action.action_id: prior})
        .ranked_actions[0]
    )
    rollback_active = bool(
        prior["negative_gate_confirmed"] is False
        and current["routing_admitted"] is True
        and current["routing_block_reason"] is None
    )
    report = {
        "version": "policy-self-correction-episode-v1",
        "test_loaded": False,
        "self_correction_opportunities": 1,
        "successful_self_corrections": int(rollback_active),
        "hypothesis": (
            "A global negative prior can safely prune model-estimator-family across "
            "unseen ADMET endpoints."
        ),
        "falsification_protocol_sha256": confirmation["protocol_sha256"],
        "falsification_outcomes": confirmation["outcome_counts"],
        "mean_positive_pruning_regret": confirmation["mean_positive_pruning_regret"],
        "maximum_positive_pruning_regret": confirmation[
            "maximum_positive_pruning_regret"
        ],
        "negative_result_retained": True,
        "deployment_blocked": confirmation["negative_memory_gate_deployment_approved"]
        is False,
        "rollback_active": rollback_active,
        "current_prior_negative_gate_confirmed": prior["negative_gate_confirmed"],
        "current_action_admitted": current["routing_admitted"],
        "regression_test": ("test_unconfirmed_global_negative_memory_cannot_prune"),
        "claim_boundary": {
            "supported": (
                "The Agent detected an overgeneralized memory policy with an independent "
                "experiment, retained the refutation, blocked deployment and restored the "
                "candidate to the action set."
            ),
            "not_supported": (
                "The correction does not demonstrate predictive superiority over AutoML "
                "or that a safe stratified memory policy has already been learned."
            ),
        },
    }
    if not rollback_active:
        raise RuntimeError("Policy rollback is not active in the current source")
    (root / "reports/policy_self_correction_v1.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    lines = [
        "# Policy self-correction episode",
        "",
        "Status: **SUCCESSFUL SAFETY ROLLBACK (1/1)**",
        "",
        "1. Prospective transport evidence suggested a global negative model prior.",
        "2. A destructive pruning rule was preregistered but not production-approved.",
        "3. Five independent endpoints on fresh seeds refuted global pruning.",
        "4. The refutation was retained; the deployment claim was denied.",
        "5. Current policy requires an independent confirmation certificate and therefore",
        "   restores the RandomForest action instead of pruning it.",
        "",
        f"- Mean positive regret caught: {report['mean_positive_pruning_regret']:.6f}",
        f"- Maximum positive regret caught: {report['maximum_positive_pruning_regret']:.6f}",
        "- Current negative_gate_confirmed: false",
        "- Current action admitted: true",
        "- Test loaded: false",
        "",
        "This is direct evidence of autonomous policy falsification and rollback. It is",
        "not evidence of a better predictor or a finished contextual memory policy.",
        "",
    ]
    (root / "reports/policy_self_correction_v1.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
