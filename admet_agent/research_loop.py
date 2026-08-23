"""Auditable hypothesis/action routing for autonomous ADMET research."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from .hardness import HardnessCard
from .research_policy import EvidenceConditionedVOIPolicy
from .research_policy_v2 import ECVOI_V2_ROUTING_POLICIES

ResearchAxis = Literal["feature", "model", "data", "replicate", "stop"]
RESEARCH_ROUTING_POLICIES = (
    "legacy",
    "ecvoi",
    "ecvoi-no-context",
    "ecvoi-no-memory",
    "ecvoi-no-voi",
    "ecvoi-no-cost",
    "ecvoi-no-risk",
    *ECVOI_V2_ROUTING_POLICIES,
)


@dataclass(frozen=True)
class ResearchAction:
    action_id: str
    axis: ResearchAxis
    hypothesis: str
    falsification_test: str
    rationale: tuple[str, ...]
    expected_effect: float
    evidence_confidence: float
    compute_cost: float
    scientific_risk: float
    payload: dict[str, Any] = field(default_factory=dict)
    parent_action_id: str | None = None

    def __post_init__(self) -> None:
        for name in (
            "expected_effect",
            "evidence_confidence",
            "compute_cost",
            "scientific_risk",
        ):
            value = getattr(self, name)
            if not math.isfinite(value) or not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be finite and within [0, 1]")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ResearchDecision:
    selected_action_id: str
    effect_priority: float
    ranked_actions: tuple[dict[str, Any], ...]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class EffectFirstResearchPlanner:
    """Translate a HardnessCard into falsifiable, single-axis actions."""

    def __init__(self, routing_policy: str = "legacy"):
        supported = set(RESEARCH_ROUTING_POLICIES)
        if routing_policy not in supported:
            raise ValueError(
                f"routing_policy must be one of {sorted(supported)}"
            )
        self.routing_policy = routing_policy
        if routing_policy in ECVOI_V2_ROUTING_POLICIES:
            self.ecvoi_policy = None
            return
        variants = {
            "legacy": {},
            "ecvoi": {},
            "ecvoi-no-context": {
                "transport_scope": "global",
                "policy_version": "ec-voi-v1-no-context",
            },
            "ecvoi-no-memory": {
                "transport_scope": "none",
                "policy_version": "ec-voi-v1-no-memory",
            },
            "ecvoi-no-voi": {
                "information_bonus": 0.0,
                "policy_version": "ec-voi-v1-no-voi",
            },
            "ecvoi-no-cost": {
                "cost_weight": 0.0,
                "risk_weight": 1.0,
                "policy_version": "ec-voi-v1-no-cost",
            },
            "ecvoi-no-risk": {
                "cost_weight": 1.0,
                "risk_weight": 0.0,
                "policy_version": "ec-voi-v1-no-risk",
            },
        }
        self.ecvoi_policy = EvidenceConditionedVOIPolicy(**variants[routing_policy])

    def propose(self, card: HardnessCard) -> list[ResearchAction]:
        dims = card.dimensions
        actions: list[ResearchAction] = []
        if (
            self.routing_policy not in ECVOI_V2_ROUTING_POLICIES
            and dims["split_leakage_risk"].severity > 0
        ):
            actions.append(
                ResearchAction(
                    "data-leakage-safe",
                    "data",
                    "Removing exact train/validation structure overlap will yield a more honest and more transferable estimate.",
                    "Compare the same model and features under official versus leakage_safe training data.",
                    ("exact molecular overlap detected",),
                    0.55,
                    0.95,
                    0.10,
                    0.05,
                    {"data_policy": "leakage_safe"},
                )
            )
        if dims["data_quality"].severity >= 0.03:
            severity = dims["data_quality"].severity
            actions.append(
                ResearchAction(
                    "data-quality-filter",
                    "data",
                    "Canonicalization and conflict-aware deduplication will reduce label noise.",
                    "Hold model, seed and features fixed and compare quality_filtered with official data.",
                    tuple(dims["data_quality"].evidence.get("warnings", ())),
                    min(0.75, 0.20 + 0.65 * severity),
                    0.80,
                    0.12,
                    0.12,
                    {"data_policy": "quality_filtered"},
                )
            )
        if dims["imbalance"].severity >= 0.25:
            severity = dims["imbalance"].severity
            actions.append(
                ResearchAction(
                    "model-imbalance-aware",
                    "model",
                    "An imbalance-aware loss or balanced ensemble will improve minority ranking.",
                    "Compare PR-AUC/AUROC across identical seeds against the unweighted baseline.",
                    (f"imbalance severity={severity:.3f}",),
                    min(0.80, 0.20 + 0.65 * severity),
                    0.85,
                    0.25,
                    0.08,
                    {
                        "imbalance": "class_weight",
                        "candidate_model": "balanced_extra_trees",
                    },
                )
            )
        if dims["domain_shift"].severity >= 0.25:
            severity = dims["domain_shift"].severity
            diagnostic_confidence = dims["domain_shift"].confidence
            actions.append(
                ResearchAction(
                    "feature-domain-shift",
                    "feature",
                    "Complementary fingerprints plus a pretrained molecular representation will improve scaffold transfer.",
                    "Compare multi-view and pretrained representations using the same estimator and seeds.",
                    (f"domain-shift severity={severity:.3f}",),
                    min(0.85, 0.20 + 0.75 * severity),
                    0.35 + 0.55 * diagnostic_confidence,
                    0.65,
                    0.15,
                    {"representations": ["unimol2", "morgan", "rdkit2d"]},
                )
            )
        if dims["local_roughness"].severity >= 0.20:
            severity = dims["local_roughness"].severity
            diagnostic_confidence = dims["local_roughness"].confidence
            actions.append(
                ResearchAction(
                    "model-estimator-family",
                    "model",
                    "A RandomForest bias-variance profile may be less brittle than ExtraTrees around activity cliffs.",
                    "Compare fixed RandomForest and ExtraTrees configurations on identical features, data and seeds.",
                    (f"local-roughness severity={severity:.3f}",),
                    min(0.70, 0.15 + 0.65 * severity),
                    0.35 + 0.55 * diagnostic_confidence,
                    0.30,
                    0.10,
                    {
                        "candidate_model": "random_forest",
                        "control_model": "extra_trees",
                    },
                )
            )
        if dims["scarcity"].severity >= 0.35:
            severity = dims["scarcity"].severity
            actions.append(
                ResearchAction(
                    "data-external-quarantine",
                    "data",
                    "Assay-aligned external observations or endpoint transfer may improve low-data learning.",
                    "After provenance and assay alignment, compare source-held-out and TDC validation performance.",
                    (f"scarcity severity={severity:.3f}",),
                    min(0.80, 0.20 + 0.70 * severity),
                    0.45,
                    0.75,
                    0.40,
                    {
                        "skill": "admet-data-collector",
                        "skill_mode": "discover",
                        "merge_status": "quarantined_pending_assay_alignment",
                        "required_release_state": "modeling_ready_discovery",
                        "automatic_download_allowed": False,
                    },
                )
            )
        actions.append(
            ResearchAction(
                "replicate-multiseed",
                "replicate",
                "The apparent winner must remain superior across validation resamples.",
                "Repeat promoted candidates over at least five seeds and compare LCB and rank stability.",
                ("selection variance is unknown before replication",),
                0.35,
                0.98,
                0.45,
                0.02,
                {"minimum_seeds": 5, "selection": "effect_first_lcb"},
            )
        )
        actions.append(
            ResearchAction(
                "stop-no-defensible-action",
                "stop",
                "Further search should stop when expected gain is smaller than uncertainty or no valid axis remains.",
                "Stop if all promoted actions fail their preregistered minimum effect and stability gates.",
                ("prevents validation overfitting",),
                0.05,
                0.90,
                0.0,
                0.0,
                {"abstain_on_ood": True},
            )
        )
        return actions

    def route(
        self,
        actions: list[ResearchAction],
        *,
        effect_priority: float = 0.80,
        action_priors: dict[str, dict[str, Any]] | None = None,
    ) -> ResearchDecision:
        if not actions:
            raise ValueError("At least one research action is required")
        if not 0.5 <= effect_priority <= 1.0:
            raise ValueError("effect_priority must be within [0.5, 1.0]")
        if self.routing_policy in ECVOI_V2_ROUTING_POLICIES:
            raise RuntimeError(
                "EC-VOI v2 must rank resolved treatment/control pairs"
            )
        if self.routing_policy != "legacy":
            ranked = self.ecvoi_policy.rank(
                actions,
                effect_priority=effect_priority,
                action_priors=action_priors,
            )
            return ResearchDecision(
                ranked[0]["action_id"], effect_priority, tuple(ranked)
            )
        ranked = []
        action_priors = action_priors or {}
        for action in actions:
            prior = action_priors.get(action.action_id)
            transport_observations = (
                int(prior.get("transport_eligible_observations", 0)) if prior else 0
            )
            refuted_rate = (
                float(prior.get("refuted", 0)) / transport_observations
                if transport_observations
                else 0.0
            )
            negative_memory_gate = bool(
                prior
                and prior.get("routing_eligible") is True
                and prior.get("negative_gate_confirmed") is True
                and transport_observations >= 5
                and refuted_rate >= 0.80
                and float(prior.get("posterior_support_probability", 0.5)) < 0.35
            )
            multiplier = (
                float(prior.get("effect_multiplier", 1.0))
                if prior and prior.get("routing_eligible") is True
                else 1.0
            )
            calibrated_effect = min(1.0, action.expected_effect * multiplier)
            scientific_value = calibrated_effect * action.evidence_confidence
            penalty = (1.0 - effect_priority) * (
                0.55 * action.compute_cost + 0.45 * action.scientific_risk
            )
            utility = effect_priority * scientific_value - penalty
            if negative_memory_gate:
                utility = -1.0
            ranked.append(
                {
                    **action.to_dict(),
                    "calibrated_expected_effect": calibrated_effect,
                    "prior_evidence": prior,
                    "routing_admitted": not negative_memory_gate,
                    "routing_block_reason": (
                        "prospective_transport_refutation_gate"
                        if negative_memory_gate
                        else None
                    ),
                    "utility": utility,
                }
            )
        ranked.sort(key=lambda item: (-item["utility"], item["action_id"]))
        return ResearchDecision(ranked[0]["action_id"], effect_priority, tuple(ranked))

    def plan(
        self,
        card: HardnessCard,
        *,
        effect_priority: float = 0.80,
        action_priors: dict[str, dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        actions = self.propose(card)
        if self.routing_policy in ECVOI_V2_ROUTING_POLICIES:
            return {
                "task": card.task,
                "hardness_evidence_sha256": card.evidence_sha256,
                "selected_action_id": "pending-ecvoi-v2-pair-resolution",
                "effect_priority": effect_priority,
                "actions": [action.to_dict() for action in actions],
                "routing_policy": self.routing_policy,
                "routing_deferred_to_resolved_pairs": True,
                "test_loaded": False,
                "leaderboard_loaded": False,
            }
        decision = self.route(
            actions,
            effect_priority=effect_priority,
            action_priors=action_priors,
        )
        output = {
            "task": card.task,
            "hardness_evidence_sha256": card.evidence_sha256,
            "selected_action_id": decision.selected_action_id,
            "effect_priority": effect_priority,
            "actions": list(decision.ranked_actions),
        }
        if self.routing_policy != "legacy":
            assert self.ecvoi_policy is not None
            output["routing_policy"] = self.ecvoi_policy.policy_version
        return output


class EvidenceLedger:
    """Append-only, hash-chained scientific evidence ledger."""

    def __init__(self, path: Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def _last_hash(self) -> str:
        if not self.path.exists():
            return "GENESIS"
        last = ""
        with self.path.open(encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    last = line
        return json.loads(last)["record_sha256"] if last else "GENESIS"

    def append(self, event_type: str, payload: dict[str, Any]) -> dict[str, Any]:
        base = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "event_type": event_type,
            "payload": payload,
            "previous_sha256": self._last_hash(),
        }
        digest = hashlib.sha256(
            json.dumps(base, sort_keys=True, separators=(",", ":"), default=str).encode(
                "utf-8"
            )
        ).hexdigest()
        record = {**base, "record_sha256": digest}
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, sort_keys=True, default=str) + "\n")
        return record

    def verify(self) -> dict[str, Any]:
        previous = "GENESIS"
        count = 0
        if not self.path.exists():
            return {"valid": True, "records": 0, "head_sha256": previous}
        with self.path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                record = json.loads(line)
                digest = record.pop("record_sha256")
                expected = hashlib.sha256(
                    json.dumps(
                        record, sort_keys=True, separators=(",", ":"), default=str
                    ).encode("utf-8")
                ).hexdigest()
                if digest != expected or record["previous_sha256"] != previous:
                    return {
                        "valid": False,
                        "records": count,
                        "failed_line": line_number,
                    }
                previous = digest
                count += 1
        return {"valid": True, "records": count, "head_sha256": previous}
