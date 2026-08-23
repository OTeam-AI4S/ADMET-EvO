"""Risk-triggered activation policy for the ADMET logical committee.

The production default is one planning agent.  Specialist reviewers are
activated only for a small set of validation-visible decisions where an
independent role can change the safety or evidential quality of the decision.
This policy never launches training and rejects test-derived inputs.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from typing import Any, Literal

CommitteeRole = Literal["data_steward", "model_researcher", "skeptic"]
ActivationMode = Literal["single_agent", "targeted_review", "full_committee"]
_SAFE_TEST_KEYS = frozenset({"falsification_test", "falsification_tests"})


def _forbidden_test_paths(value: Any, path: str = "input") -> tuple[str, ...]:
    """Reject test feedback recursively while allowing ``test_loaded=false``."""

    forbidden: list[str] = []
    if isinstance(value, Mapping):
        for key, child in value.items():
            text = str(key)
            normalized = re.sub(
                r"([a-z0-9])([A-Z])", r"\1_\2", text.strip()
            ).lower()
            child_path = f"{path}.{key}"
            safe_sentinel = normalized == "test_loaded" and child is False
            tokens = tuple(
                token for token in re.split(r"[^a-z0-9]+", normalized) if token
            )
            if (
                normalized not in _SAFE_TEST_KEYS
                and not safe_sentinel
                and any(
                    token in {"test", "testing", "heldout", "holdout"}
                    for token in tokens
                )
            ):
                forbidden.append(child_path)
            forbidden.extend(_forbidden_test_paths(child, child_path))
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        for index, child in enumerate(value):
            forbidden.extend(_forbidden_test_paths(child, f"{path}[{index}]"))
    return tuple(sorted(set(forbidden)))


@dataclass(frozen=True)
class CommitteeActivationDecision:
    """Auditable routing decision made before any committee proposal is run."""

    mode: ActivationMode
    roles: tuple[CommitteeRole, ...]
    reason_codes: tuple[str, ...]
    stage: Literal["pre_action", "post_validation"]
    decision_sha256: str
    test_loaded: bool = False

    @property
    def activate(self) -> bool:
        return self.mode != "single_agent"

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["activate"] = self.activate
        return payload


class ConditionalCommitteeActivationPolicy:
    """Keep routine tasks single-agent and escalate only material risks.

    The policy is deliberately conservative: generic hardness, imbalance or a
    merely inconclusive outcome does not activate a committee.  A scheduled
    corrective action, selected data intervention, external-data admission,
    destructive policy update, or high-cost representation does.
    """

    _role_order: tuple[CommitteeRole, ...] = (
        "data_steward",
        "model_researcher",
        "skeptic",
    )

    @staticmethod
    def _selected_action(plan: Mapping[str, Any]) -> Mapping[str, Any] | None:
        selected = plan.get("selected_action_id")
        if not selected:
            return None
        for action in plan.get("actions", ()):
            if isinstance(action, Mapping) and action.get("action_id") == selected:
                return action
        return None

    def decide(
        self,
        plan: Mapping[str, Any],
        *,
        correction: Mapping[str, Any] | None = None,
        risk_context: Mapping[str, bool] | None = None,
    ) -> CommitteeActivationDecision:
        inputs = [plan, correction or {}, risk_context or {}]
        forbidden = _forbidden_test_paths(inputs)
        if forbidden:
            raise ValueError(f"Test-like fields are forbidden: {list(forbidden)}")

        roles: set[CommitteeRole] = set()
        reasons: set[str] = set()
        stage: Literal["pre_action", "post_validation"] = "pre_action"
        action = self._selected_action(plan)

        if action is not None:
            axis = str(action.get("axis", "")).lower()
            action_id = str(action.get("action_id", "")).lower()
            if axis == "data" or action_id.startswith("data-"):
                roles.update(("data_steward", "skeptic"))
                reasons.add("selected_data_intervention_requires_provenance_review")
            # Historical feature-effect priorities can be numerically high but
            # do not imply expensive training.  The automatic cost gate is
            # therefore restricted to selected model actions; data and other
            # high-cost cases use the explicit risk flags below.
            if axis == "model" and float(
                action.get("compute_cost", 0.0) or 0.0
            ) >= 0.60:
                roles.update(("model_researcher", "skeptic"))
                reasons.add("selected_action_exceeds_high_cost_gate")

        context = dict(risk_context or {})
        unknown = set(context).difference(
            {
                "external_data_admission",
                "destructive_policy_change",
                "high_cost_representation",
                "cross_axis_disagreement",
            }
        )
        if unknown:
            raise ValueError(f"Unknown risk flags: {sorted(unknown)}")
        if context.get("external_data_admission"):
            roles.update(("data_steward", "skeptic"))
            reasons.add("external_data_requires_independent_admission_review")
        if context.get("destructive_policy_change"):
            roles.add("skeptic")
            reasons.add("destructive_policy_change_requires_falsification_review")
        if context.get("high_cost_representation"):
            roles.update(("model_researcher", "skeptic"))
            reasons.add("high_cost_representation_requires_value_of_information_review")
        if context.get("cross_axis_disagreement"):
            roles.update(self._role_order)
            reasons.add("cross_axis_disagreement_requires_full_arbitration")

        if correction and correction.get("execution_status") == "scheduled":
            stage = "post_validation"
            kind = str(correction.get("decision_kind") or correction.get("kind") or "")
            axis = str(correction.get("selected_axis") or "").lower()
            if kind in {"repair_evidence", "switch_axis"}:
                roles.add("skeptic")
                roles.add("data_steward" if axis == "data" else "model_researcher")
                reasons.add(f"scheduled_{kind}_requires_independent_review")

        ordered_roles = tuple(role for role in self._role_order if role in roles)
        if not ordered_roles:
            mode: ActivationMode = "single_agent"
            reasons.add("no_material_escalation_trigger")
        elif len(ordered_roles) == len(self._role_order):
            mode = "full_committee"
        else:
            mode = "targeted_review"

        core = {
            "mode": mode,
            "roles": ordered_roles,
            "reason_codes": tuple(sorted(reasons)),
            "stage": stage,
            "test_loaded": False,
        }
        digest = hashlib.sha256(
            json.dumps(core, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        return CommitteeActivationDecision(decision_sha256=digest, **core)
