"""Budget-preserving committee orchestration for ADMET experiment proposals.

The committee deliberately stops at the planning boundary: several logical
researchers may suggest experiments, but a single deterministic arbiter emits
at most ``max_candidates`` unique :class:`ExperimentSpec` objects.  The module
does not execute experiments and never consumes test data.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections import defaultdict
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass, field
from typing import Any, Literal, Protocol

from .brain import AgentBrain
from .guard import ProtocolViolation, validate_experiment
from .schemas import ExperimentSpec, RunResult, TaskProfile

ProposerRole = Literal["data_steward", "model_researcher", "skeptic"]
PairRole = Literal["treatment", "control"]
PairAxis = Literal["feature", "model", "data", "imbalance"]


def proposal_fingerprint(spec: ExperimentSpec) -> str:
    """Return a stable identity for a scientific configuration.

    Runtime details (seed, device, worker count and budget) and prose are not
    part of the identity.  Consequently, two agents cannot consume two slots
    merely by describing the same experiment differently.
    """

    payload = {
        "task": spec.task,
        "representations": list(spec.representations),
        "model": spec.model,
        "imbalance": spec.imbalance,
        "data_policy": spec.data_policy,
        "hyperparameters": spec.hyperparameters,
    }
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True)
class ProposalAssessment:
    """A proposer's validation-only assessment of one experiment."""

    expected_gain: float = 0.25
    confidence: float = 0.50
    compute_cost: float = 0.25
    scientific_risk: float = 0.10
    guard_risk: float = 0.0
    matched_group: str | None = None
    pair_role: PairRole | None = None
    matched_axis: PairAxis | None = None
    rationale: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        for name in (
            "expected_gain",
            "confidence",
            "compute_cost",
            "scientific_risk",
            "guard_risk",
        ):
            value = getattr(self, name)
            if not math.isfinite(value) or not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be finite and within [0, 1]")
        pair_fields = (self.matched_group, self.pair_role, self.matched_axis)
        if any(value is None for value in pair_fields) and not all(
            value is None for value in pair_fields
        ):
            raise ValueError(
                "matched_group, pair_role and matched_axis must be supplied together"
            )
        if self.matched_group is not None and not self.matched_group.strip():
            raise ValueError("matched_group must not be blank")


@dataclass(frozen=True)
class ProposalEnvelope:
    """Typed, auditable proposal exchanged by logical committee members."""

    spec: ExperimentSpec
    proposer_id: str
    proposer_role: ProposerRole
    assessment: ProposalAssessment = field(default_factory=ProposalAssessment)

    def __post_init__(self) -> None:
        if not self.proposer_id.strip():
            raise ValueError("proposer_id must not be blank")

    @property
    def fingerprint(self) -> str:
        return proposal_fingerprint(self.spec)

    @property
    def utility(self) -> float:
        """Expected evidence value after fixed compute and science penalties."""

        value = self.assessment
        return (
            value.expected_gain * value.confidence
            - 0.20 * value.compute_cost
            - 0.25 * value.scientific_risk
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "spec": self.spec.to_dict(),
            "proposer_id": self.proposer_id,
            "proposer_role": self.proposer_role,
            "assessment": asdict(self.assessment),
            "fingerprint": self.fingerprint,
            "utility": self.utility,
        }


@dataclass(frozen=True)
class ProposalRejection:
    fingerprint: str
    proposer_id: str
    reason: str


@dataclass(frozen=True)
class CommitteeDisagreement:
    kind: Literal["candidate_set", "assessment"]
    proposer_ids: tuple[str, ...]
    fingerprints: tuple[str, ...]
    detail: str


@dataclass(frozen=True)
class ArbitrationDecision:
    accepted: tuple[ProposalEnvelope, ...]
    rejected: tuple[ProposalRejection, ...]
    disagreements: tuple[CommitteeDisagreement, ...]
    max_candidates: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "accepted": [item.to_dict() for item in self.accepted],
            "rejected": [asdict(item) for item in self.rejected],
            "disagreements": [asdict(item) for item in self.disagreements],
            "max_candidates": self.max_candidates,
        }


EnvelopeAssessor = Callable[
    [ExperimentSpec, TaskProfile, Sequence[RunResult]], ProposalAssessment
]


class EnvelopeProposer(Protocol):
    proposer_id: str
    proposer_role: ProposerRole

    def propose_envelopes(
        self, profile: TaskProfile, history: Sequence[RunResult]
    ) -> list[ProposalEnvelope]: ...


class BrainProposer:
    """Adapt an existing ``AgentBrain`` into a typed logical proposer."""

    proposer_role: ProposerRole
    default_assessment = ProposalAssessment()

    def __init__(
        self,
        brain: AgentBrain,
        *,
        proposer_id: str,
        assessor: EnvelopeAssessor | None = None,
    ) -> None:
        if not proposer_id.strip():
            raise ValueError("proposer_id must not be blank")
        self.brain = brain
        self.proposer_id = proposer_id
        self.assessor = assessor

    def set_context(self, **context: Any) -> None:
        setter = getattr(self.brain, "set_context", None)
        if callable(setter):
            setter(**context)

    def propose_envelopes(
        self, profile: TaskProfile, history: Sequence[RunResult]
    ) -> list[ProposalEnvelope]:
        envelopes = []
        for spec in self.brain.propose(profile, history):
            assessment = (
                self.assessor(spec, profile, history)
                if self.assessor is not None
                else self.default_assessment
            )
            envelopes.append(
                ProposalEnvelope(
                    spec=spec,
                    proposer_id=self.proposer_id,
                    proposer_role=self.proposer_role,
                    assessment=assessment,
                )
            )
        return envelopes


class DataStewardProposer(BrainProposer):
    """Logical expert for provenance, cleaning and split-safe data actions."""

    proposer_role: ProposerRole = "data_steward"
    default_assessment = ProposalAssessment(
        expected_gain=0.30,
        confidence=0.70,
        compute_cost=0.15,
        scientific_risk=0.10,
    )


class ModelResearcherProposer(BrainProposer):
    """Logical expert for models, representations and ensembles."""

    proposer_role: ProposerRole = "model_researcher"
    default_assessment = ProposalAssessment(
        expected_gain=0.45,
        confidence=0.55,
        compute_cost=0.45,
        scientific_risk=0.15,
    )


class SkepticProposer(BrainProposer):
    """Logical expert for controls, falsification and stopping evidence."""

    proposer_role: ProposerRole = "skeptic"
    default_assessment = ProposalAssessment(
        expected_gain=0.20,
        confidence=0.85,
        compute_cost=0.10,
        scientific_risk=0.05,
    )


class DeterministicArbiter:
    """Validate, deduplicate and budget committee proposals deterministically."""

    def __init__(
        self,
        *,
        max_candidates: int,
        max_guard_risk: float = 0.25,
        validator: Callable[[ExperimentSpec], None] = validate_experiment,
    ) -> None:
        if not 1 <= max_candidates <= 20:
            raise ValueError("max_candidates must be between 1 and 20")
        if not math.isfinite(max_guard_risk) or not 0.0 <= max_guard_risk <= 1.0:
            raise ValueError("max_guard_risk must be finite and within [0, 1]")
        self.max_candidates = max_candidates
        self.max_guard_risk = max_guard_risk
        self.validator = validator

    @staticmethod
    def _envelope_key(
        item: ProposalEnvelope,
    ) -> tuple[float, int, float, str, str]:
        return (
            item.assessment.guard_risk,
            # When two roles propose the same scientific configuration, keep
            # causal pair attribution.  Otherwise a slightly higher singleton
            # utility can erase a control's identity and orphan its treatment.
            int(item.assessment.matched_group is None),
            -item.utility,
            item.fingerprint,
            item.proposer_id,
        )

    @staticmethod
    def _unit_key(items: Sequence[ProposalEnvelope]) -> tuple[float, float, str]:
        return (
            max(item.assessment.guard_risk for item in items),
            -sum(item.utility for item in items) / len(items),
            ":".join(sorted(item.fingerprint for item in items)),
        )

    def arbitrate(self, envelopes: Sequence[ProposalEnvelope]) -> ArbitrationDecision:
        rejected: list[ProposalRejection] = []
        disagreements: list[CommitteeDisagreement] = []
        eligible: list[ProposalEnvelope] = []

        source_sets: dict[str, set[str]] = defaultdict(set)
        for item in envelopes:
            source_sets[item.proposer_id].add(item.fingerprint)
            if item.assessment.guard_risk > self.max_guard_risk:
                rejected.append(
                    ProposalRejection(
                        item.fingerprint, item.proposer_id, "guard_risk_exceeded"
                    )
                )
                continue
            try:
                self.validator(item.spec)
            except (ProtocolViolation, ValueError) as exc:
                rejected.append(
                    ProposalRejection(
                        item.fingerprint,
                        item.proposer_id,
                        f"guard_violation: {exc}",
                    )
                )
                continue
            eligible.append(item)

        if len(source_sets) > 1:
            distinct_sets = {frozenset(value) for value in source_sets.values()}
            if len(distinct_sets) > 1:
                disagreements.append(
                    CommitteeDisagreement(
                        kind="candidate_set",
                        proposer_ids=tuple(sorted(source_sets)),
                        fingerprints=tuple(
                            sorted({value for values in source_sets.values() for value in values})
                        ),
                        detail="logical proposers nominated different candidate sets",
                    )
                )

        by_fingerprint: dict[str, list[ProposalEnvelope]] = defaultdict(list)
        for item in eligible:
            by_fingerprint[item.fingerprint].append(item)

        unique: list[ProposalEnvelope] = []
        for fingerprint in sorted(by_fingerprint):
            variants = sorted(by_fingerprint[fingerprint], key=self._envelope_key)
            winner = variants[0]
            unique.append(winner)
            if len(variants) > 1:
                assessment_views = {
                    (
                        item.assessment.guard_risk,
                        item.assessment.matched_group,
                        item.assessment.pair_role,
                        item.assessment.matched_axis,
                        round(item.utility, 12),
                    )
                    for item in variants
                }
                if len(assessment_views) > 1:
                    disagreements.append(
                        CommitteeDisagreement(
                            kind="assessment",
                            proposer_ids=tuple(sorted({item.proposer_id for item in variants})),
                            fingerprints=(fingerprint,),
                            detail="duplicate configuration received different risk, pair or utility assessments",
                        )
                    )
                for duplicate in variants[1:]:
                    rejected.append(
                        ProposalRejection(
                            duplicate.fingerprint,
                            duplicate.proposer_id,
                            f"duplicate_of:{winner.proposer_id}",
                        )
                    )

        singletons: list[tuple[ProposalEnvelope, ...]] = []
        pair_groups: dict[str, list[ProposalEnvelope]] = defaultdict(list)
        for item in unique:
            group = item.assessment.matched_group
            if group is None:
                singletons.append((item,))
            else:
                pair_groups[group].append(item)

        units: list[tuple[ProposalEnvelope, ...]] = list(singletons)
        for group in sorted(pair_groups):
            members = pair_groups[group]
            by_role: dict[str, list[ProposalEnvelope]] = defaultdict(list)
            for item in members:
                # ProposalAssessment guarantees a non-None role for paired items.
                by_role[str(item.assessment.pair_role)].append(item)
            if not by_role.get("treatment") or not by_role.get("control"):
                for item in members:
                    rejected.append(
                        ProposalRejection(
                            item.fingerprint,
                            item.proposer_id,
                            f"unmatched_pair:{group}",
                        )
                    )
                continue
            selected_members = []
            for role in ("control", "treatment"):
                ranked = sorted(by_role[role], key=self._envelope_key)
                selected_members.append(ranked[0])
                for surplus in ranked[1:]:
                    rejected.append(
                        ProposalRejection(
                            surplus.fingerprint,
                            surplus.proposer_id,
                            f"superseded_{role}_in_pair:{group}",
                        )
                    )
            control, treatment = selected_members
            if not self._is_matched(control, treatment):
                for item in selected_members:
                    rejected.append(
                        ProposalRejection(
                            item.fingerprint,
                            item.proposer_id,
                            f"invalid_matched_pair:{group}",
                        )
                    )
                continue
            units.append(tuple(selected_members))

        accepted: list[ProposalEnvelope] = []
        for unit in sorted(units, key=self._unit_key):
            if len(accepted) + len(unit) > self.max_candidates:
                for item in unit:
                    rejected.append(
                        ProposalRejection(
                            item.fingerprint,
                            item.proposer_id,
                            "budget_exhausted",
                        )
                    )
                continue
            accepted.extend(unit)

        rejected.sort(key=lambda item: (item.fingerprint, item.proposer_id, item.reason))
        disagreements.sort(
            key=lambda item: (item.kind, item.fingerprints, item.proposer_ids)
        )
        return ArbitrationDecision(
            accepted=tuple(accepted),
            rejected=tuple(rejected),
            disagreements=tuple(disagreements),
            max_candidates=self.max_candidates,
        )

    @staticmethod
    def _is_matched(
        control: ProposalEnvelope, treatment: ProposalEnvelope
    ) -> bool:
        """Require an attributed pair to differ on exactly its declared axis."""

        if control.assessment.matched_axis != treatment.assessment.matched_axis:
            return False
        control_spec = control.spec
        treatment_spec = treatment.spec
        dimensions = {
            "feature": control_spec.representations != treatment_spec.representations,
            "model": (
                control_spec.model,
                control_spec.hyperparameters,
            )
            != (
                treatment_spec.model,
                treatment_spec.hyperparameters,
            ),
            "data": control_spec.data_policy != treatment_spec.data_policy,
            "imbalance": control_spec.imbalance != treatment_spec.imbalance,
        }
        changed = {name for name, differs in dimensions.items() if differs}
        return changed == {control.assessment.matched_axis}


class CommitteeBrain:
    """An ``AgentBrain``-compatible, single-executor committee facade."""

    def __init__(
        self,
        proposers: Sequence[EnvelopeProposer],
        *,
        max_candidates: int,
        max_guard_risk: float = 0.25,
    ) -> None:
        if not proposers:
            raise ValueError("CommitteeBrain requires at least one proposer")
        ids = [item.proposer_id for item in proposers]
        if len(ids) != len(set(ids)):
            raise ValueError("committee proposer_id values must be unique")
        self.proposers = tuple(proposers)
        self.max_candidates = max_candidates
        self.arbiter = DeterministicArbiter(
            max_candidates=max_candidates, max_guard_risk=max_guard_risk
        )
        self.last_decision: ArbitrationDecision | None = None
        self.last_finalist_audit: dict[str, Any] | None = None
        self.errors: list[str] = []

    def set_context(self, **context: Any) -> None:
        """Forward discovery context once per logical proposer."""

        for proposer in self.proposers:
            setter = getattr(proposer, "set_context", None)
            if not callable(setter):
                continue
            try:
                setter(**context)
            except TypeError as exc:
                self.errors.append(
                    f"{proposer.proposer_id} context rejected: {type(exc).__name__}: {exc}"
                )

    def drain_errors(self) -> list[str]:
        errors, self.errors = self.errors, []
        return errors

    def propose(
        self, profile: TaskProfile, history: Sequence[RunResult]
    ) -> list[ExperimentSpec]:
        envelopes: list[ProposalEnvelope] = []
        for proposer in self.proposers:
            try:
                envelopes.extend(proposer.propose_envelopes(profile, history))
            # Third-party/LLM-backed planners can fail through heterogeneous
            # exception types; isolation is the committee's explicit boundary.
            except Exception as exc:  # noqa: BLE001
                self.errors.append(
                    f"{proposer.proposer_id} proposal failed: {type(exc).__name__}: {exc}"
                )
        self.last_decision = self.arbiter.arbitrate(envelopes)

        proposals = []
        for envelope in self.last_decision.accepted:
            payload = envelope.spec.to_dict()
            payload["budget"] = {
                "max_candidates": self.max_candidates,
                "max_minutes": envelope.spec.budget.max_minutes,
            }
            proposals.append(ExperimentSpec.from_dict(payload))
        return proposals

    def select_finalist(
        self, profile: TaskProfile, results: Sequence[RunResult]
    ) -> str:
        """Select centrally by the registered metric without extra agent calls."""

        if not results:
            raise ValueError("Cannot select a finalist without completed runs")
        direction = -1.0 if profile.metric == "mae" else 1.0
        ranked = sorted(
            results,
            key=lambda item: (-direction * item.validation_score, item.run_id),
        )
        winner = ranked[0]
        self.last_finalist_audit = {
            "winner": winner.run_id,
            "metric": profile.metric,
            "ranking": tuple(item.run_id for item in ranked),
            "policy": "registered_validation_metric_then_run_id",
        }
        return winner.run_id
