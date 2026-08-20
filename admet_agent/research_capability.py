"""Non-compensatory Research Capability Scorecard v3.

The evaluator consumes dimensionless discovery effects produced by
``metric_normalization``.  It never reads datasets, predictions, or test
scores.  Frozen-test claims rely only on certification/firewall metadata and
cannot feed any value back into discovery routing.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from statistics import mean
from typing import Any

from .tasks import TASKS, resolve_task

SCORECARD_VERSION = "admet-research-capability-v3"
EXPECTED_TASKS = frozenset(task.name for task in TASKS.values())
PROCESS_DIMENSIONS = (
    "fault_discovery",
    "remediation",
    "causal_contrast",
    "falsification_and_negative_retention",
    "routing_calibration",
    "self_correction",
    "efficiency_and_stopping",
    "data_stewardship",
    "reproducibility",
)
REQUIRED_PROCESS_CONTROLS = frozenset(
    {
        "random_action_policy",
        "deterministic_rule_policy",
        "conventional_automl",
        "expanded_space_automl",
    }
)


def _clip(value: float) -> float:
    return float(min(1.0, max(0.0, value)))


def _finite(value: float | None) -> bool:
    return value is not None and math.isfinite(value)


def _rate(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator > 0 else None


def _valid_sha256(value: str | None) -> bool:
    return bool(
        value
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _validate_count(name: str, value: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")


@dataclass(frozen=True)
class EndpointValidationEvidence:
    """Validation-only, dimensionless evidence for one frozen endpoint."""

    task: str
    normalized_application_margin: float
    normalized_effect_vs_strong_fixed: float
    validation_seed_count: int
    normalization_profile_sha256: str
    evidence_sha256: str
    expanded_automl_normalized_delta: float | None = None
    expanded_automl_one_sided_lcb: float | None = None
    expanded_automl_same_budget: bool | None = None
    expanded_automl_protocol_match: bool | None = None

    def __post_init__(self) -> None:
        canonical = resolve_task(self.task).name
        if self.task != canonical:
            raise ValueError(f"task must use canonical name {canonical!r}")
        for name in (
            "normalized_application_margin",
            "normalized_effect_vs_strong_fixed",
        ):
            if not math.isfinite(float(getattr(self, name))):
                raise ValueError(f"{name} must be finite")
        _validate_count("validation_seed_count", self.validation_seed_count)
        if not _valid_sha256(self.normalization_profile_sha256):
            raise ValueError("normalization_profile_sha256 must be a SHA-256 digest")
        if not _valid_sha256(self.evidence_sha256):
            raise ValueError("evidence_sha256 must be a SHA-256 digest")
        optional_numbers = (
            self.expanded_automl_normalized_delta,
            self.expanded_automl_one_sided_lcb,
        )
        if any(
            value is not None and not math.isfinite(value) for value in optional_numbers
        ):
            raise ValueError(
                "expanded AutoML numeric evidence must be finite when present"
            )
        for name in ("expanded_automl_same_budget", "expanded_automl_protocol_match"):
            value = getattr(self, name)
            if value is not None and type(value) is not bool:
                raise ValueError(f"{name} must be boolean when present")

    @property
    def has_expanded_automl_evidence(self) -> bool:
        return (
            _finite(self.expanded_automl_normalized_delta)
            and _finite(self.expanded_automl_one_sided_lcb)
            and self.expanded_automl_same_budget is not None
            and self.expanded_automl_protocol_match is not None
        )


@dataclass(frozen=True)
class ProcessEvidence:
    faults_present: int = 0
    faults_detected: int = 0
    false_fault_alarms: int = 0
    remediation_opportunities: int = 0
    successful_remediations: int = 0
    causal_contrasts_attempted: int = 0
    valid_causal_contrasts: int = 0
    falsification_opportunities: int = 0
    completed_falsifications: int = 0
    negative_results_observed: int = 0
    negative_results_retained: int = 0
    routing_brier_score: float | None = None
    routing_expected_calibration_error: float | None = None
    self_correction_opportunities: int = 0
    successful_self_corrections: int = 0
    attempted_actions: int = 0
    useful_actions: int = 0
    baseline_compute_seconds: float | None = None
    agent_compute_seconds: float | None = None
    stopping_decisions: int = 0
    appropriate_stops: int = 0
    external_sources_used: int = 0
    complete_source_manifests: int = 0
    data_leakage_incidents: int = 0
    data_admission_opportunities: int = 0
    correct_data_admission_decisions: int = 0
    unsafe_data_admissions: int = 0
    reproducibility_requests: int = 0
    reproducible_runs: int = 0

    def __post_init__(self) -> None:
        for name, value in asdict(self).items():
            if name in {
                "routing_brier_score",
                "routing_expected_calibration_error",
                "baseline_compute_seconds",
                "agent_compute_seconds",
            }:
                continue
            _validate_count(name, value)
        for name in ("routing_brier_score", "routing_expected_calibration_error"):
            value = getattr(self, name)
            if value is not None and (not math.isfinite(value) or not 0 <= value <= 1):
                raise ValueError(f"{name} must be within [0, 1] when present")
        for name in ("baseline_compute_seconds", "agent_compute_seconds"):
            value = getattr(self, name)
            if value is not None and (not math.isfinite(value) or value < 0):
                raise ValueError(f"{name} must be finite and non-negative when present")
        bounded_counts = (
            (
                "faults_detected",
                self.faults_detected,
                "faults_present",
                self.faults_present,
            ),
            (
                "successful_remediations",
                self.successful_remediations,
                "remediation_opportunities",
                self.remediation_opportunities,
            ),
            (
                "valid_causal_contrasts",
                self.valid_causal_contrasts,
                "causal_contrasts_attempted",
                self.causal_contrasts_attempted,
            ),
            (
                "completed_falsifications",
                self.completed_falsifications,
                "falsification_opportunities",
                self.falsification_opportunities,
            ),
            (
                "negative_results_retained",
                self.negative_results_retained,
                "negative_results_observed",
                self.negative_results_observed,
            ),
            (
                "successful_self_corrections",
                self.successful_self_corrections,
                "self_correction_opportunities",
                self.self_correction_opportunities,
            ),
            (
                "useful_actions",
                self.useful_actions,
                "attempted_actions",
                self.attempted_actions,
            ),
            (
                "appropriate_stops",
                self.appropriate_stops,
                "stopping_decisions",
                self.stopping_decisions,
            ),
            (
                "complete_source_manifests",
                self.complete_source_manifests,
                "external_sources_used",
                self.external_sources_used,
            ),
            (
                "correct_data_admission_decisions",
                self.correct_data_admission_decisions,
                "data_admission_opportunities",
                self.data_admission_opportunities,
            ),
            (
                "unsafe_data_admissions",
                self.unsafe_data_admissions,
                "data_admission_opportunities",
                self.data_admission_opportunities,
            ),
            (
                "reproducible_runs",
                self.reproducible_runs,
                "reproducibility_requests",
                self.reproducibility_requests,
            ),
        )
        for numerator_name, numerator, denominator_name, denominator in bounded_counts:
            if numerator > denominator:
                raise ValueError(f"{numerator_name} cannot exceed {denominator_name}")


@dataclass(frozen=True)
class ProcessAdvantageEvidence:
    controls_evaluated: tuple[str, ...]
    same_budget: bool
    protocol_hash_match: bool
    dimension_lower_bounds: Mapping[str, float]

    def __post_init__(self) -> None:
        if (
            type(self.same_budget) is not bool
            or type(self.protocol_hash_match) is not bool
        ):
            raise ValueError("process advantage safety flags must be boolean")
        if any(
            not isinstance(control, str) or not control
            for control in self.controls_evaluated
        ):
            raise ValueError("controls_evaluated must contain non-empty strings")
        unknown = set(self.dimension_lower_bounds) - set(PROCESS_DIMENSIONS)
        if unknown:
            raise ValueError(f"unknown process dimensions: {sorted(unknown)}")
        if any(
            not math.isfinite(value) for value in self.dimension_lower_bounds.values()
        ):
            raise ValueError("process dimension lower bounds must be finite")


@dataclass(frozen=True)
class FrozenCertificationEvidence:
    certified_tasks: tuple[str, ...]
    freeze_manifest_sha256: str | None
    certification_report_sha256: str | None
    freeze_audit_passed: bool
    finalists_locked_before_test: bool
    one_shot_certification: bool
    official_test_loaded_during_discovery: bool
    test_feedback_to_discovery: bool

    def __post_init__(self) -> None:
        if any(not isinstance(task, str) or not task for task in self.certified_tasks):
            raise ValueError("certified_tasks must contain non-empty task names")
        for name in (
            "freeze_audit_passed",
            "finalists_locked_before_test",
            "one_shot_certification",
            "official_test_loaded_during_discovery",
            "test_feedback_to_discovery",
        ):
            if type(getattr(self, name)) is not bool:
                raise ValueError(f"{name} must be boolean")


@dataclass(frozen=True)
class SafetyEvidence:
    critical_faults_present: int
    critical_faults_detected: int
    test_feedback_events: int
    protocol_violations: int
    required_hash_items: int
    verified_hash_items: int
    frozen_certification: FrozenCertificationEvidence | None = None

    def __post_init__(self) -> None:
        for name in (
            "critical_faults_present",
            "critical_faults_detected",
            "test_feedback_events",
            "protocol_violations",
            "required_hash_items",
            "verified_hash_items",
        ):
            _validate_count(name, getattr(self, name))
        if self.critical_faults_detected > self.critical_faults_present:
            raise ValueError(
                "critical_faults_detected cannot exceed critical_faults_present"
            )
        if self.verified_hash_items > self.required_hash_items:
            raise ValueError("verified_hash_items cannot exceed required_hash_items")


@dataclass(frozen=True)
class CapabilityThresholds:
    application_floor_pass_rate: float = 0.90
    maximum_application_regret: float = 0.02
    predictive_task_coverage: float = 1.0
    minimum_validation_seeds: int = 5
    expanded_automl_noninferiority_margin: float = 0.01
    expanded_automl_pass_rate: float = 1.0
    process_component_floor: float = 0.60
    critical_fault_recall: float = 0.95
    hash_coverage: float = 1.0
    effect_target: float = 0.02

    def __post_init__(self) -> None:
        for name in (
            "application_floor_pass_rate",
            "predictive_task_coverage",
            "expanded_automl_pass_rate",
            "process_component_floor",
            "critical_fault_recall",
            "hash_coverage",
        ):
            value = getattr(self, name)
            if not math.isfinite(value) or not 0 <= value <= 1:
                raise ValueError(f"{name} must be within [0, 1]")
        for name in (
            "maximum_application_regret",
            "expanded_automl_noninferiority_margin",
            "effect_target",
        ):
            value = getattr(self, name)
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and non-negative")
        if self.effect_target == 0:
            raise ValueError("effect_target must be positive")
        if (
            isinstance(self.minimum_validation_seeds, bool)
            or not isinstance(self.minimum_validation_seeds, int)
            or self.minimum_validation_seeds < 1
        ):
            raise ValueError("minimum_validation_seeds must be positive")


@dataclass(frozen=True)
class PredictiveSufficiencyGate:
    passed: bool
    task_coverage: float
    covered_tasks: int
    required_tasks: int
    application_floor_pass_rate: float
    worst_normalized_application_regret: float | None
    mean_normalized_effect_vs_strong_fixed: float | None
    missing_tasks: tuple[str, ...]
    insufficient_seed_tasks: tuple[str, ...]
    blockers: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ExpandedAutoMLGate:
    passed: bool
    evidence_coverage: float
    noninferior_pass_rate: float
    worst_normalized_regret: float | None
    mean_normalized_delta: float | None
    same_budget_all_tasks: bool
    protocol_match_all_tasks: bool
    missing_tasks: tuple[str, ...]
    blockers: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class SafetyGate:
    validation_passed: bool
    frozen_certification_passed: bool
    critical_fault_recall: float | None
    hash_coverage: float | None
    test_feedback_events: int
    protocol_violations: int
    frozen_task_coverage: float
    blockers: tuple[str, ...]
    frozen_blockers: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class CapabilityDimension:
    score: float | None
    coverage: float
    passed: bool
    components: Mapping[str, float | None]
    blockers: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ProcessAdvantageGate:
    passed: bool
    absolute_process_passed: bool
    evidence_coverage: float
    controls_complete: bool
    same_budget: bool
    protocol_hash_match: bool
    missing_dimensions: tuple[str, ...]
    nonpositive_dimensions: tuple[str, ...]
    blockers: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ResearchCapabilityScorecard:
    claim_level: str
    claim_flags: Mapping[str, bool]
    predictive_sufficiency: PredictiveSufficiencyGate
    expanded_automl_noninferiority: ExpandedAutoMLGate
    safety: SafetyGate
    process_dimensions: Mapping[str, CapabilityDimension]
    process_advantage: ProcessAdvantageGate
    process_dimension_coverage: float
    minimum_observed_process_score: float | None
    claim_blockers: tuple[str, ...]
    version: str = SCORECARD_VERSION

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "claim_level": self.claim_level,
            "claim_flags": dict(self.claim_flags),
            "predictive_sufficiency": self.predictive_sufficiency.to_dict(),
            "expanded_automl_noninferiority": self.expanded_automl_noninferiority.to_dict(),
            "safety": self.safety.to_dict(),
            "process_dimensions": {
                name: dimension.to_dict()
                for name, dimension in self.process_dimensions.items()
            },
            "process_advantage": self.process_advantage.to_dict(),
            "process_dimension_coverage": self.process_dimension_coverage,
            "minimum_observed_process_score": self.minimum_observed_process_score,
            "claim_blockers": list(self.claim_blockers),
        }


def _evaluate_predictive(
    endpoints: Sequence[EndpointValidationEvidence],
    thresholds: CapabilityThresholds,
) -> PredictiveSufficiencyGate:
    by_task = {item.task: item for item in endpoints}
    missing = tuple(sorted(EXPECTED_TASKS - set(by_task)))
    sufficient = [
        item
        for item in endpoints
        if item.validation_seed_count >= thresholds.minimum_validation_seeds
    ]
    insufficient = tuple(
        sorted(
            item.task
            for item in endpoints
            if item.validation_seed_count < thresholds.minimum_validation_seeds
        )
    )
    coverage = len(sufficient) / len(EXPECTED_TASKS)
    pass_rate = (
        mean(float(item.normalized_application_margin >= 0) for item in sufficient)
        if sufficient
        else 0.0
    )
    worst_regret = max(
        (max(0.0, -item.normalized_application_margin) for item in sufficient),
        default=None,
    )
    blockers = []
    if coverage < thresholds.predictive_task_coverage:
        blockers.append("predictive_task_coverage_below_threshold")
    if pass_rate < thresholds.application_floor_pass_rate:
        blockers.append("application_floor_pass_rate_below_threshold")
    if worst_regret is None:
        blockers.append("application_regret_evidence_missing")
    elif worst_regret > thresholds.maximum_application_regret:
        blockers.append("worst_application_regret_above_threshold")
    return PredictiveSufficiencyGate(
        passed=not blockers,
        task_coverage=coverage,
        covered_tasks=len(sufficient),
        required_tasks=len(EXPECTED_TASKS),
        application_floor_pass_rate=pass_rate,
        worst_normalized_application_regret=worst_regret,
        mean_normalized_effect_vs_strong_fixed=(
            mean(item.normalized_effect_vs_strong_fixed for item in sufficient)
            if sufficient
            else None
        ),
        missing_tasks=missing,
        insufficient_seed_tasks=insufficient,
        blockers=tuple(blockers),
    )


def _evaluate_expanded_automl(
    endpoints: Sequence[EndpointValidationEvidence],
    thresholds: CapabilityThresholds,
) -> ExpandedAutoMLGate:
    rows = [item for item in endpoints if item.has_expanded_automl_evidence]
    present = {item.task for item in rows}
    missing = tuple(sorted(EXPECTED_TASKS - present))
    coverage = len(rows) / len(EXPECTED_TASKS)
    passes = [
        float(
            item.expanded_automl_one_sided_lcb
            >= -thresholds.expanded_automl_noninferiority_margin
        )
        for item in rows
    ]
    pass_rate = mean(passes) if passes else 0.0
    same_budget = bool(rows) and all(item.expanded_automl_same_budget for item in rows)
    protocol_match = bool(rows) and all(
        item.expanded_automl_protocol_match for item in rows
    )
    deltas = [float(item.expanded_automl_normalized_delta) for item in rows]
    blockers = []
    if coverage < 1.0:
        blockers.append("expanded_automl_coverage_incomplete")
    if pass_rate < thresholds.expanded_automl_pass_rate:
        blockers.append("expanded_automl_noninferiority_rate_below_threshold")
    if not same_budget:
        blockers.append("expanded_automl_budget_not_matched")
    if not protocol_match:
        blockers.append("expanded_automl_protocol_mismatch")
    return ExpandedAutoMLGate(
        passed=not blockers,
        evidence_coverage=coverage,
        noninferior_pass_rate=pass_rate,
        worst_normalized_regret=(
            max(max(0.0, -value) for value in deltas) if deltas else None
        ),
        mean_normalized_delta=mean(deltas) if deltas else None,
        same_budget_all_tasks=same_budget,
        protocol_match_all_tasks=protocol_match,
        missing_tasks=missing,
        blockers=tuple(blockers),
    )


def _evaluate_safety(
    evidence: SafetyEvidence,
    thresholds: CapabilityThresholds,
) -> SafetyGate:
    critical_recall = _rate(
        evidence.critical_faults_detected, evidence.critical_faults_present
    )
    hash_coverage = _rate(evidence.verified_hash_items, evidence.required_hash_items)
    blockers = []
    if critical_recall is None:
        blockers.append("critical_fault_evidence_missing")
    elif critical_recall < thresholds.critical_fault_recall:
        blockers.append("critical_fault_recall_below_threshold")
    if evidence.test_feedback_events:
        blockers.append("test_feedback_detected")
    if evidence.protocol_violations:
        blockers.append("protocol_violation_detected")
    if hash_coverage is None:
        blockers.append("hash_evidence_missing")
    elif hash_coverage < thresholds.hash_coverage:
        blockers.append("hash_coverage_below_threshold")

    frozen = evidence.frozen_certification
    frozen_blockers = []
    frozen_coverage = 0.0
    if frozen is None:
        frozen_blockers.append("frozen_certification_evidence_missing")
    else:
        canonical = set()
        for task in frozen.certified_tasks:
            try:
                canonical.add(resolve_task(task).name)
            except ValueError:
                frozen_blockers.append("frozen_certification_contains_unknown_task")
        frozen_coverage = len(canonical & EXPECTED_TASKS) / len(EXPECTED_TASKS)
        if len(frozen.certified_tasks) != len(canonical):
            frozen_blockers.append("frozen_certification_contains_duplicate_task")
        if canonical != EXPECTED_TASKS:
            frozen_blockers.append("frozen_certification_task_coverage_incomplete")
        if not _valid_sha256(frozen.freeze_manifest_sha256):
            frozen_blockers.append("freeze_manifest_hash_missing_or_invalid")
        if not _valid_sha256(frozen.certification_report_sha256):
            frozen_blockers.append("certification_report_hash_missing_or_invalid")
        if not frozen.freeze_audit_passed:
            frozen_blockers.append("freeze_audit_not_passed")
        if not frozen.finalists_locked_before_test:
            frozen_blockers.append("finalists_not_locked_before_test")
        if not frozen.one_shot_certification:
            frozen_blockers.append("certification_not_one_shot")
        if frozen.official_test_loaded_during_discovery:
            frozen_blockers.append("official_test_loaded_during_discovery")
        if frozen.test_feedback_to_discovery:
            frozen_blockers.append("frozen_test_feedback_to_discovery")
    validation_passed = not blockers
    return SafetyGate(
        validation_passed=validation_passed,
        frozen_certification_passed=validation_passed and not frozen_blockers,
        critical_fault_recall=critical_recall,
        hash_coverage=hash_coverage,
        test_feedback_events=evidence.test_feedback_events,
        protocol_violations=evidence.protocol_violations,
        frozen_task_coverage=frozen_coverage,
        blockers=tuple(blockers),
        frozen_blockers=tuple(frozen_blockers),
    )


def _dimension(
    components: Mapping[str, float | None],
    thresholds: CapabilityThresholds,
) -> CapabilityDimension:
    available = [value for value in components.values() if value is not None]
    coverage = len(available) / len(components)
    score = min(available) if available else None
    blockers = [name for name, value in components.items() if value is None]
    if score is not None and score < thresholds.process_component_floor:
        blockers.append("component_score_below_floor")
    return CapabilityDimension(
        score=score,
        coverage=coverage,
        passed=coverage == 1.0
        and score is not None
        and score >= thresholds.process_component_floor,
        components=dict(components),
        blockers=tuple(blockers),
    )


def _evaluate_process(
    evidence: ProcessEvidence,
    endpoints: Sequence[EndpointValidationEvidence],
    thresholds: CapabilityThresholds,
) -> dict[str, CapabilityDimension]:
    if evidence.faults_present > 0:
        recall = evidence.faults_detected / evidence.faults_present
        predicted = evidence.faults_detected + evidence.false_fault_alarms
        precision = evidence.faults_detected / predicted if predicted else 0.0
        fault_f1 = (
            2 * precision * recall / (precision + recall)
            if precision + recall > 0
            else 0.0
        )
    else:
        fault_f1 = None
    positive_effect = (
        mean(max(0.0, item.normalized_effect_vs_strong_fixed) for item in endpoints)
        if endpoints
        else None
    )
    action_yield = _rate(evidence.useful_actions, evidence.attempted_actions)
    compute_efficiency = None
    if (
        positive_effect is not None
        and evidence.baseline_compute_seconds is not None
        and evidence.agent_compute_seconds is not None
        and evidence.baseline_compute_seconds > 0
    ):
        cost_ratio = evidence.agent_compute_seconds / evidence.baseline_compute_seconds
        compute_efficiency = _clip(
            (positive_effect / thresholds.effect_target)
            / max(1.0, math.log2(1.0 + cost_ratio))
        )
    source_manifest_rate = _rate(
        evidence.complete_source_manifests, evidence.external_sources_used
    )
    admission_accuracy = _rate(
        evidence.correct_data_admission_decisions,
        evidence.data_admission_opportunities,
    )
    unsafe_merge_avoidance = (
        None
        if evidence.data_admission_opportunities == 0
        else float(evidence.unsafe_data_admissions == 0)
    )
    if evidence.data_leakage_incidents or evidence.unsafe_data_admissions:
        source_manifest_rate = 0.0
        unsafe_merge_avoidance = 0.0
    routing_components = {
        "one_minus_brier": (
            None
            if evidence.routing_brier_score is None
            else 1.0 - evidence.routing_brier_score
        ),
        "one_minus_ece": (
            None
            if evidence.routing_expected_calibration_error is None
            else 1.0 - evidence.routing_expected_calibration_error
        ),
    }
    return {
        "fault_discovery": _dimension({"fault_f1": fault_f1}, thresholds),
        "remediation": _dimension(
            {
                "successful_remediation_rate": _rate(
                    evidence.successful_remediations, evidence.remediation_opportunities
                )
            },
            thresholds,
        ),
        "causal_contrast": _dimension(
            {
                "valid_contrast_rate": _rate(
                    evidence.valid_causal_contrasts, evidence.causal_contrasts_attempted
                )
            },
            thresholds,
        ),
        "falsification_and_negative_retention": _dimension(
            {
                "falsification_completion_rate": _rate(
                    evidence.completed_falsifications,
                    evidence.falsification_opportunities,
                ),
                "negative_result_retention_rate": _rate(
                    evidence.negative_results_retained,
                    evidence.negative_results_observed,
                ),
            },
            thresholds,
        ),
        "routing_calibration": _dimension(routing_components, thresholds),
        "self_correction": _dimension(
            {
                "self_correction_rate": _rate(
                    evidence.successful_self_corrections,
                    evidence.self_correction_opportunities,
                )
            },
            thresholds,
        ),
        "efficiency_and_stopping": _dimension(
            {
                "action_yield": action_yield,
                "compute_efficiency": compute_efficiency,
                "appropriate_stopping_rate": _rate(
                    evidence.appropriate_stops, evidence.stopping_decisions
                ),
            },
            thresholds,
        ),
        "data_stewardship": _dimension(
            {
                "source_manifest_completeness": source_manifest_rate,
                "data_admission_accuracy": admission_accuracy,
                "unsafe_merge_avoidance": unsafe_merge_avoidance,
            },
            thresholds,
        ),
        "reproducibility": _dimension(
            {
                "reproducible_run_rate": _rate(
                    evidence.reproducible_runs, evidence.reproducibility_requests
                )
            },
            thresholds,
        ),
    }


def _evaluate_process_advantage(
    dimensions: Mapping[str, CapabilityDimension],
    evidence: ProcessAdvantageEvidence | None,
) -> ProcessAdvantageGate:
    absolute_passed = all(item.passed for item in dimensions.values())
    if evidence is None:
        return ProcessAdvantageGate(
            passed=False,
            absolute_process_passed=absolute_passed,
            evidence_coverage=0.0,
            controls_complete=False,
            same_budget=False,
            protocol_hash_match=False,
            missing_dimensions=PROCESS_DIMENSIONS,
            nonpositive_dimensions=(),
            blockers=("process_advantage_evidence_missing",),
        )
    supplied = set(evidence.dimension_lower_bounds)
    missing = tuple(sorted(set(PROCESS_DIMENSIONS) - supplied))
    nonpositive = tuple(
        sorted(
            name
            for name, value in evidence.dimension_lower_bounds.items()
            if name in PROCESS_DIMENSIONS and value <= 0
        )
    )
    controls_complete = REQUIRED_PROCESS_CONTROLS <= set(evidence.controls_evaluated)
    blockers = []
    if not absolute_passed:
        blockers.append("absolute_process_component_floor_failed")
    if missing:
        blockers.append("process_advantage_dimension_coverage_incomplete")
    if nonpositive:
        blockers.append("process_advantage_not_positive_for_every_dimension")
    if not controls_complete:
        blockers.append("required_process_controls_missing")
    if not evidence.same_budget:
        blockers.append("process_controls_not_same_budget")
    if not evidence.protocol_hash_match:
        blockers.append("process_control_protocol_mismatch")
    return ProcessAdvantageGate(
        passed=not blockers,
        absolute_process_passed=absolute_passed,
        evidence_coverage=len(supplied & set(PROCESS_DIMENSIONS))
        / len(PROCESS_DIMENSIONS),
        controls_complete=controls_complete,
        same_budget=evidence.same_budget,
        protocol_hash_match=evidence.protocol_hash_match,
        missing_dimensions=missing,
        nonpositive_dimensions=nonpositive,
        blockers=tuple(blockers),
    )


def evaluate_research_capability(
    endpoints: Sequence[EndpointValidationEvidence],
    process: ProcessEvidence,
    safety: SafetyEvidence,
    *,
    process_advantage: ProcessAdvantageEvidence | None = None,
    thresholds: CapabilityThresholds | None = None,
) -> ResearchCapabilityScorecard:
    """Evaluate v2 evidence without loading or accepting any test score."""

    thresholds = thresholds or CapabilityThresholds()
    names = [item.task for item in endpoints]
    if len(names) != len(set(names)):
        raise ValueError("Only one endpoint validation record per TDC task is allowed")
    predictive = _evaluate_predictive(endpoints, thresholds)
    automl = _evaluate_expanded_automl(endpoints, thresholds)
    safety_gate = _evaluate_safety(safety, thresholds)
    dimensions = _evaluate_process(process, endpoints, thresholds)
    advantage = _evaluate_process_advantage(dimensions, process_advantage)
    observed_scores = [
        item.score for item in dimensions.values() if item.score is not None
    ]
    dimension_coverage = mean(item.coverage for item in dimensions.values())

    blockers = []
    if not safety_gate.validation_passed:
        blockers.extend(safety_gate.blockers)
    if not predictive.passed:
        blockers.extend(predictive.blockers)
    foundation = safety_gate.validation_passed and predictive.passed
    flags = {
        "validation_application_performance_qualified": foundation,
        "frozen_application_performance_certified": (
            foundation and safety_gate.frozen_certification_passed
        ),
        "validation_process_advantage": foundation and advantage.passed,
        "expanded_automl_predictive_noninferiority": foundation and automl.passed,
        "frozen_application_qualified_process_advantage": (
            foundation and safety_gate.frozen_certification_passed and advantage.passed
        ),
        "frozen_process_advantage_plus_automl_noninferiority": (
            foundation
            and safety_gate.frozen_certification_passed
            and advantage.passed
            and automl.passed
        ),
    }
    if not foundation:
        claim = "insufficient_or_unsafe_evidence"
    elif flags["frozen_process_advantage_plus_automl_noninferiority"]:
        claim = "frozen_process_advantage_plus_automl_noninferiority"
    elif flags["frozen_application_qualified_process_advantage"]:
        claim = "frozen_application_qualified_process_advantage"
    elif advantage.passed and automl.passed:
        claim = "validation_process_advantage_plus_automl_noninferiority"
    elif advantage.passed:
        claim = "validation_process_advantage"
    elif safety_gate.frozen_certification_passed:
        claim = "frozen_application_performance_qualified"
    elif automl.passed:
        claim = "validation_expanded_automl_noninferior"
    else:
        claim = "validation_predictive_sufficiency"
    if foundation and not advantage.passed:
        blockers.extend(advantage.blockers)
    if foundation and not automl.passed:
        blockers.extend(automl.blockers)
    if foundation and not safety_gate.frozen_certification_passed:
        blockers.extend(safety_gate.frozen_blockers)
    return ResearchCapabilityScorecard(
        claim_level=claim,
        claim_flags=flags,
        predictive_sufficiency=predictive,
        expanded_automl_noninferiority=automl,
        safety=safety_gate,
        process_dimensions=dimensions,
        process_advantage=advantage,
        process_dimension_coverage=dimension_coverage,
        minimum_observed_process_score=min(observed_scores)
        if observed_scores
        else None,
        claim_blockers=tuple(dict.fromkeys(blockers)),
    )


__all__ = [
    "PROCESS_DIMENSIONS",
    "SCORECARD_VERSION",
    "CapabilityDimension",
    "CapabilityThresholds",
    "EndpointValidationEvidence",
    "ExpandedAutoMLGate",
    "FrozenCertificationEvidence",
    "PredictiveSufficiencyGate",
    "ProcessAdvantageEvidence",
    "ProcessAdvantageGate",
    "ProcessEvidence",
    "ResearchCapabilityScorecard",
    "SafetyEvidence",
    "SafetyGate",
    "evaluate_research_capability",
]
