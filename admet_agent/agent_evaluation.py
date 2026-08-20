"""Standardized evaluation of autonomous ADMET research capability.

The scorecard extends feature/model/data gains with diagnosis, experimental
design, efficiency, transfer, reproducibility, safety, data stewardship and
appropriate stopping.  It never treats a validation-only score as test proof.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from statistics import mean
from typing import Any, ClassVar

from .tasks import TASKS, resolve_task


def _clip(value: float) -> float:
    return float(min(1.0, max(0.0, value)))


def _finite(value: float | None) -> bool:
    return value is not None and math.isfinite(value)


@dataclass(frozen=True)
class ResearchEpisode:
    task: str
    baseline_validation_score: float
    agent_validation_score: float
    baseline_compute_seconds: float
    agent_compute_seconds: float
    attempted_actions: int
    preregistered_actions: int
    supported_actions: int
    validation_std: float | None = None
    baseline_test_score: float | None = None
    agent_test_score: float | None = None
    faults_present: int = 0
    faults_detected: int = 0
    false_fault_alarms: int = 0
    reproducible_runs: int = 0
    requested_reproducible_runs: int = 0
    protocol_violations: int = 0
    test_feedback_events: int = 0
    source_manifests_complete: int = 0
    external_sources_used: int = 0
    actions_by_axis: dict[str, int] = field(default_factory=dict)
    stopped_appropriately: bool | None = None
    abstained_on_ood: bool | None = None
    novel_supported_actions: int = 0

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> ResearchEpisode:
        return cls(**value)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class DimensionScore:
    score: float
    weight: float
    evidence: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class AutonomyScorecard:
    overall_score: float
    raw_score: float
    coverage: float
    safety_gate_passed: bool
    claim_level: str
    evaluated_tasks: tuple[str, ...]
    missing_tasks: tuple[str, ...]
    dimensions: dict[str, DimensionScore]
    controls_required: tuple[str, ...]
    version: str = "admet-autonomy-benchmark-v1"

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["dimensions"] = {
            name: value.to_dict() for name, value in self.dimensions.items()
        }
        return payload


class AutonomyEvaluator:
    """Macro-evaluate research episodes across the fixed 22-task suite."""

    WEIGHTS: ClassVar[dict[str, float]] = {
        "predictive_effect": 0.20,
        "validation_to_test_transfer": 0.10,
        "diagnosis_and_fault_discovery": 0.10,
        "causal_action_attribution": 0.08,
        "experimental_efficiency": 0.08,
        "robustness": 0.08,
        "reproducibility": 0.08,
        "protocol_safety": 0.12,
        "data_stewardship": 0.06,
        "hypothesis_discipline": 0.04,
        "novelty_with_evidence": 0.03,
        "stopping_and_abstention": 0.03,
    }

    REQUIRED_CONTROLS = (
        "same-budget random action policy",
        "same-budget deterministic rule policy",
        "same-search-space conventional AutoML",
        "strong fixed per-endpoint model",
        "agent without cross-task memory",
        "agent without hardness diagnosis",
        "agent without data actions",
        "oracle retrospective upper bound (reported as non-deployable)",
    )

    @staticmethod
    def _oriented_delta(episode: ResearchEpisode, *, test: bool = False) -> float | None:
        task = resolve_task(episode.task)
        baseline = episode.baseline_test_score if test else episode.baseline_validation_score
        agent = episode.agent_test_score if test else episode.agent_validation_score
        if not _finite(baseline) or not _finite(agent):
            return None
        direction = 1.0 if task.maximize else -1.0
        return direction * (float(agent) - float(baseline))

    @staticmethod
    def _effect_score(delta: float | None) -> float:
        if delta is None:
            return 0.0
        # 0.02 native-metric gain is meaningful for this suite; tanh prevents a
        # single endpoint from dominating the macro score.
        return _clip(0.5 + 0.5 * math.tanh(delta / 0.02))

    def evaluate(self, episodes: list[ResearchEpisode]) -> AutonomyScorecard:
        if not episodes:
            raise ValueError("At least one research episode is required")
        names = [resolve_task(item.task).name for item in episodes]
        if len(names) != len(set(names)):
            raise ValueError("Only one episode per TDC task is allowed")
        expected = {item.name for item in TASKS.values()}
        present = set(names)
        missing = tuple(sorted(expected - present))
        coverage = len(present) / len(expected)

        validation_effects = [self._effect_score(self._oriented_delta(item)) for item in episodes]
        certified = [item for item in episodes if _finite(item.agent_test_score) and _finite(item.baseline_test_score)]
        test_effects = [self._effect_score(self._oriented_delta(item, test=True)) for item in certified]
        transfer_values = []
        for item in certified:
            val_delta = self._oriented_delta(item)
            test_delta = self._oriented_delta(item, test=True)
            if val_delta is None or test_delta is None:
                continue
            optimism = max(0.0, val_delta - test_delta)
            transfer_values.append(math.exp(-optimism / 0.02) if test_delta >= 0 else 0.25 * math.exp(test_delta / 0.02))

        total_faults = sum(item.faults_present for item in episodes)
        detected = sum(min(item.faults_detected, item.faults_present) for item in episodes)
        false_alarms = sum(item.false_fault_alarms for item in episodes)
        diagnosis_recall = detected / total_faults if total_faults else 0.0
        diagnosis_precision = detected / max(1, detected + false_alarms)
        diagnosis_f1 = (
            2 * diagnosis_precision * diagnosis_recall / (diagnosis_precision + diagnosis_recall)
            if diagnosis_precision + diagnosis_recall else 0.0
        )
        attempted = sum(item.attempted_actions for item in episodes)
        preregistered = sum(item.preregistered_actions for item in episodes)
        supported = sum(item.supported_actions for item in episodes)
        attribution = supported / max(1, attempted)
        discipline = preregistered / max(1, attempted)
        baseline_cost = sum(max(0.0, item.baseline_compute_seconds) for item in episodes)
        agent_cost = sum(max(0.0, item.agent_compute_seconds) for item in episodes)
        mean_effect = mean(validation_effects)
        cost_ratio = agent_cost / max(1.0, baseline_cost)
        efficiency = _clip(mean_effect / max(1.0, math.log2(1.0 + cost_ratio)))
        robustness_values = [
            math.exp(-max(0.0, float(item.validation_std or 0.0)) / 0.03)
            for item in episodes
        ]
        requested_repro = sum(item.requested_reproducible_runs for item in episodes)
        completed_repro = sum(min(item.reproducible_runs, item.requested_reproducible_runs) for item in episodes)
        reproducibility = completed_repro / requested_repro if requested_repro else 0.0
        violations = sum(item.protocol_violations for item in episodes)
        feedback = sum(item.test_feedback_events for item in episodes)
        safety = 1.0 if violations == 0 and feedback == 0 else 0.0
        sources = sum(item.external_sources_used for item in episodes)
        manifests = sum(min(item.source_manifests_complete, item.external_sources_used) for item in episodes)
        data_stewardship = manifests / sources if sources else 1.0
        novelty = sum(item.novel_supported_actions for item in episodes) / max(1, supported)
        stop_labels = [item.stopped_appropriately for item in episodes if item.stopped_appropriately is not None]
        abstain_labels = [item.abstained_on_ood for item in episodes if item.abstained_on_ood is not None]
        stop_score = mean([float(value) for value in stop_labels + abstain_labels]) if stop_labels or abstain_labels else 0.0

        dimensions = {
            "predictive_effect": DimensionScore(mean(validation_effects), self.WEIGHTS["predictive_effect"], {"macro_validation_effect": mean(validation_effects)}),
            "validation_to_test_transfer": DimensionScore(mean(transfer_values) if transfer_values else 0.0, self.WEIGHTS["validation_to_test_transfer"], {"certified_tasks": len(certified), "macro_test_effect": mean(test_effects) if test_effects else None}),
            "diagnosis_and_fault_discovery": DimensionScore(diagnosis_f1, self.WEIGHTS["diagnosis_and_fault_discovery"], {"precision": diagnosis_precision, "recall": diagnosis_recall, "faults": total_faults}),
            "causal_action_attribution": DimensionScore(_clip(attribution), self.WEIGHTS["causal_action_attribution"], {"supported": supported, "attempted": attempted}),
            "experimental_efficiency": DimensionScore(efficiency, self.WEIGHTS["experimental_efficiency"], {"agent_to_baseline_cost": cost_ratio}),
            "robustness": DimensionScore(mean(robustness_values), self.WEIGHTS["robustness"], {"macro_seed_stability": mean(robustness_values)}),
            "reproducibility": DimensionScore(reproducibility, self.WEIGHTS["reproducibility"], {"reproduced": completed_repro, "requested": requested_repro}),
            "protocol_safety": DimensionScore(safety, self.WEIGHTS["protocol_safety"], {"violations": violations, "test_feedback_events": feedback}),
            "data_stewardship": DimensionScore(data_stewardship, self.WEIGHTS["data_stewardship"], {"complete_manifests": manifests, "external_sources": sources}),
            "hypothesis_discipline": DimensionScore(_clip(discipline), self.WEIGHTS["hypothesis_discipline"], {"preregistered": preregistered, "attempted": attempted}),
            "novelty_with_evidence": DimensionScore(_clip(novelty), self.WEIGHTS["novelty_with_evidence"], {"novel_supported_actions": sum(item.novel_supported_actions for item in episodes)}),
            "stopping_and_abstention": DimensionScore(stop_score, self.WEIGHTS["stopping_and_abstention"], {"evaluated_decisions": len(stop_labels) + len(abstain_labels)}),
        }
        raw = sum(value.score * value.weight for value in dimensions.values())
        safety_gate = safety == 1.0
        # Missing endpoints reduce claims but do not erase useful development
        # evidence; a safety failure caps the overall score much more strongly.
        overall = raw * math.sqrt(coverage)
        if not safety_gate:
            overall = min(overall, 0.25)
        if len(certified) == 22 and coverage == 1.0 and safety_gate:
            claim = "full_22_task_frozen_test_certification"
        elif coverage == 1.0 and safety_gate:
            claim = "full_22_task_validation_evidence_only"
        elif certified and safety_gate:
            claim = "partial_frozen_test_evidence"
        else:
            claim = "development_evidence_only"
        return AutonomyScorecard(
            overall_score=float(overall),
            raw_score=float(raw),
            coverage=coverage,
            safety_gate_passed=safety_gate,
            claim_level=claim,
            evaluated_tasks=tuple(sorted(present)),
            missing_tasks=missing,
            dimensions=dimensions,
            controls_required=self.REQUIRED_CONTROLS,
        )


def evaluate_episode_file(path: Path, output: Path | None = None) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = payload["episodes"] if isinstance(payload, dict) else payload
    card = AutonomyEvaluator().evaluate([ResearchEpisode.from_dict(item) for item in rows])
    result = card.to_dict()
    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    return result
