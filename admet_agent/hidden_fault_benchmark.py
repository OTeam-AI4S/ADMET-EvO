"""Standard scoring for hidden scientific/data fault challenges."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Any

SUPPORTED_FAULTS = frozenset(
    {
        "invalid_molecular_structure",
        "invalid_target",
        "conflicting_class_labels",
        "replicate_measurements",
        "high_variance_replicate_measurements",
        "exact_split_overlap",
        "extreme_class_imbalance",
        "chemical_domain_shift",
        "local_label_roughness",
        "species_mismatch",
        "assay_mismatch",
        "unit_or_transform_mismatch",
        "wrong_metric_direction",
        "provenance_missing",
    }
)

CRITICAL_FAULTS = frozenset(
    {
        "exact_split_overlap",
        "unit_or_transform_mismatch",
        "wrong_metric_direction",
        "species_mismatch",
        "assay_mismatch",
        "provenance_missing",
    }
)


@dataclass(frozen=True)
class FaultChallengeResult:
    challenge_id: str
    task: str
    expected_faults: tuple[str, ...]
    detected_faults: tuple[str, ...]
    detection_seconds: float
    remediations_proposed: int
    valid_remediations: int

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> FaultChallengeResult:
        payload = dict(value)
        payload["expected_faults"] = tuple(payload["expected_faults"])
        payload["detected_faults"] = tuple(payload["detected_faults"])
        return cls(**payload)

    def validate(self) -> None:
        unknown = (set(self.expected_faults) | set(self.detected_faults)) - SUPPORTED_FAULTS
        if unknown:
            raise ValueError(f"Unknown fault types: {sorted(unknown)}")
        if self.detection_seconds < 0:
            raise ValueError("detection_seconds must be non-negative")
        if not 0 <= self.valid_remediations <= self.remediations_proposed:
            raise ValueError("valid_remediations must be within proposed remediation count")


def score_hidden_faults(rows: list[FaultChallengeResult]) -> dict[str, Any]:
    if not rows:
        raise ValueError("Hidden-fault results are required")
    ids = [item.challenge_id for item in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("Hidden-fault challenge IDs must be unique")
    tp = fp = fn = 0
    critical_expected = critical_detected = 0
    case_scores = []
    remediation_scores = []
    for item in rows:
        item.validate()
        expected = set(item.expected_faults)
        detected = set(item.detected_faults)
        item_tp = len(expected & detected)
        item_fp = len(detected - expected)
        item_fn = len(expected - detected)
        tp += item_tp
        fp += item_fp
        fn += item_fn
        if not expected:
            f1 = 1.0 if not detected else 0.0
        else:
            precision = item_tp / max(1, item_tp + item_fp)
            recall = item_tp / max(1, item_tp + item_fn)
            f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        case_scores.append(f1)
        critical = expected & CRITICAL_FAULTS
        critical_expected += len(critical)
        critical_detected += len(critical & detected)
        remediation_scores.append(
            1.0
            if not expected and not detected
            else item.valid_remediations / max(1, item.remediations_proposed)
        )
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    critical_recall = critical_detected / critical_expected if critical_expected else 0.0
    remediation = mean(remediation_scores)
    latency = mean(item.detection_seconds for item in rows)
    # Safety-critical misses are gated more strongly than ordinary taxonomy
    # errors.  A benchmark with no critical cases cannot claim safety evidence.
    safety_gate = critical_expected > 0 and critical_recall == 1.0
    overall = 0.45 * f1 + 0.25 * critical_recall + 0.20 * remediation + 0.10 * mean(case_scores)
    if not safety_gate:
        overall = min(overall, 0.5)
    return {
        "version": "hidden-scientific-fault-benchmark-v1",
        "challenges": len(rows),
        "micro_precision": precision,
        "micro_recall": recall,
        "micro_f1": f1,
        "macro_case_f1": mean(case_scores),
        "critical_recall": critical_recall,
        "critical_expected": critical_expected,
        "remediation_validity": remediation,
        "mean_detection_seconds": latency,
        "safety_gate_passed": safety_gate,
        "overall_score": overall,
    }


def evaluate_fault_file(path: Path, output: Path | None = None) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    values = payload["challenges"] if isinstance(payload, dict) else payload
    score = score_hidden_faults(
        [FaultChallengeResult.from_dict(item) for item in values]
    )
    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(score, indent=2, sort_keys=True), encoding="utf-8")
    return score
