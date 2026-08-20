"""Train-only metric normalization for cross-endpoint ADMET evidence.

Profiles are created exclusively from a task contract and its training labels.
They contain no validation or test statistics, so the same frozen profile can
be used for discovery-side action evidence and later one-shot certification.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
from scipy.stats import t as student_t

from .tasks import TaskSpec, resolve_task

NORMALIZATION_VERSION = "tdc22-train-only-metric-normalization-v2"
SUPPORTED_METRICS = frozenset({"auroc", "auprc", "spearman", "mae"})
DEFAULT_MINIMUM_NORMALIZED_EFFECT = 0.01
DEFAULT_LCB_CONFIDENCE = 0.90
_AUPRC_MINIMUM_GAIN_SCALE = 0.10
_MINIMUM_REGRESSION_SCALE = 1e-8


def _canonical_sha256(payload: dict[str, Any]) -> str:
    encoded = json.dumps(
        payload,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _training_reference_sha256(values: np.ndarray) -> str:
    # float.hex is independent of locale and JSON float formatting details.
    payload = {"ordered_float64_hex": [float(value).hex() for value in values]}
    return _canonical_sha256(payload)


def _finite_training_targets(train_targets: Sequence[float]) -> np.ndarray:
    values = np.asarray(train_targets, dtype=float)
    if values.ndim != 1 or len(values) < 2:
        raise ValueError(
            "train_targets must be a one-dimensional sequence of at least two values"
        )
    if not np.all(np.isfinite(values)):
        raise ValueError("train_targets must contain only finite values")
    return values


def _robust_regression_scale(values: np.ndarray) -> tuple[float, str]:
    median = float(np.median(values))
    mad_scale = 1.4826 * float(np.median(np.abs(values - median)))
    q25, q75 = np.quantile(values, [0.25, 0.75])
    iqr_scale = float(q75 - q25) / 1.349
    robust = max(mad_scale, iqr_scale)
    if math.isfinite(robust) and robust >= _MINIMUM_REGRESSION_SCALE:
        return robust, "max(1.4826*MAD(train_y),IQR(train_y)/1.349)"
    fallback = float(np.std(values, ddof=1))
    if math.isfinite(fallback) and fallback >= _MINIMUM_REGRESSION_SCALE:
        return fallback, "sample_std(train_y)_degenerate_robust_fallback"
    raise ValueError("MAE normalization requires non-constant training targets")


@dataclass(frozen=True)
class MetricNormalizationProfile:
    """Immutable discovery-preregistered metric normalization profile."""

    task: str
    metric: str
    maximize: bool
    null_reference: float
    gain_scale: float
    gain_scale_method: str
    train_count: int
    train_positive_rate: float | None
    training_reference_sha256: str
    version: str = NORMALIZATION_VERSION
    scale_source: str = "train_only"

    def __post_init__(self) -> None:
        task = resolve_task(self.task)
        if self.task != task.name:
            raise ValueError(f"task must use canonical name {task.name!r}")
        if self.metric not in SUPPORTED_METRICS or self.metric != task.metric:
            raise ValueError(f"metric must match the task contract ({task.metric})")
        if self.maximize != task.maximize:
            raise ValueError("maximize must match the task metric direction")
        if not math.isfinite(self.null_reference):
            raise ValueError("null_reference must be finite")
        if not math.isfinite(self.gain_scale) or self.gain_scale <= 0:
            raise ValueError("gain_scale must be finite and positive")
        if self.train_count < 2:
            raise ValueError("train_count must be at least two")
        if (
            self.train_positive_rate is not None
            and not 0 < self.train_positive_rate < 1
        ):
            raise ValueError(
                "train_positive_rate must be strictly between zero and one"
            )
        if self.version != NORMALIZATION_VERSION or self.scale_source != "train_only":
            raise ValueError(
                "only the train-only v2 normalization protocol is accepted"
            )
        if len(self.training_reference_sha256) != 64 or any(
            character not in "0123456789abcdef"
            for character in self.training_reference_sha256
        ):
            raise ValueError("training_reference_sha256 must be a SHA-256 digest")

    @property
    def direction(self) -> float:
        return 1.0 if self.maximize else -1.0

    def _payload(self) -> dict[str, Any]:
        return asdict(self)

    @property
    def profile_sha256(self) -> str:
        return _canonical_sha256(self._payload())

    def to_dict(self) -> dict[str, Any]:
        return {**self._payload(), "profile_sha256": self.profile_sha256}

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> MetricNormalizationProfile:
        values = dict(payload)
        expected = values.pop("profile_sha256", None)
        if expected is None:
            raise ValueError("Metric normalization profile SHA-256 is required")
        profile = cls(**values)
        if expected != profile.profile_sha256:
            raise ValueError("Metric normalization profile SHA-256 mismatch")
        return profile

    def utility(self, score: float) -> float:
        """Map a native score to dimensionless, higher-is-better utility."""

        value = float(score)
        if not math.isfinite(value):
            raise ValueError("metric score must be finite")
        if self.metric in {"auroc", "auprc"} and not 0 <= value <= 1:
            raise ValueError(f"{self.metric} score must be within [0, 1]")
        if self.metric == "spearman" and not -1 <= value <= 1:
            raise ValueError("spearman score must be within [-1, 1]")
        if self.metric == "mae" and value < 0:
            raise ValueError("mae score must be non-negative")
        return self.direction * (value - self.null_reference) / self.gain_scale


def build_metric_normalization_profile(
    task: str | TaskSpec,
    train_targets: Sequence[float],
) -> MetricNormalizationProfile:
    """Create a profile using training labels only.

    AUROC uses its 0.5 chance baseline; AUPRC uses training prevalence;
    Spearman uses zero correlation; MAE uses a robust training-target scale.
    """

    spec = resolve_task(task) if isinstance(task, str) else task
    canonical = resolve_task(spec.name)
    if spec != canonical:
        raise ValueError("task specification must match the canonical TDC contract")
    values = _finite_training_targets(train_targets)
    positive_rate: float | None = None
    if spec.task_type == "classification":
        labels = set(np.unique(values).tolist())
        if not labels <= {0.0, 1.0} or labels != {0.0, 1.0}:
            raise ValueError(
                "classification train_targets must contain both binary classes 0 and 1"
            )
        positive_rate = float(np.mean(values))

    if spec.metric == "auroc":
        null_reference = 0.5
        gain_scale = 0.5
        method = "perfect_minus_chance_auroc"
    elif spec.metric == "auprc":
        assert positive_rate is not None
        null_reference = positive_rate
        gain_scale = max(1.0 - positive_rate, _AUPRC_MINIMUM_GAIN_SCALE)
        method = "max(perfect_minus_train_prevalence,0.10)"
    elif spec.metric == "spearman":
        null_reference = 0.0
        gain_scale = 1.0
        method = "perfect_minus_zero_correlation"
    elif spec.metric == "mae":
        null_reference = 0.0
        gain_scale, method = _robust_regression_scale(values)
    else:  # protected by canonical TaskSpec, retained as a defensive boundary
        raise ValueError(f"Unsupported metric {spec.metric!r}")

    return MetricNormalizationProfile(
        task=spec.name,
        metric=spec.metric,
        maximize=spec.maximize,
        null_reference=null_reference,
        gain_scale=gain_scale,
        gain_scale_method=method,
        train_count=len(values),
        train_positive_rate=positive_rate,
        training_reference_sha256=_training_reference_sha256(values),
    )


def normalized_utility(profile: MetricNormalizationProfile, score: float) -> float:
    return profile.utility(score)


def paired_normalized_deltas(
    profile: MetricNormalizationProfile,
    treatment_scores: Sequence[float],
    control_scores: Sequence[float],
) -> tuple[float, ...]:
    """Return treatment-minus-control utility for matched validation seeds."""

    treatments = tuple(treatment_scores)
    controls = tuple(control_scores)
    if not treatments or len(treatments) != len(controls):
        raise ValueError(
            "treatment_scores and control_scores must have equal non-zero length"
        )
    return tuple(
        profile.utility(treatment) - profile.utility(control)
        for treatment, control in zip(treatments, controls, strict=True)
    )


@dataclass(frozen=True)
class NormalizedEffectSummary:
    normalized_seed_deltas: tuple[float, ...]
    paired_seed_count: int
    mean_normalized_delta: float
    normalized_delta_std: float | None
    positive_delta_rate: float
    negative_delta_rate: float
    directional_consistency: float
    one_sided_lcb: float | None
    one_sided_lcb_confidence: float
    stability_score: float | None
    minimum_normalized_effect: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def summarize_normalized_deltas(
    normalized_deltas: Sequence[float],
    *,
    minimum_effect: float = DEFAULT_MINIMUM_NORMALIZED_EFFECT,
    lcb_confidence: float = DEFAULT_LCB_CONFIDENCE,
) -> NormalizedEffectSummary:
    """Summarize normalized paired effects with sample (ddof=1) uncertainty."""

    values = np.asarray(normalized_deltas, dtype=float)
    if values.ndim != 1 or len(values) == 0 or not np.all(np.isfinite(values)):
        raise ValueError(
            "normalized_deltas must be a non-empty finite one-dimensional sequence"
        )
    if not math.isfinite(minimum_effect) or minimum_effect <= 0:
        raise ValueError("minimum_effect must be finite and positive")
    if not math.isfinite(lcb_confidence) or not 0.5 < lcb_confidence < 1.0:
        raise ValueError("lcb_confidence must be within (0.5, 1.0)")

    count = len(values)
    effect = float(np.mean(values))
    positive_rate = float(np.mean(values > 0.0))
    negative_rate = float(np.mean(values < 0.0))
    directional_consistency = (
        positive_rate if effect > 0 else negative_rate if effect < 0 else 0.0
    )
    if count < 2:
        deviation = None
        lcb = None
        stability = None
    else:
        deviation = float(np.std(values, ddof=1))
        critical = float(student_t.ppf(lcb_confidence, df=count - 1))
        lcb = effect - critical * deviation / math.sqrt(count)
        stability = directional_consistency * math.exp(
            -deviation / (abs(effect) + minimum_effect)
        )
    return NormalizedEffectSummary(
        normalized_seed_deltas=tuple(float(value) for value in values),
        paired_seed_count=count,
        mean_normalized_delta=effect,
        normalized_delta_std=deviation,
        positive_delta_rate=positive_rate,
        negative_delta_rate=negative_rate,
        directional_consistency=directional_consistency,
        one_sided_lcb=lcb,
        one_sided_lcb_confidence=lcb_confidence,
        stability_score=stability,
        minimum_normalized_effect=minimum_effect,
    )


def summarize_paired_scores(
    profile: MetricNormalizationProfile,
    treatment_scores: Sequence[float],
    control_scores: Sequence[float],
    *,
    minimum_effect: float = DEFAULT_MINIMUM_NORMALIZED_EFFECT,
    lcb_confidence: float = DEFAULT_LCB_CONFIDENCE,
) -> NormalizedEffectSummary:
    deltas = paired_normalized_deltas(profile, treatment_scores, control_scores)
    return summarize_normalized_deltas(
        deltas,
        minimum_effect=minimum_effect,
        lcb_confidence=lcb_confidence,
    )


__all__ = [
    "DEFAULT_LCB_CONFIDENCE",
    "DEFAULT_MINIMUM_NORMALIZED_EFFECT",
    "NORMALIZATION_VERSION",
    "MetricNormalizationProfile",
    "NormalizedEffectSummary",
    "build_metric_normalization_profile",
    "normalized_utility",
    "paired_normalized_deltas",
    "summarize_normalized_deltas",
    "summarize_paired_scores",
]
