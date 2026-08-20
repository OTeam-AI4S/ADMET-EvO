from __future__ import annotations

import math

import numpy as np
import pytest

from admet_agent.action_memory import ResearchOutcomeMemory
from admet_agent.metric_normalization import (
    MetricNormalizationProfile,
    build_metric_normalization_profile,
    paired_normalized_deltas,
    summarize_normalized_deltas,
    summarize_paired_scores,
)


def test_profiles_use_metric_specific_train_only_gain_scales():
    auroc = build_metric_normalization_profile("AMES", [0, 1, 0, 1])
    auprc = build_metric_normalization_profile("CYP2C9_Veith", [0, 0, 0, 1])
    spearman = build_metric_normalization_profile("Half_Life_Obach", [1, 2, 3, 4])
    mae = build_metric_normalization_profile("LD50_Zhu", [0, 1, 2, 3, 4])

    assert (auroc.null_reference, auroc.gain_scale) == (0.5, 0.5)
    assert auprc.null_reference == pytest.approx(0.25)
    assert auprc.gain_scale == pytest.approx(0.75)
    assert (spearman.null_reference, spearman.gain_scale) == (0.0, 1.0)
    assert mae.maximize is False
    assert mae.gain_scale > 0
    assert all(
        profile.scale_source == "train_only"
        for profile in (auroc, auprc, spearman, mae)
    )


def test_equal_fractional_gains_are_equal_across_all_four_metrics():
    profiles = [
        build_metric_normalization_profile("AMES", [0, 1, 0, 1]),
        build_metric_normalization_profile("CYP2C9_Veith", [0, 0, 0, 1]),
        build_metric_normalization_profile("Half_Life_Obach", [1, 2, 3, 4]),
        build_metric_normalization_profile("LD50_Zhu", [0, 1, 2, 3, 4]),
    ]
    deltas = []
    for profile in profiles:
        if profile.metric == "mae":
            control = 0.5 * profile.gain_scale
            treatment = control - 0.02 * profile.gain_scale
        else:
            control = profile.null_reference + 0.25 * profile.gain_scale
            treatment = control + 0.02 * profile.gain_scale
        deltas.append(paired_normalized_deltas(profile, [treatment], [control])[0])
    assert deltas == pytest.approx([0.02, 0.02, 0.02, 0.02])


def test_mae_native_magnitude_cannot_dominate_after_normalization():
    small_units = build_metric_normalization_profile("Caco2_Wang", [0, 1, 2, 3, 4])
    large_units = build_metric_normalization_profile(
        "LD50_Zhu", [0, 100, 200, 300, 400]
    )
    small_delta = paired_normalized_deltas(
        small_units, [0.5 - 0.05 * small_units.gain_scale], [0.5]
    )
    large_delta = paired_normalized_deltas(
        large_units, [50 - 0.05 * large_units.gain_scale], [50]
    )
    assert small_delta == pytest.approx(large_delta)
    assert small_delta == pytest.approx((0.05,))


def test_summary_uses_sample_standard_deviation_lcb_and_stability():
    values = [0.01, 0.02, 0.03, 0.04, 0.05]
    summary = summarize_normalized_deltas(values)
    expected_std = float(np.std(values, ddof=1))
    assert summary.normalized_delta_std == pytest.approx(expected_std)
    assert summary.normalized_delta_std != pytest.approx(float(np.std(values, ddof=0)))
    assert summary.one_sided_lcb is not None
    assert summary.one_sided_lcb < summary.mean_normalized_delta
    assert 0 < summary.stability_score <= 1
    assert summary.positive_delta_rate == 1.0


def test_single_pair_reports_uncertainty_as_missing_not_perfect_stability():
    summary = summarize_normalized_deltas([0.02])
    assert summary.normalized_delta_std is None
    assert summary.one_sided_lcb is None
    assert summary.stability_score is None


def test_paired_score_summary_is_direction_correct_for_mae():
    profile = build_metric_normalization_profile("LD50_Zhu", [0, 1, 2, 3, 4])
    summary = summarize_paired_scores(profile, [0.8, 0.9, 1.0], [1.0, 1.1, 1.2])
    assert summary.mean_normalized_delta > 0
    assert summary.positive_delta_rate == 1.0


def test_profile_serialization_hash_is_deterministic_and_tamper_evident():
    left = build_metric_normalization_profile("AMES", [0, 1, 0, 1])
    right = build_metric_normalization_profile("AMES", [0, 1, 0, 1])
    assert left.profile_sha256 == right.profile_sha256
    assert MetricNormalizationProfile.from_dict(left.to_dict()) == left

    tampered = left.to_dict()
    tampered["gain_scale"] = 0.75
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        MetricNormalizationProfile.from_dict(tampered)

    unsigned = left.to_dict()
    unsigned.pop("profile_sha256")
    with pytest.raises(ValueError, match="SHA-256 is required"):
        MetricNormalizationProfile.from_dict(unsigned)


def test_training_reference_changes_profile_hash_without_storing_labels():
    first = build_metric_normalization_profile("AMES", [0, 1, 0, 1])
    second = build_metric_normalization_profile("AMES", [1, 0, 1, 0])
    assert first.training_reference_sha256 != second.training_reference_sha256
    assert first.profile_sha256 != second.profile_sha256
    assert "train_targets" not in first.to_dict()


@pytest.mark.parametrize(
    "task,targets",
    [
        ("AMES", [0, 0, 0]),
        ("AMES", [0, 1, math.nan]),
        ("LD50_Zhu", [1, 1, 1]),
    ],
)
def test_invalid_training_profiles_are_rejected(task, targets):
    with pytest.raises(ValueError):
        build_metric_normalization_profile(task, targets)


def test_metric_score_ranges_and_pair_lengths_are_validated():
    profile = build_metric_normalization_profile("AMES", [0, 1, 0, 1])
    with pytest.raises(ValueError, match=r"within \[0, 1\]"):
        profile.utility(1.1)
    with pytest.raises(ValueError, match="equal non-zero length"):
        paired_normalized_deltas(profile, [0.8], [0.7, 0.6])


def test_action_memory_rejects_nested_test_keys_without_writing(tmp_path):
    memory = ResearchOutcomeMemory(tmp_path / "action_outcomes.jsonl")
    evidence = {
        "action_id": "feature-domain-shift",
        "axis": "feature",
        "status": "supported",
        "single_axis_verified": True,
        "details": [{"certification": {"agent_test_score": 0.9}}],
    }
    with pytest.raises(
        ValueError, match=r"details\[0\].certification.agent_test_score"
    ):
        memory.append(task="AMES", category="toxicity", evidence=evidence)
    assert not memory.path.exists()


def test_action_memory_allows_nested_validation_only_metadata(tmp_path):
    memory = ResearchOutcomeMemory(tmp_path / "action_outcomes.jsonl")
    evidence = {
        "action_id": "feature-domain-shift",
        "axis": "feature",
        "status": "supported",
        "single_axis_verified": True,
        "oriented_seed_deltas": [0.01, 0.02, 0.03],
        "normalization": {"profile_sha256": "a" * 64, "scale_source": "train_only"},
    }
    record = memory.append(task="AMES", category="toxicity", evidence=evidence)
    assert record["paired_seeds"] == 3
