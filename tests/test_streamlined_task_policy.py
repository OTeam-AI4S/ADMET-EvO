import pytest

from admet_agent.schemas import TaskProfile
from admet_agent.streamlined_task_policy import (
    admit_minimal_pair,
    build_minimal_experiment_bundle,
    build_minimal_pair_specs,
    build_minimal_task_policy,
    minimal_frozen_anchor_recipe,
    size_bucket,
)


@pytest.mark.parametrize(
    ("n_train", "expected"),
    [(999, "small"), (1000, "medium"), (4999, "medium"), (5000, "large")],
)
def test_size_buckets_are_explicit(n_train, expected):
    assert size_bucket(n_train) == expected


def test_small_absorption_classification_is_feature_focused_and_pretrained():
    policy = build_minimal_task_policy("HIA_Hou", 403)
    assert policy.category == "absorption"
    assert policy.size_bucket == "small"
    assert policy.scientific_axis == "feature"
    assert policy.optional_strong_anchor == "maplight_gnn_frozen"
    assert policy.maximum_hpo_slots == 0
    assert policy.typical_fit_ceiling == 12


def test_small_metabolism_is_imbalance_matched_not_broad_search():
    policy = build_minimal_task_policy("CYP2C9_Substrate_CarbonMangels", 467)
    assert policy.scientific_axis == "model"
    assert policy.matched_control.endswith("unweighted")
    assert policy.matched_treatment.endswith("class_weighted")
    assert policy.maximum_hypotheses == 1


def test_excretion_uses_metric_aligned_rank_pair():
    policy = build_minimal_task_policy("Half_Life_Obach", 465)
    assert policy.category == "excretion"
    assert policy.scientific_axis == "feature"
    assert policy.matched_control == "basic2d_extra_trees"
    assert policy.matched_treatment == "rich2d_extra_trees"


def test_large_task_uses_one_seed_screen_and_no_pretrained_default():
    policy = build_minimal_task_policy("CYP2C9_Veith", 8463)
    assert policy.size_bucket == "large"
    assert policy.screen_seeds == (1,)
    assert policy.maximum_pretrained_anchor_slots == 0
    assert policy.typical_fit_ceiling == 10


def test_every_default_path_removes_expensive_optional_subsystems():
    policy = build_minimal_task_policy("DILI", 325)
    assert policy.default_committee is False
    assert policy.default_ensemble is False
    assert policy.default_unimol2 is False
    assert policy.default_external_data_search is False
    assert policy.maximum_corrections == 1
    assert policy.test_loaded is False


def _profile(task, task_type, metric, n_train):
    return TaskProfile(task, task_type, metric, n_train, 100, 0.2 if task_type == "classification" else None, 0.0, 1.0, 0, 10, 0.5)


def test_minimal_pair_materializes_as_exact_single_axis_specs():
    profile = _profile("HIA_Hou", "classification", "auroc", 403)
    control, treatment = build_minimal_pair_specs(profile)
    assert control.model == treatment.model == "extra_trees"
    assert control.representations == ("morgan", "rdkit2d")
    assert treatment.representations == ("morgan", "avalon", "erg", "rdkit2d")
    assert control.data_policy == treatment.data_policy == "official"
    assert minimal_frozen_anchor_recipe(profile) == "maplight_gnn"


def test_large_pair_changes_only_model_and_has_no_optional_anchor():
    profile = _profile("LD50_Zhu", "regression", "mae", 5168)
    control, treatment = build_minimal_pair_specs(profile)
    assert control.representations == treatment.representations
    assert (control.model, treatment.model) == ("lightgbm", "extra_trees")
    assert minimal_frozen_anchor_recipe(profile) is None


def test_exact_supported_and_refuted_evidence_prevents_retraining():
    caco = build_minimal_task_policy("Caco2_Wang", 637)
    common = {
        "action_id": "feature-domain-shift",
        "single_axis_verified": True,
        "validation_seeds": [1, 2, 3, 4, 5],
        "test_loaded": False,
    }
    supported = admit_minimal_pair(caco, {**common, "task": "Caco2_Wang", "status": "supported"})
    assert supported["decision"] == "reuse_supported_pair_and_stop"
    assert supported["new_training_required"] is False
    bbb = build_minimal_task_policy("BBB_Martins", 1421)
    refuted = admit_minimal_pair(bbb, {**common, "task": "BBB_Martins", "status": "refuted"})
    assert refuted["decision"] == "block_refuted_pair_use_anchor_or_stop"
    assert refuted["new_training_required"] is False


def test_pair_admission_does_not_transport_another_tasks_outcome_and_rejects_test():
    policy = build_minimal_task_policy("Pgp_Broccatelli", 851)
    other = admit_minimal_pair(
        policy,
        {
            "task": "Caco2_Wang",
            "action_id": "feature-domain-shift",
            "status": "supported",
            "single_axis_verified": True,
            "validation_seeds": [1, 2, 3, 4, 5],
        },
    )
    assert other["decision"] == "screen_new_pair"
    with pytest.raises(ValueError, match="Test-like"):
        admit_minimal_pair(policy, {"test_score": 0.9})


def test_executable_bundle_never_exceeds_three_configs_and_separates_frozen_anchor():
    profile = _profile("HIA_Hou", "classification", "auroc", 403)
    bundle = build_minimal_experiment_bundle(profile)
    assert len(bundle.matched_specs) == 2
    assert bundle.frozen_anchor_recipe == "maplight_gnn"
    assert len(bundle.matched_specs) + 1 <= bundle.maximum_total_configurations == 3
    assert bundle.test_loaded is False


def test_supported_bundle_stops_without_duplicate_pair_or_anchor():
    profile = _profile("Caco2_Wang", "regression", "mae", 637)
    evidence = {
        "task": "Caco2_Wang",
        "action_id": "feature-domain-shift",
        "status": "supported",
        "single_axis_verified": True,
        "validation_seeds": [1, 2, 3, 4, 5],
        "test_loaded": False,
    }
    bundle = build_minimal_experiment_bundle(profile, evidence)
    assert bundle.matched_specs == ()
    assert bundle.frozen_anchor_recipe is None
    assert bundle.pair_admission["new_training_required"] is False


def test_sealed_pilot_memory_blocks_duplicate_metabolism_and_toxicity_work():
    cyp = build_minimal_experiment_bundle(
        _profile("CYP2C9_Veith", "classification", "auprc", 8463)
    )
    assert cyp.matched_specs == ()
    assert cyp.pair_admission["evidence_status"] == "inconclusive"
    ld50 = build_minimal_experiment_bundle(
        _profile("LD50_Zhu", "regression", "mae", 5168)
    )
    assert ld50.matched_specs == ()
    assert ld50.pair_admission["evidence_status"] == "refuted"
