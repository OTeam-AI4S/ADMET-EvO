import importlib.util
from pathlib import Path


ROOT = Path(__file__).parents[1]
SCRIPT = ROOT / "scripts/audit_third_party_release_v27.py"


def _module():
    spec = importlib.util.spec_from_file_location("third_party_release_v27", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_third_party_release_scope_access_and_license_gate_are_fail_closed():
    report = _module().build_audit(ROOT)
    assert report["scope"] == {
        "frozen_task_count": 22,
        "essential_dependency_count": 2,
        "final_mapping_minimol_used": False,
        "rejected_experiments_are_not_release_dependencies": True,
    }
    assert set(report["dependencies"]) == {"unimol2_84m", "gin_supervised_masking"}
    assert report["dependencies"]["unimol2_84m"]["used_by_tasks"] == [
        "DILI",
        "Half_Life_Obach",
        "Lipophilicity_AstraZeneca",
        "hERG",
    ]
    assert report["dependencies"]["gin_supervised_masking"]["used_by_tasks"] == [
        "Bioavailability_Ma",
        "CYP2C9_Substrate_CarbonMangels",
    ]
    assert report["dependencies"]["unimol2_84m"]["checkpoint"]["sha256"] == (
        "5b9241630f1cf0b173fb06d1e76096e5daf5f91c3e51b066ba69524dafe60e35"
    )
    assert report["dependencies"]["gin_supervised_masking"]["checkpoint"]["sha256"] == (
        "6d0f8febad73e437772ebffc2ac32253d79f86ee138cfc233590ae50fb1cfeb9"
    )
    assert report["dependencies"]["gin_supervised_masking"]["checkpoint"][
        "official_parameter_equivalence"
    ] == {
        "verified": True,
        "parameter_tensor_count": 57,
        "parameter_value_count": 1_860_905,
        "all_parameter_tensors_bitwise_equal": True,
        "maximum_absolute_parameter_difference": 0.0,
    }
    assert report["gates"] == {
        "registration_free_checkpoint_access_verified": True,
        "source_code_licenses_verified": True,
        "checkpoint_specific_licenses_verified": False,
        "third_party_checkpoints_in_bundle": False,
        "safe_to_redistribute_third_party_checkpoints": False,
        "bundle_safe_for_external_deposit": True,
        "ready_for_external_deposit": True,
        "remaining_external_gate": (
            "deposit the prepared link-only bundle and mint a DOI; do not add third-party "
            "checkpoint files unless their artifact licenses are made explicit"
        ),
    }
    assert report["policy"]["official_test_loaded"] is False
    assert report["policy"]["selection_or_freeze_changed"] is False


def test_third_party_audit_markdown_states_the_non_inference_boundary():
    module = _module()
    text = module.markdown(module.build_audit(ROOT))
    assert "publicly downloadable and content-addressed" in text
    assert "not promoted into an artifact redistribution claim" in text
    assert "redistributes neither" in text
    assert "57 parameter tensors" in text


def test_persisted_gin_equivalence_record_is_exact_and_firewalled():
    import json

    report = json.loads(
        (ROOT / "reports/gin_checkpoint_equivalence_v27.json").read_text(encoding="utf-8")
    )
    assert report["verified"] is True
    assert report["comparison"]["all_parameter_tensors_bitwise_equal"] is True
    assert report["comparison"]["maximum_absolute_parameter_difference"] == 0.0
    assert report["comparison"]["parameter_tensor_count"] == 57
    assert report["comparison"]["parameter_value_count"] == 1_860_905
    assert report["official_test_loaded"] is False
    assert report["selection_or_freeze_changed"] is False
