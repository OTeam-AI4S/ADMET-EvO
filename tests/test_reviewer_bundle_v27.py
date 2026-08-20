import importlib.util
import json
import tarfile
from pathlib import Path


ROOT = Path(__file__).parents[1]
SCRIPT = ROOT / "scripts/build_reviewer_bundle_v27.py"


def _module():
    spec = importlib.util.spec_from_file_location("reviewer_bundle_v27", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_reviewer_bundle_is_small_label_free_and_claim_complete(tmp_path):
    module = _module()
    receipt = module.build_bundle(ROOT, tmp_path)
    bundle = tmp_path / module.BUNDLE_NAME
    manifest = json.loads((bundle / "MANIFEST.json").read_text(encoding="utf-8"))
    assert manifest["scorecard_status"] == {
        "top5": 18,
        "top10": 21,
        "top5_deficit": 0,
        "top10_deficit": 1,
        "goal_met": False,
    }
    assert manifest["official_test_labels_included"] is False
    assert manifest["official_predictions_included"] is True
    assert manifest["credentials_included"] is False
    assert manifest["local_data_cache_included"] is False
    assert manifest["historical_run_cache_included"] is False
    assert manifest["third_party_checkpoints_included"] is False
    assert manifest["registration_free_checkpoint_access_verified"] is True
    assert manifest["checkpoint_specific_licenses_verified"] is False
    assert manifest["official_prediction_file_count"] == 110
    assert manifest["validation_npz_file_count"] == 1
    assert manifest["total_uncompressed_bytes"] < 30_000_000
    assert manifest["file_count"] >= 250
    assert manifest["source_audit"] == {
        "audited_file_count": manifest["file_count"] - 1,
        "secret_patterns_detected": 0,
        "label_like_prediction_arrays_detected": 0,
    }
    required = {
        "MANUSCRIPT_V24.md",
        "reports/goal_scorecard_v27.json",
        "reports/paper_evidence_manifest_v27.json",
        "reports/third_party_release_audit_v27.json",
        "reports/third_party_release_audit_v27.md",
        "paper/submission/jcheminformatics/manuscript.md",
        "paper/submission/jcheminformatics/supplementary_information.md",
        "runs/v27-official/Bioavailability_Ma/final/results.json",
        "runs/v27-official/CYP2C9_Substrate_CarbonMangels/final/results.json",
        "vendor/tdc-admet-bench-audit/models/maplight_gnn/wrapper/environment.yml",
    }
    assert required <= set(manifest["files"])
    assert not any(path.startswith(("data/", "logs/", ".venvs/")) for path in manifest["files"])
    assert receipt["ready_for_external_deposit"] is True


def test_archive_is_deterministic_and_matches_bundle_manifest(tmp_path):
    module = _module()
    first = module.build_bundle(ROOT, tmp_path)
    first_hash = first["archive"]["sha256"]
    second = module.build_bundle(ROOT, tmp_path)
    assert second["archive"]["sha256"] == first_hash
    archive = tmp_path / f"{module.BUNDLE_NAME}.tar.gz"
    with tarfile.open(archive, "r:gz") as handle:
        members = {member.name: member for member in handle.getmembers() if member.isfile()}
        manifest_name = f"{module.BUNDLE_NAME}/MANIFEST.json"
        assert manifest_name in members
        extracted = handle.extractfile(members[manifest_name])
        assert extracted is not None
        archived_manifest = json.loads(extracted.read().decode("utf-8"))
    assert archived_manifest["manifest_sha256"] == json.loads(
        (tmp_path / module.BUNDLE_NAME / "MANIFEST.json").read_text(encoding="utf-8")
    )["manifest_sha256"]


def test_gpu_environment_has_no_accidental_patch_marker():
    text = (ROOT / "environment-gpu.yml").read_text(encoding="utf-8")
    assert "\n+#" not in text
