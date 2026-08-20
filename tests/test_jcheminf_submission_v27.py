import importlib.util
import struct
from pathlib import Path


ROOT = Path(__file__).parents[1]
SCRIPT = ROOT / "scripts/build_jcheminf_submission_v27.py"


def _module():
    spec = importlib.util.spec_from_file_location("jcheminf_submission_v27", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _png_size(path: Path) -> tuple[int, int]:
    data = path.read_bytes()
    assert data[:8] == b"\x89PNG\r\n\x1a\n"
    return struct.unpack(">II", data[16:24])


def test_submission_package_meets_machine_verifiable_journal_gates():
    manifest = _module().build_submission(ROOT)
    out = ROOT / "paper/submission/jcheminformatics"
    assert manifest["target_journal"] == "Journal of Cheminformatics"
    assert manifest["article_type"] == "Research Article"
    assert manifest["manuscript"]["abstract_word_count"] <= 350
    assert manifest["manuscript"]["contribution_heading_present"] is True
    assert manifest["manuscript"]["citation_count"] == 15
    assert manifest["manuscript"]["reference_count"] == 15
    assert manifest["supplement"]["task_count"] == 22
    assert manifest["supplement"]["top5"] == 18
    assert manifest["supplement"]["top10"] == 21
    assert manifest["supplement"]["router_control_task_count"] == 22
    assert manifest["supplement"]["fresh_guard_task_count"] == 22
    assert manifest["supplement"]["chemical_space_task_count"] == 22
    assert manifest["supplement"]["identity_overlap_total"] == 0
    assert manifest["supplement"]["scaffold_overlap_total"] == 0
    assert manifest["official_test_loaded_during_generation"] is False
    assert manifest["selection_or_freeze_changed"] is False
    assert manifest["third_party_release_audit"] == {
        "essential_dependency_count": 2,
        "registration_free_checkpoint_access_verified": True,
        "checkpoint_specific_licenses_verified": False,
        "third_party_checkpoints_in_bundle": False,
        "gin_official_parameter_equivalence_verified": True,
        "ready_for_external_deposit": True,
    }
    assert _png_size(out / "graphical_abstract.png") == (920, 300)
    assert (out / "graphical_abstract.png").stat().st_size <= 150_000
    manuscript = (out / "manuscript.md").read_text(encoding="utf-8")
    assert "### Contribution" in manuscript
    assert "ARCHIVE_DOI_REQUIRED" in manuscript
    assert "## Declarations" in manuscript
    references = manuscript.split("## References", 1)[1]
    assert "\\\"" not in references
    assert "https://doi.org/10.1186/s13321-026-01217-2" in references
    availability = (out / "data_and_code_availability.md").read_text(encoding="utf-8")
    assert "artifact metadata has no explicit license field" in availability
    assert "57\nparameter tensors are bitwise identical" in availability
    assert "redistributes neither third-party checkpoint" in availability


def test_submission_manifest_hashes_all_sources_and_outputs():
    module = _module()
    manifest = module.build_submission(ROOT)
    for item in manifest["sources"].values():
        assert module.hash_file(ROOT / item["path"]) == {"bytes": item["bytes"], "sha256": item["sha256"]}
    for relative, expected in manifest["outputs"].items():
        assert module.hash_file(ROOT / relative) == expected
    assert len(manifest["manifest_sha256"]) == 64
