import hashlib
import importlib.util
import json
from pathlib import Path


ROOT = Path(__file__).parents[1]
SCRIPT = ROOT / "scripts/build_paper_figures_v27.py"


def _module():
    spec = importlib.util.spec_from_file_location("build_paper_figures_v27", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_task_label_shortener_preserves_endpoint_identity():
    module = _module()
    assert module._short_task("CYP2C9_Substrate_CarbonMangels") == "CYP2C9 substrate"
    assert module._short_task("Lipophilicity_AstraZeneca") == "Lipophilicity AZ"
    assert module._short_task("BBB_Martins") == "BBB"


def test_persisted_figure_manifest_matches_sources_outputs_and_claim_boundaries():
    manifest_path = ROOT / "paper/figures/manifest.json"
    assert manifest_path.exists()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["figure_count"] == 4
    assert manifest["formats"] == ["svg", "png"]
    assert manifest["official_test_loaded"] is False
    assert manifest["leaderboard_used_for_selection"] is False
    assert manifest["selection_or_freeze_changed"] is False
    assert len(manifest["outputs"]) == 9
    for record in manifest["sources"].values():
        path = ROOT / record["path"]
        assert hashlib.sha256(path.read_bytes()).hexdigest() == record["sha256"]
    for relative, record in manifest["outputs"].items():
        path = ROOT / relative
        assert path.exists() and path.stat().st_size == record["bytes"]
        assert hashlib.sha256(path.read_bytes()).hexdigest() == record["sha256"]
    ranks = manifest["figure_data"]["figure2_leaderboard_ranks"]
    assert (ranks["task_count"], ranks["top5"], ranks["top10"]) == (22, 18, 21)
    assert ranks["conservative_deficit_task"] == "Bioavailability_Ma"
    router = manifest["figure_data"]["figure3_router_tradeoff"]
    assert router["task_count"] == 22
    assert router["fresh_validation_seeds"] == [4, 5]
    assert router["guard_minus_unguarded_mean"] == 0
    assert router["claim_gate_passed"] is False
    assert router["aggregate_cost_ratio"] < 0.35
    chemistry = manifest["figure_data"]["figure4_chemical_space_risk"]
    assert chemistry["classification_worse_low_similarity"] == 11
    assert chemistry["regression_worse_low_similarity"] == 7
    assert chemistry["identity_overlap_total"] == 0
    assert chemistry["scaffold_overlap_total"] == 0
