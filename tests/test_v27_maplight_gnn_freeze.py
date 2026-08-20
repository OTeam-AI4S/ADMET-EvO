import importlib.util
from pathlib import Path


SCRIPT = Path(__file__).parents[1] / "scripts/build_v27_maplight_gnn_freeze.py"


def _load_script():
    spec = importlib.util.spec_from_file_location("v27_maplight_gnn_freeze", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_v27_freeze_derives_from_v26_and_upgrades_only_two_tasks():
    module = _load_script()
    root = Path(__file__).parents[1]
    report = module.build(root)
    assert report["coverage"] == {"expected": 22, "actual": 22, "complete": True}
    assert report["upgrade_tasks"] == [
        "Bioavailability_Ma",
        "CYP2C9_Substrate_CarbonMangels",
    ]
    assert report["finalists"]["Bioavailability_Ma"]["recipe"] == "maplight_gnn"
    assert (
        report["finalists"]["CYP2C9_Substrate_CarbonMangels"]["recipe"]
        == "maplight_gnn"
    )
    assert report["official_test_loaded_during_build"] is False
    for task in report["upgrade_tasks"]:
        confirmation = report["provenance"][task]["fresh_confirmation"]
        assert confirmation["gate_passed"] is True
        assert confirmation["paired_seed_improvements"] >= 3
        assert confirmation["maplight_gnn_mean"] > confirmation["maplight_mean"]
