import importlib.util
import json
from pathlib import Path


SCRIPT = Path(__file__).parents[1] / "scripts/build_compute_memory_appendix_v27.py"


def _load_script():
    spec = importlib.util.spec_from_file_location("compute_appendix_v27", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_compute_appendix_rebuilds_with_complete_router_and_explicit_missingness():
    module = _load_script()
    root = Path(__file__).parents[1]
    report = module.build_report(root)
    assert report["router_replay"]["task_count"] == 22
    assert report["official_test_loaded"] is False
    assert report["official_results_used"] is False
    assert report["selection_or_freeze_changed"] is False
    assert report["router_replay"]["selected_to_score_total_cost_ratio"] < 1
    coverage = report["peak_memory_coverage"]
    assert coverage["records_total"] == coverage["records_missing"]
    assert coverage["records_measured"] == 0


def test_persisted_compute_appendix_matches_rebuild():
    module = _load_script()
    root = Path(__file__).parents[1]
    persisted = json.loads(
        (root / "reports/compute_memory_appendix_v27.json").read_text(encoding="utf-8")
    )
    assert persisted == module.build_report(root)
