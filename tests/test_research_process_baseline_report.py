import importlib.util
from pathlib import Path

SCRIPT = Path(__file__).parents[1] / "scripts/build_research_process_baseline.py"


def _module():
    spec = importlib.util.spec_from_file_location("process_baseline", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_existing_v3_run_builds_conservative_validation_only_baseline():
    root = Path(__file__).parents[1]
    payload = _module().build(root / "runs/v3-agent-tdc22-v5")
    metrics = payload["metrics"]
    assert metrics["tasks"] and len(metrics["tasks"]) == 22
    assert metrics["single_axis_contrast_validity"]["rate"] == 1.0
    assert metrics["self_correction_rate"]["rate"] == 0.0
    assert metrics["self_correction_rate"]["coverage"] == 0.0
    assert metrics["resources"]["total_fits"] == 790
    assert metrics["resources"]["planned_fits_avoided"] == 0
    assert payload["test_loaded"] is False
    assert "same-budget" in " ".join(payload["limitations"])
