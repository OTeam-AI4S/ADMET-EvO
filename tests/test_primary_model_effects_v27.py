import importlib.util
import math
from pathlib import Path


SCRIPT = Path(__file__).parents[1] / "scripts/summarize_primary_model_effects_v27.py"


def _load_script():
    spec = importlib.util.spec_from_file_location("primary_effects_v27", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_paired_effect_has_exact_sign_flip_test_and_oriented_counts():
    module = _load_script()
    result = module.paired_effect([1.0, 2.0, 3.0])
    assert result["mean_oriented_gain"] == 2.0
    assert result["wins"] == 3
    assert result["ties"] == 0
    assert result["losses"] == 0
    assert result["exact_two_sided_sign_flip_p"] == 0.25
    assert math.isfinite(result["small_sample_corrected_dz"])


def test_generated_effect_report_is_validation_only_and_does_not_change_freeze():
    module = _load_script()
    root = Path(__file__).parents[1]
    report = module.build_report(
        root / "runs/v27-maplight-gnn-confirmation/result.json",
        root / "runs/v23-ablation-study/lipophilicity_scale_comparison.json",
    )
    assert report["official_test_loaded"] is False
    assert report["test_labels_used"] is False
    assert report["selection_or_freeze_changed"] is False
    assert len(report["contrasts"]) == 3
    assert all(item["effect"]["n"] == 5 for item in report["contrasts"].values())
