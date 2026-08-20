import importlib.util
from pathlib import Path


SCRIPT = Path(__file__).parents[1] / "scripts/build_v27_goal_scorecard.py"


def _load_script():
    spec = importlib.util.spec_from_file_location("v27_goal_scorecard", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_v27_scorecard_audits_official_artifacts_and_leaves_only_bio_gap():
    module = _load_script()
    root = Path(__file__).parents[1]
    scorecard = module.build_v27_scorecard(
        root, root / "reports/leaderboard_snapshot_2026-08-17.json"
    )
    assert scorecard["status"] == {
        "top5": 18,
        "top10": 21,
        "top5_deficit": 0,
        "top10_deficit": 1,
        "goal_met": False,
    }
    missing = [row["task"] for row in scorecard["tasks"] if row["estimated_rank"] > 10]
    assert missing == ["Bioavailability_Ma"]
    cyp = scorecard["rescue_outcomes"]["CYP2C9_Substrate_CarbonMangels"]
    assert cyp["candidate_official"] == 0.434
    assert cyp["estimated_rank"] <= 5
    bio = scorecard["rescue_outcomes"]["Bioavailability_Ma"]
    assert bio["candidate_official"] == 0.731
    assert bio["decision"] == "accept_improvement_below_top10"
    assert scorecard["official_results_complete"] is True
    for task in scorecard["upgrade_tasks"]:
        provenance = scorecard["provenance"][task]
        assert provenance["test_labels_passed_to_executor"] is False
        assert len(provenance["prediction_hashes"]) == 5
