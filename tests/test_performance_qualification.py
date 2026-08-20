import json
from pathlib import Path

import pytest

from admet_agent.performance_qualification import (
    PerformanceQualificationPolicy,
    qualify_frozen_scorecard,
    qualify_frozen_scorecard_file,
)

ROOT = Path(__file__).parents[1]


def test_current_v27_scorecard_passes_application_qualification_without_feedback():
    qualification = qualify_frozen_scorecard_file(
        ROOT / "reports/goal_scorecard_v27.json"
    )
    payload = qualification.to_dict()
    assert qualification.passed is True
    assert qualification.tasks_observed == 22
    assert qualification.top5_tasks == 18
    assert qualification.top10_tasks == 21
    assert qualification.exception_tasks == ("Bioavailability_Ma",)
    assert qualification.provenance_coverage == 1.0
    assert qualification.test_labels_loaded is False
    assert qualification.selection_feedback_to_discovery is False
    assert payload["evidence_role"] == "application_performance_qualification_only"
    assert len(payload["seal_sha256"]) == 64


def test_feedback_policy_is_rejected():
    with pytest.raises(ValueError, match="cannot feed back"):
        PerformanceQualificationPolicy(selection_feedback_to_discovery=True)


def test_rank_count_tampering_is_rejected():
    scorecard = json.loads(
        (ROOT / "reports/goal_scorecard_v27.json").read_text(encoding="utf-8")
    )
    scorecard["status"]["top5"] = 22
    with pytest.raises(ValueError, match="rank counts"):
        qualify_frozen_scorecard(scorecard)
