from scripts.summarize_v25_bio_rescue import markdown


def test_v25_markdown_reports_rejected_candidate_and_retained_score():
    report = {
        "decision": "reject_keep_previous_official",
        "scores": {
            "previous_official": 0.697,
            "candidate_official": 0.675,
            "official_delta": -0.022,
            "retained_scorecard_score": 0.697,
            "fresh_scaffold_confirmation_mean": 0.754,
            "validation_to_official_delta": -0.079,
        },
        "preregistration": {"full_22_task_freeze_sha256": "full"},
        "execution": {"freeze_sha256": "execution"},
    }
    rendered = markdown(report)
    assert "reject_keep_previous_official" in rendered
    assert "Retained scorecard AUROC: 0.6970" in rendered
    assert "Frozen candidate official AUROC: 0.6750" in rendered
