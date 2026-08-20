from scripts.summarize_v24_rescue_experiments import markdown


def test_rescue_markdown_keeps_negative_decisions_visible():
    report = {
        "experiments": [
            {
                "task": "Example",
                "method": "teacher",
                "baseline": 0.5,
                "candidate": 0.4,
                "gain": -0.1,
                "decision": "reject",
            }
        ]
    }

    rendered = markdown(report)
    assert "-0.1000" in rendered
    assert "reject" in rendered
