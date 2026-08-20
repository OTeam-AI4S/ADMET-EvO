from scripts.summarize_unimol2_scale import compare_scales


def _records(size, scores, seconds):
    return [
        {
            "task": "Example",
            "metric": "mae",
            "learning_rate": 0.0001,
            "epochs": 20,
            "inner_folds": 5,
            "seed": seed,
            "score": score,
            "elapsed_seconds": seconds,
            "model_size": size,
            "test_labels_used": False,
        }
        for seed, score in enumerate(scores, start=1)
    ]


def test_scale_summary_uses_paired_direction_and_conservative_lcb():
    smaller = _records("84m", [0.4, 0.41, 0.39, 0.42, 0.38], 10)
    larger = _records("164m", [0.41, 0.42, 0.40, 0.43, 0.39], 20)

    report = compare_scales(smaller, larger)

    assert report["paired_gain_larger_over_smaller"]["mean"] < 0
    assert report["paired_gain_larger_over_smaller"]["larger_wins"] == 0
    assert report["larger_runtime_ratio"] == 2.0
    assert report["recommendation"] == "retain_smaller"
