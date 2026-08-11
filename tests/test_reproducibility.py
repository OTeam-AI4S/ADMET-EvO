import json

from scripts.verify_top1_results import verify_runs


def test_verify_runs_checks_scores_seeds_and_freeze_hashes(tmp_path):
    manifest = {
        "protocol": {"seeds": [1, 2, 3, 4, 5]},
        "results": {
            "AMES": {
                "dataset_key": "ames",
                "mean": 0.874,
                "std": 0.001,
                "freeze_sha256": "frozen",
            }
        },
    }
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    result_dir = tmp_path / "runs" / "AMES" / "final"
    result_dir.mkdir(parents=True)
    (result_dir / "results.json").write_text(
        json.dumps(
            {
                "seeds": [1, 2, 3, 4, 5],
                "freeze_sha256": "frozen",
                "results": {"ames": [0.874, 0.001]},
            }
        ),
        encoding="utf-8",
    )

    messages = verify_runs(tmp_path / "runs", manifest_path)

    assert messages == ["AMES: verified 0.874 +/- 0.001 and freeze frozen"]
