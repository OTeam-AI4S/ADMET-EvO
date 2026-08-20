import json
import subprocess
import sys
from pathlib import Path

SCRIPT = (
    Path(__file__).parents[1]
    / "skills"
    / "optimize-admet-agent"
    / "scripts"
    / "compare_ablation.py"
)


def test_compare_ablation_reports_direction_aware_paired_gain(tmp_path):
    source = tmp_path / "input.json"
    output = tmp_path / "summary.json"
    source.write_text(
        json.dumps(
            {
                "metric": "mae",
                "higher_is_better": False,
                "seeds": [1, 2, 3, 4, 5],
                "baseline": {
                    "name": "baseline",
                    "scores": [0.31, 0.30, 0.29, 0.32, 0.30],
                    "cost": {"wall_seconds": 10.0},
                },
                "candidates": [
                    {
                        "name": "candidate",
                        "scores": [0.29, 0.28, 0.28, 0.30, 0.29],
                        "cost": {"wall_seconds": 12.0},
                    }
                ],
                "protocol": {"test_labels_used": False},
            }
        ),
        encoding="utf-8",
    )
    subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            str(source),
            "--output",
            str(output),
            "--bootstrap-samples",
            "2000",
            "--random-seed",
            "7",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    report = json.loads(output.read_text(encoding="utf-8"))
    candidate = report["candidates"][0]
    assert candidate["mean_improvement"] > 0
    assert candidate["wins_ties_losses"] == [5, 0, 0]
    assert candidate["sign_flip_pvalue_two_sided"] == 0.0625
    assert candidate["cost_delta"] == {"wall_seconds": 2.0}
    assert report["positive_improvement_means_better"] is True
    assert report["warning"]


def test_compare_ablation_rejects_unpaired_scores(tmp_path):
    source = tmp_path / "bad.json"
    source.write_text(
        json.dumps(
            {
                "metric": "auroc",
                "higher_is_better": True,
                "baseline": {"name": "base", "scores": [0.7, 0.8]},
                "candidates": [{"name": "bad", "scores": [0.9, 0.8, 0.7]}],
            }
        ),
        encoding="utf-8",
    )
    completed = subprocess.run(
        [sys.executable, str(SCRIPT), str(source)],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode != 0
    assert "do not match the baseline length" in completed.stderr
