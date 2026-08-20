"""Paired, validation-only comparison of two Uni-Mol2 model scales."""

from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path

from scipy.stats import t


def load_results(root: Path) -> list[dict]:
    paths = sorted(root.glob("seed_*/results.json"))
    records = [json.loads(path.read_text(encoding="utf-8")) for path in paths]
    seeds = [int(record["seed"]) for record in records]
    if seeds != [1, 2, 3, 4, 5]:
        raise ValueError(f"Expected complete seeds 1-5 under {root}, got {seeds}")
    if any(record.get("test_labels_used") is not False for record in records):
        raise ValueError("Scale comparison requires validation-only results")
    return records


def compare_scales(smaller: list[dict], larger: list[dict]) -> dict:
    if len(smaller) != len(larger) or len(smaller) < 2:
        raise ValueError("Paired comparison requires equal groups with at least two seeds")
    smaller_by_seed = {int(item["seed"]): item for item in smaller}
    larger_by_seed = {int(item["seed"]): item for item in larger}
    if smaller_by_seed.keys() != larger_by_seed.keys():
        raise ValueError("Scale groups must contain the same seeds")

    reference = smaller[0]
    matched_fields = ("task", "metric", "learning_rate", "epochs", "inner_folds")
    for item in [*smaller, *larger]:
        if any(item[field] != reference[field] for field in matched_fields):
            raise ValueError("Scale groups do not share the same experimental protocol")

    minimize = reference["metric"] == "mae"
    gains = []
    for seed in sorted(smaller_by_seed):
        small_score = float(smaller_by_seed[seed]["score"])
        large_score = float(larger_by_seed[seed]["score"])
        gains.append(small_score - large_score if minimize else large_score - small_score)

    mean_gain = statistics.mean(gains)
    standard_error = statistics.stdev(gains) / math.sqrt(len(gains))
    radius = float(t.ppf(0.975, df=len(gains) - 1)) * standard_error
    small_seconds = sum(float(item["elapsed_seconds"]) for item in smaller)
    large_seconds = sum(float(item["elapsed_seconds"]) for item in larger)

    def summary(records: list[dict]) -> dict:
        scores = [float(item["score"]) for item in records]
        seconds = sum(float(item["elapsed_seconds"]) for item in records)
        return {
            "model_size": records[0]["model_size"],
            "scores": scores,
            "mean": statistics.mean(scores),
            "std": statistics.pstdev(scores),
            "total_gpu_hours": seconds / 3600.0,
        }

    lower_bound = mean_gain - radius
    return {
        "task": reference["task"],
        "metric": reference["metric"],
        "protocol": {field: reference[field] for field in matched_fields[2:]},
        "seeds": sorted(smaller_by_seed),
        "test_labels_used": False,
        "smaller": summary(smaller),
        "larger": summary(larger),
        "paired_gain_larger_over_smaller": {
            "values": gains,
            "mean": mean_gain,
            "ci95": [mean_gain - radius, mean_gain + radius],
            "larger_wins": sum(value > 0 for value in gains),
        },
        "larger_runtime_ratio": large_seconds / small_seconds,
        "recommendation": (
            "retain_larger" if lower_bound > 0 else "retain_smaller"
        ),
        "decision_rule": "retain larger only when paired gain 95% LCB is positive",
    }


def markdown(report: dict) -> str:
    small = report["smaller"]
    large = report["larger"]
    gain = report["paired_gain_larger_over_smaller"]
    return "\n".join(
        [
            f"# {report['task']} Uni-Mol2 scale comparison",
            "",
            "Validation-only paired comparison on scaffold seeds 1-5.",
            "",
            "| Scale | Score | Total GPU-hours |",
            "|---|---:|---:|",
            (
                f"| {small['model_size']} | {small['mean']:.4f} +/- "
                f"{small['std']:.4f} | {small['total_gpu_hours']:.2f} |"
            ),
            (
                f"| {large['model_size']} | {large['mean']:.4f} +/- "
                f"{large['std']:.4f} | {large['total_gpu_hours']:.2f} |"
            ),
            "",
            (
                f"Paired gain of larger over smaller: {gain['mean']:.4f}; "
                f"95% CI [{gain['ci95'][0]:.4f}, {gain['ci95'][1]:.4f}]."
            ),
            (
                f"Larger model wins: {gain['larger_wins']}/5 seeds; runtime ratio: "
                f"{report['larger_runtime_ratio']:.2f}x."
            ),
            f"Decision: **{report['recommendation']}** ({report['decision_rule']}).",
            "",
        ]
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--smaller-root",
        type=Path,
        default=Path(
            "runs/unimol2-scale-ablation-84m-b8/84m/"
            "Lipophilicity_AstraZeneca/lr_0.0001"
        ),
    )
    parser.add_argument(
        "--larger-root",
        type=Path,
        default=Path(
            "runs/unimol2-scale-ablation-b8/164m/"
            "Lipophilicity_AstraZeneca/lr_0.0001"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("runs/v23-ablation-study/lipophilicity_scale_comparison"),
    )
    args = parser.parse_args()

    report = compare_scales(load_results(args.smaller_root), load_results(args.larger_root))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.with_suffix(".json").write_text(
        json.dumps(report, indent=2, sort_keys=True), encoding="utf-8"
    )
    args.output.with_suffix(".md").write_text(markdown(report), encoding="utf-8")
    print(
        json.dumps(
            {
                "recommendation": report["recommendation"],
                "gain": report["paired_gain_larger_over_smaller"]["mean"],
                "runtime_ratio": report["larger_runtime_ratio"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
