"""Descriptive paired effect sizes for the primary v2.7 model ablations.

The input experiments are already completed validation-only comparisons.  This
script does not select a model; it standardizes their paired uncertainty and
effect-size reporting for the manuscript.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import statistics
from pathlib import Path
from typing import Any

from scipy.stats import t


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def paired_effect(values: list[float]) -> dict[str, Any]:
    """Summarize oriented paired gains; positive always favors the candidate."""
    if len(values) < 2 or not all(math.isfinite(value) for value in values):
        raise ValueError("Paired effects require at least two finite values")
    n = len(values)
    mean = statistics.mean(values)
    sample_sd = statistics.stdev(values)
    standard_error = sample_sd / math.sqrt(n)
    radius = float(t.ppf(0.975, df=n - 1)) * standard_error
    cohen_dz = mean / sample_sd if sample_sd > 0 else math.copysign(math.inf, mean)
    correction = 1.0 - 3.0 / (4.0 * (n - 1) - 1.0)
    observed = abs(mean)
    permuted = [
        abs(statistics.mean(sign * value for sign, value in zip(signs, values)))
        for signs in itertools.product((-1.0, 1.0), repeat=n)
    ]
    exact_p = sum(value >= observed - 1e-15 for value in permuted) / len(permuted)
    return {
        "n": n,
        "values": values,
        "mean_oriented_gain": mean,
        "sample_sd": sample_sd,
        "standard_error": standard_error,
        "paired_t_95_ci": [mean - radius, mean + radius],
        "cohen_dz": cohen_dz,
        "small_sample_corrected_dz": correction * cohen_dz,
        "exact_two_sided_sign_flip_p": exact_p,
        "wins": sum(value > 0 for value in values),
        "ties": sum(value == 0 for value in values),
        "losses": sum(value < 0 for value in values),
    }


def build_report(confirmation_path: Path, scale_path: Path) -> dict[str, Any]:
    confirmation = json.loads(confirmation_path.read_text(encoding="utf-8"))
    scale = json.loads(scale_path.read_text(encoding="utf-8"))
    protocol = confirmation["protocol"]
    if protocol["official_test_loaded"] is not False or protocol["test_labels_used"] is not False:
        raise ValueError("MapLight+GNN comparison is not validation-only")
    if scale["test_labels_used"] is not False:
        raise ValueError("Uni-Mol2 scale comparison is not validation-only")

    contrasts: dict[str, Any] = {}
    for task in ("Bioavailability_Ma", "CYP2C9_Substrate_CarbonMangels"):
        task_result = confirmation["tasks"][task]
        seeds = [str(seed) for seed in protocol["fresh_seeds"]]
        values = [float(task_result[seed]["paired_gain"]) for seed in seeds]
        contrasts[f"{task}:maplight_gnn_minus_maplight"] = {
            "task": task,
            "metric": task_result["summary"]["metric"],
            "candidate": "maplight_gnn",
            "control": "maplight",
            "seeds": [int(seed) for seed in seeds],
            "effect": paired_effect(values),
        }

    scale_values = [
        float(value) for value in scale["paired_gain_larger_over_smaller"]["values"]
    ]
    contrasts["Lipophilicity_AstraZeneca:unimol2_164m_minus_84m"] = {
        "task": scale["task"],
        "metric": scale["metric"],
        "candidate": "unimol2_164m",
        "control": "unimol2_84m",
        "seeds": scale["seeds"],
        "effect": paired_effect(scale_values),
        "candidate_to_control_runtime_ratio": scale["larger_runtime_ratio"],
    }
    return {
        "version": "v27-primary-model-effects-1",
        "analysis_role": "post_hoc_descriptive_summary_of_completed_paired_validation_experiments",
        "selection_or_freeze_changed": False,
        "official_test_loaded": False,
        "test_labels_used": False,
        "caution": (
            "Five split seeds are paired computational replicates, not independent "
            "samples of chemical tasks; intervals and standardized effects are descriptive."
        ),
        "sources": {
            "maplight_gnn_confirmation": {
                "path": confirmation_path.as_posix(),
                "sha256": sha256(confirmation_path),
            },
            "unimol2_scale": {
                "path": scale_path.as_posix(),
                "sha256": sha256(scale_path),
            },
        },
        "contrasts": contrasts,
    }


def markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Primary model paired effects (v2.7)",
        "",
        report["analysis_role"].replace("_", " ") + ".",
        "",
        "| Contrast | Mean oriented gain | Paired t 95% CI | corrected dz | W/T/L | exact sign-flip p |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name, contrast in report["contrasts"].items():
        effect = contrast["effect"]
        interval = effect["paired_t_95_ci"]
        lines.append(
            f"| {name} | {effect['mean_oriented_gain']:.5f} | "
            f"[{interval[0]:.5f}, {interval[1]:.5f}] | "
            f"{effect['small_sample_corrected_dz']:.3f} | "
            f"{effect['wins']}/{effect['ties']}/{effect['losses']} | "
            f"{effect['exact_two_sided_sign_flip_p']:.4f} |"
        )
    lines.extend(["", report["caution"], ""])
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--confirmation",
        type=Path,
        default=Path("runs/v27-maplight-gnn-confirmation/result.json"),
    )
    parser.add_argument(
        "--scale",
        type=Path,
        default=Path("runs/v23-ablation-study/lipophilicity_scale_comparison.json"),
    )
    parser.add_argument(
        "--output", type=Path, default=Path("reports/primary_model_effects_v27")
    )
    args = parser.parse_args()
    report = build_report(args.confirmation, args.scale)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.with_suffix(".json").write_text(
        json.dumps(report, indent=2, sort_keys=True), encoding="utf-8"
    )
    args.output.with_suffix(".md").write_text(markdown(report), encoding="utf-8")
    print(json.dumps({name: value["effect"] for name, value in report["contrasts"].items()}, sort_keys=True))


if __name__ == "__main__":
    main()
