"""Summarize validation-only rescue attempts, including negative results."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def build_report(repo_root: Path) -> dict[str, Any]:
    transfer_root = repo_root / "runs" / "v24-endpoint-transfer"
    bio = _load(transfer_root / "Bioavailability_Ma" / "results.json")
    cyp = _load(
        transfer_root / "CYP2C9_Substrate_CarbonMangels" / "results.json"
    )
    rank = _load(repo_root / "runs" / "v24-cyp2c9-rank-confirmation" / "results.json")[
        "tasks"
    ]["CYP2C9_Substrate_CarbonMangels"]
    balanced = _load(repo_root / "runs" / "v24-cyp2c9-balanced" / "results.json")[
        "tasks"
    ]["CYP2C9_Substrate_CarbonMangels"]
    minimol_root = (
        repo_root
        / "runs"
        / "v24-minimol-validation"
        / "CYP2C9_Substrate_CarbonMangels"
    )
    minimol = _load(minimol_root / "results.json")
    minimol_fusion = _load(minimol_root / "rank_fusion.json")
    minimol_inner = _load(minimol_root / "inner_ensemble.json")
    maplight_path = minimol_root / "maplight_minimol.json"
    maplight = _load(maplight_path) if maplight_path.exists() else None
    bio_baseline = bio["summaries"]["baseline_lightgbm"]["mean"]
    bio_teacher = bio["summaries"]["teacher_aug_svm"]["mean"]
    cyp_baseline = cyp["summaries"]["baseline_lightgbm"]["mean"]
    cyp_teacher = cyp["summaries"]["teacher_aug_lightgbm"]["mean"]
    fresh_best = rank["results"][rank["best"]]["mean"]
    fresh_single = rank["results"][rank["best_single_model"]]["mean"]
    experiments = [
            {
                "task": "Bioavailability_Ma",
                "method": "exact-overlap-excluded cross-endpoint teacher features",
                "baseline": bio_baseline,
                "candidate": bio_teacher,
                "gain": bio_teacher - bio_baseline,
                "decision": "retain_for_ablation_not_freeze",
                "reason": (
                    "small mean gain but insufficient evidence for the 0.225 "
                    "Top-10 gap"
                ),
            },
            {
                "task": "CYP2C9_Substrate_CarbonMangels",
                "method": "exact-overlap-excluded cross-endpoint teacher features",
                "baseline": cyp_baseline,
                "candidate": cyp_teacher,
                "gain": cyp_teacher - cyp_baseline,
                "decision": "reject",
                "reason": "teacher augmentation reduced the five-seed mean",
            },
            {
                "task": "CYP2C9_Substrate_CarbonMangels",
                "method": "fresh-seed 6-10 fixed heterogeneous rank ensemble",
                "baseline": fresh_single,
                "candidate": fresh_best,
                "gain": fresh_best - fresh_single,
                "decision": "reject",
                "reason": "gain is small and variance remains high",
            },
            {
                "task": "CYP2C9_Substrate_CarbonMangels",
                "method": "balanced bagged Extra Trees",
                "baseline": balanced["candidates"][balanced["best_conventional"]][
                    "mean"
                ],
                "candidate": max(
                    row["mean"] for row in balanced["candidates"].values()
                ),
                "gain": balanced["best_delta_vs_conventional"],
                "decision": "reject",
                "reason": "no gain over conventional LightGBM",
            },
            {
                "task": "CYP2C9_Substrate_CarbonMangels",
                "method": "MiniMol 512-d fingerprint with fixed official MLP head",
                "baseline": cyp_baseline,
                "candidate": minimol["mean"],
                "gain": minimol["mean"] - cyp_baseline,
                "decision": "reject",
                "reason": "pretrained fingerprint reduced five-seed validation mean",
            },
            {
                "task": "CYP2C9_Substrate_CarbonMangels",
                "method": "equal-rank MiniMol and LightGBM fusion",
                "baseline": minimol_fusion["summary"]["2d_lightgbm"]["mean"],
                "candidate": minimol_fusion["summary"]["equal_rank_fusion"]["mean"],
                "gain": minimol_fusion["gain_over_baseline"],
                "decision": "reject",
                "reason": "single preregistered fusion reduced validation mean",
            },
            {
                "task": "CYP2C9_Substrate_CarbonMangels",
                "method": "MiniMol inner-scaffold five-head ensemble",
                "baseline": cyp_baseline,
                "candidate": minimol_inner["mean"],
                "gain": minimol_inner["mean"] - cyp_baseline,
                "decision": "reject",
                "reason": "paper-aligned ensemble remained below LightGBM",
            },
        ]
    if maplight is not None:
        best_name = maplight["best_fixed_candidate"]
        best = maplight["fixed_candidates"][best_name]
        baseline_scores = cyp["summaries"]["baseline_lightgbm"]["scores"]
        paired_gains = [
            candidate - baseline
            for candidate, baseline in zip(best["scores"], baseline_scores)
        ]
        experiments.append(
            {
                "task": "CYP2C9_Substrate_CarbonMangels",
                "method": "balanced eight-member MapLight+MiniMol CatBoost",
                "baseline": cyp_baseline,
                "candidate": best["mean"],
                "gain": best["mean"] - cyp_baseline,
                "decision": "reject",
                "reason": (
                    "fixed and leave-one-seed-out selection both remained below "
                    "LightGBM; only two of five paired scaffold seeds improved"
                ),
                "candidate_name": best_name,
                "candidate_scores": best["scores"],
                "candidate_std": best["std"],
                "baseline_scores": baseline_scores,
                "paired_gains": paired_gains,
                "paired_improvements": sum(value > 0 for value in paired_gains),
                "artifact": str(maplight_path.relative_to(repo_root)),
                "official_test_loaded": False,
            }
        )
    return {
        "official_test_loaded": False,
        "test_labels_used": False,
        "experiments": experiments,
    }


def markdown(report: dict[str, Any]) -> str:
    rows = []
    for item in report["experiments"]:
        rows.append(
            f"| {item['task']} | {item['method']} | {item['baseline']:.4f} | "
            f"{item['candidate']:.4f} | {item['gain']:+.4f} | {item['decision']} |"
        )
    return "\n".join(
        [
            "# v2.4 weak-task rescue ledger",
            "",
            "All results are validation-only; negative results are retained.",
            "",
            "| Task | Method | Baseline | Candidate | Gain | Decision |",
            "|---|---|---:|---:|---:|---|",
            *rows,
            "",
            "Bioavailability teacher features remove every canonical exact target-development molecule from each auxiliary teacher training set.",
            "",
        ]
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    parser.add_argument(
        "--output", type=Path, default=Path("runs/v24-rescue-ledger/report")
    )
    args = parser.parse_args()
    report = build_report(args.repo_root.resolve())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.with_suffix(".json").write_text(
        json.dumps(report, indent=2, sort_keys=True), encoding="utf-8"
    )
    args.output.with_suffix(".md").write_text(markdown(report), encoding="utf-8")
    print(json.dumps({"experiments": len(report["experiments"])}, sort_keys=True))


if __name__ == "__main__":
    main()
