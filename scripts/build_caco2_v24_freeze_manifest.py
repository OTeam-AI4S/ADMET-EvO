"""Build an auditable, validation-only Caco2 v2.4 freeze manifest.

The builder validates discovery/confirmation evidence and hashes the frozen
recipe's prediction artifacts.  It never imports or loads the official TDC test
split.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from admet_agent.frozen_recipe import build_execution_plan
from admet_agent.tasks import resolve_task

DISCOVERY_SEEDS = tuple(range(1, 6))
CONFIRMATION_SEEDS = tuple(range(6, 11))
BASELINE_NAME = "lightgbm_current_binary_rich"
FROZEN_RECIPE = "2d_lightgbm"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Required Caco2 evidence is missing: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"Expected a JSON object in {path}")
    return value


def _artifact(repo_root: Path, path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Required frozen artifact is missing: {path}")
    return {
        "path": path.relative_to(repo_root).as_posix(),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def _assert_data_firewall(result: dict[str, Any], seeds: tuple[int, ...]) -> None:
    if result.get("official_test_loaded") is not False:
        raise ValueError("Caco2 evidence does not explicitly keep the official test sealed")
    preregistration = result.get("preregistration", {})
    if preregistration.get("test_labels_used") is not False:
        raise ValueError("Caco2 preregistration does not explicitly reject test labels")
    observed_seeds = tuple(int(row["seed"]) for row in result["data_isolation"])
    if observed_seeds != seeds:
        raise ValueError(f"Expected scaffold seeds {seeds}, got {observed_seeds}")
    for row in result["data_isolation"]:
        for field in (
            "smiles_overlap",
            "bemis_murcko_scaffold_overlap",
            "invalid_smiles_train",
            "invalid_smiles_valid",
        ):
            if int(row[field]) != 0:
                raise ValueError(f"Data-isolation audit failed for seed {row['seed']}: {field}")


def _candidate(summary: dict[str, Any], name: str) -> dict[str, Any]:
    for row in summary["candidates"]:
        if row["name"] == name:
            return row
    raise ValueError(f"Candidate {name!r} is missing from paired comparison")


def _reject_decision(
    comparison: dict[str, Any], name: str, reason: str
) -> dict[str, Any]:
    row = _candidate(comparison, name)
    return {
        "status": "reject",
        "reason": reason,
        "mean_mae": row["mean"],
        "mean_improvement": row["mean_improvement"],
        "bootstrap_95_ci": row["bootstrap_95_ci"],
        "wins_ties_losses": row["wins_ties_losses"],
        "cost_delta": row["cost_delta"],
    }


def build_manifest(repo_root: Path) -> dict[str, Any]:
    discovery_root = repo_root / "runs" / "caco2-feature-ablation-v24"
    confirmation_root = repo_root / "runs" / "caco2-feature-confirmation-v24"
    nested_root = repo_root / "runs" / "caco2-nested-selection-v24"

    discovery_results_path = discovery_root / "results.json"
    confirmation_results_path = confirmation_root / "results.json"
    discovery_comparison_path = (
        discovery_root / "comparisons" / "current_baseline.summary.json"
    )
    discovery_ladder_path = (
        discovery_root / "comparisons" / "feature_ladder.summary.json"
    )
    confirmation_comparison_path = (
        confirmation_root
        / "comparisons"
        / "current_baseline_confirmation.summary.json"
    )
    confirmation_ladder_path = (
        confirmation_root
        / "comparisons"
        / "feature_ladder_confirmation.summary.json"
    )
    nested_comparison_path = (
        nested_root / "comparisons" / "nested_selection.summary.json"
    )

    discovery_results = _load_json(discovery_results_path)
    confirmation_results = _load_json(confirmation_results_path)
    discovery_comparison = _load_json(discovery_comparison_path)
    discovery_ladder = _load_json(discovery_ladder_path)
    confirmation_comparison = _load_json(confirmation_comparison_path)
    confirmation_ladder = _load_json(confirmation_ladder_path)
    nested_comparison = _load_json(nested_comparison_path)

    _assert_data_firewall(discovery_results, DISCOVERY_SEEDS)
    _assert_data_firewall(confirmation_results, CONFIRMATION_SEEDS)
    if confirmation_results["preregistration"].get("phase") != "confirmation":
        raise ValueError("Fresh-seed evidence is not marked as confirmation")
    practical_threshold = float(
        confirmation_results["preregistration"]["minimum_useful_mae_reduction"]
    )
    if practical_threshold != 0.01:
        raise ValueError("The Caco2 practical threshold must remain preregistered at 0.01")
    if confirmation_comparison["baseline"]["name"] != BASELINE_NAME:
        raise ValueError("Confirmation comparison does not use the frozen baseline")
    if any(
        comparison["protocol"].get("official_test_loaded") is not False
        or comparison["protocol"].get("test_labels_used") is not False
        for comparison in (
            discovery_comparison,
            discovery_ladder,
            confirmation_comparison,
            confirmation_ladder,
            nested_comparison,
        )
    ):
        raise ValueError("A paired comparison crossed the official-test firewall")

    baseline_discovery = discovery_comparison["baseline"]
    baseline_confirmation = confirmation_comparison["baseline"]
    source_paths = [
        discovery_results_path,
        confirmation_results_path,
        discovery_comparison_path,
        discovery_ladder_path,
        confirmation_comparison_path,
        confirmation_ladder_path,
        nested_comparison_path,
        repo_root / "scripts" / "run_caco2_feature_ablation.py",
        repo_root / "scripts" / "run_caco2_nested_selection.py",
        repo_root / "scripts" / "build_caco2_v24_freeze_manifest.py",
        repo_root / "admet_agent" / "features.py",
        repo_root / "admet_agent" / "feature_selection.py",
        repo_root / "admet_agent" / "frozen_recipe.py",
    ]
    for seed in (*DISCOVERY_SEEDS, *CONFIRMATION_SEEDS):
        phase_root = discovery_root if seed in DISCOVERY_SEEDS else confirmation_root
        source_paths.append(
            phase_root / "artifacts" / f"seed_{seed}" / f"{BASELINE_NAME}.npz"
        )

    decisions = {
        "current_binary_rich_lightgbm": {
            "status": "promote",
            "claim": "retain_and_freeze_current_validation_baseline; no new improvement claimed",
            "reason": (
                "No challenger cleared the preregistered 0.01-MAE confirmation gate; "
                "the current recipe remains the cost-effective validation control"
            ),
            "discovery_mean_mae": baseline_discovery["mean"],
            "discovery_sample_std": baseline_discovery["sample_std"],
            "confirmation_mean_mae": baseline_confirmation["mean"],
            "confirmation_sample_std": baseline_confirmation["sample_std"],
        },
        "full_count_lightgbm": _reject_decision(
            confirmation_comparison,
            "lightgbm_full_count",
            "Confirmation gain is below 0.01 MAE and its interval crosses zero",
        ),
        "full_count_extra_trees": _reject_decision(
            confirmation_comparison,
            "extra_trees_full_count",
            "Confirmation mean regresses and costs substantially more",
        ),
        "catboost_rdkit_erg_morgan_count": _reject_decision(
            confirmation_comparison,
            "catboost_rdkit_erg_morgan_count",
            "Fresh-seed comparison loses to the current baseline on all five seeds",
        ),
        "nested_rdkit2d_selection": _reject_decision(
            nested_comparison,
            "nested_selected_rdkit2d",
            "Nested selection regresses, never wins, and adds substantial runtime",
        ),
        "catboost_plus_avalon": _reject_decision(
            discovery_ladder,
            "catboost_rdkit_erg_morgan_count_avalon",
            "Avalon addition misses the practical threshold in discovery",
        ),
    }

    manifest: dict[str, Any] = {
        "schema_version": 1,
        "task": "Caco2_Wang",
        "metric": "mae",
        "lower_is_better": True,
        "evidence_class": "repository_internal_validation",
        "freeze": {
            "status": "validation_candidate_frozen",
            "recipe": FROZEN_RECIPE,
            "model": "lightgbm",
            "representations": ["morgan", "avalon", "erg", "rdkit2d"],
            "execution_plan": build_execution_plan(
                resolve_task("Caco2_Wang"), FROZEN_RECIPE
            ),
            "discovery_seeds": list(DISCOVERY_SEEDS),
            "confirmation_seeds": list(CONFIRMATION_SEEDS),
            "minimum_useful_mae_reduction": practical_threshold,
            "official_test_loaded": False,
            "test_labels_used": False,
            "official_test_authorized": False,
            "leaderboard_feedback_used_for_selection": False,
        },
        "data_sha256": confirmation_results["environment"]["data_train_val_sha256"],
        "code_revision": confirmation_results["environment"]["git"],
        "environment": confirmation_results["environment"]["packages"],
        "decisions": decisions,
        "source_artifacts": [_artifact(repo_root, path) for path in source_paths],
    }
    canonical = json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode()
    manifest["manifest_sha256"] = hashlib.sha256(canonical).hexdigest()
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    parser.add_argument(
        "--output", type=Path, default=Path("reports/caco2_v24_freeze_manifest.json")
    )
    args = parser.parse_args()
    repo_root = args.repo_root.resolve()
    output = args.output if args.output.is_absolute() else repo_root / args.output
    manifest = build_manifest(repo_root)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
