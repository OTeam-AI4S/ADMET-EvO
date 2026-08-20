"""Build a validation-only freeze manifest for the first v2.4 finalists.

The manifest is deliberately stricter than a result summary: every source artifact is
hashed, scaffold seeds 1-5 must be present, and execution readiness is kept separate
from validation evidence.  It does not load the TDC official test split.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
from pathlib import Path
from typing import Any

from admet_agent.frozen_recipe import build_execution_plan
from admet_agent.tasks import resolve_task

SEEDS = tuple(range(1, 6))


def _finalist_config(recipe: str, protocol: dict[str, Any]) -> dict[str, Any]:
    return {"kind": "recipe", "recipe": recipe, "protocol": protocol}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def require_seed_artifacts(root: Path, filename: str) -> list[Path]:
    paths = [root / f"seed_{seed}" / filename for seed in SEEDS]
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing frozen seed artifacts: {missing}")
    return paths


def _artifact_records(repo_root: Path, paths: list[Path]) -> list[dict[str, Any]]:
    records = []
    for path in sorted(set(paths)):
        records.append(
            {
                "path": path.relative_to(repo_root).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    return records


def attach_smoke_evidence(
    repo_root: Path,
    candidate: dict[str, Any],
    mapping_path: Path,
) -> None:
    result_path = (
        repo_root
        / "runs"
        / "v24-recipe-smoke"
        / candidate["task"]
        / "seed_1"
        / "result.json"
    )
    if not result_path.is_file():
        candidate["smoke_test"] = {"status": "missing"}
        return
    result = _load_json(result_path)
    expected = candidate["finalist_config"]
    observed = result.get("frozen_config", {})
    reproducibility = result.get("reproducibility", {})
    checks = {
        "passed": result.get("status") == "passed",
        "task_matches": result.get("task") == candidate["task"],
        "recipe_matches": observed.get("recipe") == expected["recipe"],
        "protocol_matches": observed.get("protocol") == expected["protocol"],
        "official_test_not_loaded": result.get("official_test_loaded") is False,
        "test_labels_not_passed": (
            result.get("test_labels_passed_to_executor") is False
        ),
        "mapping_hash_matches": (
            mapping_path.is_file()
            and result.get("mapping_sha256") == sha256_file(mapping_path)
        ),
        "cuda_recorded": bool(result.get("environment", {}).get("cuda")),
        "gpu_recorded": bool(result.get("environment", {}).get("gpu")),
        "unimol_tools_recorded": bool(
            result.get("environment", {}).get("unimol_tools")
        ),
        "training_seed_matches": reproducibility.get("training_seed") == 1,
        "split_seed_matches": reproducibility.get("split_seed") == 1,
        "requested_seed_matches": reproducibility.get("requested_seed") == 1,
    }
    valid = all(checks.values())
    candidate["smoke_test"] = {
        "status": "valid" if valid else "invalid",
        "checks": checks,
        "score": result.get("score"),
        "elapsed_seconds": result.get("elapsed_seconds"),
        "artifact": _artifact_records(repo_root, [result_path])[0],
    }
    if valid:
        candidate["execution_ready"] = True
        candidate["ready_for_official_test"] = True
        candidate["execution_blocker"] = None


def _pilot(repo_root: Path, task: str) -> tuple[Path, dict[str, Any]]:
    path = repo_root / "runs" / "unimol2-pilot-v22" / task / "results.json"
    result = _load_json(path)
    protocol = result.get("protocol", {})
    if protocol.get("test_labels_used") is not False:
        raise ValueError(f"{task} pilot is not explicitly validation-only")
    if tuple(protocol.get("outer_seeds", ())) != SEEDS:
        raise ValueError(f"{task} pilot does not contain scaffold seeds 1-5")
    return path, result


def _router_candidate(
    repo_root: Path,
    task: str,
    recipe: str,
    adapter: str,
) -> dict[str, Any]:
    pilot_path, pilot = _pilot(repo_root, task)
    pilot_root = pilot_path.parent
    prediction_paths = require_seed_artifacts(
        pilot_root, "validation_predictions.npz"
    )
    router_path = repo_root / "runs" / "unimol2-pilot-v22" / "router_ablation.json"
    routed = _load_json(router_path)["tasks"][task]
    if routed["best_blend_in_sample"] != recipe:
        raise ValueError(f"Frozen {task} recipe no longer matches router evidence")
    loso = routed["loso"]["blend_lcb_0.25" if task == "DILI" else "blend_lcb_0.0"]
    selected_count = sum(value == recipe for value in loso["selected"].values())
    runtime_protocol = {
        "model_size": "84m",
        "learning_rate": 1e-4,
        "epochs": int(pilot["protocol"]["epochs"]),
        "batch_size": int(pilot["protocol"]["batch_size"]),
        "inner_folds": 5,
    }
    return {
        "task": task,
        "metric": routed["metric"],
        "recipe": recipe,
        "execution_plan": build_execution_plan(resolve_task(task), recipe),
        "finalist_config": _finalist_config(recipe, runtime_protocol),
        "protocol": {
            "outer_split": pilot["protocol"]["outer_split"],
            "outer_seeds": list(SEEDS),
            "selection": "fixed recipe chosen before any official-test access",
        },
        "validation": {
            "fixed_recipe_mean": routed["best_blend_in_sample_mean"],
            "loso_policy_mean": loso["mean"],
            "loso_policy_std": loso["std"],
            "fixed_recipe_selected_in_loso_folds": selected_count,
            "loso_folds": len(SEEDS),
        },
        "test_labels_used": False,
        "evidence_complete": True,
        "adapter_implemented": True,
        "execution_ready": False,
        "ready_for_official_test": False,
        "execution_blocker": adapter,
        "artifacts": _artifact_records(
            repo_root, [pilot_path, router_path, *prediction_paths]
        ),
    }


def _lipophilicity_candidate(repo_root: Path) -> dict[str, Any]:
    root = (
        repo_root
        / "runs"
        / "unimol2-scale-ablation-84m-b8"
        / "84m"
        / "Lipophilicity_AstraZeneca"
        / "lr_0.0001"
    )
    result_paths = require_seed_artifacts(root, "results.json")
    prediction_paths = require_seed_artifacts(root, "predictions.npz")
    records = [_load_json(path) for path in result_paths]
    if any(record.get("test_labels_used") is not False for record in records):
        raise ValueError("Lipophilicity scale evidence is not validation-only")
    if [int(record["seed"]) for record in records] != list(SEEDS):
        raise ValueError("Lipophilicity scale evidence does not contain seeds 1-5")
    protocol_fields = ("model_size", "learning_rate", "epochs", "inner_folds")
    protocol = {field: records[0][field] for field in protocol_fields}
    protocol["batch_size"] = 8
    if any(
        any(record[field] != protocol[field] for field in protocol_fields)
        for record in records
    ):
        raise ValueError("Lipophilicity seed protocols do not match")
    scores = [float(record["score"]) for record in records]
    scale_path = (
        repo_root
        / "runs"
        / "v23-ablation-study"
        / "lipophilicity_scale_comparison.json"
    )
    scale = _load_json(scale_path)
    if scale.get("recommendation") != "retain_smaller":
        raise ValueError("84m scale is not the frozen scale-comparison winner")
    return {
        "task": "Lipophilicity_AstraZeneca",
        "metric": records[0]["metric"],
        "recipe": "finetuned_unimol2",
        "execution_plan": build_execution_plan(
            resolve_task("Lipophilicity_AstraZeneca"), "finetuned_unimol2"
        ),
        "finalist_config": _finalist_config(
            "finetuned_unimol2",
            {
                "model_size": protocol["model_size"],
                "learning_rate": protocol["learning_rate"],
                "epochs": protocol["epochs"],
                "batch_size": protocol["batch_size"],
                "inner_folds": protocol["inner_folds"],
            },
        ),
        "protocol": {**protocol, "outer_seeds": list(SEEDS)},
        "validation": {
            "scores": scores,
            "mean": statistics.mean(scores),
            "std": statistics.pstdev(scores),
            "scale_decision": scale["recommendation"],
        },
        "test_labels_used": False,
        "evidence_complete": True,
        "adapter_implemented": True,
        "execution_ready": False,
        "ready_for_official_test": False,
        "execution_blocker": (
            "run a validation-only smoke test in the pinned unimol-tools GPU environment"
        ),
        "artifacts": _artifact_records(
            repo_root, [scale_path, *result_paths, *prediction_paths]
        ),
    }


def _half_life_candidate(repo_root: Path) -> dict[str, Any]:
    pilot_path, pilot = _pilot(repo_root, "Half_Life_Obach")
    root = repo_root / "runs" / "half-life-fusion-ablation"
    result_path = root / "results.json"
    result = _load_json(result_path)
    if result.get("test_labels_used") is not False:
        raise ValueError("Half-Life fusion evidence is not validation-only")
    recipe = result["best_fixed_candidate"]
    summary = result["summaries"][recipe]
    prediction_paths = require_seed_artifacts(root, "validation_predictions.npz")
    loso = result["loso"]["lcb_0.25"]
    selected_count = sum(value == recipe for value in loso["selected"].values())
    runtime_protocol = {
        "model_size": "84m",
        "learning_rate": 1e-4,
        "epochs": int(pilot["protocol"]["epochs"]),
        "batch_size": int(pilot["protocol"]["batch_size"]),
        "inner_folds": 5,
    }
    return {
        "task": "Half_Life_Obach",
        "metric": pilot["metric"],
        "recipe": recipe,
        "execution_plan": build_execution_plan(
            resolve_task("Half_Life_Obach"), recipe
        ),
        "finalist_config": _finalist_config(recipe, runtime_protocol),
        "protocol": {
            "outer_split": pilot["protocol"]["outer_split"],
            "outer_seeds": list(SEEDS),
            "selection": result["selection_protocol"],
        },
        "validation": {
            "fixed_recipe_scores": summary["scores"],
            "fixed_recipe_mean": summary["mean"],
            "fixed_recipe_std": summary["std"],
            "loso_policy_mean": loso["mean"],
            "loso_policy_std": loso["std"],
            "fixed_recipe_selected_in_loso_folds": selected_count,
            "loso_folds": len(SEEDS),
        },
        "test_labels_used": False,
        "evidence_complete": True,
        "adapter_implemented": True,
        "execution_ready": False,
        "ready_for_official_test": False,
        "execution_blocker": (
            "run a validation-only heterogeneous smoke test in the pinned GPU environment"
        ),
        "artifacts": _artifact_records(
            repo_root, [pilot_path, result_path, *prediction_paths]
        ),
    }


def build_manifest(repo_root: Path) -> dict[str, Any]:
    repo_root = repo_root.resolve()
    candidates = [
        _router_candidate(
            repo_root,
            "DILI",
            "raw(2d_lightgbm@0.33+2d_extra_trees@0.33+finetuned_unimol2@0.33)",
            "run a validation-only blend smoke test in the pinned unimol-tools GPU environment",
        ),
        _router_candidate(
            repo_root,
            "hERG",
            "raw(2d_lightgbm@0.50+finetuned_unimol2@0.50)",
            "run a validation-only blend smoke test in the pinned unimol-tools GPU environment",
        ),
        _lipophilicity_candidate(repo_root),
        _half_life_candidate(repo_root),
    ]
    finalists = {item["task"]: item["finalist_config"] for item in candidates}
    finalists_sha256 = hashlib.sha256(
        json.dumps(finalists, sort_keys=True).encode()
    ).hexdigest()
    mapping_path = repo_root / "runs" / "v24-freeze" / "finalists.mapping.json"
    for candidate in candidates:
        attach_smoke_evidence(repo_root, candidate, mapping_path)
    coverage_complete = len(candidates) == 22
    return {
        "manifest_version": 1,
        "phase": "v2.4-preregistration",
        "official_test_loaded": False,
        "official_test_runs": 0,
        "decision_rule": (
            "Do not access the official test split until every candidate has a "
            "frozen runtime adapter, deterministic dry-run, and passing hash audit."
        ),
        "all_evidence_complete": all(item["evidence_complete"] for item in candidates),
        "all_adapters_implemented": all(
            item["adapter_implemented"] for item in candidates
        ),
        "all_execution_ready": all(item["execution_ready"] for item in candidates),
        "target_task_count": 22,
        "frozen_task_count": len(candidates),
        "coverage_complete": coverage_complete,
        "ready_for_official_test": coverage_complete
        and all(item["ready_for_official_test"] for item in candidates),
        "runtime_sources": _artifact_records(
            repo_root,
            [
                repo_root / "admet_agent" / "frozen_executor.py",
                repo_root / "admet_agent" / "frozen_recipe.py",
                repo_root / "admet_agent" / "runner.py",
            ],
        ),
        "finalists": finalists,
        "finalists_sha256": finalists_sha256,
        "candidates": candidates,
    }


def markdown(report: dict[str, Any]) -> str:
    rows = []
    for item in report["candidates"]:
        validation = item["validation"]
        mean = validation.get("fixed_recipe_mean", validation.get("mean"))
        rows.append(
            f"| {item['task']} | `{item['recipe']}` | {mean:.4f} | "
            f"yes | {'yes' if item['execution_ready'] else 'no'} |"
        )
    blockers = [
        f"- **{item['task']}**: {item['execution_blocker']}"
        for item in report["candidates"]
        if not item["execution_ready"]
    ]
    return "\n".join(
        [
            "# v2.4 finalist freeze manifest",
            "",
            "Validation-only preregistration. The official test split has not been loaded.",
            "",
            "| Task | Frozen recipe | Validation mean | Evidence | Executable |",
            "|---|---|---:|:---:|:---:|",
            *rows,
            "",
            f"Ready for official test: **{report['ready_for_official_test']}**.",
            "",
            "## Remaining execution blockers",
            "",
            *blockers,
            "",
            "Every listed evidence artifact is content-addressed by SHA-256 in the JSON manifest.",
            (
                "The CLI-ready mapping is `finalists.mapping.json` and its canonical "
                f"digest is `{report['finalists_sha256']}`."
            ),
            "",
        ]
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    parser.add_argument(
        "--output", type=Path, default=Path("runs/v24-freeze/finalists")
    )
    args = parser.parse_args()
    report = build_manifest(args.repo_root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.with_suffix(".json").write_text(
        json.dumps(report, indent=2, sort_keys=True), encoding="utf-8"
    )
    args.output.with_suffix(".mapping.json").write_text(
        json.dumps(report["finalists"], indent=2, sort_keys=True), encoding="utf-8"
    )
    args.output.with_suffix(".md").write_text(markdown(report), encoding="utf-8")
    print(
        json.dumps(
            {
                "candidates": len(report["candidates"]),
                "official_test_loaded": report["official_test_loaded"],
                "ready_for_official_test": report["ready_for_official_test"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
