"""Derive an auditable best-mean ratio selection from completed Development runs.

The raw Development artifacts are never modified.  For each task with a complete
ratio sweep, this script chooses the ratio with the largest mean oriented delta
across Development seeds.  Exact ties are resolved in favour of the smaller
ratio.  The derived artifact also records sample variance and standard deviation
for the selected ratio and can be consumed directly by the existing independent
Confirmation runner.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import statistics
from pathlib import Path
from typing import Any


POLICY = "maximum_development_mean_oriented_delta_ties_choose_smaller_ratio"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _derive(payload: dict[str, Any], source_path: Path) -> dict[str, Any]:
    derived = copy.deepcopy(payload)
    derived["selection_policy"] = POLICY
    derived["original_selection"] = copy.deepcopy(payload.get("selection"))
    derived["derived_from_development_artifact"] = str(source_path.resolve())
    derived["derived_from_development_artifact_sha256"] = _sha256(source_path)

    summaries = payload.get("ratio_summaries")
    rows = payload.get("per_seed_ratio_model")
    if not isinstance(summaries, dict) or not summaries or not isinstance(rows, list):
        derived["best_mean_selection"] = None
        return derived

    candidates: list[tuple[float, dict[str, Any]]] = []
    for ratio_text, summary in summaries.items():
        if not isinstance(summary, dict):
            continue
        mean = summary.get("oriented_delta_mean")
        if mean is None or not math.isfinite(float(mean)):
            continue
        candidates.append((float(ratio_text), summary))
    if not candidates:
        raise RuntimeError(f"No finite ratio summaries in {source_path}")

    best_mean = max(float(summary["oriented_delta_mean"]) for _, summary in candidates)
    tied = [
        (ratio, summary)
        for ratio, summary in candidates
        if math.isclose(
            float(summary["oriented_delta_mean"]),
            best_mean,
            rel_tol=0.0,
            abs_tol=1e-12,
        )
    ]
    best_ratio, best_summary = min(tied, key=lambda item: item[0])
    deltas = [
        float(row["oriented_delta"])
        for row in rows
        if math.isclose(
            float(row["ratio"]), best_ratio, rel_tol=0.0, abs_tol=1e-12
        )
    ]
    if not deltas:
        raise RuntimeError(
            f"Selected ratio {best_ratio:g} has no per-seed rows in {source_path}"
        )
    mean = statistics.fmean(deltas)
    sample_variance = statistics.variance(deltas) if len(deltas) > 1 else 0.0
    sample_std = math.sqrt(sample_variance)
    if not math.isclose(mean, best_mean, rel_tol=0.0, abs_tol=1e-12):
        raise RuntimeError(f"Ratio summary mean disagrees with per-seed rows in {source_path}")

    selection = {
        "status": "selected_best_mean",
        "selected_ratio": best_ratio,
        "best_ratio": best_ratio,
        "best_mean": best_mean,
        "rule": POLICY,
        "tie_count": len(tied),
    }
    statistics_payload = {
        "ratio": best_ratio,
        "n_seeds": len(deltas),
        "oriented_delta_mean": mean,
        "oriented_delta_sample_variance": sample_variance,
        "oriented_delta_sample_standard_deviation": sample_std,
        "oriented_delta_values": deltas,
        "baseline_mean": best_summary.get("baseline_mean"),
        "augmented_mean": best_summary.get("augmented_mean"),
        "positive_seed_fraction": best_summary.get("positive_seed_fraction"),
    }
    derived["status"] = "selected_best_mean"
    derived["selection"] = selection
    derived["best_mean_selection"] = statistics_payload
    return derived


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    manifest: dict[str, Any] = {"selection_policy": POLICY, "tasks": {}}
    for source_path in sorted(args.source_dir.glob("*.json")):
        if source_path.name.endswith(".error.json"):
            continue
        payload = json.loads(source_path.read_text(encoding="utf-8"))
        task = payload.get("task")
        if not isinstance(task, str) or not task:
            continue
        derived = _derive(payload, source_path)
        output_path = args.output_dir / f"{task}.json"
        output_path.write_text(
            json.dumps(derived, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        manifest["tasks"][task] = {
            "status": derived.get("status"),
            "selection": derived.get("selection"),
            "best_mean_selection": derived.get("best_mean_selection"),
            "artifact": str(output_path.resolve()),
            "artifact_sha256": _sha256(output_path),
        }

    manifest_path = args.output_dir / "_best_mean_selection_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({"tasks": len(manifest["tasks"]), "manifest": str(manifest_path)}))


if __name__ == "__main__":
    main()
