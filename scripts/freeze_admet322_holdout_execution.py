"""Freeze hashes for all recipes and confirmation code before holdout execution."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--confirmation-cohort", type=Path, required=True)
    parser.add_argument("--development-cohort", type=Path, required=True)
    parser.add_argument("--recipe-dir", type=Path, required=True)
    parser.add_argument("--preregistration", type=Path, required=True)
    parser.add_argument("--amendment", type=Path, required=True)
    parser.add_argument("--runner", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    confirmation_ids = set(
        pd.read_csv(args.confirmation_cohort, usecols=["endpoint_id"])["endpoint_id"].astype(str)
    )
    development_ids = set(
        pd.read_csv(args.development_cohort, usecols=["endpoint_id"])["endpoint_id"].astype(str)
    )
    manifests: dict[str, Path] = {}
    for path in args.recipe_dir.glob("*.manifest.json"):
        payload = json.loads(path.read_text())
        endpoint_id = str(payload["endpoint_id"])
        if endpoint_id in manifests:
            raise RuntimeError(f"Duplicate recipe manifest for {endpoint_id}")
        if payload.get("holdout_labels_loaded") is not False:
            raise RuntimeError(f"Holdout firewall not certified for {endpoint_id}")
        if payload.get("integrity_passed") is not True:
            raise RuntimeError(f"Development integrity failed for {endpoint_id}")
        manifests[endpoint_id] = path
    if set(manifests) != development_ids:
        raise RuntimeError(
            f"Recipe set incomplete: expected={len(development_ids)}, observed={len(manifests)}"
        )
    if not confirmation_ids.issubset(set(manifests)):
        raise RuntimeError("A fresh confirmation endpoint lacks a frozen recipe")
    recipe_hash_lines = [
        f"{endpoint_id}\t{_sha256(manifests[endpoint_id])}" for endpoint_id in sorted(manifests)
    ]
    recipe_set_sha256 = hashlib.sha256(("\n".join(recipe_hash_lines) + "\n").encode()).hexdigest()
    output = {
        "version": "admet322-holdout-execution-freeze-v1",
        "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
        "confirmation_endpoint_count": len(confirmation_ids),
        "development_recipe_count": len(manifests),
        "confirmation_cohort": str(args.confirmation_cohort.resolve()),
        "confirmation_cohort_sha256": _sha256(args.confirmation_cohort),
        "development_cohort": str(args.development_cohort.resolve()),
        "development_cohort_sha256": _sha256(args.development_cohort),
        "recipe_dir": str(args.recipe_dir.resolve()),
        "recipe_set_sha256": recipe_set_sha256,
        "preregistration": str(args.preregistration.resolve()),
        "preregistration_sha256": _sha256(args.preregistration),
        "threshold_amendment": str(args.amendment.resolve()),
        "threshold_amendment_sha256": _sha256(args.amendment),
        "holdout_runner": str(args.runner.resolve()),
        "holdout_runner_sha256": _sha256(args.runner),
        "development_seed": 17,
        "confirmation_seeds": [29, 43],
        "train_folds": [0, 1, 2],
        "holdout_folds": [3, 4],
        "holdout_executed_when_frozen": False,
    }
    args.output.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    print(json.dumps(output, sort_keys=True))


if __name__ == "__main__":
    main()
