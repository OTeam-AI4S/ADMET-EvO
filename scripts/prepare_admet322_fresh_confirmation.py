"""Freeze the endpoint set that was unseen in the earlier ADMET50 experiment."""

from __future__ import annotations

import argparse
import hashlib
import json
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
    parser.add_argument("--full-cohort", type=Path, required=True)
    parser.add_argument("--previous-cohort", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()

    full = pd.read_csv(args.full_cohort)
    previous_ids = set(
        pd.read_csv(args.previous_cohort, usecols=["endpoint_id"])["endpoint_id"].astype(str)
    )
    all_ids = set(full["endpoint_id"].astype(str))
    fresh_ids = sorted(all_ids - previous_ids)
    fresh = full.loc[full["endpoint_id"].astype(str).isin(fresh_ids)].copy()
    if set(fresh["endpoint_id"].astype(str)) != set(fresh_ids):
        raise RuntimeError("Fresh endpoint filtering failed")
    if sorted(fresh["outer_fold"].astype(int).unique()) != [0, 1, 2, 3, 4]:
        raise RuntimeError("Expected all five frozen folds")
    if int((fresh.groupby("inchikey")["outer_fold"].nunique() > 1).sum()):
        raise RuntimeError("Identity leakage in fresh confirmation cohort")
    if int((fresh.groupby("chemistry_group")["outer_fold"].nunique() > 1).sum()):
        raise RuntimeError("Chemistry-group leakage in fresh confirmation cohort")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fresh.to_csv(args.output, index=False)
    manifest = {
        "version": "admet322-fresh-outside-admet50-confirmation-v1",
        "full_cohort": str(args.full_cohort.resolve()),
        "full_cohort_sha256": _sha256(args.full_cohort),
        "previous_cohort": str(args.previous_cohort.resolve()),
        "previous_cohort_sha256": _sha256(args.previous_cohort),
        "output": str(args.output.resolve()),
        "output_sha256": _sha256(args.output),
        "all_endpoint_count": len(all_ids),
        "previously_exposed_endpoint_count": len(all_ids & previous_ids),
        "fresh_endpoint_count": len(fresh_ids),
        "fresh_rows": len(fresh),
        "development_folds": [0, 1, 2],
        "untouched_confirmation_folds": [3, 4],
        "selection_basis": "set difference from the previously evaluated ADMET50 endpoint IDs only",
        "confirmation_outcomes_used_for_endpoint_selection": False,
        "identity_fold_overlap": 0,
        "chemistry_group_fold_overlap": 0,
    }
    args.manifest.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps(manifest, sort_keys=True))


if __name__ == "__main__":
    main()
