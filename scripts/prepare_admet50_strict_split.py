"""Freeze model-selection and untouched-confirmation partitions for ADMET50."""

from __future__ import annotations

import hashlib
import json
import argparse
from pathlib import Path

import pandas as pd


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=Path("reports/admet50_discovery_cohort_v1.csv"))
    parser.add_argument("--output", type=Path, default=Path("reports/admet50_model_selection_cohort_v2.csv"))
    parser.add_argument("--manifest", type=Path, default=Path("reports/admet50_strict_split_manifest_v2.json"))
    parser.add_argument("--version", default="admet50-strict-selection-confirmation-split-v2")
    args = parser.parse_args()
    source = args.source
    cohort = pd.read_csv(source)
    if sorted(cohort["outer_fold"].unique()) != [0, 1, 2, 3, 4]:
        raise RuntimeError("Expected the frozen five-fold ADMET50 cohort")
    selection = cohort.loc[cohort["outer_fold"].isin([0, 1, 2])].copy()
    output = args.output
    selection.to_csv(output, index=False)
    source_endpoints = int(cohort["endpoint_id"].nunique())
    if selection["endpoint_id"].nunique() != source_endpoints:
        raise RuntimeError("All source endpoints must survive in the selection partition")
    if int((cohort.groupby("chemistry_group")["outer_fold"].nunique() > 1).sum()):
        raise RuntimeError("Chemistry-group fold leakage")
    manifest = {
        "version": args.version,
        "source_cohort": str(source.resolve()), "source_sha256": _sha256(source),
        "model_selection_folds": [0, 1, 2], "untouched_confirmation_folds": [3, 4],
        "model_selection_protocol": (
            "Three-fold CV entirely within folds 0,1,2; neither labels nor structures "
            "from folds 3,4 enter action or endpoint selection models."
        ),
        "confirmation_protocol": (
            "Fit the frozen selected action on all rows from folds 0,1,2 and evaluate "
            "once on the union of folds 3,4; seeds 29 and 43 quantify fit stochasticity."
        ),
        "selection_cohort": str(output.resolve()), "selection_sha256": _sha256(output),
        "selection_rows": len(selection),
        "confirmation_rows": int(cohort["outer_fold"].isin([3, 4]).sum()),
        "endpoints": source_endpoints,
        "confirmation_labels_used_for_selection": False,
    }
    path = args.manifest
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps(manifest, sort_keys=True))


if __name__ == "__main__":
    main()
