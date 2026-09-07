"""Prepare direct-label ADMET endpoints that are outside the canonical TDC-22."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd

from admet_agent.cohort_governance_v4 import assign_identity_locked_folds
from admet_agent.fresh_cohort_v6 import prepare_endpoint, standardize_structure_column
from admet_agent.tasks import TASKS


SOURCES = (
    {
        "endpoint_id": "tdc_extra::CYP1A2_Veith",
        "dataset": "CYP1A2_Veith",
        "path": Path("external/new_admet_tdc_raw/cyp1a2_veith.tab"),
        "source_id": "veith_qhts_2009",
        "task_type": "classification",
        "metric": "mcc",
        "label_contract": "TDC direct binary inhibition label",
    },
    {
        "endpoint_id": "tdc_extra::CYP2C19_Veith",
        "dataset": "CYP2C19_Veith",
        "path": Path("external/new_admet_tdc_raw/cyp2c19_veith.tab"),
        "source_id": "veith_qhts_2009",
        "task_type": "classification",
        "metric": "mcc",
        "label_contract": "TDC direct binary inhibition label",
    },
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    canonical = {name.lower() for name in TASKS}
    if any(item["dataset"].lower() in canonical for item in SOURCES):
        raise RuntimeError("Same-lineage cohort unexpectedly contains a canonical TDC-22 task")
    cache: dict = {}
    prepared = []
    audits = {}
    raw_hashes = {}
    for spec in SOURCES:
        path = spec["path"]
        raw_hashes[str(path)] = _sha256(path)
        raw = pd.read_csv(path, sep="\t")
        standardized, structure_audit = standardize_structure_column(
            raw,
            smiles_column="Drug",
            cache=cache,
        )
        endpoint, endpoint_audit = prepare_endpoint(
            standardized,
            target_column="Y",
            task_type=spec["task_type"],
        )
        endpoint["endpoint_id"] = spec["endpoint_id"]
        endpoint["source_id"] = spec["source_id"]
        endpoint["task_type"] = spec["task_type"]
        endpoint["metric"] = spec["metric"]
        endpoint["metric_direction"] = "maximize"
        prepared.append(endpoint)
        audits[spec["endpoint_id"]] = {
            "dataset": spec["dataset"],
            "label_contract": spec["label_contract"],
            "structure": structure_audit,
            "endpoint": endpoint_audit,
        }

    all_rows = pd.concat(prepared, ignore_index=True)
    identity_groups = all_rows[["inchikey", "chemistry_group"]].drop_duplicates()
    assignment = assign_identity_locked_folds(identity_groups, folds=5, seed=20260827)
    all_rows["outer_fold"] = all_rows["inchikey"].map(assignment).astype(int)
    columns = [
        "endpoint_id",
        "source_id",
        "task_type",
        "metric",
        "metric_direction",
        "inchikey",
        "standard_smiles",
        "chemistry_group",
        "target",
        "replicate_count",
        "outer_fold",
    ]
    cohort = all_rows[columns].sort_values(
        ["endpoint_id", "outer_fold", "inchikey"], kind="stable"
    ).reset_index(drop=True)
    output = Path("reports/same_lineage_admet_cohort_v1.csv")
    output.parent.mkdir(parents=True, exist_ok=True)
    cohort.to_csv(output, index=False)
    manifest = {
        "version": "same-lineage-admet-cohort-v1",
        "purpose": "generalization to direct-label, same-publication endpoints outside TDC-22",
        "raw_sha256": raw_hashes,
        "cohort_path": str(output.resolve()),
        "cohort_sha256": _sha256(output),
        "rows": len(cohort),
        "endpoints": int(cohort["endpoint_id"].nunique()),
        "endpoint_audits": audits,
        "split": {
            "outer_folds": 5,
            "seed": 20260827,
            "identity_and_chemistry_group_locked": True,
            "official_holdout": False,
        },
        "endpoint_selection_used_model_results": False,
        "direct_labels_used_without_relabeling": True,
        "tdc_test_labels_loaded": False,
        "leaderboard_feedback_used": False,
    }
    Path("reports/same_lineage_admet_cohort_manifest_v1.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps({key: manifest[key] for key in ("rows", "endpoints", "cohort_sha256")}, sort_keys=True))


if __name__ == "__main__":
    main()
