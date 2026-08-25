"""Repair and rerun the two non-roundtripping regression screen tasks."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from run_scaling_resilient_adapters import run_regression

TASKS = {
    "Solubility_AqSolDB": ("solubility_aqsoldb", "mae"),
    "VDss_Lombardo": ("vdss_lombardo", "spearman"),
}


def main() -> None:
    cohort_dir = Path("data/external/chembl/chembl_37/tdc22_cohorts")
    output_dir = Path("reports/chembl37_tdc22_scaling_screen_v1")
    summary_path = Path("reports/chembl37_tdc22_scaling_screen_v1.json")
    summary = json.loads(summary_path.read_text())
    for task, (benchmark, metric) in TASKS.items():
        output = output_dir / f"{task}.json"
        payload = run_regression(
            SimpleNamespace(
                task=task,
                benchmark=benchmark,
                metric=metric,
                source=cohort_dir / f"{task}.tab",
                source_name=f"ChEMBL37_{task}",
                smiles_column="X",
                label_column="Y",
                id_column="ID",
                separator="\t",
                expected_source_sha256=None,
                lineage_status="pending_document_lineage_clearance",
                data_dir=Path("data"),
                output=output,
                minimum_external_rows=30,
                near_threshold=0.80,
                seed=(1,),
                n_jobs=8,
            )
        )
        summary["tasks"][task] = {
            "status": payload["status"],
            "summary": payload["summary"],
            "output": str(output),
            "tdc_feature_fallback_count": payload["tdc_feature_fallback_count"],
        }
        summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
        print(json.dumps({"task": task, **summary["tasks"][task]}), flush=True)


if __name__ == "__main__":
    main()
