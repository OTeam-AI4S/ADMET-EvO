"""Run the one-seed ChEMBL 37 external-data screen across 19 TDC tasks."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

from run_scaling_parallel_adapters import run_binary, run_regression

REGRESSION = {
    "Caco2_Wang": ("caco2_wang", "mae"),
    "Lipophilicity_AstraZeneca": ("lipophilicity_astrazeneca", "mae"),
    "Solubility_AqSolDB": ("solubility_aqsoldb", "mae"),
    "PPBR_AZ": ("ppbr_az", "mae"),
    "VDss_Lombardo": ("vdss_lombardo", "spearman"),
    "Half_Life_Obach": ("half_life_obach", "spearman"),
    "Clearance_Hepatocyte_AZ": ("clearance_hepatocyte_az", "spearman"),
    "Clearance_Microsome_AZ": ("clearance_microsome_az", "spearman"),
    "LD50_Zhu": ("ld50_zhu", "mae"),
}

BINARY = {
    "HIA_Hou": "hia_hou",
    "Pgp_Broccatelli": "pgp_broccatelli",
    "Bioavailability_Ma": "bioavailability_ma",
    "CYP2C9_Veith": "cyp2c9_veith",
    "CYP2D6_Veith": "cyp2d6_veith",
    "CYP3A4_Veith": "cyp3a4_veith",
    "CYP2C9_Substrate_CarbonMangels": "cyp2c9_substrate_carbonmangels",
    "CYP2D6_Substrate_CarbonMangels": "cyp2d6_substrate_carbonmangels",
    "CYP3A4_Substrate_CarbonMangels": "cyp3a4_substrate_carbonmangels",
    "AMES": "ames",
}


def run(args: argparse.Namespace) -> dict:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "protocol": "chembl37-tdc22-one-seed-scaling-screen-v1",
        "seed": args.seed,
        "near_neighbor_rule": "max Morgan Tanimoto < 0.80",
        "test_labels_loaded": False,
        "tasks": {},
    }
    ordered = [*REGRESSION, *BINARY]
    for task in ordered:
        source = args.cohort_dir / f"{task}.tab"
        output = args.output_dir / f"{task}.json"
        try:
            if task in REGRESSION:
                benchmark, metric = REGRESSION[task]
                payload = run_regression(
                    SimpleNamespace(
                        task=task, benchmark=benchmark, metric=metric,
                        source=source, source_name=f"ChEMBL37_{task}",
                        smiles_column="X", label_column="Y", id_column="ID",
                        separator="\t", expected_source_sha256=None,
                        lineage_status="pending_document_lineage_clearance",
                        data_dir=args.data_dir, output=output,
                        minimum_external_rows=args.minimum_external_rows,
                        near_threshold=0.80, seed=(args.seed,), n_jobs=args.n_jobs,
                    )
                )
                task_summary = payload["summary"]
            else:
                benchmark = BINARY[task]
                payload = run_binary(
                    SimpleNamespace(
                        task=task, benchmark=benchmark, source=source,
                        source_name=f"ChEMBL37_{task}", smiles_column="X",
                        label_column="Y", id_column="ID", separator="\t",
                        expected_source_sha256=None,
                        lineage_status="pending_document_lineage_clearance",
                        data_dir=args.data_dir, output=output,
                        majority_cap=args.majority_cap, near_threshold=0.80,
                        seed=(args.seed,), n_jobs=args.n_jobs,
                    )
                )
                task_summary = payload["summary"]
            summary["tasks"][task] = {"status": payload["status"], "summary": task_summary, "output": str(output)}
        except Exception as exc:
            failure = {"task": task, "status": "failed_closed", "error_type": type(exc).__name__, "error": str(exc)}
            output.write_text(json.dumps(failure, indent=2, sort_keys=True) + "\n")
            summary["tasks"][task] = failure
        args.summary.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
        print(json.dumps({"task": task, **summary["tasks"][task]}, default=str), flush=True)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cohort-dir", type=Path, default=Path("data/external/chembl/chembl_37/tdc22_cohorts"))
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--output-dir", type=Path, default=Path("reports/chembl37_tdc22_scaling_screen_v1"))
    parser.add_argument("--summary", type=Path, default=Path("reports/chembl37_tdc22_scaling_screen_v1.json"))
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--majority-cap", type=int, default=15_000)
    parser.add_argument("--minimum-external-rows", type=int, default=30)
    parser.add_argument("--n-jobs", type=int, default=8)
    return parser.parse_args()


if __name__ == "__main__":
    result = run(parse_args())
    print(json.dumps({"tasks": len(result["tasks"])}, indent=2))
