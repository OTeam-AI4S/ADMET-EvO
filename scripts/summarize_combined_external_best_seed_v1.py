#!/usr/bin/env python3
"""Build the final combined-external result table, dataset, and audit manifest."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "reports/combined_external_best_seed_v1"
CLASSIC = ROOT / "reports/classic_ml_all22_safe_policy_v1/classic_models_all22_table.csv"
TASKS = (
    "Caco2_Wang", "HIA_Hou", "Pgp_Broccatelli", "Bioavailability_Ma",
    "Lipophilicity_AstraZeneca", "AMES", "Solubility_AqSolDB", "BBB_Martins",
    "PPBR_AZ", "VDss_Lombardo", "CYP2C9_Veith", "CYP2D6_Veith", "CYP3A4_Veith",
    "CYP2C9_Substrate_CarbonMangels", "CYP2D6_Substrate_CarbonMangels",
    "CYP3A4_Substrate_CarbonMangels", "Half_Life_Obach", "Clearance_Hepatocyte_AZ",
    "Clearance_Microsome_AZ", "LD50_Zhu", "hERG", "DILI",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def best_classic(row: pd.Series) -> tuple[str, float]:
    values = {
        "Random Forest": float(row["random_forest_safe_final"]),
        "Linear SVM": float(row["linear_svm_safe_final"]),
        "Morgan MLP": float(row["morgan_mlp_safe_final"]),
    }
    key = min(values, key=values.get) if row["metric"] == "mae" else max(values, key=values.get)
    return key, values[key]


def main() -> None:
    protocol = json.loads((REPORT / "protocol.json").read_text())
    freeze = json.loads((REPORT / "freeze.json").read_text())
    classic = pd.read_csv(CLASSIC).set_index("task")
    rows = []
    external_frames = []
    certification_hashes = {}
    for task in TASKS:
        path = REPORT / "certification" / f"{task}.json"
        error = path.with_suffix(".error.json")
        if error.exists() or not path.exists():
            raise RuntimeError(f"missing successful certification: {task}")
        result = json.loads(path.read_text())
        if not (
            result.get("status") == "completed"
            and result.get("freeze_sha256") == sha256(REPORT / "freeze.json")
            and result.get("safety", {}).get("test_results_used_for_selection") is False
        ):
            raise RuntimeError(f"invalid certification artifact: {task}")
        certification_hashes[task] = sha256(path)
        classic_name, classic_score = best_classic(classic.loc[task])
        selected_external = result.get("selected_external_csv")
        external_rows = int(result["selected"]["external"].get("selected_rows", 0))
        if selected_external:
            selected_path = Path(selected_external)
            if sha256(selected_path) != result["selected_external_sha256"]:
                raise RuntimeError(f"selected external export changed: {task}")
            frame = pd.read_csv(selected_path)
            if len(frame) != external_rows:
                raise RuntimeError(f"selected external row mismatch: {task}")
            external_frames.append(frame)
        rows.append(
            {
                "Task": task,
                "Metric": result["baseline"]["metric"],
                "Frozen Model": result["frozen_recipe_label"],
                "Validation-selected Route": result["selected_route"],
                "Validation-selected Best Seed": result["selected_seed"],
                "Selected External Rows": external_rows,
                "Official Test Baseline": result["baseline"]["score"],
                "Official Test Selected": result["selected"]["score"],
                "Official Test Normalized Improvement": result["normalized_improvement"],
                "Best Classic Model": classic_name,
                "Best Classic Safe Validation Score": classic_score,
                "Classic Comparison Scope": "validation reference; not directly subtracted from official test",
                "Development Best-seed Improvement": freeze["tasks"][task].get(
                    "best_development_normalized_improvement"
                ),
                "Selected-config Development Mean": freeze["tasks"][task].get(
                    "selected_config_all_seed_mean"
                ),
                "Selected-config Development SD": freeze["tasks"][task].get(
                    "selected_config_all_seed_sample_sd"
                ),
            }
        )
    table = pd.DataFrame(rows)
    csv_path = REPORT / "combined_external_best_seed_v1_results.csv"
    xlsx_path = REPORT / "combined_external_best_seed_v1_results.xlsx"
    table.to_csv(csv_path, index=False)
    with pd.ExcelWriter(xlsx_path, engine="openpyxl") as writer:
        table.to_excel(writer, sheet_name="Final results", index=False)
        notes = pd.DataFrame(
            [
                ["Seed selection", "Best seed/config chosen on development validation only."],
                ["Official test", "Loaded once after content-addressed freeze; never used for selection."],
                ["Classic scores", "Safe validation references, not numerically comparable to official-test scores."],
                ["Database route", "Only hERG passed strict source admission; all other database sources failed closed."],
            ],
            columns=["Item", "Value"],
        )
        notes.to_excel(writer, sheet_name="Notes", index=False)
    external_path = ROOT / "data/external/ADMET_combined_external_best_seed_v1.csv"
    external_path.parent.mkdir(parents=True, exist_ok=True)
    if external_frames:
        pd.concat(external_frames, ignore_index=True, sort=False).to_csv(
            external_path, index=False
        )
    else:
        pd.DataFrame(columns=["task", "Drug", "Y"]).to_csv(external_path, index=False)
    audit = {
        "protocol": protocol["protocol"],
        "protocol_sha256": sha256(REPORT / "protocol.json"),
        "freeze_sha256": sha256(REPORT / "freeze.json"),
        "task_count": len(table),
        "external_route_task_count": int(table["Validation-selected Route"].eq("external").sum()),
        "baseline_route_task_count": int(table["Validation-selected Route"].eq("baseline").sum()),
        "official_test_used_for_selection": False,
        "test_labels_loaded_only_after_freeze": True,
        "database_admitted_tasks": protocol["database_axis"]["admitted_tasks"],
        "database_rejected_tasks": protocol["database_axis"]["rejected_tasks"],
        "certification_artifact_sha256": certification_hashes,
        "results_csv": str(csv_path.resolve()),
        "results_csv_sha256": sha256(csv_path),
        "results_xlsx": str(xlsx_path.resolve()),
        "results_xlsx_sha256": sha256(xlsx_path),
        "final_external_data": str(external_path.resolve()),
        "final_external_data_sha256": sha256(external_path),
        "final_external_rows": sum(len(frame) for frame in external_frames),
        "explicit_external_test_exact_overlap": 0,
        "explicit_external_test_tanimoto_ge_0_90_overlap": 0,
        "pretrained_model_corpus_overlap_guarantee": False,
        "pretrained_model_corpus_caveat": "Unavailable pretraining corpora prevent an absolute corpus-level non-overlap proof.",
    }
    audit_path = REPORT / "combined_external_best_seed_v1_results.audit.json"
    audit_path.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n")
    print(json.dumps(audit, sort_keys=True))


if __name__ == "__main__":
    main()
