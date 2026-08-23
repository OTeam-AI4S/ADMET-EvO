"""Build the v6 fresh cohort under the frozen v2 source registry."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd

from admet_agent.cohort_governance_v3 import protected_structure_firewall
from admet_agent.cohort_governance_v4 import assign_identity_locked_folds
from admet_agent.fresh_cohort_v6 import (
    endpoint_admission,
    prepare_endpoint,
    standardize_structure_column,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _self_hashed(path: Path) -> dict:
    payload = json.loads(path.read_text())
    unbound = {key: value for key, value in payload.items() if key != "sha256"}
    expected = hashlib.sha256(
        (json.dumps(unbound, indent=2, sort_keys=True) + "\n").encode()
    ).hexdigest()
    if payload.get("sha256") != expected:
        raise RuntimeError(f"Self-hash mismatch: {path}")
    return payload


def _raw_spec(dataset: str) -> tuple[Path, str, list[tuple[str, str]]]:
    root = Path("external/ecvoi_v6_tdc_cache")
    fixed = {
        "ToxCast": ("toxcast.tab", "X", None),
        "hERG_Karim": ("herg_karim.tab", "Drug", [("Y", "classification")]),
        "hERG_Central": ("herg_central.tab", "X", [
            ("hERG_at_10uM", "regression"), ("hERG_at_1uM", "regression"),
            ("hERG_inhib", "classification"),
        ]),
        "HIV": ("hiv.tab", "Drug", [("Y", "classification")]),
        "SARSCoV2_Vitro_Touret": ("sarscov2_vitro_touret.tab", "Drug", [("Y", "classification")]),
        "SARSCoV2_3CLPro_Diamond": ("sarscov2_3clpro_diamond.tab", "Drug", [("Y", "classification")]),
        "orexin1_receptor_butkiewicz": ("orexin1_receptor_butkiewicz.tab", "Drug", [("Y", "classification")]),
        "m1_muscarinic_receptor_agonists_butkiewicz": ("m1_muscarinic_receptor_agonists_butkiewicz.tab", "Drug", [("Y", "classification")]),
        "potassium_ion_channel_kir2.1_butkiewicz": ("potassium_ion_channel_kir2.1_butkiewicz.tab", "Drug", [("Y", "classification")]),
        "serine_threonine_kinase_33_butkiewicz": ("serine_threonine_kinase_33_butkiewicz.tab", "Drug", [("Y", "classification")]),
        "BACE": ("bace.csv", "mol", [("Class", "classification"), ("pIC50", "regression")]),
        "SIDER": ("sider.csv.gz", "smiles", None),
        "MUV": ("muv.csv.gz", "smiles", None),
    }
    filename, smiles, targets = fixed[dataset]
    path = root / filename
    if targets is None:
        preview = pd.read_csv(path, sep="\t" if path.suffix == ".tab" else ",", nrows=1)
        excluded = {smiles, "ID", "Drug_ID", "mol_id"}
        targets = [(column, "classification") for column in preview.columns if column not in excluded]
    return path, smiles, sorted(targets)


def _read(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, sep="\t" if path.suffix == ".tab" else ",")


def _protected_sets(v5_protocol: dict) -> tuple[list[str], list[str]]:
    firewall = v5_protocol["firewall"]
    development = []
    for raw, digest in firewall["development_exact_identity_sources"].items():
        path = Path(raw)
        if _sha256(path) != digest:
            raise RuntimeError(f"Development structure hash changed: {path}")
        development.extend(pd.read_csv(path, usecols=["Drug"])["Drug"].dropna().astype(str))
    development.extend(
        pd.read_csv("reports/ecvoi_v5_public_cohort_v4.csv", usecols=["standard_smiles"])["standard_smiles"]
        .dropna().astype(str)
    )
    protected = []
    for raw, digest in firewall["protected_tdc_test_structure_sources"].items():
        path = Path(raw)
        if _sha256(path) != digest:
            raise RuntimeError(f"Protected structure hash changed: {path}")
        protected.extend(pd.read_csv(path, usecols=["Drug"])["Drug"].dropna().astype(str))
    blind = firewall["protected_openadmet_blind_structure_source"]
    blind_path = Path(blind["path"])
    if _sha256(blind_path) != blind["sha256"] or blind["contains_labels"] is not False:
        raise RuntimeError("OpenADMET blind structure binding failed")
    protected.extend(pd.read_csv(blind_path, usecols=["SMILES"])["SMILES"].dropna().astype(str))
    return development, protected


def main() -> None:
    registry = _self_hashed(Path("reports/ecvoi_v6_fresh_source_registry_v2.json"))
    acquisition = _self_hashed(Path("reports/ecvoi_v6_fresh_source_acquisition_v2.json"))
    if acquisition["source_registry_sha256"] != registry["sha256"]:
        raise RuntimeError("Registry/acquisition binding mismatch")
    for raw, item in acquisition["raw_files"].items():
        if _sha256(Path(raw)) != item["sha256"]:
            raise RuntimeError(f"Raw cache changed: {raw}")
    v5_protocol = _self_hashed(Path("reports/ecvoi_v5_public_cohort_protocol_v4.json"))

    structure_cache: dict[str, tuple[str, str, str] | None] = {}
    candidates: list[dict] = []
    dataset_audits = {}
    endpoint_audits = {}
    skipped = []
    for spec in registry["candidates"]:
        dataset = spec["dataset"]
        if dataset in registry.get("quarantined_before_v2", []):
            skipped.append({"dataset": dataset, "reason": "non_SMILES_representation"})
            continue
        path, smiles_column, targets = _raw_spec(dataset)
        raw = _read(path)
        standardized, structure_audit = standardize_structure_column(
            raw, smiles_column=smiles_column, cache=structure_cache
        )
        dataset_audits[dataset] = {**structure_audit, "raw_sha256": _sha256(path)}
        prepared = []
        for target_column, task_type in targets:
            frame, audit = prepare_endpoint(
                standardized, target_column=target_column, task_type=task_type
            )
            endpoint_id = f"{dataset.lower()}::{target_column}"
            endpoint_audits[endpoint_id] = audit
            frame["endpoint_id"] = endpoint_id
            frame["source_id"] = spec["source_id"]
            frame["task_type"] = task_type
            frame["metric"] = "mcc" if task_type == "classification" else "mae"
            frame["metric_direction"] = "maximize" if task_type == "classification" else "minimize"
            prepared.append({
                "endpoint_id": endpoint_id, "target_column": target_column,
                "task_type": task_type, "frame": frame,
                "valid_fraction": structure_audit["valid_fraction"],
            })
        candidates.append({"spec": spec, "prepared": prepared})

    preliminary = [item["frame"] for group in candidates for item in group["prepared"]]
    identity_rows = pd.concat(preliminary, ignore_index=True)[
        ["inchikey", "standard_smiles", "chemistry_group"]
    ].drop_duplicates("inchikey")
    development, protected = _protected_sets(v5_protocol)
    allowed, firewall_audit = protected_structure_firewall(
        identity_rows,
        development_smiles=development,
        protected_test_smiles=protected,
        tanimoto_threshold=registry["firewall"]["near_neighbor_tanimoto_threshold"],
    )
    filtered_pairs = identity_rows.loc[identity_rows["inchikey"].isin(allowed), ["inchikey", "chemistry_group"]]
    assignment = assign_identity_locked_folds(
        filtered_pairs, folds=registry["split"]["outer_folds"], seed=registry["split"]["seed"]
    )

    selected = []
    admission_audits = {}
    for group in candidates:
        spec = group["spec"]
        admitted = []
        for item in sorted(group["prepared"], key=lambda value: value["target_column"]):
            frame = item["frame"].loc[item["frame"]["inchikey"].isin(allowed)].copy()
            frame["outer_fold"] = frame["inchikey"].map(assignment).astype(int)
            passed, reasons = endpoint_admission(
                frame, task_type=item["task_type"], valid_smiles_fraction=item["valid_fraction"]
            )
            admission_audits[item["endpoint_id"]] = {
                "passed": passed, "reasons": reasons, "post_firewall_identities": len(frame),
            }
            if passed:
                admitted.append(frame)
        selected.extend(admitted[: int(spec["max_endpoints"])])

    cohort = pd.concat(selected, ignore_index=True) if selected else pd.DataFrame()
    cohort = cohort[[
        "endpoint_id", "source_id", "task_type", "metric", "metric_direction",
        "inchikey", "standard_smiles", "chemistry_group", "target", "replicate_count", "outer_fold",
    ]].sort_values(["endpoint_id", "outer_fold", "inchikey"], kind="stable").reset_index(drop=True)
    output = Path("reports/ecvoi_v6_fresh_cohort_v1.csv")
    cohort.to_csv(output, index=False)
    sources = int(cohort["source_id"].nunique()) if len(cohort) else 0
    endpoints = int(cohort["endpoint_id"].nunique()) if len(cohort) else 0
    task_types = sorted(cohort["task_type"].unique()) if len(cohort) else []
    identity_overlap = int((cohort.groupby("inchikey")["outer_fold"].nunique() > 1).sum()) if len(cohort) else 0
    group_overlap = int((cohort.groupby("chemistry_group")["outer_fold"].nunique() > 1).sum()) if len(cohort) else 0
    checks = [
        {"name": "minimum_sources", "observed": sources, "requirement": 9, "passed": sources >= 9},
        {"name": "minimum_endpoints", "observed": endpoints, "requirement": 18, "passed": endpoints >= 18},
        {"name": "classification_and_regression", "observed": task_types, "requirement": ["classification", "regression"], "passed": task_types == ["classification", "regression"]},
        {"name": "identity_fold_overlap", "observed": identity_overlap, "requirement": 0, "passed": identity_overlap == 0},
        {"name": "chemistry_group_fold_overlap", "observed": group_overlap, "requirement": 0, "passed": group_overlap == 0},
        {"name": "protected_labels_loaded", "observed": False, "requirement": False, "passed": True},
    ]
    manifest = {
        "version": "ecvoi-v6-fresh-cohort-manifest-v1",
        "source_registry_sha256": registry["sha256"],
        "acquisition_sha256": acquisition["sha256"],
        "cohort_path": str(output), "cohort_sha256": _sha256(output),
        "rows": len(cohort), "endpoints": endpoints, "sources": sources,
        "selected_endpoints": sorted(cohort["endpoint_id"].unique()),
        "source_counts": cohort.groupby("source_id")["endpoint_id"].nunique().sort_index().to_dict(),
        "dataset_audits": dataset_audits, "endpoint_audits": endpoint_audits,
        "admission_audits": admission_audits, "skipped": skipped,
        "firewall_audit": firewall_audit, "checks": checks,
        "promotion_ready_for_action_freeze": all(item["passed"] for item in checks),
        "model_actions_run": False, "action_outcomes_computed": False,
        "protected_labels_loaded": False, "test_or_leaderboard_feedback_used": False,
        "tdc_19_unlocked": False,
    }
    manifest_path = Path("reports/ecvoi_v6_fresh_cohort_manifest_v1.json")
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "rows": len(cohort), "endpoints": endpoints, "sources": sources,
        "checks_passed": all(item["passed"] for item in checks),
        "cohort_sha256": manifest["cohort_sha256"],
    }, sort_keys=True))


if __name__ == "__main__":
    main()
