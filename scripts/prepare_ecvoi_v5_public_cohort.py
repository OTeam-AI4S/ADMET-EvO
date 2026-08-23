"""Prepare the protocol-bound, decontaminated EC-VOI v5 public cohort."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd

from admet_agent.cohort_governance import (
    assign_identity_locked_folds,
    collapse_endpoint_records,
    protected_structure_firewall,
    standardize_records,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _protocol(path: Path) -> dict:
    payload = json.loads(path.read_text())
    unbound = {key:value for key,value in payload.items() if key != "sha256"}
    expected = hashlib.sha256((json.dumps(unbound,indent=2,sort_keys=True)+"\n").encode()).hexdigest()
    if payload.get("sha256") != expected:
        raise RuntimeError("Cohort protocol self-hash mismatch")
    for source, digest in payload["implementation_sha256"].items():
        if _sha256(Path(source)) != digest:
            raise RuntimeError(f"Frozen cohort implementation changed: {source}")
    return payload


def _read_smiles(paths: dict[str, str], column: str) -> list[str]:
    values = []
    for raw_path, expected in paths.items():
        path = Path(raw_path)
        if _sha256(path) != expected:
            raise RuntimeError(f"Protected structure file changed: {path}")
        values.extend(pd.read_csv(path, usecols=[column])[column].dropna().astype(str))
    return values


def run(args: argparse.Namespace) -> dict:
    protocol_path = args.protocol.resolve()
    protocol = _protocol(protocol_path)
    raw_root = Path(protocol["raw_root"])
    for name, expected in protocol["raw_files_sha256"].items():
        if _sha256(raw_root / name) != expected:
            raise RuntimeError(f"Raw cohort file changed: {name}")
    endpoint_frames = []
    endpoint_audits = {}
    invalid_by_source = {}
    standardized_cache = {}
    for endpoint in protocol["endpoints"]:
        raw_file = endpoint["raw_file"]
        if raw_file not in standardized_cache:
            raw = pd.read_csv(raw_root / raw_file, sep="\t")
            standardized, invalid = standardize_records(
                raw, smiles_column=endpoint["smiles_column"]
            )
            standardized_cache[raw_file] = standardized
            invalid_by_source[raw_file] = invalid
        selected = standardized_cache[raw_file].copy()
        selected["target"] = selected[endpoint["target_column"]]
        collapsed, audit = collapse_endpoint_records(
            selected, task_type=endpoint["task_type"]
        )
        collapsed["endpoint_id"] = endpoint["endpoint_id"]
        collapsed["source_id"] = endpoint["source_id"]
        collapsed["task_type"] = endpoint["task_type"]
        collapsed["metric"] = endpoint["metric"]
        collapsed["metric_direction"] = endpoint["metric_direction"]
        endpoint_frames.append(
            collapsed[["endpoint_id","source_id","task_type","metric","metric_direction","inchikey","standard_smiles","chemistry_group","target","replicate_count"]]
        )
        endpoint_audits[endpoint["endpoint_id"]] = audit
    longform = pd.concat(endpoint_frames, ignore_index=True)
    identities = longform[["inchikey","standard_smiles","chemistry_group"]].drop_duplicates("inchikey")
    firewall = protocol["firewall"]
    development_smiles = _read_smiles(
        firewall["development_exact_identity_sources"], "Drug"
    )
    protected_smiles = _read_smiles(
        firewall["protected_tdc_test_structure_sources"], "Drug"
    )
    blind = firewall["protected_openadmet_blind_structure_source"]
    blind_path = Path(blind["path"])
    if _sha256(blind_path) != blind["sha256"] or blind["contains_labels"] is not False:
        raise RuntimeError("OpenADMET protected structure binding failed")
    protected_smiles.extend(
        pd.read_csv(blind_path, usecols=["SMILES"])["SMILES"].dropna().astype(str)
    )
    allowed, firewall_audit = protected_structure_firewall(
        identities,
        development_smiles=development_smiles,
        protected_test_smiles=protected_smiles,
        tanimoto_threshold=firewall["near_neighbor_tanimoto_threshold"],
    )
    longform = longform.loc[longform["inchikey"].isin(allowed)].copy()
    identity_rows = longform[["inchikey","chemistry_group"]].drop_duplicates()
    assignment = assign_identity_locked_folds(
        identity_rows,
        folds=protocol["split"]["outer_folds"],
        seed=protocol["split"]["seed"],
    )
    longform["outer_fold"] = longform["inchikey"].map(assignment).astype(int)
    longform = longform.sort_values(
        ["endpoint_id","outer_fold","inchikey"], kind="stable"
    ).reset_index(drop=True)
    endpoint_summary = {}
    for endpoint, frame in longform.groupby("endpoint_id", sort=True):
        summary = {
            "rows": len(frame),
            "identities": int(frame["inchikey"].nunique()),
            "chemistry_groups": int(frame["chemistry_group"].nunique()),
            "fold_rows": {str(int(k)):int(v) for k,v in frame.groupby("outer_fold").size().items()},
            "task_type": str(frame["task_type"].iloc[0]),
            "source_id": str(frame["source_id"].iloc[0]),
        }
        if summary["task_type"] == "classification":
            summary["positive"] = int(frame["target"].sum())
            summary["negative"] = int(len(frame)-frame["target"].sum())
            summary["fold_positive"] = {str(int(k)):int(v) for k,v in frame.groupby("outer_fold")["target"].sum().items()}
        endpoint_summary[endpoint] = summary
    group_fold_counts = longform.groupby("chemistry_group")["outer_fold"].nunique()
    identity_fold_counts = longform.groupby("inchikey")["outer_fold"].nunique()
    scientific_endpoints = [value for value in endpoint_summary.values() if value["rows"] >= 100]
    source_families = {value["source_id"] for value in scientific_endpoints}
    checks = [
        {"name":"minimum_endpoint_source_units","observed":len(scientific_endpoints),"requirement":">=8","passed":len(scientific_endpoints)>=8},
        {"name":"classification_and_regression","observed":sorted({v["task_type"] for v in scientific_endpoints}),"requirement":["classification","regression"],"passed":{v["task_type"] for v in scientific_endpoints}=={"classification","regression"}},
        {"name":"minimum_source_families","observed":len(source_families),"requirement":">=4","passed":len(source_families)>=4},
        {"name":"identity_fold_overlap","observed":int((identity_fold_counts>1).sum()),"requirement":0,"passed":not (identity_fold_counts>1).any()},
        {"name":"chemistry_group_fold_overlap","observed":int((group_fold_counts>1).sum()),"requirement":0,"passed":not (group_fold_counts>1).any()},
        {"name":"all_endpoints_have_five_folds","observed":sum(len(v["fold_rows"])==5 for v in scientific_endpoints),"requirement":len(scientific_endpoints),"passed":all(len(v["fold_rows"])==5 for v in scientific_endpoints)},
        {"name":"test_labels_loaded","observed":False,"requirement":False,"passed":True},
    ]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    longform.to_csv(args.output, index=False)
    manifest = {
        "version":"ecvoi-v5-public-cohort-manifest-v1",
        "protocol_path":str(protocol_path),
        "protocol_sha256":protocol["sha256"],
        "output_path":str(args.output.resolve()),
        "output_sha256":_sha256(args.output),
        "rows":len(longform),
        "endpoint_summary":endpoint_summary,
        "endpoint_audits":endpoint_audits,
        "invalid_smiles_by_raw_file":invalid_by_source,
        "firewall_audit":firewall_audit,
        "quarantined_raw_files":protocol["quarantined_raw_files"],
        "source_licenses":protocol["sources"],
        "checks":checks,
        "promotion_passed":all(item["passed"] for item in checks),
        "release_state":"modeling_ready_development" if all(item["passed"] for item in checks) else "quarantine",
        "test_structures_used_for_record_removal":True,
        "test_labels_loaded":False,
        "blind_test_opened":False,
        "leaderboard_feedback_used":False,
    }
    args.manifest.write_text(json.dumps(manifest,indent=2,sort_keys=True)+"\n")
    return manifest


def parse_args() -> argparse.Namespace:
    parser=argparse.ArgumentParser()
    parser.add_argument("--protocol",type=Path,default=Path("reports/ecvoi_v5_public_cohort_protocol_v1.json"))
    parser.add_argument("--output",type=Path,default=Path("reports/ecvoi_v5_public_cohort_v1.csv"))
    parser.add_argument("--manifest",type=Path,default=Path("reports/ecvoi_v5_public_cohort_manifest_v1.json"))
    return parser.parse_args()


if __name__=="__main__":
    print(json.dumps(run(parse_args()),indent=2,sort_keys=True))
