"""Build a fixed 50-endpoint discovery cohort without using model outcomes.

The cohort contains the frozen 17-endpoint v5 cohort plus 20 ToxCast and
13 SIDER endpoints.  New endpoints are selected only by label availability,
identity count, minority-class support, and five-fold class coverage.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import pandas as pd

from admet_agent.external_structure_firewall import _standard_molecule
from admet_agent.fresh_cohort_v6 import endpoint_admission, prepare_endpoint
from scripts.build_openadmet_cyp_split_manifest import _chemistry_group


VERSION = "admet50-discovery-cohort-v1"
FOLD_SALT = "admet50-shared-chemistry-group-fold-v1"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _standardize_one(raw: str) -> tuple[str, tuple[str, str, str] | None]:
    from rdkit import RDLogger

    RDLogger.DisableLog("rdApp.*")
    molecule = _standard_molecule(raw)
    if molecule is None:
        return raw, None
    try:
        group = _chemistry_group(molecule.smiles)
    except Exception:
        return raw, None
    return raw, (molecule.smiles, molecule.inchikey, group)


def _parallel_standardize(
    frames: list[tuple[pd.DataFrame, str]], workers: int
) -> dict[str, tuple[str, str, str] | None]:
    values = sorted(
        {
            raw
            for frame, column in frames
            for raw in frame[column].dropna().astype(str).tolist()
        }
    )
    with ProcessPoolExecutor(max_workers=workers) as pool:
        pairs = pool.map(_standardize_one, values, chunksize=16)
        return dict(pairs)


def _map_structures(
    raw: pd.DataFrame,
    smiles_column: str,
    cache: dict[str, tuple[str, str, str] | None],
) -> tuple[pd.DataFrame, dict]:
    strings = raw[smiles_column].astype(str)
    mapped = strings.map(cache)
    valid = mapped.notna()
    frame = raw.loc[valid].copy()
    values = mapped.loc[valid]
    frame["standard_smiles"] = values.map(lambda item: item[0])
    frame["inchikey"] = values.map(lambda item: item[1])
    frame["chemistry_group"] = values.map(lambda item: item[2])
    return frame, {
        "input_rows": len(raw),
        "valid_rows": int(valid.sum()),
        "invalid_rows": int((~valid).sum()),
        "valid_fraction": float(valid.mean()) if len(valid) else 0.0,
    }


def _fold_assignment(base: pd.DataFrame, groups: set[str]) -> dict[str, int]:
    observed = base.groupby("chemistry_group")["outer_fold"].nunique()
    if int((observed > 1).sum()):
        raise RuntimeError("Frozen v5 cohort has chemistry-group fold leakage")
    assignment = (
        base[["chemistry_group", "outer_fold"]]
        .drop_duplicates("chemistry_group")
        .set_index("chemistry_group")["outer_fold"]
        .astype(int)
        .to_dict()
    )
    for group in sorted(groups - set(assignment)):
        digest = hashlib.sha256(f"{FOLD_SALT}|{group}".encode()).hexdigest()
        assignment[group] = int(digest[:16], 16) % 5
    return assignment


def _eligible_endpoints(
    standardized: pd.DataFrame,
    *,
    source_id: str,
    endpoint_prefix: str,
    smiles_column: str,
    base_identities: set[str],
    fold_map: dict[str, int],
    valid_fraction: float,
) -> tuple[list[dict], dict]:
    excluded = {smiles_column, "ID", "Drug_ID", "mol_id"}
    candidates: list[dict] = []
    audits = {}
    for target_column in sorted(set(standardized.columns) - excluded - {
        "standard_smiles", "inchikey", "chemistry_group"
    }):
        numeric = pd.to_numeric(standardized[target_column], errors="coerce")
        if set(numeric.dropna().unique()) - {0.0, 1.0}:
            continue
        work = standardized.copy()
        work[target_column] = numeric
        prepared, collapse_audit = prepare_endpoint(
            work, target_column=target_column, task_type="classification"
        )
        overlap = int(prepared["inchikey"].isin(base_identities).sum())
        prepared = prepared.loc[~prepared["inchikey"].isin(base_identities)].copy()
        prepared["outer_fold"] = prepared["chemistry_group"].map(fold_map).astype(int)
        passed, reasons = endpoint_admission(
            prepared, task_type="classification", valid_smiles_fraction=valid_fraction
        )
        counts = prepared["target"].value_counts()
        fold_table = prepared.groupby(["outer_fold", "target"]).size().unstack(fill_value=0)
        minimum_fold_class = int(fold_table.min().min()) if fold_table.shape == (5, 2) else 0
        endpoint_id = f"{endpoint_prefix}::{target_column}"
        audits[endpoint_id] = {
            **collapse_audit,
            "base_v5_identity_overlap_removed": overlap,
            "post_overlap_identities": len(prepared),
            "minority_count": int(counts.min()) if len(counts) == 2 else 0,
            "minimum_fold_class_count": minimum_fold_class,
            "passed": passed,
            "reasons": reasons,
        }
        if not passed:
            continue
        prepared["endpoint_id"] = endpoint_id
        prepared["source_id"] = source_id
        prepared["task_type"] = "classification"
        prepared["metric"] = "mcc"
        prepared["metric_direction"] = "maximize"
        candidates.append({
            "endpoint_id": endpoint_id,
            "frame": prepared,
            "minority_count": int(counts.min()),
            "minimum_fold_class_count": minimum_fold_class,
            "identities": len(prepared),
        })
    return candidates, audits


def _select(candidates: list[dict], count: int) -> list[dict]:
    # Performance-blind deterministic priority: worst-fold class support first,
    # then total minority support and total identities, then endpoint name.
    return sorted(
        candidates,
        key=lambda item: (
            -item["minimum_fold_class_count"],
            -item["minority_count"],
            -item["identities"],
            item["endpoint_id"],
        ),
    )[:count]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=32)
    parser.add_argument(
        "--output", type=Path, default=Path("reports/admet50_discovery_cohort_v1.csv")
    )
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--toxcast-count", type=int, default=20)
    parser.add_argument("--sider-count", type=int, default=13)
    parser.add_argument("--all-eligible", action="store_true")
    parser.add_argument("--version", default=VERSION)
    args = parser.parse_args()

    base_path = Path("reports/ecvoi_v5_public_cohort_v4.csv")
    toxcast_path = Path("external/ecvoi_v6_tdc_cache/toxcast.tab")
    sider_path = Path("external/ecvoi_v6_tdc_cache/sider.csv.gz")
    base = pd.read_csv(base_path)
    toxcast_raw = pd.read_csv(toxcast_path, sep="\t")
    sider_raw = pd.read_csv(sider_path)
    cache = _parallel_standardize(
        [(toxcast_raw, "X"), (sider_raw, "smiles")], args.workers
    )
    toxcast, toxcast_structure_audit = _map_structures(toxcast_raw, "X", cache)
    sider, sider_structure_audit = _map_structures(sider_raw, "smiles", cache)
    new_groups = set(toxcast["chemistry_group"]) | set(sider["chemistry_group"])
    fold_map = _fold_assignment(base, new_groups)
    base_identities = set(base["inchikey"].astype(str))

    toxcast_candidates, toxcast_audits = _eligible_endpoints(
        toxcast,
        source_id="ToxCast",
        endpoint_prefix="toxcast",
        smiles_column="X",
        base_identities=base_identities,
        fold_map=fold_map,
        valid_fraction=toxcast_structure_audit["valid_fraction"],
    )
    sider_candidates, sider_audits = _eligible_endpoints(
        sider,
        source_id="SIDER",
        endpoint_prefix="sider",
        smiles_column="smiles",
        base_identities=base_identities,
        fold_map=fold_map,
        valid_fraction=sider_structure_audit["valid_fraction"],
    )
    toxcast_count = len(toxcast_candidates) if args.all_eligible else args.toxcast_count
    sider_count = len(sider_candidates) if args.all_eligible else args.sider_count
    selected = _select(toxcast_candidates, toxcast_count) + _select(sider_candidates, sider_count)
    expected_new = toxcast_count + sider_count
    if len(selected) != expected_new:
        raise RuntimeError(f"Expected {expected_new} new endpoints, got {len(selected)}")
    columns = [
        "endpoint_id", "source_id", "task_type", "metric", "metric_direction",
        "inchikey", "standard_smiles", "chemistry_group", "target", "replicate_count",
        "outer_fold",
    ]
    cohort = pd.concat([base] + [item["frame"][columns] for item in selected], ignore_index=True)
    cohort = cohort[columns].sort_values(
        ["endpoint_id", "outer_fold", "inchikey"], kind="stable"
    ).reset_index(drop=True)
    expected_total = int(base["endpoint_id"].nunique()) + expected_new
    if cohort["endpoint_id"].nunique() != expected_total:
        raise RuntimeError(f"Cohort does not contain exactly {expected_total} endpoints")
    if int((cohort.groupby("inchikey")["outer_fold"].nunique() > 1).sum()):
        raise RuntimeError("Identity leakage across outer folds")
    if int((cohort.groupby("chemistry_group")["outer_fold"].nunique() > 1).sum()):
        raise RuntimeError("Chemistry-group leakage across outer folds")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    cohort.to_csv(args.output, index=False)
    selected_ids = [item["endpoint_id"] for item in selected]
    manifest = {
        "version": args.version,
        "cohort_path": str(args.output.resolve()),
        "cohort_sha256": _sha256(args.output),
        "rows": len(cohort),
        "endpoint_count": int(cohort["endpoint_id"].nunique()),
        "frozen_v5_endpoint_count": int(base["endpoint_id"].nunique()),
        "new_endpoint_count": len(selected_ids),
        "selected_new_endpoints": selected_ids,
        "selection_rule": (
            "Within each source, rank endpoints by minimum five-fold class support, "
            "total minority support, identity count, then endpoint_id; select "
            f"ToxCast {toxcast_count} and SIDER {sider_count}. No model outcomes are used."
        ),
        "confirmation_selection_rule": (
            "Top20 is selected later using seed-17 development outcomes only; seeds 29 "
            "and 43 remain untouched confirmation data."
        ),
        "standardization": {
            "unique_raw_structures": len(cache),
            "toxcast": toxcast_structure_audit,
            "sider": sider_structure_audit,
        },
        "source_files": {
            str(base_path): _sha256(base_path),
            str(toxcast_path): _sha256(toxcast_path),
            str(sider_path): _sha256(sider_path),
        },
        "eligible_counts": {
            "ToxCast": len(toxcast_candidates), "SIDER": len(sider_candidates)
        },
        "endpoint_audits": {**toxcast_audits, **sider_audits},
        "checks": {
            "expected_total_endpoints": expected_total,
            "expected_new_endpoints": expected_new,
            "identity_fold_overlap": 0,
            "chemistry_group_fold_overlap": 0,
            "model_outcomes_used_for_cohort_selection": False,
            "confirmation_seeds_used": False,
        },
    }
    manifest_path = args.manifest or args.output.with_suffix(".manifest.json")
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "rows": len(cohort), "endpoints": expected_total, "new_endpoints": expected_new,
        "eligible": manifest["eligible_counts"], "cohort_sha256": manifest["cohort_sha256"],
    }, sort_keys=True))


if __name__ == "__main__":
    main()
