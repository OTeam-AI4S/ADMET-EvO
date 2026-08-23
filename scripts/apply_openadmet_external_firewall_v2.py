"""Apply the preregistered OpenADMET blind-structure firewall to one source.

The sealed blind table is verified against an explicit hash/schema manifest.
Only record-level exact-identity and fixed-threshold similarity exclusions are
made.  Blind-set coverage never accepts, rejects, ranks or reweights a source.
"""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
from pathlib import Path

import pandas as pd
from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator

from admet_agent.external_structure_firewall import _standard_molecule


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _standardizer_hash() -> str:
    source = inspect.getsource(_standard_molecule).encode("utf-8")
    return hashlib.sha256(source).hexdigest()


def _load_sealed_blind_structures(
    manifest_path: Path,
) -> tuple[tuple[str, ...], dict, str]:
    manifest_path = manifest_path.resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    blind = manifest["blind_table"]
    if blind.get("contains_label_columns") is not False:
        raise RuntimeError("Blind manifest must explicitly declare no label columns")
    if manifest.get("test_labels_loaded") is not False:
        raise RuntimeError("Blind manifest test_labels_loaded flag is not false")
    threshold = manifest["record_exclusions"].get(
        "tanimoto_greater_than_or_equal"
    )
    if threshold != 0.9:
        raise RuntimeError("The preregistered OpenADMET Tanimoto threshold is 0.90")

    repository = manifest_path.parent.parent
    table_path = (repository / blind["path"]).resolve()
    if _sha256(table_path) != blind["sha256"]:
        raise RuntimeError("Blind table hash mismatch")
    frame = pd.read_csv(table_path, dtype="string")
    expected_columns = list(blind["ordered_columns"])
    if list(frame.columns) != expected_columns:
        raise RuntimeError("Blind table ordered schema mismatch")
    if len(frame) != int(blind["expected_rows"]):
        raise RuntimeError("Blind table row-count mismatch")
    for required in (blind["molecule_name_column"], blind["smiles_column"]):
        if required not in frame:
            raise RuntimeError(f"Blind table missing required column {required!r}")

    expected_implementation_hash = manifest["standardization"].get(
        "implementation_hash"
    )
    if expected_implementation_hash in {None, "", "pending_v2_source_freeze"}:
        raise RuntimeError("Standardizer implementation hash has not been frozen")
    if expected_implementation_hash != _standardizer_hash():
        raise RuntimeError("Standardizer implementation hash mismatch")
    structures = tuple(str(value) for value in frame[blind["smiles_column"]].dropna())
    return structures, manifest, _sha256(manifest_path)


def apply_firewall(
    *,
    source_path: Path,
    expected_source_sha256: str,
    smiles_column: str,
    output_path: Path,
    manifest_path: Path,
) -> dict:
    source_path = source_path.resolve()
    source_sha256 = _sha256(source_path)
    if source_sha256 != expected_source_sha256:
        raise RuntimeError("External source hash mismatch")
    blind_smiles, manifest, manifest_sha256 = _load_sealed_blind_structures(
        manifest_path
    )
    blind = {
        molecule.inchikey: molecule
        for value in blind_smiles
        if (molecule := _standard_molecule(value)) is not None
    }
    if not blind:
        raise RuntimeError("No valid structures in the sealed blind table")

    external = pd.read_csv(source_path)
    if smiles_column not in external:
        raise ValueError(f"Missing external SMILES column {smiles_column!r}")
    working = external.copy().reset_index(drop=True)
    working["_standard_molecule"] = working[smiles_column].map(
        lambda value: _standard_molecule(str(value))
    )
    invalid_mask = working["_standard_molecule"].isna()
    valid = working.loc[~invalid_mask].copy()
    exact_mask = valid["_standard_molecule"].map(
        lambda molecule: molecule.inchikey in blind
    )
    eligible = valid.loc[~exact_mask].copy()

    generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    blind_fingerprints = [
        generator.GetFingerprint(Chem.MolFromSmiles(molecule.smiles))
        for molecule in blind.values()
    ]
    near_mask: list[bool] = []
    threshold = float(
        manifest["record_exclusions"]["tanimoto_greater_than_or_equal"]
    )
    for molecule in eligible["_standard_molecule"]:
        fingerprint = generator.GetFingerprint(Chem.MolFromSmiles(molecule.smiles))
        similarities = DataStructs.BulkTanimotoSimilarity(
            fingerprint, blind_fingerprints
        )
        near_mask.append(max(similarities) >= threshold)

    kept = eligible.loc[[not value for value in near_mask]].copy()
    kept[smiles_column] = kept["_standard_molecule"].map(
        lambda molecule: molecule.smiles
    )
    kept["standard_inchikey"] = kept["_standard_molecule"].map(
        lambda molecule: molecule.inchikey
    )
    kept = kept.drop(columns=["_standard_molecule"]).reset_index(drop=True)

    output_path = output_path.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    kept.to_csv(output_path, index=False)
    audit = {
        "version": "openadmet-external-record-firewall-v2",
        "source_path": str(source_path),
        "source_sha256": source_sha256,
        "blind_manifest_path": str(manifest_path.resolve()),
        "blind_manifest_sha256": manifest_sha256,
        "standardizer_implementation_sha256": _standardizer_hash(),
        "fingerprint": "Morgan_radius_2_2048_bits",
        "tanimoto_greater_than_or_equal": threshold,
        "external_input_rows": len(working),
        "invalid_external_rows": int(invalid_mask.sum()),
        "exact_blind_identity_exclusion_rows": int(exact_mask.sum()),
        "near_blind_neighbor_exclusion_rows": int(sum(near_mask)),
        "filtered_external_rows": len(kept),
        "filtered_external_path": str(output_path),
        "filtered_external_table_sha256": _sha256(output_path),
        "dataset_rejected_from_blind_distribution": False,
        "blind_distribution_used_for_source_or_model_selection": False,
        "test_labels_loaded": False,
        "test_structures_exposed_to_planner": False,
    }
    audit_path = output_path.with_suffix(".firewall.json")
    audit_path.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n")
    return {**audit, "audit_path": str(audit_path)}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--expected-source-sha256", required=True)
    parser.add_argument("--smiles-column", default="Drug")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--blind-manifest",
        type=Path,
        default=Path("reports/openadmet_blind_structure_firewall_manifest_v1.json"),
    )
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    print(
        json.dumps(
            apply_firewall(
                source_path=arguments.source,
                expected_source_sha256=arguments.expected_source_sha256,
                smiles_column=arguments.smiles_column,
                output_path=arguments.output,
                manifest_path=arguments.blind_manifest,
            ),
            indent=2,
            sort_keys=True,
        )
    )
