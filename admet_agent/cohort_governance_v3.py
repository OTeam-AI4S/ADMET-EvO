"""Deduplicated, fail-closed fingerprint compatibility adapter for cohort firewall."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import pandas as pd
from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator

from admet_agent.external_structure_firewall import _standard_molecule


def _standard_unique_raw(values: Sequence[str]):
    output = {}
    for raw in sorted(set(map(str, values))):
        molecule = _standard_molecule(raw)
        if molecule is not None:
            output[molecule.inchikey] = molecule
    return output


def protected_structure_firewall(
    identities: pd.DataFrame,
    *,
    development_smiles: Sequence[str],
    protected_test_smiles: Sequence[str],
    tanimoto_threshold: float = 0.80,
) -> tuple[set[str], dict[str, Any]]:
    if not 0 <= tanimoto_threshold <= 1:
        raise ValueError("Tanimoto threshold must be in [0, 1]")
    required = {"inchikey", "standard_smiles"}
    if missing := required - set(identities):
        raise ValueError(f"Missing identity columns: {sorted(missing)}")
    development = _standard_unique_raw(development_smiles)
    protected = _standard_unique_raw(protected_test_smiles)
    if not protected:
        raise ValueError("Protected test structure set is empty")
    generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    protected_fingerprints = []
    protected_fingerprint_failures = 0
    for value in protected.values():
        molecule = Chem.MolFromSmiles(value.smiles)
        if molecule is None:
            protected_fingerprint_failures += 1
            continue
        protected_fingerprints.append(generator.GetFingerprint(molecule))
    if not protected_fingerprints:
        raise RuntimeError("No protected structure could produce a fingerprint")
    allowed: set[str] = set()
    exact_development = exact_test = near_test = 0
    candidate_fingerprint_failures = 0
    for row in identities.drop_duplicates("inchikey").itertuples(index=False):
        identity = str(row.inchikey)
        if identity in development:
            exact_development += 1
            continue
        if identity in protected:
            exact_test += 1
            continue
        molecule = Chem.MolFromSmiles(str(row.standard_smiles))
        if molecule is None:
            candidate_fingerprint_failures += 1
            continue
        fingerprint = generator.GetFingerprint(molecule)
        maximum = max(
            DataStructs.BulkTanimotoSimilarity(fingerprint, protected_fingerprints)
        )
        if maximum >= tanimoto_threshold:
            near_test += 1
            continue
        allowed.add(identity)
    return allowed, {
        "candidate_identities": int(identities["inchikey"].nunique()),
        "protected_development_identities": len(development),
        "protected_test_identities": len(protected),
        "protected_fingerprint_failures_exact_identity_only": protected_fingerprint_failures,
        "candidate_fingerprint_failures_removed": candidate_fingerprint_failures,
        "exact_development_overlap_removed": exact_development,
        "exact_test_overlap_removed": exact_test,
        "near_test_neighbor_removed": near_test,
        "tanimoto_threshold": tanimoto_threshold,
        "allowed_identities": len(allowed),
        "raw_structure_deduplication_before_standardization": True,
        "source_acceptance_depends_on_removal_counts": False,
        "test_labels_loaded": False,
        "protected_structures_exposed_to_policy": False,
    }


__all__ = ["protected_structure_firewall"]
