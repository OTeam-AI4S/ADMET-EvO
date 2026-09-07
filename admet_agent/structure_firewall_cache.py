"""Input-bound on-disk cache for label-blind structure firewall references."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from pathlib import Path
from types import MappingProxyType

from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator

from .external_structure_firewall import (
    StructureFirewallReference,
    _sequence_sha256,
    _StandardMolecule,
    prepare_structure_firewall_reference,
)

CACHE_PROTOCOL = "structure-firewall-reference-cache-v1"
DEFAULT_CACHE_ROOT = Path("checkpoints/evidence_packet_v5/structure_firewall")


def _restore(rows: list[dict[str, str]]) -> dict[str, _StandardMolecule]:
    restored = {}
    for row in rows:
        molecule = _StandardMolecule(**row)
        restored[molecule.inchikey] = molecule
    return restored


def prepare_structure_firewall_reference_cached(
    *,
    train_smiles,
    validation_smiles,
    test_smiles,
    cache_root: str | Path = DEFAULT_CACHE_ROOT,
) -> StructureFirewallReference:
    """Load or create a cache bound to all three ordered structure inputs.

    The payload contains standardized structures only.  Test labels are never
    loaded, accepted, or persisted by this layer.
    """

    hashes = {
        "train_input_sha256": _sequence_sha256(train_smiles),
        "validation_input_sha256": _sequence_sha256(validation_smiles),
        "test_input_sha256": _sequence_sha256(test_smiles),
    }
    key = hashlib.sha256("|".join(hashes.values()).encode("ascii")).hexdigest()
    root = Path(cache_root).resolve()
    path = root / f"{key}.json"
    if path.is_file():
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("protocol") != CACHE_PROTOCOL:
            raise ValueError("Unknown structure firewall cache protocol")
        if payload.get("test_labels_loaded") is not False:
            raise ValueError("Structure firewall cache lacks label-blind attestation")
        if any(payload.get(name) != value for name, value in hashes.items()):
            raise ValueError("Structure firewall cache input hash mismatch")
        train = _restore(payload["train"])
        validation = _restore(payload["validation"])
        test = _restore(payload["test"])
        if not test:
            raise ValueError("Cached structure firewall has no valid test structures")
        generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
        return StructureFirewallReference(
            train=MappingProxyType(train),
            validation=MappingProxyType(validation),
            test=MappingProxyType(test),
            test_fingerprints=tuple(
                generator.GetFingerprint(Chem.MolFromSmiles(value.smiles))
                for value in test.values()
            ),
            **hashes,
        )

    reference = prepare_structure_firewall_reference(
        train_smiles=train_smiles,
        validation_smiles=validation_smiles,
        test_smiles=test_smiles,
    )
    payload = {
        "protocol": CACHE_PROTOCOL,
        **hashes,
        "train": [asdict(value) for value in reference.train.values()],
        "validation": [asdict(value) for value in reference.validation.values()],
        "test": [asdict(value) for value in reference.test.values()],
        "test_labels_loaded": False,
    }
    root.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)
    return reference


__all__ = ["prepare_structure_firewall_reference_cached"]
