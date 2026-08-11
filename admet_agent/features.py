"""Leakage-free, composable 2D molecular representations."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Sequence

import numpy as np
from rdkit import Chem, DataStructs, rdBase
from rdkit.Avalon import pyAvalonTools
from rdkit.Chem import Descriptors, MACCSkeys, rdFingerprintGenerator, rdReducedGraphs


FEATURE_SCHEMA_VERSION = "v21-canonical-smiles-cache"


class MolecularFeaturizer:
    def __init__(self, representations: Sequence[str], n_bits: int = 2048):
        self.representations = tuple(representations)
        self.n_bits = n_bits
        self._morgan = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=n_bits)
        self._morgan_r3 = rdFingerprintGenerator.GetMorganGenerator(radius=3, fpSize=n_bits)
        self._atom_pair = rdFingerprintGenerator.GetAtomPairGenerator(fpSize=n_bits)
        self._torsion = rdFingerprintGenerator.GetTopologicalTorsionGenerator(fpSize=n_bits)
        self._descriptor_functions = Descriptors._descList

    def _bits(self, fingerprint) -> np.ndarray:
        values = np.zeros(fingerprint.GetNumBits(), dtype=np.float32)
        DataStructs.ConvertToNumpyArray(fingerprint, values)
        return values

    def _one(self, mol) -> np.ndarray:
        blocks: list[np.ndarray] = []
        if "morgan" in self.representations:
            blocks.append(self._bits(self._morgan.GetFingerprint(mol)))
        if "morgan_r3" in self.representations:
            blocks.append(self._bits(self._morgan_r3.GetFingerprint(mol)))
        if "morgan_count" in self.representations:
            values = np.zeros(self.n_bits, dtype=np.float32)
            DataStructs.ConvertToNumpyArray(self._morgan.GetCountFingerprint(mol), values)
            blocks.append(values)
        if "avalon" in self.representations:
            blocks.append(self._bits(pyAvalonTools.GetAvalonFP(mol, nBits=1024)))
        if "erg" in self.representations:
            blocks.append(np.asarray(rdReducedGraphs.GetErGFingerprint(mol), dtype=np.float32))
        if "maccs" in self.representations:
            blocks.append(self._bits(MACCSkeys.GenMACCSKeys(mol)))
        if "rdkit_fp" in self.representations:
            blocks.append(self._bits(Chem.RDKFingerprint(mol, fpSize=self.n_bits)))
        if "atom_pair" in self.representations:
            blocks.append(self._bits(self._atom_pair.GetFingerprint(mol)))
        if "torsion" in self.representations:
            blocks.append(self._bits(self._torsion.GetFingerprint(mol)))
        if "rdkit2d" in self.representations:
            values = []
            for _, function in self._descriptor_functions:
                try:
                    values.append(float(function(mol)))
                except Exception:
                    values.append(0.0)
            blocks.append(np.asarray(values, dtype=np.float32))
        return np.nan_to_num(np.concatenate(blocks), nan=0.0, posinf=0.0, neginf=0.0)

    def transform(self, smiles: Sequence[str]) -> np.ndarray:
        valid_rows: list[np.ndarray | None] = []
        width = None
        for value in smiles:
            mol = Chem.MolFromSmiles(str(value))
            row = None if mol is None else self._one(mol)
            if row is not None:
                width = len(row)
            valid_rows.append(row)
        if width is None:
            raise ValueError("No valid SMILES were available for featurization")
        return np.asarray(
            [np.zeros(width, dtype=np.float32) if row is None else row for row in valid_rows],
            dtype=np.float32,
        )


class FeatureCache:
    def __init__(self, directory: Path):
        self.directory = directory
        self.directory.mkdir(parents=True, exist_ok=True)

    def _path(self, smiles: Sequence[str], representations: Sequence[str]) -> Path:
        digest = hashlib.sha256()
        digest.update(FEATURE_SCHEMA_VERSION.encode())
        digest.update(rdBase.rdkitVersion.encode())
        digest.update(json.dumps(list(smiles), separators=(",", ":")).encode())
        digest.update("|".join(representations).encode())
        return self.directory / f"{digest.hexdigest()}.npy"

    def get_or_compute(self, smiles: Sequence[str], representations: Sequence[str]) -> np.ndarray:
        requested = [str(value) for value in smiles]
        canonical = sorted(set(requested))
        path = self._path(canonical, representations)
        if path.exists():
            canonical_values = np.load(path, allow_pickle=False)
        else:
            canonical_values = MolecularFeaturizer(representations).transform(canonical)
            np.save(path, canonical_values, allow_pickle=False)
        row_by_smiles = {
            value: canonical_values[index] for index, value in enumerate(canonical)
        }
        return np.asarray([row_by_smiles[value] for value in requested], dtype=np.float32)
