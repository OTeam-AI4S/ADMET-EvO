"""Leakage-free, composable 2D molecular representations."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import tempfile
from collections.abc import Sequence
from functools import lru_cache
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import numpy as np
from rdkit import Chem, DataStructs, rdBase
from rdkit.Avalon import pyAvalonTools
from rdkit.Chem import Descriptors, MACCSkeys, rdFingerprintGenerator, rdReducedGraphs

FEATURE_SCHEMA_VERSION = "v24-unimol2-minimol-cache"
UNIMOL2_REPRESENTATION = "unimol2"
MINIMOL_REPRESENTATION = "minimol"
MINIMOL_PINNED_VERSION = "1.3.5"


@lru_cache(maxsize=1)
def _minimol_subprocess_identity() -> str:
    repo_root = Path(__file__).resolve().parents[1]
    weight_candidates = sorted(
        (repo_root / ".venvs/minimol/lib").glob(
            "python*/site-packages/minimol/ckpts/minimol_v1/state_dict.pth"
        )
    )
    if not weight_candidates:
        return "minimol-subprocess-missing"
    digest = hashlib.sha256()
    with weight_candidates[0].open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return f"minimol-subprocess-{MINIMOL_PINNED_VERSION}-{digest.hexdigest()}"


class MiniMolFeaturizer:
    """Lazy adapter around the optional frozen MiniMol fingerprint model."""

    def __init__(self, batch_size: int = 128):
        if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size < 1:
            raise ValueError("batch_size must be a positive integer")
        self.batch_size = batch_size

    def transform(self, smiles: Sequence[str]) -> np.ndarray:
        try:
            from minimol import Minimol
        except ImportError:
            return self._transform_subprocess(smiles)
        requested = [str(value) for value in smiles]
        model = Minimol(batch_size=self.batch_size)
        output = model(requested)
        values = np.stack(
            [value.detach().cpu().numpy() for value in output]
        ).astype(np.float32)
        if values.shape != (len(requested), 512):
            raise RuntimeError("MiniMol returned an unexpected fingerprint shape")
        if not np.isfinite(values).all():
            raise RuntimeError("MiniMol returned non-finite fingerprints")
        return values

    def _transform_subprocess(self, smiles: Sequence[str]) -> np.ndarray:
        repo_root = Path(__file__).resolve().parents[1]
        python = repo_root / ".venvs/minimol/bin/python"
        extractor = repo_root / "scripts/extract_minimol_embeddings.py"
        if not python.exists() or not extractor.exists():
            raise RuntimeError(
                "Install the `minimol` project extra or create .venvs/minimol"
            )
        requested = [str(value) for value in smiles]
        with tempfile.TemporaryDirectory(prefix="admet-minimol-") as directory:
            temporary = Path(directory)
            manifest = temporary / "smiles.json"
            output = temporary / "embeddings.npy"
            manifest.write_text(
                json.dumps(
                    {
                        "task": "feature_cache",
                        "smiles": requested,
                        "contains_labels": False,
                        "official_test_loaded": False,
                    },
                    sort_keys=True,
                ),
                encoding="utf-8",
            )
            completed = subprocess.run(
                [
                    str(python),
                    str(extractor),
                    "--input",
                    str(manifest),
                    "--output",
                    str(output),
                    "--batch-size",
                    str(self.batch_size),
                ],
                cwd=repo_root,
                check=False,
                capture_output=True,
                text=True,
            )
            if completed.returncode != 0:
                raise RuntimeError(
                    "Isolated MiniMol extraction failed: "
                    + completed.stderr[-2000:]
                )
            values = np.load(output, allow_pickle=False)
        if values.shape != (len(requested), 512) or not np.isfinite(values).all():
            raise RuntimeError("Isolated MiniMol returned invalid fingerprints")
        return np.asarray(values, dtype=np.float32)


class UniMol2Featurizer:
    """Lazy adapter around the optional official ``unimol_tools`` package."""

    def __init__(self, device: str = "cpu", batch_size: int = 32):
        if device not in {"cpu", "cuda"}:
            raise ValueError("device must be cpu or cuda")
        if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size < 1:
            raise ValueError("batch_size must be a positive integer")
        self.device = device
        self.batch_size = batch_size

    def transform(self, smiles: Sequence[str]) -> np.ndarray:
        try:
            from unimol_tools import UniMolRepr
        except ImportError as exc:
            raise RuntimeError(
                "Install the `unimol` project extra to use the Uni-Mol2 representation"
            ) from exc

        requested = [str(value) for value in smiles]
        valid_indices = [
            index for index, value in enumerate(requested)
            if Chem.MolFromSmiles(value) is not None
        ]
        if not valid_indices:
            raise ValueError("No valid SMILES were available for Uni-Mol2 featurization")
        valid_smiles = [requested[index] for index in valid_indices]
        encoder = UniMolRepr(
            data_type="molecule",
            model_name="unimolv2",
            model_size="84m",
            batch_size=self.batch_size,
            use_cuda=self.device == "cuda",
            use_ddp=False,
        )
        output = encoder.get_repr(valid_smiles, return_atomic_reprs=False)
        # unimol_tools <=0.1.5 returned a dictionary; newer releases return the
        # molecular representations directly. Supporting both keeps artifacts
        # reproducible across the currently supported 0.1.x series.
        if isinstance(output, dict):
            output = output.get("cls_repr")
        values = np.asarray(output, dtype=np.float32)
        if values.ndim != 2 or values.shape[0] != len(valid_smiles):
            raise RuntimeError(
                "Uni-Mol2 returned an unexpected molecular representation shape"
            )
        if not np.isfinite(values).all():
            raise RuntimeError("Uni-Mol2 returned non-finite molecular representations")
        rows = np.zeros((len(requested), values.shape[1]), dtype=np.float32)
        rows[valid_indices] = values
        return rows


class MolecularFeaturizer:
    def __init__(
        self,
        representations: Sequence[str],
        n_bits: int = 2048,
        device: str = "cpu",
        unimol_batch_size: int = 32,
        minimol_batch_size: int = 128,
    ):
        self.representations = tuple(representations)
        self.n_bits = n_bits
        self.device = device
        self.unimol_batch_size = unimol_batch_size
        self.minimol_batch_size = minimol_batch_size
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

    def _transform_rdkit(self, smiles: Sequence[str]) -> np.ndarray:
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

    def transform(self, smiles: Sequence[str]) -> np.ndarray:
        blocks: list[np.ndarray] = []
        rdkit_representations = tuple(
            name
            for name in self.representations
            if name not in {UNIMOL2_REPRESENTATION, MINIMOL_REPRESENTATION}
        )
        if rdkit_representations:
            rdkit_featurizer = MolecularFeaturizer(
                rdkit_representations,
                n_bits=self.n_bits,
                device=self.device,
                unimol_batch_size=self.unimol_batch_size,
                minimol_batch_size=self.minimol_batch_size,
            )
            blocks.append(rdkit_featurizer._transform_rdkit(smiles))
        if UNIMOL2_REPRESENTATION in self.representations:
            blocks.append(
                UniMol2Featurizer(
                    device=self.device,
                    batch_size=self.unimol_batch_size,
                ).transform(smiles)
            )
        if MINIMOL_REPRESENTATION in self.representations:
            blocks.append(
                MiniMolFeaturizer(batch_size=self.minimol_batch_size).transform(smiles)
            )
        if not blocks:
            raise ValueError("At least one molecular representation is required")
        return np.concatenate(blocks, axis=1).astype(np.float32, copy=False)


class FeatureCache:
    def __init__(
        self,
        directory: Path,
        device: str = "cpu",
        unimol_batch_size: int = 32,
        minimol_batch_size: int = 128,
    ):
        self.directory = directory
        self.device = device
        self.unimol_batch_size = unimol_batch_size
        self.minimol_batch_size = minimol_batch_size
        self.directory.mkdir(parents=True, exist_ok=True)

    def _path(self, smiles: Sequence[str], representations: Sequence[str]) -> Path:
        digest = hashlib.sha256()
        digest.update(FEATURE_SCHEMA_VERSION.encode())
        digest.update(rdBase.rdkitVersion.encode())
        if UNIMOL2_REPRESENTATION in representations:
            try:
                unimol_tools_version = version("unimol-tools")
            except PackageNotFoundError:
                unimol_tools_version = "not-installed"
            digest.update(unimol_tools_version.encode())
        if MINIMOL_REPRESENTATION in representations:
            try:
                minimol_version = version("minimol")
            except PackageNotFoundError:
                minimol_version = _minimol_subprocess_identity()
            digest.update(minimol_version.encode())
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
            canonical_values = MolecularFeaturizer(
                representations,
                device=self.device,
                unimol_batch_size=self.unimol_batch_size,
                minimol_batch_size=self.minimol_batch_size,
            ).transform(canonical)
            # A shared cache may be populated by several benchmark arms.  Write
            # into the same filesystem and publish atomically so readers never
            # observe a partial NumPy payload.
            descriptor, temporary_name = tempfile.mkstemp(
                prefix=f".{path.stem}-", suffix=".npy", dir=self.directory
            )
            temporary_path = Path(temporary_name)
            try:
                with os.fdopen(descriptor, "wb") as handle:
                    np.save(handle, canonical_values, allow_pickle=False)
                os.replace(temporary_path, path)
            finally:
                temporary_path.unlink(missing_ok=True)
        row_by_smiles = {
            value: canonical_values[index] for index, value in enumerate(canonical)
        }
        return np.asarray([row_by_smiles[value] for value in requested], dtype=np.float32)
