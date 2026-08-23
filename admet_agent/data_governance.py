"""Acquisition, curation, provenance and split-isolation utilities.

This module never downloads or opens a test split.  It accepts discovery data
or explicitly supplied external tables and produces deterministic manifests.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Protocol

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem.MolStandardize import rdMolStandardize

from .endpoint_contracts import EndpointContract

ALLOWED_DATA_POLICIES = frozenset(
    {
        "official",
        "identity_safe",
        "canonical_deduplicate",
        "quality_filtered",
        "leakage_safe",
    }
)


def canonical_smiles(value: object, *, largest_fragment: bool = True) -> str | None:
    mol = Chem.MolFromSmiles(str(value))
    if mol is None:
        return None
    if largest_fragment:
        mol = rdMolStandardize.LargestFragmentChooser().choose(mol)
    try:
        Chem.SanitizeMol(mol)
        return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)
    except Exception:  # noqa: BLE001 - RDKit exposes several wrapper exceptions
        return None


def frame_sha256(frame: pd.DataFrame) -> str:
    normalized = frame.copy()
    normalized.columns = [str(value) for value in normalized.columns]
    normalized = normalized.reindex(sorted(normalized.columns), axis=1)
    values = pd.util.hash_pandas_object(normalized, index=True).to_numpy(dtype=np.uint64)
    return hashlib.sha256(values.tobytes()).hexdigest()


@dataclass(frozen=True)
class SourceManifest:
    source_id: str
    endpoint: str
    origin: str
    license: str
    acquired_at: str
    assay_system: str | None
    species: str | None
    unit: str | None
    label_transform: str
    citation: str
    checksum: str | None = None

    def validate_for(self, contract: EndpointContract) -> None:
        errors = []
        if self.endpoint not in {contract.tdc_name, contract.endpoint_id}:
            errors.append("endpoint does not match the endpoint contract")
        for name in ("source_id", "origin", "license", "acquired_at", "citation"):
            if not str(getattr(self, name)).strip():
                errors.append(f"{name} is required")
        if not self.label_transform.strip() or self.label_transform == "unspecified":
            errors.append("label_transform must be explicit")
        if not self.assay_system:
            errors.append("assay_system is required for external observations")
        if not self.species:
            errors.append("species is required; use not_applicable when appropriate")
        if contract.task_type == "regression" and not self.unit:
            errors.append("unit is required for an external regression source")
        if errors:
            raise ValueError("; ".join(errors))

    def compatibility_issues(self, contract: EndpointContract) -> tuple[str, ...]:
        issues = []
        if (
            contract.species
            and self.species
            and self.species.lower() != contract.species.lower()
        ):
            issues.append(
                f"species mismatch: source={self.species}, contract={contract.species}"
            )
        if contract.assay_system and self.assay_system != contract.assay_system:
            issues.append("assay system requires an explicit bridging/alignment study")
        if contract.task_type == "regression" and contract.tdc_native_unit is None:
            issues.append("TDC native unit/transform must be reconstructed before merging")
        return tuple(issues)


@dataclass(frozen=True)
class DataAudit:
    policy: str
    input_rows: int
    output_rows: int
    invalid_smiles: int
    duplicate_rows: int
    duplicate_structures: int
    conflicting_structures: int
    missing_targets: int
    non_finite_targets: int
    replicate_rows: int = 0
    replicate_target_dispersion: float | None = None
    removed_validation_overlap: int = 0
    input_sha256: str = ""
    output_sha256: str = ""
    warnings: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class TableSource(Protocol):
    def acquire(self) -> tuple[pd.DataFrame, SourceManifest]: ...


@dataclass(frozen=True)
class LocalTableSource:
    path: Path
    manifest: SourceManifest
    read_options: dict[str, Any] = field(default_factory=dict)

    def acquire(self) -> tuple[pd.DataFrame, SourceManifest]:
        suffix = self.path.suffix.lower()
        if suffix == ".csv":
            frame = pd.read_csv(self.path, **self.read_options)
        elif suffix in {".tsv", ".txt"}:
            frame = pd.read_csv(self.path, sep="\t", **self.read_options)
        elif suffix == ".parquet":
            frame = pd.read_parquet(self.path, **self.read_options)
        else:
            raise ValueError(f"Unsupported table format: {suffix}")
        checksum = hashlib.sha256(self.path.read_bytes()).hexdigest()
        if self.manifest.checksum and checksum != self.manifest.checksum:
            raise ValueError("External source checksum does not match its manifest")
        return frame, self.manifest


class MolecularDataCurator:
    """Deterministic molecule/label curation with a complete audit report."""

    def __init__(self, smiles_col: str = "Drug", target_col: str = "Y"):
        self.smiles_col = smiles_col
        self.target_col = target_col

    def audit(
        self,
        frame: pd.DataFrame,
        *,
        task_type: str | None = None,
        policy: str = "official",
    ) -> DataAudit:
        _, report = self.curate(
            frame, task_type=task_type, policy=policy, dry_run=True
        )
        return report

    def curate(
        self,
        frame: pd.DataFrame,
        *,
        task_type: str | None,
        policy: str,
        validation_smiles: set[str] | None = None,
        dry_run: bool = False,
    ) -> tuple[pd.DataFrame, DataAudit]:
        if policy not in ALLOWED_DATA_POLICIES:
            raise ValueError(f"Unknown data policy {policy!r}")
        required = {self.smiles_col, self.target_col}
        missing = required - set(frame.columns)
        if missing:
            raise ValueError(f"Missing required data columns: {sorted(missing)}")
        working = frame.copy().reset_index(drop=True)
        input_hash = frame_sha256(working)
        canonical = working[self.smiles_col].map(canonical_smiles)
        numeric = pd.to_numeric(working[self.target_col], errors="coerce")
        invalid = int(canonical.isna().sum())
        missing_targets = int(working[self.target_col].isna().sum())
        non_finite = int(np.isinf(numeric.to_numpy(dtype=float)).sum())
        valid_view = pd.DataFrame({"canonical": canonical, "target": numeric}).dropna()
        duplicate_rows = int(valid_view.duplicated(["canonical", "target"]).sum())
        counts = valid_view.groupby("canonical").size()
        duplicate_structures = int((counts > 1).sum())
        replicate_rows = int((counts[counts > 1] - 1).sum())
        target_counts = valid_view.groupby("canonical")["target"].nunique()
        replicate_dispersion = None
        if task_type == "classification":
            conflicting = int((target_counts > 1).sum())
        elif task_type == "regression":
            grouped_targets = valid_view.groupby("canonical")["target"]
            global_scale = float(valid_view["target"].std()) or 1.0
            ranges = grouped_targets.max() - grouped_targets.min()
            conflicting = int((ranges > global_scale).sum())
            repeated_std = grouped_targets.std().dropna()
            replicate_dispersion = (
                float(repeated_std.median() / global_scale)
                if len(repeated_std)
                else 0.0
            )
        else:
            conflicting = int((target_counts > 1).sum())
        overlap = 0
        if validation_smiles:
            overlap = int(canonical.isin(validation_smiles).sum())

        warnings = []
        if invalid:
            warnings.append("invalid SMILES present")
        if duplicate_structures and task_type == "regression":
            warnings.append("same canonical structure has repeated measurements")
        if conflicting and task_type != "regression":
            warnings.append("same canonical structure has conflicting labels")
        elif conflicting:
            warnings.append("some repeated measurements differ by more than one target standard deviation")
        if overlap:
            warnings.append("training structures overlap validation structures")

        output = working.copy()
        if policy != "official" and not dry_run:
            output["_canonical_smiles"] = canonical
            output[self.target_col] = numeric
            output = output[
                output["_canonical_smiles"].notna()
                & output[self.target_col].notna()
                & np.isfinite(output[self.target_col])
            ].copy()
            if policy in {"canonical_deduplicate", "quality_filtered", "leakage_safe"}:
                if task_type == "classification":
                    grouped = output.groupby("_canonical_smiles")[self.target_col]
                    ambiguous = set(grouped.nunique()[lambda value: value > 1].index)
                    if policy in {"quality_filtered", "leakage_safe"}:
                        output = output[~output["_canonical_smiles"].isin(ambiguous)]
                    output = output.drop_duplicates("_canonical_smiles", keep="first")
                elif task_type == "regression":
                    medians = output.groupby("_canonical_smiles")[self.target_col].median()
                    output = output.drop_duplicates("_canonical_smiles", keep="first")
                    output[self.target_col] = output["_canonical_smiles"].map(medians)
                else:
                    output = output.drop_duplicates("_canonical_smiles", keep="first")
            if policy in {"identity_safe", "leakage_safe"} and validation_smiles:
                output = output[~output["_canonical_smiles"].isin(validation_smiles)]
            output[self.smiles_col] = output["_canonical_smiles"]
            output = output.drop(columns=["_canonical_smiles"]).reset_index(drop=True)

        report = DataAudit(
            policy=policy,
            input_rows=len(working),
            output_rows=len(output),
            invalid_smiles=invalid,
            duplicate_rows=duplicate_rows,
            duplicate_structures=duplicate_structures,
            conflicting_structures=conflicting,
            missing_targets=missing_targets,
            non_finite_targets=non_finite,
            replicate_rows=replicate_rows,
            replicate_target_dispersion=replicate_dispersion,
            removed_validation_overlap=(
                overlap
                if policy in {"identity_safe", "leakage_safe"} and not dry_run
                else 0
            ),
            input_sha256=input_hash,
            output_sha256=frame_sha256(output),
            warnings=tuple(warnings),
        )
        return output, report

    def ingest_external(
        self,
        source: TableSource,
        contract: EndpointContract,
        *,
        smiles_col: str,
        target_col: str,
        output_dir: Path,
    ) -> tuple[pd.DataFrame, dict[str, Any]]:
        frame, manifest = source.acquire()
        manifest.validate_for(contract)
        renamed = frame.rename(columns={smiles_col: self.smiles_col, target_col: self.target_col})
        curated, audit = self.curate(
            renamed,
            task_type=contract.task_type,
            policy="quality_filtered",
        )
        output_dir.mkdir(parents=True, exist_ok=True)
        curated_path = output_dir / f"{manifest.source_id}.curated.csv"
        curated.to_csv(curated_path, index=False)
        packet = {
            "contract": contract.to_dict(),
            "source": asdict(manifest),
            "audit": audit.to_dict(),
            "compatibility_issues": list(manifest.compatibility_issues(contract)),
            "curated_path": str(curated_path),
            "merge_status": "quarantined_pending_assay_alignment",
        }
        (output_dir / f"{manifest.source_id}.manifest.json").write_text(
            json.dumps(packet, indent=2, sort_keys=True), encoding="utf-8"
        )
        return curated, packet
