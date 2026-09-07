"""Paper-style, fail-closed evidence packets for external ADMET training data.

An evidence packet is deliberately smaller than an endpoint-level database.  It
binds one independently reviewable FDA document or literature assay to its
records, endpoint semantics, label transform, checksum, and human review.  The
legacy bulk pools are not accepted by this module.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import numpy as np
import pandas as pd

from .external_structure_firewall import (
    StructureFirewallReference,
    _standard_molecule,
    apply_external_structure_firewall,
    load_label_blind_test_structures,
)
from .tasks import resolve_task

SCHEMA_VERSION = "admet-evidence-packet-v4.1"
MAX_PACKET_ROWS = 5_000
SAME_SOURCE_TEST_SKELETON_LIMIT = 0.05
# Match the audited reference harness: ECFP4 neighbours at or above 0.90
# are excluded from external training data.
TEST_TANIMOTO_LIMIT = 0.90

ALLOWED_SOURCE_KINDS = frozenset(
    {
        "fda_label",
        "fda_review",
        "fda_guidance_assay",
        "fda_regulatory_dataset",
        "independent_peer_reviewed_assay",
        "regulatory_scientific_assay_report",
    }
)
ALLOWED_CITATION_TYPES = frozenset({"doi", "pubmed", "fda_document"})
ALLOWED_WEIGHT_MODES = frozenset({"paper_equal", "native_priority"})
_PACKET_ID = re.compile(r"^[a-z0-9][a-z0-9._-]{2,119}$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _required(mapping: dict[str, Any], key: str, context: str) -> Any:
    if key not in mapping:
        raise ValueError(f"{context} is missing required field {key!r}")
    value = mapping[key]
    if value is None or (isinstance(value, str) and not value.strip()):
        raise ValueError(f"{context}.{key} must not be empty")
    return value


def _https_url(value: str, field: str) -> str:
    parsed = urlparse(value)
    if parsed.scheme != "https" or not parsed.netloc:
        raise ValueError(f"{field} must be a stable HTTPS URL")
    return value


@dataclass(frozen=True)
class ValidatedEvidencePacket:
    manifest_path: Path
    records_path: Path
    packet_id: str
    task: str
    task_type: str
    source_kind: str
    native_priority_weight: float
    manifest: dict[str, Any]
    records: pd.DataFrame


@dataclass(frozen=True)
class PreparedEvidencePacket:
    packet: ValidatedEvidencePacket
    records: pd.DataFrame
    audit: dict[str, Any]

    @property
    def admitted(self) -> bool:
        return bool(self.audit["admitted_for_training"])


def validate_evidence_packet(
    manifest_path: str | Path,
    *,
    require_modeling_ready: bool = True,
) -> ValidatedEvidencePacket:
    """Validate provenance, semantic review, file binding, and record schema.

    No row-count minimum is imposed: a precise one-row FDA/literature packet is
    preferable to a large heterogeneous pool.  The 5,000-row ceiling prevents a
    bulk endpoint dump from being silently relabelled as a packet.
    """

    path = Path(manifest_path).resolve()
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Evidence packet manifest must be a JSON object")
    if payload.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"schema_version must equal {SCHEMA_VERSION!r}")

    packet_id = str(_required(payload, "packet_id", "manifest"))
    if not _PACKET_ID.fullmatch(packet_id):
        raise ValueError("packet_id must be a lowercase stable identifier")
    task = str(_required(payload, "task", "manifest"))
    task_spec = resolve_task(task)
    if task != task_spec.name:
        raise ValueError(f"task must use canonical name {task_spec.name!r}")
    task_type = str(_required(payload, "task_type", "manifest"))
    if task_type != task_spec.task_type:
        raise ValueError("task_type does not match the canonical task contract")

    source = _required(payload, "source", "manifest")
    if not isinstance(source, dict):
        raise ValueError("manifest.source must be an object")
    source_kind = str(_required(source, "kind", "source"))
    if source_kind not in ALLOWED_SOURCE_KINDS:
        raise ValueError(
            "source.kind is not an allowed FDA/independent-literature type"
        )
    _required(source, "name", "source")
    _required(source, "title", "source")
    citation_url = _https_url(
        str(_required(source, "citation_url", "source")), "source.citation_url"
    )
    citation_type = str(_required(source, "citation_type", "source"))
    if citation_type not in ALLOWED_CITATION_TYPES:
        raise ValueError("source.citation_type must be DOI, PubMed, or FDA document")
    stable_identifier = str(_required(source, "stable_identifier", "source"))
    citation_host = (urlparse(citation_url).hostname or "").lower()
    if citation_type == "doi" and not (
        citation_host == "doi.org"
        and stable_identifier.lower().removeprefix("doi:").startswith("10.")
    ):
        raise ValueError("DOI evidence requires a doi.org URL and DOI identifier")
    if citation_type == "pubmed" and not (
        citation_host in {"pubmed.ncbi.nlm.nih.gov", "www.ncbi.nlm.nih.gov"}
        and stable_identifier.upper().removeprefix("PMID:").isdigit()
    ):
        raise ValueError("PubMed evidence requires an NCBI URL and PMID identifier")
    if citation_type == "fda_document" and not citation_host.endswith("fda.gov"):
        raise ValueError("FDA document evidence requires an fda.gov citation URL")
    if source_kind.startswith("fda_") and citation_type != "fda_document":
        raise ValueError("FDA packet source kinds require an FDA document citation")
    if source_kind == "independent_peer_reviewed_assay" and citation_type not in {
        "doi",
        "pubmed",
    }:
        raise ValueError(
            "independent literature packets require DOI or PubMed evidence"
        )
    if (
        source_kind == "regulatory_scientific_assay_report"
        and citation_type != "doi"
    ):
        raise ValueError("regulatory assay reports require a DOI citation")
    _required(source, "license_or_access_terms", "source")
    if source.get("independence_status") != "verified_independent":
        raise ValueError("source independence must be explicitly verified")
    _required(source, "independence_evidence", "source")

    assay = _required(payload, "assay", "manifest")
    if not isinstance(assay, dict):
        raise ValueError("manifest.assay must be an object")
    for field in (
        "endpoint_definition",
        "assay_type",
        "organism",
        "biological_system",
        "source_record_locator",
    ):
        _required(assay, field, "assay")

    label = _required(payload, "label", "manifest")
    if not isinstance(label, dict):
        raise ValueError("manifest.label must be an object")
    for field in ("source_value_definition", "mapping_rule", "direction", "transform"):
        _required(label, field, "label")
    if task_type == "regression":
        _required(label, "unit", "label")
    elif label.get("mapping_rule") in {"binary", "active/inactive", "unspecified"}:
        raise ValueError("classification mapping_rule must state direction/threshold")

    review = _required(payload, "review", "manifest")
    if not isinstance(review, dict):
        raise ValueError("manifest.review must be an object")
    for field in ("status", "reviewer", "reviewed_at", "endpoint_match", "notes"):
        _required(review, field, "review")
    if require_modeling_ready and review.get("status") != "modeling_ready":
        raise ValueError("packet has not passed modeling-ready review")
    if require_modeling_ready and review.get("endpoint_match") != "direct":
        raise ValueError("only directly endpoint-matched packets enter v4 training")
    native_priority_weight = float(
        _required(review, "native_priority_external_weight", "review")
    )
    if not math.isfinite(native_priority_weight) or not 0 < native_priority_weight < 1:
        raise ValueError(
            "external weight must be finite, positive, and below native weight"
        )

    records_spec = _required(payload, "records", "manifest")
    if not isinstance(records_spec, dict):
        raise ValueError("manifest.records must be an object")
    relative = Path(str(_required(records_spec, "path", "records")))
    if relative.is_absolute():
        raise ValueError("records.path must be relative to the manifest")
    records_path = (path.parent / relative).resolve()
    if not records_path.is_relative_to(path.parent):
        raise ValueError("records.path may not escape the packet directory")
    expected_sha = str(_required(records_spec, "sha256", "records"))
    if not _SHA256.fullmatch(expected_sha):
        raise ValueError("records.sha256 must be a lowercase SHA-256 digest")
    if not records_path.is_file():
        raise ValueError("bound packet records file does not exist")
    observed_sha = sha256_file(records_path)
    if observed_sha != expected_sha:
        raise ValueError("packet records SHA-256 mismatch")

    records = pd.read_csv(records_path)
    required_columns = {"Drug", "Y", "source_record_id"}
    if not required_columns.issubset(records.columns):
        missing = sorted(required_columns - set(records.columns))
        raise ValueError(f"packet records are missing columns: {missing}")
    expected_rows = int(_required(records_spec, "row_count", "records"))
    if expected_rows != len(records):
        raise ValueError("records.row_count does not match the bound CSV")
    if not 1 <= len(records) <= MAX_PACKET_ROWS:
        raise ValueError(f"evidence packets must contain 1-{MAX_PACKET_ROWS} rows")
    if records["source_record_id"].isna().any():
        raise ValueError("source_record_id may not be missing")
    if records["source_record_id"].astype(str).duplicated().any():
        raise ValueError("source_record_id must be unique within a packet")
    numeric = pd.to_numeric(records["Y"], errors="coerce")
    if not np.isfinite(numeric.to_numpy(dtype=float)).all():
        raise ValueError("Y must contain only finite numeric values")
    if task_type == "classification" and not set(numeric.unique()) <= {0, 1}:
        raise ValueError("classification Evidence Packet labels must be exactly 0/1")
    records = records.copy()
    records["Y"] = numeric.astype(int if task_type == "classification" else float)
    return ValidatedEvidencePacket(
        manifest_path=path,
        records_path=records_path,
        packet_id=packet_id,
        task=task,
        task_type=task_type,
        source_kind=source_kind,
        native_priority_weight=native_priority_weight,
        manifest=payload,
        records=records,
    )


def _collapse_internal_identities(
    records: pd.DataFrame, *, task_type: str
) -> tuple[pd.DataFrame, dict[str, int]]:
    work = records.copy().reset_index(drop=True)
    work["_standard"] = work["Drug"].map(lambda value: _standard_molecule(str(value)))
    invalid = int(work["_standard"].isna().sum())
    work = work.loc[work["_standard"].notna()].copy()
    work["_identity"] = work["_standard"].map(lambda value: value.inchikey)
    work["Drug"] = work["_standard"].map(lambda value: value.smiles)

    conflicts: set[str] = set()
    if task_type == "classification":
        counts = work.groupby("_identity")["Y"].nunique()
        conflicts = set(counts[counts > 1].index)
        work = work.loc[~work["_identity"].isin(conflicts)].copy()
        output = work.drop_duplicates("_identity", keep="first")
    else:
        rows = []
        for _, group in work.groupby("_identity", sort=True):
            row = group.iloc[0].copy()
            row["Y"] = float(group["Y"].median())
            row["source_record_id"] = "|".join(
                sorted(group["source_record_id"].astype(str))
            )
            rows.append(row)
        output = pd.DataFrame(rows, columns=work.columns)
    output = output.drop(columns=["_standard", "_identity"]).reset_index(drop=True)
    return output, {
        "invalid_structure_rows": invalid,
        "conflicting_identity_count": len(conflicts),
        "duplicate_or_replicate_rows_collapsed": int(
            len(records) - invalid - len(conflicts) - len(output)
        ),
    }


def prepare_evidence_packet(
    packet: ValidatedEvidencePacket,
    *,
    train_smiles: Sequence[str],
    validation_smiles: Sequence[str],
    test_path: str | Path | None = None,
    test_smiles: Sequence[str] | None = None,
    firewall_reference: StructureFirewallReference | None = None,
) -> PreparedEvidencePacket:
    """Apply packet-level curation and the label-blind three-layer firewall."""

    if (test_path is None) == (test_smiles is None):
        raise ValueError("provide exactly one of test_path or test_smiles")
    test_loader_audit: dict[str, Any]
    if test_path is not None:
        structures = load_label_blind_test_structures(test_path)
        test_loader_audit = {
            "path": str(Path(test_path).resolve()),
            "columns_loaded": ["Drug"],
            "label_columns_loaded": [],
        }
    else:
        structures = tuple(str(value) for value in test_smiles or ())
        test_loader_audit = {
            "path": None,
            "columns_loaded": ["Drug"],
            "label_columns_loaded": [],
        }

    # Label validation happens in validate_evidence_packet; regression trimming
    # happens next; L2 same-source rejection is then computed on that raw source.
    # Internal conflict/replicate collapse must not run before L2, otherwise
    # conflicting rows could hide benchmark-source overlap.
    curated = packet.records.copy().reset_index(drop=True)
    trim_audit: dict[str, Any] = {"applied": False, "removed_rows": 0}
    if packet.task_type == "regression" and len(curated):
        lower, upper = curated["Y"].quantile([0.01, 0.99])
        keep = curated["Y"].between(lower, upper, inclusive="both")
        trim_audit = {
            "applied": True,
            "lower_quantile": float(lower),
            "upper_quantile": float(upper),
            "removed_rows": int((~keep).sum()),
        }
        curated = curated.loc[keep].reset_index(drop=True)

    safe_before_collapse, firewall = apply_external_structure_firewall(
        curated,
        train_smiles=train_smiles,
        validation_smiles=validation_smiles,
        test_smiles=structures,
        same_source_test_overlap_limit=SAME_SOURCE_TEST_SKELETON_LIMIT,
        test_tanimoto_limit=TEST_TANIMOTO_LIMIT,
        reference=firewall_reference,
    )
    safe, curation = _collapse_internal_identities(
        safe_before_collapse, task_type=packet.task_type
    )
    admitted = bool(not firewall.dataset_rejected and len(safe) > 0)
    safe = safe.assign(
        evidence_packet_id=packet.packet_id,
        evidence_source_kind=packet.source_kind,
        evidence_base_weight=packet.native_priority_weight,
    )
    audit = {
        "schema_version": SCHEMA_VERSION,
        "packet_id": packet.packet_id,
        "task": packet.task,
        "manifest_path": str(packet.manifest_path),
        "manifest_sha256": sha256_file(packet.manifest_path),
        "records_path": str(packet.records_path),
        "records_sha256": sha256_file(packet.records_path),
        "curation": curation,
        "regression_trim": trim_audit,
        "structure_firewall": firewall.to_dict(),
        "admitted_for_training": admitted,
        "admitted_rows": len(safe),
        "test_loader": test_loader_audit,
        "test_labels_loaded": False,
        "legacy_bulk_pool_used": False,
    }
    return PreparedEvidencePacket(packet=packet, records=safe, audit=audit)


def assemble_evidence_training_data(
    native_train: pd.DataFrame,
    prepared_packets: Iterable[PreparedEvidencePacket],
    *,
    weight_mode: str = "paper_equal",
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Append every admitted packet without coreset selection or bulk sampling."""

    if weight_mode not in ALLOWED_WEIGHT_MODES:
        raise ValueError(f"weight_mode must be one of {sorted(ALLOWED_WEIGHT_MODES)}")
    required = {"Drug", "Y"}
    if not required.issubset(native_train):
        raise ValueError("native_train must contain Drug and Y")
    native = native_train.copy().reset_index(drop=True)
    native["sample_weight"] = 1.0
    native["data_origin"] = "native_tdc"
    native["evidence_packet_id"] = pd.NA
    frames = [native]
    packet_rows: dict[str, int] = {}
    external_weights: dict[str, float] = {}
    observed_tasks: set[str] = set()
    for prepared in prepared_packets:
        if not prepared.admitted:
            continue
        packet = prepared.packet
        observed_tasks.add(packet.task)
        external = prepared.records.copy().reset_index(drop=True)
        weight = 1.0 if weight_mode == "paper_equal" else packet.native_priority_weight
        external["sample_weight"] = weight
        external["data_origin"] = "evidence_packet"
        frames.append(external)
        packet_rows[packet.packet_id] = len(external)
        external_weights[packet.packet_id] = weight
    if len(observed_tasks) > 1:
        raise ValueError("one training table may only combine packets for one task")
    output = pd.concat(frames, ignore_index=True, sort=False)
    return output, {
        "weight_mode": weight_mode,
        "native_rows": len(native),
        "external_rows": int(sum(packet_rows.values())),
        "packet_rows": packet_rows,
        "external_weights": external_weights,
        "coreset_selection_used": False,
        "legacy_bulk_pool_used": False,
        "native_weight": 1.0,
        "native_strictly_higher_than_external": bool(
            weight_mode == "native_priority"
            and external_weights
            and max(external_weights.values()) < 1.0
        ),
    }


def write_packet_audit(audit: dict[str, Any], output: str | Path) -> None:
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


__all__ = [
    "ALLOWED_WEIGHT_MODES",
    "MAX_PACKET_ROWS",
    "SAME_SOURCE_TEST_SKELETON_LIMIT",
    "SCHEMA_VERSION",
    "TEST_TANIMOTO_LIMIT",
    "PreparedEvidencePacket",
    "ValidatedEvidencePacket",
    "assemble_evidence_training_data",
    "prepare_evidence_packet",
    "sha256_file",
    "validate_evidence_packet",
    "write_packet_audit",
]
