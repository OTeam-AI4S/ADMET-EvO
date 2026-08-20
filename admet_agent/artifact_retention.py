"""Read-only artifact retention planning.

This module deliberately has no delete, move, or archive operation.  It only
inventories selected model/evidence artifacts and assigns conservative
retention tiers that can be reviewed by a human before any later action.
"""

from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
from enum import Enum
import hashlib
import json
import os
from pathlib import Path
import re
from typing import Any, Iterable, Iterator, Sequence


SCHEMA_VERSION = "admet-artifact-retention-dry-run-v1"
DEFAULT_HASH_MAX_BYTES = 8 * 1024 * 1024
_REFERENCE_JSON_MAX_BYTES = 32 * 1024 * 1024


class RetentionTier(str, Enum):
    """Mutually exclusive, increasingly disposable retention tiers."""

    EVIDENCE = "evidence"
    DEPLOYMENT = "deployment"
    DEBUG_TTL = "debug_ttl"
    COLD_STORE = "cold_store"
    RECLAIM_CANDIDATE = "reclaim_candidate"


_MODEL_SUFFIXES = {".pth", ".pt", ".ckpt"}
_CONFIG_SUFFIXES = {".json", ".yaml", ".yml", ".toml"}
_FORMAL_TOKENS = {"official", "final", "finalist", "finalists", "freeze", "frozen", "deployment", "production"}
_COLD_TOKENS = {"ablation", "pilot", "shard", "shards", "historical", "archive"}
_RECLAIM_TOKENS = {"smoke", "screen", "screens", "scratch", "temporary", "tmp"}


def _tokens(path: Path) -> set[str]:
    return {
        token
        for part in path.parts
        for token in re.split(r"[^a-z0-9]+", part.lower())
        if token
    }


def _is_feature_cache(path: Path) -> bool:
    return any(
        part.lower() in {"feature_cache", "shared_feature_cache"}
        or part.lower().endswith("feature_cache")
        for part in path.parts
    )


def _is_base_checkpoint(path: Path) -> bool:
    tokens = _tokens(path)
    name = path.name.lower()
    return (
        name in {"checkpoint.pt", "checkpoint.pth", "base_checkpoint.pt", "base_checkpoint.pth"}
        and bool(tokens & {"modelzoo", "weights", "pretrained", "pretrain", "unimol2"})
    )


def artifact_type(path: Path) -> str | None:
    """Return the supported artifact type, or ``None`` for an ignored file."""

    name = path.name.lower()
    stem = path.stem.lower()
    suffix = path.suffix.lower()
    if _is_feature_cache(path):
        return "feature_cache"
    if name == "model.joblib":
        return "trained_model"
    if suffix in _MODEL_SUFFIXES:
        # Avoid treating Python's distutils-precedence.pth as a model.
        if suffix == ".pth" and not (
            "unimol" in str(path).lower()
            or stem.startswith("model_")
            or "checkpoint" in stem
            or _is_base_checkpoint(path)
        ):
            return None
        return "base_checkpoint" if _is_base_checkpoint(path) else "unimol2_checkpoint"
    if "prediction" in stem and suffix in {".npz", ".npy", ".json", ".csv", ".parquet"}:
        return "predictions"
    if suffix == ".json" and (
        stem in {"result", "results"}
        or stem.endswith("_result")
        or stem.endswith("_results")
        or stem.startswith("search_seed_")
    ):
        return "results"
    if suffix in _CONFIG_SUFFIXES and any(
        token in stem
        for token in ("manifest", "finalist", "freeze", "mapping", "registry")
    ):
        return "manifest"
    if suffix in _CONFIG_SUFFIXES and (
        stem in {"config", "configuration", "protocol"}
        or stem.endswith("_config")
        or stem.endswith("_configuration")
        or stem.endswith("_protocol")
    ):
        return "config"
    return None


def _is_reference_document(path: Path) -> bool:
    """Select metadata documents without reading metric/result payloads."""

    if path.suffix.lower() != ".json":
        return False
    stem = path.stem.lower()
    return any(
        token in stem
        for token in ("manifest", "finalist", "freeze", "mapping", "config", "protocol", "registry")
    )


def _walk_strings(value: Any) -> Iterator[str]:
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for child in value.values():
            yield from _walk_strings(child)
    elif isinstance(value, list):
        for child in value:
            yield from _walk_strings(child)


def _display_path(path: Path, project_root: Path) -> str:
    try:
        return path.relative_to(project_root).as_posix()
    except ValueError:
        return path.as_posix()


def _classification_path(path: Path, project_root: Path, run_roots: Sequence[Path]) -> Path:
    """Remove unrelated host prefixes (for example ``/tmp``) from policy tokens."""

    try:
        return path.relative_to(project_root)
    except ValueError:
        pass
    for root in run_roots:
        try:
            return Path(root.name) / path.relative_to(root)
        except ValueError:
            continue
    # An artifact outside every declared scope should not normally be possible;
    # falling back to its basename is safer than inferring policy from /tmp or a
    # user's home-directory name.
    return Path(path.name)


def _iter_files(root: Path) -> Iterator[Path]:
    if root.is_file():
        yield root
        return
    if not root.is_dir():
        return
    for directory, dirnames, filenames in os.walk(root, followlinks=False):
        dirnames.sort()
        filenames.sort()
        base = Path(directory)
        for filename in filenames:
            candidate = base / filename
            if candidate.is_file() and not candidate.is_symlink():
                yield candidate


def _collect_artifacts(run_roots: Sequence[Path]) -> list[Path]:
    seen: set[Path] = set()
    artifacts: list[Path] = []
    for root in run_roots:
        for path in _iter_files(root):
            resolved = path.resolve()
            if resolved not in seen and artifact_type(resolved) is not None:
                seen.add(resolved)
                artifacts.append(resolved)
    return sorted(artifacts, key=lambda path: path.as_posix())


def _resolve_reference(value: str, document: Path, project_root: Path) -> Path | None:
    if not value or "\x00" in value or len(value) > 4096:
        return None
    raw = Path(value).expanduser()
    candidates = [raw] if raw.is_absolute() else [project_root / raw, document.parent / raw]
    for candidate in candidates:
        try:
            if candidate.is_file():
                return candidate.resolve()
        except OSError:
            continue
    return None


def _collect_references(
    reference_roots: Sequence[Path],
    project_root: Path,
    artifact_paths: set[Path],
) -> tuple[dict[Path, set[str]], list[str]]:
    references: dict[Path, set[str]] = {path: set() for path in artifact_paths}
    warnings: list[str] = []
    seen_documents: set[Path] = set()
    for root in reference_roots:
        for document in _iter_files(root):
            document = document.resolve()
            if document in seen_documents or not _is_reference_document(document):
                continue
            seen_documents.add(document)
            try:
                if document.stat().st_size > _REFERENCE_JSON_MAX_BYTES:
                    warnings.append(
                        f"reference document skipped because it exceeds {_REFERENCE_JSON_MAX_BYTES} bytes: "
                        f"{_display_path(document, project_root)}"
                    )
                    continue
                payload = json.loads(document.read_text(encoding="utf-8"))
            except (OSError, UnicodeError, json.JSONDecodeError) as exc:
                warnings.append(
                    f"reference document unreadable: {_display_path(document, project_root)} ({type(exc).__name__})"
                )
                continue
            for value in _walk_strings(payload):
                target = _resolve_reference(value, document, project_root)
                if target in artifact_paths and target != document:
                    references[target].add(_display_path(document, project_root))
    return references, warnings


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _classify(path: Path, kind: str, referenced: bool) -> tuple[RetentionTier, str]:
    tokens = _tokens(path)
    formal = bool(tokens & _FORMAL_TOKENS)
    cold = bool(tokens & _COLD_TOKENS)
    reclaim = bool(tokens & _RECLAIM_TOKENS)

    if kind in {"results", "predictions", "manifest", "config"}:
        return RetentionTier.EVIDENCE, "Result, prediction, configuration, or manifest required for reproducible evidence."
    if kind == "base_checkpoint":
        return RetentionTier.DEPLOYMENT, "Shared public/base checkpoint; retained and never proposed for deletion."
    if referenced and kind in {"trained_model", "unimol2_checkpoint"}:
        return RetentionTier.DEPLOYMENT, "Model checkpoint is directly referenced by retained metadata."
    if formal and kind in {"trained_model", "unimol2_checkpoint"}:
        return RetentionTier.DEPLOYMENT, "Checkpoint belongs to an official/final/frozen run."
    if referenced:
        return RetentionTier.EVIDENCE, "Artifact is directly referenced by retained metadata."
    if reclaim:
        return RetentionTier.RECLAIM_CANDIDATE, "Unreferenced smoke/screen/scratch artifact; review before reclamation."
    if cold:
        return RetentionTier.COLD_STORE, "Unreferenced pilot/ablation/shard artifact; preserve in cold storage before local removal."
    if formal:
        return RetentionTier.COLD_STORE, "Regenerable cache from a formal run; keep conservatively or archive after verification."
    return RetentionTier.DEBUG_TTL, "Unreferenced working artifact; retain under a future TTL policy pending review."


def build_retention_manifest(
    run_roots: Sequence[str | Path],
    *,
    project_root: str | Path | None = None,
    reference_roots: Sequence[str | Path] | None = None,
    hash_max_bytes: int | None = DEFAULT_HASH_MAX_BYTES,
    hash_files: bool = True,
) -> dict[str, Any]:
    """Build a non-destructive retention manifest for selected roots.

    ``hash_max_bytes=None`` hashes every selected file.  By default only small
    files are hashed, avoiding expensive reads of large checkpoints.  No file
    is ever modified, moved, or deleted.
    """

    if not run_roots:
        raise ValueError("at least one run root is required")
    if hash_max_bytes is not None and hash_max_bytes < 0:
        raise ValueError("hash_max_bytes must be non-negative or None")

    project = Path(project_root or Path.cwd()).expanduser().resolve()
    roots = tuple(Path(root).expanduser().resolve() for root in run_roots)
    refs = tuple(Path(root).expanduser().resolve() for root in (reference_roots or roots))
    missing_roots = [_display_path(root, project) for root in roots if not root.exists()]
    artifacts = _collect_artifacts(roots)
    reference_map, warnings = _collect_references(refs, project, set(artifacts))
    warnings.extend(f"run root does not exist: {root}" for root in missing_roots)

    records: list[dict[str, Any]] = []
    for path in artifacts:
        size = path.stat().st_size
        kind = artifact_type(path)
        assert kind is not None
        referenced_by = sorted(reference_map[path])
        tier, reason = _classify(
            _classification_path(path, project, roots), kind, bool(referenced_by)
        )
        digest: str | None = None
        if not hash_files:
            hash_status = "disabled"
        elif hash_max_bytes is not None and size > hash_max_bytes:
            hash_status = "skipped_size"
        else:
            try:
                digest = _sha256(path)
                hash_status = "computed"
            except OSError as exc:
                hash_status = f"error:{type(exc).__name__}"
                warnings.append(f"hash failed: {_display_path(path, project)} ({type(exc).__name__})")
        records.append(
            {
                "path": _display_path(path, project),
                "artifact_type": kind,
                "bytes": size,
                "sha256": digest,
                "sha256_status": hash_status,
                "retention_tier": tier.value,
                "reason": reason,
                "referenced": bool(referenced_by),
                "reference_status": "direct" if referenced_by else "unreferenced",
                "referenced_by": referenced_by,
                "reclaimable": tier is RetentionTier.RECLAIM_CANDIDATE,
            }
        )

    counts = Counter(record["retention_tier"] for record in records)
    bytes_by_tier = Counter()
    for record in records:
        bytes_by_tier[record["retention_tier"]] += record["bytes"]
    all_tiers = [tier.value for tier in RetentionTier]
    reclaimable = bytes_by_tier[RetentionTier.RECLAIM_CANDIDATE.value]
    cold_store = bytes_by_tier[RetentionTier.COLD_STORE.value]
    return {
        "schema_version": SCHEMA_VERSION,
        "mode": "dry_run_read_only",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "project_root": project.as_posix(),
        "run_roots": [_display_path(root, project) for root in roots],
        "reference_roots": [_display_path(root, project) for root in refs],
        "policy": {
            "mutates_files": False,
            "deletes_files": False,
            "large_file_hashing_default": False,
            "hash_files": hash_files,
            "hash_max_bytes": hash_max_bytes,
            "formal_finalists_reclaimable": False,
            "public_base_checkpoints_reclaimable": False,
            "cold_store_requires_verified_archive_before_local_removal": True,
        },
        "summary": {
            "artifact_count": len(records),
            "total_bytes": sum(record["bytes"] for record in records),
            "counts_by_tier": {tier: counts[tier] for tier in all_tiers},
            "bytes_by_tier": {tier: bytes_by_tier[tier] for tier in all_tiers},
            "estimated_reclaimable_bytes": reclaimable,
            "estimated_cold_store_bytes": cold_store,
            "estimated_local_release_after_approved_archival_bytes": reclaimable + cold_store,
        },
        "artifacts": records,
        "warnings": sorted(set(warnings)),
    }


def write_retention_manifest(manifest: dict[str, Any], output: str | Path) -> Path:
    """Write the JSON plan; artifact paths themselves remain untouched."""

    destination = Path(output).expanduser()
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return destination


__all__ = [
    "DEFAULT_HASH_MAX_BYTES",
    "RetentionTier",
    "artifact_type",
    "build_retention_manifest",
    "write_retention_manifest",
]
