"""Immutable production-source snapshots for reproducible benchmark runs."""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path
from typing import Any


def _hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def production_source_manifest() -> dict[str, Any]:
    package_dir = Path(__file__).resolve().parent
    project_dir = package_dir.parent
    paths = sorted(package_dir.glob("*.py"))
    if (project_dir / "pyproject.toml").exists():
        paths.append(project_dir / "pyproject.toml")
    files = {
        str(path.relative_to(project_dir)): _hash(path)
        for path in sorted(paths)
    }
    combined = hashlib.sha256(
        json.dumps(files, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return {
        "protocol": "admet-agent-production-source-v1",
        "project_dir": str(project_dir),
        "files": files,
        "source_sha256": combined,
    }


def snapshot_source(run_dir: Path) -> dict[str, Any]:
    """Copy production sources once and reject code drift in the same run dir."""
    current = production_source_manifest()
    snapshot_dir = run_dir / "source_snapshot"
    manifest_path = snapshot_dir / "manifest.json"
    if manifest_path.exists():
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        if existing.get("source_sha256") != current["source_sha256"]:
            raise RuntimeError(
                "Source firewall refused code drift in an existing run directory"
            )
        return existing
    snapshot_dir.mkdir(parents=True, exist_ok=False)
    project_dir = Path(current["project_dir"])
    for relative in current["files"]:
        source = project_dir / relative
        target = snapshot_dir / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
    manifest = {**current, "project_dir": "."}
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )
    return manifest
