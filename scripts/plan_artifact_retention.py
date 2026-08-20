#!/usr/bin/env python3
"""Generate a read-only artifact retention plan (never deletes or moves files)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from admet_agent.artifact_retention import (
    DEFAULT_HASH_MAX_BYTES,
    build_retention_manifest,
    write_retention_manifest,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "roots",
        nargs="*",
        type=Path,
        help="Run roots to scan (default: runs).",
    )
    parser.add_argument(
        "--run-root",
        action="append",
        default=[],
        type=Path,
        help="Additional run root; repeat for multiple roots.",
    )
    parser.add_argument(
        "--reference-root",
        action="append",
        default=[],
        type=Path,
        help="Root containing freeze/finalist/manifests used to detect references; repeatable.",
    )
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("reports/artifact_retention_dry_run.json"),
    )
    hashing = parser.add_mutually_exclusive_group()
    hashing.add_argument("--no-hash", action="store_true", help="Do not hash any artifact.")
    hashing.add_argument("--hash-all", action="store_true", help="Hash all artifacts, including large checkpoints.")
    parser.add_argument(
        "--hash-max-bytes",
        type=int,
        default=DEFAULT_HASH_MAX_BYTES,
        help=f"Hash files no larger than this many bytes (default: {DEFAULT_HASH_MAX_BYTES}).",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    roots = [*args.roots, *args.run_root] or [Path("runs")]
    hash_max_bytes = None if args.hash_all else args.hash_max_bytes
    manifest = build_retention_manifest(
        roots,
        project_root=args.project_root,
        reference_roots=args.reference_root or roots,
        hash_max_bytes=hash_max_bytes,
        hash_files=not args.no_hash,
    )
    destination = write_retention_manifest(manifest, args.output)
    print(
        json.dumps(
            {
                "mode": manifest["mode"],
                "output": str(destination),
                **manifest["summary"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
