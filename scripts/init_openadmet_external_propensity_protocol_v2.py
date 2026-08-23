"""Freeze protocol v2 after the v1 integral-float parser failure."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build(args: argparse.Namespace) -> dict[str, Any]:
    predecessor = json.loads(args.predecessor.read_text(encoding="utf-8"))
    predecessor_sha = predecessor.get("sha256")
    body = {key: value for key, value in predecessor.items() if key != "sha256"}
    if hashlib.sha256(json.dumps(body, sort_keys=True).encode()).hexdigest() != predecessor_sha:
        raise RuntimeError("Predecessor protocol hash is invalid")
    implementation = Path(predecessor["implementation_path"])
    if _sha256(implementation) != predecessor["implementation_source_sha256"]:
        raise RuntimeError("Frozen v1 implementation has changed")
    failure_audit = json.loads(args.failure_audit.read_text(encoding="utf-8"))
    if failure_audit.get("protocol_sha256") != predecessor_sha:
        raise RuntimeError("Failure audit does not identify the predecessor protocol")
    if failure_audit.get("model_fits_started") is not False:
        raise RuntimeError("The v1 failure was not pre-training")
    payload = {
        **body,
        "version": "openadmet-cyp-external-propensity-protocol-v2",
        "compatibility_adapter_path": str(args.adapter.resolve()),
        "compatibility_adapter_sha256": _sha256(args.adapter.resolve()),
        "compatibility_adapter_scope": "integral_float_csv_fold_parser_only",
        "aborted_predecessor_protocol_sha256": predecessor_sha,
        "aborted_predecessor_failure_audit_sha256": _sha256(args.failure_audit),
        "scientific_design_changed_from_predecessor": False,
    }
    payload["sha256"] = hashlib.sha256(
        json.dumps(payload, sort_keys=True).encode()
    ).hexdigest()
    args.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--predecessor",
        type=Path,
        default=Path("reports/openadmet_external_propensity_protocol_v1.json"),
    )
    parser.add_argument(
        "--failure-audit",
        type=Path,
        default=Path("reports/openadmet_external_propensity_v1_aborted_audit.json"),
    )
    parser.add_argument(
        "--adapter",
        type=Path,
        default=Path("scripts/run_openadmet_external_propensity_bridge_v2.py"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("reports/openadmet_external_propensity_protocol_v2.json"),
    )
    return parser.parse_args()


if __name__ == "__main__":
    print(json.dumps(build(parse_args()), indent=2, sort_keys=True))
