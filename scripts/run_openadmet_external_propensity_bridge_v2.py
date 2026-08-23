"""Serialization-only adapter for the frozen external-propensity v1 trial.

The v1 implementation interprets a pandas float column such as ``2.0`` with
``int('2.0')`` and therefore aborts before model fitting.  This adapter changes
only that parser, binds itself in protocol v2, and delegates every scientific
operation to the still-frozen v1 implementation.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

import pandas as pd

from scripts import run_openadmet_external_propensity_bridge as v1


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def excluded_outer_folds(value: Any) -> set[int]:
    if pd.isna(value) or str(value).strip() == "":
        return set()
    folds = set()
    for token in str(value).split(";"):
        token = token.strip()
        if not token:
            continue
        parsed = float(token)
        if not math.isfinite(parsed) or not parsed.is_integer():
            raise ValueError(f"Invalid integral outer-fold token: {token!r}")
        fold = int(parsed)
        if fold not in {0, 1, 2, 3, 4}:
            raise ValueError(f"Outer-fold token is outside [0, 4]: {fold}")
        folds.add(fold)
    return folds


def run() -> dict[str, Any]:
    args = v1.parse_args()
    protocol = v1._protocol(args.protocol.resolve())
    adapter_path = Path(__file__).resolve()
    if protocol.get("compatibility_adapter_sha256") != _sha256(adapter_path):
        raise RuntimeError("External propensity compatibility-adapter mismatch")
    if protocol.get("compatibility_adapter_scope") != "integral_float_csv_fold_parser_only":
        raise RuntimeError("External propensity adapter scope is not frozen")
    v1._excluded_outer_folds = excluded_outer_folds
    result = v1.run(args)
    result["version"] = "openadmet-cyp-external-propensity-bridge-v2"
    result["compatibility_adapter_sha256"] = _sha256(adapter_path)
    result["aborted_predecessor_protocol_sha256"] = protocol[
        "aborted_predecessor_protocol_sha256"
    ]
    result_path = Path(result["result_path"])
    persisted = {key: value for key, value in result.items() if key != "result_path"}
    result_path.write_text(
        json.dumps(persisted, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return result


if __name__ == "__main__":
    print(json.dumps(run(), indent=2, sort_keys=True))
