"""Migrate eligible v1 matched effects into explicit EC-VOI v2 observations."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

FROZEN_EFFECT_STANDARDIZATION_SCALE = 0.005
FROZEN_STANDARD_ERROR_FLOOR_FRACTION = 0.10


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def _valid_sha256(value: Any) -> bool:
    return bool(
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def migrate(
    source: Path,
    *,
    effect_standardization_scale: float = FROZEN_EFFECT_STANDARDIZATION_SCALE,
    standard_error_floor_fraction: float = FROZEN_STANDARD_ERROR_FLOOR_FRACTION,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if not math.isfinite(effect_standardization_scale) or effect_standardization_scale <= 0:
        raise ValueError("Effect standardization scale must be positive")
    if (
        not math.isfinite(standard_error_floor_fraction)
        or not 0 < standard_error_floor_fraction <= 1
    ):
        raise ValueError("Standard-error floor fraction must be within (0, 1]")
    if not math.isclose(
        effect_standardization_scale, FROZEN_EFFECT_STANDARDIZATION_SCALE
    ):
        raise ValueError("v1 migration scale is frozen at 0.005")
    if not math.isclose(
        standard_error_floor_fraction, FROZEN_STANDARD_ERROR_FLOOR_FRACTION
    ):
        raise ValueError("v1 migration standard-error floor fraction is frozen at 0.10")
    source_bytes = source.read_bytes()
    input_rows = [
        json.loads(line)
        for line in source_bytes.decode().splitlines()
        if line.strip()
    ]
    admitted: list[dict[str, Any]] = []
    exclusions: dict[str, int] = {}

    def exclude(reason: str) -> None:
        exclusions[reason] = exclusions.get(reason, 0) + 1

    for row in input_rows:
        if row.get("evidence_protocol") != "discovery_only_matched_action_v1":
            exclude("unsupported_evidence_protocol")
            continue
        if row.get("transport_eligible") is not True:
            exclude("not_transport_eligible")
            continue
        if row.get("single_axis_verified") is not True:
            exclude("single_axis_not_verified")
            continue
        paired_seeds = int(row.get("paired_seeds", 0))
        if paired_seeds < 3:
            exclude("insufficient_paired_seeds")
            continue
        if not _valid_sha256(row.get("record_sha256")) or not _valid_sha256(
            row.get("transport_protocol_sha256")
        ):
            exclude("invalid_provenance_hash")
            continue
        mean = row.get("mean_oriented_delta")
        delta_std = row.get("delta_std")
        if mean is None or delta_std is None:
            exclude("matched_effect_statistics_missing")
            continue
        mean = float(mean)
        delta_std = float(delta_std)
        if not math.isfinite(mean) or not math.isfinite(delta_std) or delta_std < 0:
            exclude("matched_effect_statistics_invalid")
            continue
        empirical_se = delta_std / math.sqrt(paired_seeds)
        se_floor = standard_error_floor_fraction * effect_standardization_scale
        native_se = max(empirical_se, se_floor)
        base = {
            "schema_version": "ecvoi-v2-matched-effect-observation-v1",
            "endpoint_id": str(row["task"]),
            "pair_id": f"legacy-grid:{row['record_sha256']}",
            "action_id": str(row["action_id"]),
            "axis": str(row["axis"]),
            "native_oriented_delta": mean,
            "native_standard_error": native_se,
            "empirical_native_standard_error": empirical_se,
            "standard_error_floor": se_floor,
            "effect_standardization_scale": effect_standardization_scale,
            "standardization_basis": "frozen_v1_minimum_practical_effect_0.005",
            "paired_seeds": paired_seeds,
            "single_axis_verified": True,
            "complete_pair": True,
            "transport_eligible": True,
            "transport_scope": row.get("transport_scope"),
            "transport_protocol_sha256": row["transport_protocol_sha256"],
            "test_derived": False,
            "leaderboard_derived": False,
            "source_record_sha256": row["record_sha256"],
        }
        base["observation_sha256"] = _sha256_bytes(_canonical(base))
        admitted.append(base)
    keys = [(row["endpoint_id"], row["action_id"]) for row in admitted]
    if len(keys) != len(set(keys)):
        raise ValueError("Source contains duplicate endpoint/action observations")
    admitted.sort(key=lambda row: (row["endpoint_id"], row["action_id"]))
    output_bytes = b"".join(_canonical(row) + b"\n" for row in admitted)
    manifest = {
        "migration_version": "action-memory-v1-to-ecvoi-v2-v1",
        "source_path": str(source),
        "source_sha256": _sha256_bytes(source_bytes),
        "input_records": len(input_rows),
        "admitted_records": len(admitted),
        "excluded_records": len(input_rows) - len(admitted),
        "exclusions_by_reason": dict(sorted(exclusions.items())),
        "effect_standardization_scale": effect_standardization_scale,
        "standard_error_floor_fraction": standard_error_floor_fraction,
        "output_sha256": _sha256_bytes(output_bytes),
        "test_loaded": False,
        "leaderboard_loaded": False,
    }
    manifest["manifest_sha256"] = _sha256_bytes(_canonical(manifest))
    return admitted, manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument(
        "--effect-standardization-scale",
        type=float,
        default=FROZEN_EFFECT_STANDARDIZATION_SCALE,
    )
    parser.add_argument(
        "--standard-error-floor-fraction",
        type=float,
        default=FROZEN_STANDARD_ERROR_FLOOR_FRACTION,
    )
    args = parser.parse_args()
    rows, manifest = migrate(
        args.input,
        effect_standardization_scale=args.effect_standardization_scale,
        standard_error_floor_fraction=args.standard_error_floor_fraction,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    output = b"".join(_canonical(row) + b"\n" for row in rows)
    args.output.write_bytes(output)
    if _sha256_bytes(args.output.read_bytes()) != manifest["output_sha256"]:
        raise RuntimeError("Written v2 observation snapshot hash mismatch")
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
