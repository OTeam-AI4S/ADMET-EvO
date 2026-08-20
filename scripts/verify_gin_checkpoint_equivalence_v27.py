"""Verify tensor-level equivalence of DGL and Molfeat GIN serializations."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


OFFICIAL_URL = "https://data.dgl.ai/dgllife/pre_trained/gin_supervised_masking.pth"
MOLFEAT_URL = "https://fs.molfeat.datamol.io/artifacts/dgllife/gin_supervised_masking/0/model.save"
OFFICIAL_SHA256 = "a43274c29be5a8cf38f990794d0dc3fcc0572dfa5524240e77ed1c6aa1b54fb8"
MOLFEAT_SHA256 = "6d0f8febad73e437772ebffc2ac32253d79f86ee138cfc233590ae50fb1cfeb9"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify(official_path: Path, molfeat_path: Path) -> dict[str, Any]:
    # Imports are intentionally local: ordinary evidence-manifest tests should
    # not require the isolated DGL/PyTorch environment.
    import joblib
    import torch

    official_hash = _sha256(official_path)
    molfeat_hash = _sha256(molfeat_path)
    if official_hash != OFFICIAL_SHA256:
        raise ValueError(f"Unexpected official DGL checkpoint hash: {official_hash}")
    if molfeat_hash != MOLFEAT_SHA256:
        raise ValueError(f"Unexpected Molfeat checkpoint hash: {molfeat_hash}")

    # Both inputs are trusted upstream pickle-based formats and must only be
    # loaded after the content hashes above have been checked.
    official = torch.load(official_path, map_location="cpu")
    official_state = official.get("model_state_dict", official)
    molfeat_model = joblib.load(molfeat_path)
    molfeat_state = molfeat_model.state_dict()
    official_keys = set(official_state)
    molfeat_keys = set(molfeat_state)
    same_keys = official_keys == molfeat_keys
    common = sorted(official_keys & molfeat_keys)
    same_shapes = same_keys and all(
        tuple(official_state[key].shape) == tuple(molfeat_state[key].shape)
        for key in common
    )
    exact = same_shapes and all(
        torch.equal(official_state[key], molfeat_state[key]) for key in common
    )
    max_abs_diff = (
        max(
            float((official_state[key] - molfeat_state[key]).abs().max())
            for key in common
        )
        if common and same_shapes
        else None
    )
    if not exact or max_abs_diff != 0.0:
        raise ValueError("DGL and Molfeat GIN parameter tensors are not exactly equal")

    return {
        "schema_version": "gin-checkpoint-equivalence-v27-1",
        "observed_at": "2026-08-17",
        "official_dgllife": {
            "url": OFFICIAL_URL,
            "sha256": official_hash,
            "bytes": official_path.stat().st_size,
            "state_dict_key_count": len(official_state),
        },
        "molfeat": {
            "url": MOLFEAT_URL,
            "sha256": molfeat_hash,
            "bytes": molfeat_path.stat().st_size,
            "state_dict_key_count": len(molfeat_state),
            "deserialized_type": (
                f"{type(molfeat_model).__module__}.{type(molfeat_model).__name__}"
            ),
        },
        "comparison": {
            "same_parameter_keys": same_keys,
            "same_parameter_shapes": same_shapes,
            "all_parameter_tensors_bitwise_equal": exact,
            "maximum_absolute_parameter_difference": max_abs_diff,
            "parameter_tensor_count": len(common),
            "parameter_value_count": sum(
                int(official_state[key].numel()) for key in common
            ),
            "serialization_bytes_expected_to_differ": True,
        },
        "provenance": {
            "dgllife_version": "0.3.2",
            "official_url_declaration": (
                "https://github.com/awslabs/dgl-lifesci/blob/v0.3.2/"
                "python/dgllife/model/pretrain/property_prediction.py"
            ),
            "official_downloader": (
                "https://github.com/awslabs/dgl-lifesci/blob/v0.3.2/"
                "python/dgllife/model/pretrain/__init__.py"
            ),
            "source_spdx": "Apache-2.0",
            "artifact_specific_license_metadata_present": False,
        },
        "verified": True,
        "official_test_loaded": False,
        "selection_or_freeze_changed": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--official", type=Path, required=True)
    parser.add_argument("--molfeat", type=Path, required=True)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("reports/gin_checkpoint_equivalence_v27.json"),
    )
    args = parser.parse_args()
    report = verify(args.official, args.molfeat)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps(report["comparison"], sort_keys=True))


if __name__ == "__main__":
    main()
