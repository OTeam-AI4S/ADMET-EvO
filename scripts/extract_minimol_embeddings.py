"""Extract MiniMol fingerprints from an unlabeled SMILES-only manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
from importlib.metadata import distribution, version
from pathlib import Path

import numpy as np
import torch
from minimol import Minimol


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=128)
    args = parser.parse_args()
    payload = json.loads(args.input.read_text(encoding="utf-8"))
    if payload.get("contains_labels") is not False:
        raise RuntimeError("MiniMol extractor accepts only a label-free manifest")
    if payload.get("official_test_loaded") is not False:
        raise RuntimeError("MiniMol extractor refused a test-bearing manifest")
    model = Minimol(batch_size=args.batch_size)
    fingerprints = model(payload["smiles"])
    embeddings = np.stack(
        [value.detach().cpu().numpy() for value in fingerprints]
    ).astype(np.float32)
    if embeddings.shape != (len(payload["smiles"]), 512):
        raise RuntimeError(f"Unexpected MiniMol embedding shape: {embeddings.shape}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.save(args.output, embeddings)
    weight_path = (
        Path(__file__).resolve().parents[1]
        / ".venvs"
        / "minimol"
        / "lib"
        / "python3.11"
        / "site-packages"
        / "minimol"
        / "ckpts"
        / "minimol_v1"
        / "state_dict.pth"
    )
    checkpoint_dir = weight_path.parent
    minimol_distribution = distribution("minimol")
    license_files = [
        Path(minimol_distribution.locate_file(item))
        for item in (minimol_distribution.files or [])
        if str(item).endswith(".dist-info/LICENSE")
    ]
    if len(license_files) != 1:
        raise RuntimeError("Expected exactly one packaged MiniMol license file")
    license_path = license_files[0]
    direct_url_files = [
        Path(minimol_distribution.locate_file(item))
        for item in (minimol_distribution.files or [])
        if str(item).endswith(".dist-info/direct_url.json")
    ]
    direct_url = (
        json.loads(direct_url_files[0].read_text(encoding="utf-8"))
        if len(direct_url_files) == 1
        else None
    )
    manifest = {
        "task": payload["task"],
        "rows": embeddings.shape[0],
        "columns": embeddings.shape[1],
        "input_sha256": sha256_file(args.input),
        "output_sha256": sha256_file(args.output),
        "weight_sha256": sha256_file(weight_path),
        "checkpoint_config_sha256": sha256_file(checkpoint_dir / "config.yaml"),
        "checkpoint_base_shape_sha256": sha256_file(
            checkpoint_dir / "base_shape.yaml"
        ),
        "license": "MIT",
        "license_sha256": sha256_file(license_path),
        "package_archive_sha256": (
            direct_url.get("archive_info", {}).get("hashes", {}).get("sha256")
            if direct_url is not None
            else None
        ),
        "minimol": version("minimol"),
        "graphium": version("graphium"),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "gpu_available": torch.cuda.is_available(),
        "official_test_loaded": False,
        "labels_received": False,
    }
    args.output.with_suffix(".manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps(manifest, sort_keys=True))


if __name__ == "__main__":
    main()
