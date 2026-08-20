"""Extract MiniMol fingerprints from an unlabeled SMILES-only manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
from importlib.metadata import version
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
    manifest = {
        "task": payload["task"],
        "rows": embeddings.shape[0],
        "columns": embeddings.shape[1],
        "input_sha256": sha256_file(args.input),
        "weight_sha256": sha256_file(weight_path),
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
