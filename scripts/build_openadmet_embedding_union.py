"""Build a label-free MiniMol manifest for main training plus bridge identities."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build(args: argparse.Namespace) -> dict:
    main_path = args.main.resolve()
    main = json.loads(main_path.read_text())
    if main.get("contains_labels") is not False:
        raise RuntimeError("Main embedding manifest is not label-free")
    if main.get("official_test_loaded") is not False:
        raise RuntimeError("Main embedding manifest indicates blind-test access")
    identities = dict(zip(main["inchikeys"], main["smiles"], strict=True))
    main_identities = set(identities)
    representation_reconciliations: set[str] = set()
    parents = {"main": {"path": str(main_path), "sha256": _sha256(main_path)}}

    if args.bridge is not None:
        bridge_path = args.bridge.resolve()
        bridge_manifest_path = bridge_path.with_suffix(".manifest.json")
        bridge_manifest = json.loads(bridge_manifest_path.read_text())
        if bridge_manifest.get("training_only") is not True:
            raise RuntimeError("Bridge manifest is not training-only")
        if bridge_manifest.get("blind_test_opened") is not False:
            raise RuntimeError("Bridge manifest indicates blind-test access")
        if bridge_manifest["output_sha256"] != _sha256(bridge_path):
            raise RuntimeError("Bridge table hash mismatch")
        bridge = pd.read_csv(bridge_path)
        for identity, smiles in bridge[
            ["inchikey", "standard_smiles"]
        ].drop_duplicates().itertuples(index=False, name=None):
            existing = identities.get(str(identity))
            if existing is not None and existing != str(smiles):
                if str(identity) not in main_identities:
                    raise RuntimeError(
                        "Bridge canonical identity maps to conflicting SMILES"
                    )
                # Both parent tables have already been standardized to the
                # same frozen InChIKey.  Preserve the main-training SMILES so
                # an auxiliary source cannot replace the primary molecular
                # representation merely because one source retained optional
                # bond-stereo syntax and the other did not.
                representation_reconciliations.add(str(identity))
                continue
            identities[str(identity)] = str(smiles)
        parents["bridge"] = {
            "path": str(bridge_path),
            "sha256": _sha256(bridge_path),
            "manifest_path": str(bridge_manifest_path.resolve()),
            "manifest_sha256": _sha256(bridge_manifest_path),
            "shared_identity_representation_policy": (
                "prefer_main_training_smiles_for_equal_frozen_inchikey"
            ),
            "shared_identity_representation_reconciliations": len(
                representation_reconciliations
            ),
            "reconciled_identity_set_sha256": hashlib.sha256(
                json.dumps(sorted(representation_reconciliations)).encode()
            ).hexdigest(),
        }

    ordered = sorted(identities)
    payload = {
        "task": "OpenADMET_CYP_training_and_admitted_bridge_identity_union",
        "contains_labels": False,
        "official_test_loaded": False,
        "inchikeys": ordered,
        "smiles": [identities[identity] for identity in ordered],
        "identities": len(ordered),
        "parents": parents,
        "test_labels_loaded": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return {**payload, "output_path": str(args.output.resolve()), "output_sha256": _sha256(args.output)}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--main",
        type=Path,
        default=Path("reports/openadmet_cyp_training_longform_v1.minimol_input.json"),
    )
    parser.add_argument("--bridge", type=Path)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("reports/openadmet_cyp_training_bridge_minimol_input_v1.json"),
    )
    return parser.parse_args()


if __name__ == "__main__":
    print(json.dumps(build(parse_args()), indent=2, sort_keys=True))
