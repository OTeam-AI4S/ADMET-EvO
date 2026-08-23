"""Build a training-only, identity-locked OpenADMET CYP split manifest.

The script never opens the blinded test table.  It assigns the union of all
training molecules to one sealed certification partition or one shared outer
fold, so observations from direct inhibition, TDI, Emax and single-
concentration assays cannot cross partitions under alternate raw SMILES.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path

import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import rdFingerprintGenerator
from rdkit.Chem.Scaffolds import MurckoScaffold

from admet_agent.external_structure_firewall import _standard_molecule

DEFAULT_INPUTS = {
    "direct": "external/openadmet/cyp_challenge_data/cyp-challenge-TRAIN_inhibition.csv",
    "tdi": "external/openadmet/cyp_challenge_data/cyp-challenge-TRAIN_TDI.csv",
    "emax": "external/openadmet/cyp_challenge_data/cyp-challenge-TRAIN_Emax.csv",
    "single": "external/openadmet/cyp_challenge_data/cyp-challenge-single-concentration-TRAIN.csv",
}
EXPECTED_SHA256 = {
    "direct": "b8f79addd266fb6f9f4c222c5e4e73d926362328b6a8d2841871a54e46bd2278",
    "tdi": "b458f599a792412292664386e8f18adc5d4a4129d6bd212ae80a60fb9b96bb60",
    "emax": "70e9eb41f2401d8c79b684c42da3721a7aa4fd5c7f561636025c5b91cc29a3b9",
    "single": "cf275440aa33d10b16e7a3a19f1b196d59dc379052175ab79b47ca43db663c7a",
}
DIRECT_TARGETS = tuple(
    f"{isoform}_pIC50_direct_inhibition"
    for isoform in ("CYP1A2", "CYP2C9", "CYP2D6", "CYP3A4")
)
TDI_TARGETS = ("CYP2D6_is_TDI", "CYP3A4_is_TDI")
TDI_PIC50_TARGETS = tuple(
    f"{isoform}_pIC50_TDI_condition"
    for isoform in ("CYP1A2", "CYP2C9", "CYP2D6", "CYP3A4")
)
EMAX_TARGETS = tuple(
    f"{isoform}_EmaxVsPosCtrl_{condition}"
    for isoform in ("CYP1A2", "CYP2C9", "CYP2D6", "CYP3A4")
    for condition in ("direct_inhibition", "TDI_condition")
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _stable_digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _acyclic_family(mol: Chem.Mol) -> str:
    generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    bits = list(generator.GetFingerprint(mol).GetOnBits())
    if not bits:
        return "acyclic-empty"
    ordered = sorted(
        (_stable_digest(f"openadmet-acyclic-minhash-v1:{bit}"), bit)
        for bit in bits
    )
    # Two deterministic MinHash values create a conservative LSH family.  This
    # is used only when a molecule has no Murcko scaffold.
    return "acyclic-minhash:" + ":".join(str(bit) for _, bit in ordered[:2])


def _chemistry_group(standard_smiles: str) -> str:
    mol = Chem.MolFromSmiles(standard_smiles)
    if mol is None:
        raise ValueError("Standardized SMILES unexpectedly failed RDKit parsing")
    scaffold = MurckoScaffold.GetScaffoldForMol(mol)
    if scaffold.GetNumAtoms() == 0:
        return _acyclic_family(mol)
    generic = MurckoScaffold.MakeScaffoldGeneric(scaffold)
    value = Chem.MolToSmiles(generic, canonical=True, isomericSmiles=False)
    return f"generic-murcko:{value}"


def _freeze_partitions(
    rows: pd.DataFrame,
    *,
    certification_fraction: float,
    seed: int,
    outer_folds: int,
    inner_folds: int,
) -> pd.DataFrame:
    sizes = rows.groupby("chemistry_group").size().to_dict()
    groups = sorted(sizes, key=lambda group: _stable_digest(f"cert:{seed}:{group}"))
    minimum = math.ceil(len(rows) * 0.10)
    target = round(len(rows) * certification_fraction)
    maximum = math.floor(len(rows) * 0.15)
    certification: set[str] = set()
    count = 0
    for group in groups:
        size = int(sizes[group])
        if count >= target:
            break
        # A chemistry group is indivisible.  Skip groups that would make the
        # sealed set violate its preregistered upper bound rather than silently
        # splitting a scaffold family or accepting an oversized holdout.
        if count + size <= maximum:
            certification.add(group)
            count += size
    if count < minimum:
        raise RuntimeError(
            "No whole-chemistry-group certification assignment exists within "
            f"[10%, 15%] under seed {seed}: selected {count}/{len(rows)} rows"
        )

    development = [group for group in groups if group not in certification]
    fold_sizes = [0] * outer_folds
    outer_assignment: dict[str, int] = {}
    ordered = sorted(
        development,
        key=lambda group: (-int(sizes[group]), _stable_digest(f"outer:{seed}:{group}")),
    )
    for group in ordered:
        fold = min(
            range(outer_folds),
            key=lambda index: (fold_sizes[index], index),
        )
        outer_assignment[group] = fold
        fold_sizes[fold] += int(sizes[group])

    output = rows.copy()
    output["partition"] = output["chemistry_group"].map(
        lambda group: "certification" if group in certification else "development"
    )
    output["outer_fold"] = output["chemistry_group"].map(
        lambda group: -1 if group in certification else outer_assignment[group]
    )
    for outer in range(outer_folds):
        column = f"inner_fold_when_outer_{outer}"
        inner_sizes = [0] * inner_folds
        inner_assignment: dict[str, int] = {}
        eligible = [
            group
            for group in development
            if outer_assignment[group] != outer
        ]
        inner_ordered = sorted(
            eligible,
            key=lambda group: (
                -int(sizes[group]),
                _stable_digest(f"inner:{seed}:{outer}:{group}"),
            ),
        )
        for group in inner_ordered:
            fold = min(
                range(inner_folds),
                key=lambda index: (inner_sizes[index], index),
            )
            inner_assignment[group] = fold
            inner_sizes[fold] += int(sizes[group])
        fold_map = {
            group: (
                -1
                if group in certification or outer_assignment[group] == outer
                else inner_assignment[group]
            )
            for group in groups
        }
        output[column] = output["chemistry_group"].map(fold_map)
    return output


def _availability_by_partition(
    frames: dict[str, pd.DataFrame],
    raw_to_identity: dict[str, str],
    split: pd.DataFrame,
) -> dict[str, dict[str, int]]:
    identity_partition = split.set_index("inchikey")["partition"].to_dict()
    identity_outer = split.set_index("inchikey")["outer_fold"].to_dict()
    result: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    target_sets = (
        ("direct", DIRECT_TARGETS, ""),
        ("tdi", TDI_TARGETS + TDI_PIC50_TARGETS, ""),
        ("tdi", DIRECT_TARGETS, "tdi_paired_direct:"),
        ("emax", EMAX_TARGETS, ""),
    )
    for source, targets, key_prefix in target_sets:
        frame = frames[source]
        for target in targets:
            if target not in frame:
                continue
            for raw_smiles in frame.loc[frame[target].notna(), "SMILES"].astype(str):
                identity = raw_to_identity[raw_smiles]
                partition = identity_partition[identity]
                key = (
                    "certification"
                    if partition == "certification"
                    else f"outer_{int(identity_outer[identity])}"
                )
                result[f"{key_prefix}{target}"][key] += 1
    single = frames["single"]
    if {"enzyme", "log2fc_estimate"}.issubset(single.columns):
        labeled = single.loc[
            single["log2fc_estimate"].notna(), ["SMILES", "enzyme"]
        ]
        for raw_smiles, enzyme in labeled.itertuples(index=False, name=None):
            identity = raw_to_identity[str(raw_smiles)]
            partition = identity_partition[identity]
            key = (
                "certification"
                if partition == "certification"
                else f"outer_{int(identity_outer[identity])}"
            )
            target = f"{enzyme}_single_concentration_log2fc_estimate"
            result[target][key] += 1
    return {target: dict(counts) for target, counts in sorted(result.items())}


def build(args: argparse.Namespace) -> dict:
    if not 0.10 <= args.certification_fraction <= 0.15:
        raise ValueError("certification_fraction must be within [0.10, 0.15]")
    paths = {name: Path(getattr(args, name)).resolve() for name in DEFAULT_INPUTS}
    for name, path in paths.items():
        digest = _sha256(path)
        if digest != EXPECTED_SHA256[name]:
            raise RuntimeError(f"Frozen input hash mismatch for {name}: {digest}")

    frames = {name: pd.read_csv(path) for name, path in paths.items()}
    all_raw = sorted(
        {
            str(value)
            for frame in frames.values()
            for value in frame["SMILES"].dropna()
        }
    )
    RDLogger.DisableLog("rdApp.warning")
    standardized = {raw: _standard_molecule(raw) for raw in all_raw}
    invalid = sorted(raw for raw, molecule in standardized.items() if molecule is None)
    if invalid:
        raise RuntimeError(f"Invalid training structures after standardization: {len(invalid)}")
    raw_to_identity = {
        raw: molecule.inchikey
        for raw, molecule in standardized.items()
        if molecule is not None
    }
    identities: dict[str, dict] = {}
    sources: dict[str, set[str]] = defaultdict(set)
    raw_forms: dict[str, set[str]] = defaultdict(set)
    for source, frame in frames.items():
        for raw in frame["SMILES"].dropna().astype(str).unique():
            molecule = standardized[raw]
            assert molecule is not None
            sources[molecule.inchikey].add(source)
            raw_forms[molecule.inchikey].add(raw)
            identities[molecule.inchikey] = {
                "inchikey": molecule.inchikey,
                "standard_smiles": molecule.smiles,
            }

    rows = pd.DataFrame(
        [
            {
                **identities[identity],
                "chemistry_group": _chemistry_group(
                    identities[identity]["standard_smiles"]
                ),
                "source_modalities": ";".join(sorted(sources[identity])),
                "raw_smiles_forms": len(raw_forms[identity]),
            }
            for identity in sorted(identities)
        ]
    )
    split = _freeze_partitions(
        rows,
        certification_fraction=args.certification_fraction,
        seed=args.seed,
        outer_folds=args.outer_folds,
        inner_folds=args.inner_folds,
    ).sort_values("inchikey", kind="stable").reset_index(drop=True)
    actual_certification_fraction = float(
        (split["partition"] == "certification").mean()
    )
    if not 0.10 <= actual_certification_fraction <= 0.15:
        raise RuntimeError(
            "Whole-group certification assignment fell outside [0.10, 0.15]: "
            f"{actual_certification_fraction:.6f}"
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    split.to_csv(args.output, index=False)
    output_sha = _sha256(args.output)
    manifest = {
        "version": "openadmet-cyp-identity-locked-splits-v1",
        "training_only": True,
        "blind_test_opened": False,
        "seed": args.seed,
        "outer_folds": args.outer_folds,
        "inner_folds": args.inner_folds,
        "requested_certification_fraction": args.certification_fraction,
        "actual_certification_fraction": actual_certification_fraction,
        "identity_method": "external_structure_firewall._standard_molecule",
        "chemistry_group_method": "generic_murcko_else_two_value_morgan_minhash_v1",
        "inputs": {
            name: {"path": str(paths[name]), "sha256": EXPECTED_SHA256[name]}
            for name in sorted(paths)
        },
        "raw_unique_smiles_union": len(all_raw),
        "canonical_identities": len(split),
        "canonical_identity_collisions": len(all_raw) - len(split),
        "chemistry_groups": int(split["chemistry_group"].nunique()),
        "certification_identities": int((split["partition"] == "certification").sum()),
        "development_identities": int((split["partition"] == "development").sum()),
        "outer_fold_identity_counts": {
            str(int(fold)): int(count)
            for fold, count in split.loc[
                split["partition"] == "development", "outer_fold"
            ].value_counts().sort_index().items()
        },
        "label_availability": _availability_by_partition(
            frames, raw_to_identity, split
        ),
        "label_availability_unit": "non_missing_observations",
        "output": {"path": str(args.output.resolve()), "sha256": output_sha},
    }
    manifest_path = args.output.with_suffix(".manifest.json")
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True))
    return {**manifest, "manifest_path": str(manifest_path.resolve())}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    for name, default in DEFAULT_INPUTS.items():
        parser.add_argument(f"--{name}", default=default)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("reports/openadmet_cyp_identity_locked_splits_v1.csv"),
    )
    parser.add_argument("--seed", type=int, default=20260821)
    parser.add_argument("--certification-fraction", type=float, default=0.125)
    parser.add_argument("--outer-folds", type=int, default=5)
    parser.add_argument("--inner-folds", type=int, default=4)
    return parser.parse_args()


if __name__ == "__main__":
    print(json.dumps(build(parse_args()), indent=2, sort_keys=True))
