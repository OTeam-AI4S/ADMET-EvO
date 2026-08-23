"""Build a sparse, training-only OpenADMET CYP observation table.

Every observation inherits the canonical identity and frozen chemistry-group
partition from the split manifest.  The script has no blinded-test argument and
also emits a separate label-free manifest for frozen MiniMol extraction.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from functools import cache
from pathlib import Path
from typing import Any

import pandas as pd

from admet_agent.external_structure_firewall import _standard_molecule
from scripts.build_openadmet_cyp_split_manifest import (
    DEFAULT_INPUTS,
    DIRECT_TARGETS,
    EMAX_TARGETS,
    EXPECTED_SHA256,
    TDI_PIC50_TARGETS,
    TDI_TARGETS,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@cache
def _identity(raw_smiles: str) -> str:
    molecule = _standard_molecule(str(raw_smiles))
    if molecule is None:
        raise RuntimeError("Training SMILES failed frozen standardization")
    return molecule.inchikey


def _base_observation(
    *,
    identity: str,
    source_table: str,
    source_row: int,
    isoform: str,
    assay_arm: str,
    readout: str,
    fidelity: str,
    task_type: str,
    target: float,
    lower: float | None,
    upper: float | None,
    measurement_std: float | None,
    concentration_m: float | None,
) -> dict[str, Any]:
    return {
        "inchikey": identity,
        "source_table": source_table,
        "source_row": source_row,
        "isoform": isoform,
        "assay_arm": assay_arm,
        "readout": readout,
        "fidelity": fidelity,
        "task_type": task_type,
        "target": float(target),
        "credible_low": lower,
        "credible_high": upper,
        "measurement_std": measurement_std,
        "concentration_M": concentration_m,
        "endpoint_key": f"{isoform}|{assay_arm}|{readout}|{fidelity}",
    }


def _optional_float(row: pd.Series, column: str) -> float | None:
    if column not in row or pd.isna(row[column]):
        return None
    return float(row[column])


def _wide_observations(
    frame: pd.DataFrame,
    *,
    source_table: str,
    targets: tuple[str, ...],
    assay_arm: str,
    readout: str,
    fidelity: str,
    task_type: str,
) -> list[dict[str, Any]]:
    observations: list[dict[str, Any]] = []
    for source_row, row in frame.iterrows():
        identity = _identity(str(row["SMILES"]))
        for target_column in targets:
            if target_column not in frame or pd.isna(row[target_column]):
                continue
            isoform = target_column.split("_", 1)[0]
            observations.append(
                _base_observation(
                    identity=identity,
                    source_table=source_table,
                    source_row=int(source_row),
                    isoform=isoform,
                    assay_arm=assay_arm,
                    readout=readout,
                    fidelity=fidelity,
                    task_type=task_type,
                    target=float(row[target_column]),
                    lower=_optional_float(row, f"{target_column}_conf_low"),
                    upper=_optional_float(row, f"{target_column}_conf_high"),
                    measurement_std=_optional_float(row, f"{target_column}_std"),
                    concentration_m=None,
                )
            )
    return observations


def _single_observations(frame: pd.DataFrame) -> list[dict[str, Any]]:
    observations: list[dict[str, Any]] = []
    for source_row, row in frame.iterrows():
        if pd.isna(row.get("log2fc_estimate")):
            continue
        observations.append(
            _base_observation(
                identity=_identity(str(row["SMILES"])),
                source_table="single",
                source_row=int(source_row),
                isoform=str(row["enzyme"]),
                assay_arm="single_concentration_active_preincubation",
                readout="log2fc",
                fidelity="low_single_concentration",
                task_type="regression",
                target=float(row["log2fc_estimate"]),
                lower=None,
                upper=None,
                measurement_std=_optional_float(row, "log2fc_std_error"),
                concentration_m=_optional_float(row, "concentration_M"),
            )
        )
    return observations


def build(args: argparse.Namespace) -> dict:
    input_paths = {name: Path(getattr(args, name)).resolve() for name in DEFAULT_INPUTS}
    for name, path in input_paths.items():
        digest = _sha256(path)
        if digest != EXPECTED_SHA256[name]:
            raise RuntimeError(f"Frozen input hash mismatch for {name}: {digest}")

    split_path = args.split.resolve()
    split_manifest_path = split_path.with_suffix(".manifest.json")
    split_manifest = json.loads(split_manifest_path.read_text(encoding="utf-8"))
    if split_manifest.get("training_only") is not True:
        raise RuntimeError("Split manifest is not training-only")
    if split_manifest.get("blind_test_opened") is not False:
        raise RuntimeError("Split manifest indicates blind-test access")
    if split_manifest["output"]["sha256"] != _sha256(split_path):
        raise RuntimeError("Split table hash mismatch")
    split = pd.read_csv(split_path)
    if split["inchikey"].duplicated().any():
        raise RuntimeError("Split table contains duplicate canonical identities")

    frames = {name: pd.read_csv(path) for name, path in input_paths.items()}
    observations: list[dict[str, Any]] = []
    observations.extend(
        _wide_observations(
            frames["direct"],
            source_table="direct",
            targets=DIRECT_TARGETS,
            assay_arm="direct_inhibition",
            readout="pIC50",
            fidelity="primary_direct",
            task_type="regression",
        )
    )
    observations.extend(
        _wide_observations(
            frames["tdi"],
            source_table="tdi",
            targets=TDI_PIC50_TARGETS,
            assay_arm="TDI_preincubation",
            readout="pIC50",
            fidelity="auxiliary_paired_condition",
            task_type="regression",
        )
    )
    observations.extend(
        _wide_observations(
            frames["tdi"],
            source_table="tdi",
            targets=DIRECT_TARGETS,
            assay_arm="direct_inhibition",
            readout="pIC50",
            fidelity="auxiliary_paired_direct",
            task_type="regression",
        )
    )
    observations.extend(
        _wide_observations(
            frames["tdi"],
            source_table="tdi",
            targets=TDI_TARGETS,
            assay_arm="TDI_shift",
            readout="is_TDI",
            fidelity="primary_classification",
            task_type="classification",
        )
    )
    observations.extend(
        _wide_observations(
            frames["emax"],
            source_table="emax",
            targets=tuple(
                target for target in EMAX_TARGETS if target.endswith("direct_inhibition")
            ),
            assay_arm="direct_inhibition",
            readout="Emax",
            fidelity="auxiliary_emax",
            task_type="regression",
        )
    )
    observations.extend(
        _wide_observations(
            frames["emax"],
            source_table="emax",
            targets=tuple(
                target for target in EMAX_TARGETS if target.endswith("TDI_condition")
            ),
            assay_arm="TDI_preincubation",
            readout="Emax",
            fidelity="auxiliary_emax",
            task_type="regression",
        )
    )
    observations.extend(_single_observations(frames["single"]))
    longform = pd.DataFrame(observations)
    split_columns = list(split.columns)
    longform = longform.merge(
        split[split_columns], on="inchikey", how="left", validate="many_to_one"
    )
    if longform["partition"].isna().any():
        missing = int(longform["partition"].isna().sum())
        raise RuntimeError(f"Observations missing frozen split identity: {missing}")
    longform = longform.sort_values(
        ["inchikey", "endpoint_key", "source_table", "source_row"], kind="stable"
    ).reset_index(drop=True)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    longform.to_csv(args.output, index=False)
    embeddings_manifest = {
        "task": "OpenADMET_CYP_training_identity_union",
        "contains_labels": False,
        "official_test_loaded": False,
        "inchikeys": split["inchikey"].astype(str).tolist(),
        "smiles": split["standard_smiles"].astype(str).tolist(),
        "split_table_sha256": _sha256(split_path),
    }
    embedding_path = args.output.with_suffix(".minimol_input.json")
    embedding_path.write_text(
        json.dumps(embeddings_manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    endpoint_counts = {
        key: int(value)
        for key, value in longform["endpoint_key"].value_counts().sort_index().items()
    }
    manifest = {
        "version": "openadmet-cyp-training-longform-v1",
        "training_only": True,
        "blind_test_opened": False,
        "split_path": str(split_path),
        "split_sha256": _sha256(split_path),
        "split_manifest_sha256": _sha256(split_manifest_path),
        "inputs": {
            name: {"path": str(path), "sha256": EXPECTED_SHA256[name]}
            for name, path in sorted(input_paths.items())
        },
        "identities": int(split["inchikey"].nunique()),
        "observations": len(longform),
        "endpoint_observation_counts": endpoint_counts,
        "output_path": str(args.output.resolve()),
        "output_sha256": _sha256(args.output),
        "minimol_input_path": str(embedding_path.resolve()),
        "minimol_input_sha256": _sha256(embedding_path),
        "test_labels_loaded": False,
    }
    manifest_path = args.output.with_suffix(".manifest.json")
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return {**manifest, "manifest_path": str(manifest_path.resolve())}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    for name, default in DEFAULT_INPUTS.items():
        parser.add_argument(f"--{name}", default=default)
    parser.add_argument(
        "--split",
        type=Path,
        default=Path("reports/openadmet_cyp_identity_locked_splits_v1.csv"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("reports/openadmet_cyp_training_longform_v1.csv"),
    )
    return parser.parse_args()


if __name__ == "__main__":
    print(json.dumps(build(parse_args()), indent=2, sort_keys=True))
