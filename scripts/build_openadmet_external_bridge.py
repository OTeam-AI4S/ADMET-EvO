"""Convert firewall-filtered Veith CYP records into a fold-safe bridge table."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd

from scripts.build_openadmet_cyp_split_manifest import _chemistry_group


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _parse_source(value: str) -> tuple[str, Path]:
    isoform, raw_path = value.split("=", 1)
    if isoform not in {"CYP2C9", "CYP2D6", "CYP3A4"}:
        raise ValueError(f"Unsupported Veith bridge isoform {isoform!r}")
    return isoform, Path(raw_path).resolve()


def build(args: argparse.Namespace) -> dict:
    split_path = args.split.resolve()
    split_manifest_path = split_path.with_suffix(".manifest.json")
    split_manifest = json.loads(split_manifest_path.read_text())
    if split_manifest.get("training_only") is not True:
        raise RuntimeError("OpenADMET split is not training-only")
    if split_manifest["output"]["sha256"] != _sha256(split_path):
        raise RuntimeError("OpenADMET split hash mismatch")
    split = pd.read_csv(split_path)
    identity_rows = split.set_index("inchikey").to_dict("index")
    group_rows = {
        group: group_frame.iloc[0].to_dict()
        for group, group_frame in split.groupby("chemistry_group")
    }

    rows = []
    source_audits = {}
    for raw_source in args.source:
        isoform, source_path = _parse_source(raw_source)
        audit_path = source_path.with_suffix(".firewall.json")
        audit = json.loads(audit_path.read_text())
        if audit.get("test_labels_loaded") is not False:
            raise RuntimeError("External firewall audit indicates test-label access")
        if audit.get("dataset_rejected_from_blind_distribution") is not False:
            raise RuntimeError("External source was selected from blind distribution")
        if audit["filtered_external_table_sha256"] != _sha256(source_path):
            raise RuntimeError("Filtered external source hash mismatch")
        frame = pd.read_csv(source_path)
        required = {"Drug_ID", "Drug", "Y", "standard_inchikey"}
        if not required.issubset(frame.columns):
            raise RuntimeError(f"Filtered source missing columns {sorted(required - set(frame))}")
        admitted = 0
        certification_exclusions = 0
        fold_exclusions: dict[str, int] = {}
        for source_row, row in frame.iterrows():
            identity = str(row["standard_inchikey"])
            standard_smiles = str(row["Drug"])
            chemistry_group = _chemistry_group(standard_smiles)
            matched = identity_rows.get(identity) or group_rows.get(chemistry_group)
            if matched and matched["partition"] == "certification":
                certification_exclusions += 1
                continue
            excluded_folds = []
            if matched and matched["partition"] == "development":
                excluded_folds = [int(matched["outer_fold"])]
                key = str(excluded_folds[0])
                fold_exclusions[key] = fold_exclusions.get(key, 0) + 1
            rows.append(
                {
                    "inchikey": identity,
                    "standard_smiles": standard_smiles,
                    "chemistry_group": chemistry_group,
                    "partition": "external_bridge",
                    "outer_fold": -2,
                    "excluded_outer_folds": ";".join(map(str, excluded_folds)),
                    "source_table": f"TDC_Veith_2009_{isoform}",
                    "source_row": int(source_row),
                    "source_record_id": str(row["Drug_ID"]),
                    "source_license": "CC_BY_4.0",
                    "isoform": isoform,
                    "assay_arm": "Veith_qHTS_activity",
                    "readout": "binary_activity",
                    "fidelity": "external_bridge",
                    "task_type": "classification",
                    "target": int(row["Y"]),
                    "credible_low": None,
                    "credible_high": None,
                    "measurement_std": None,
                    "concentration_M": None,
                    "endpoint_key": (
                        f"{isoform}|Veith_qHTS_activity|binary_activity|external_bridge"
                    ),
                }
            )
            admitted += 1
        source_audits[isoform] = {
            "filtered_source_path": str(source_path),
            "filtered_source_sha256": _sha256(source_path),
            "firewall_audit_path": str(audit_path.resolve()),
            "firewall_audit_sha256": _sha256(audit_path),
            "input_filtered_rows": len(frame),
            "admitted_bridge_rows": admitted,
            "internal_certification_group_exclusions": certification_exclusions,
            "outer_fold_group_exclusions": fold_exclusions,
        }

    bridge = pd.DataFrame(rows).sort_values(
        ["isoform", "inchikey", "source_record_id"], kind="stable"
    ).reset_index(drop=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    bridge.to_csv(args.output, index=False)
    manifest = {
        "version": "openadmet-cyp-veith-external-bridge-v1",
        "training_only": True,
        "blind_test_opened": False,
        "route": "source_conditioned_auxiliary_binary_head_only",
        "exact_primary_endpoint_merge_forbidden": True,
        "split_path": str(split_path),
        "split_sha256": _sha256(split_path),
        "split_manifest_sha256": _sha256(split_manifest_path),
        "source_audits": source_audits,
        "rows": len(bridge),
        "output_path": str(args.output.resolve()),
        "output_sha256": _sha256(args.output),
        "test_labels_loaded": False,
        "leaderboard_feedback_used": False,
    }
    manifest_path = args.output.with_suffix(".manifest.json")
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return {**manifest, "manifest_path": str(manifest_path.resolve())}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source",
        action="append",
        required=True,
        help="ISOFORM=path/to/firewall-filtered.csv; repeat for each source",
    )
    parser.add_argument(
        "--split",
        type=Path,
        default=Path("reports/openadmet_cyp_identity_locked_splits_v1.csv"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("reports/openadmet_tdc_veith_external_bridge_v1.csv"),
    )
    return parser.parse_args()


if __name__ == "__main__":
    print(json.dumps(build(parse_args()), indent=2, sort_keys=True))
