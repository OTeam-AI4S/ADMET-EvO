#!/usr/bin/env python3
"""Initialize an auditable ADMET data-collection bundle."""

from __future__ import annotations

import argparse
import csv
import json
import re
from datetime import datetime, timezone
from pathlib import Path


SOURCE_FIELDS = [
    "source_id", "title", "source_type", "doi", "pmid", "url",
    "publication_year", "oa_status", "full_text_status", "data_access_status",
    "license", "license_url", "primary_or_secondary", "provenance_url",
    "retrieved_at", "access_notes",
]

ACCESS_FIELDS = [
    "source_id", "access_status", "license_status", "verified_content",
    "unverified_content", "importance", "next_lawful_route", "bias_risk",
    "checked_at",
]

RECORD_FIELDS = [
    "record_id", "source_id", "source_record_id", "compound_name",
    "compound_id", "submitted_structure", "canonical_smiles",
    "standard_inchikey", "inchikey_skeleton", "endpoint_id", "property",
    "species", "matrix_or_tissue", "assay_system", "assay_protocol",
    "readout", "timepoint", "raw_value", "raw_unit", "raw_qualifier",
    "normalized_value", "normalized_unit", "transform", "label",
    "label_definition", "censoring", "publication_id", "evidence_location",
    "evidence_quote", "extraction_method", "compatibility_grade",
    "admission_status", "extraction_date", "curator", "notes",
]

QUERY_FIELDS = [
    "query_id", "endpoint_id", "source", "query_text_or_parameters", "date",
    "filters", "result_count", "retrieved_count", "error", "retry",
    "reviewer", "stopping_reason",
]

EXCLUSION_FIELDS = [
    "record_id_or_source_id", "level", "gate", "reason", "evidence",
    "decision", "date", "reviewer", "reversible",
]


def write_csv_header(path: Path, fields: list[str]) -> None:
    if path.exists() and path.stat().st_size > 0:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        csv.DictWriter(handle, fieldnames=fields).writeheader()


def write_if_absent(path: Path, text: str) -> None:
    if path.exists() and path.stat().st_size > 0:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--endpoint-id", required=True)
    parser.add_argument(
        "--allow-existing", action="store_true",
        help="Create only missing templates in an existing directory.",
    )
    args = parser.parse_args()

    if not re.fullmatch(r"[a-z0-9][a-z0-9_.-]{1,127}", args.endpoint_id):
        parser.error("--endpoint-id must be 2-128 lowercase letters, digits, _, . or -")

    out = args.out.resolve()
    if out.exists() and any(out.iterdir()) and not args.allow_existing:
        parser.error(f"Refusing to modify non-empty directory: {out}. Use --allow-existing.")

    directories = [
        "contracts", "queries", "sources", "raw", "extracted", "normalized",
        "audits", "hardness", "reports",
    ]
    for name in directories:
        (out / name).mkdir(parents=True, exist_ok=True)

    write_csv_header(out / "queries/query_ledger.csv", QUERY_FIELDS)
    write_csv_header(out / "sources/source_registry.csv", SOURCE_FIELDS)
    write_csv_header(out / "sources/access_report.csv", ACCESS_FIELDS)
    write_csv_header(out / "extracted/raw_records.csv", RECORD_FIELDS)
    write_csv_header(out / "normalized/normalized_records.csv", RECORD_FIELDS)
    write_csv_header(out / "audits/exclusion_log.csv", EXCLUSION_FIELDS)
    write_csv_header(
        out / "audits/conflict_log.csv",
        ["conflict_id", "record_ids", "identity_key", "endpoint_id", "conditions",
         "conflict_type", "resolution", "reviewer", "date"],
    )

    endpoint_yaml = f"""endpoint_id: {args.endpoint_id}
property: TODO
admet_domain: TODO
task_type: TODO
biological_entity: TODO
species: TODO
matrix_or_tissue: TODO
assay_system: TODO
readout: TODO
measurement:
  original_unit: TODO
  canonical_unit: TODO
  scale: TODO
  direction: TODO
label_definition:
  type: TODO
  threshold: TODO
censoring_allowed: TODO
intended_role: TODO
split_contract:
  discovery: TODO
  certification: evaluator_owned_holdout
  future_audit: new_independent_evidence
explicit_exclusions: []
"""
    split_yaml = """discovery:
  owner: research_agent
  labels_visible: true
  repeated_access: budgeted
certification:
  owner: evaluator
  labels_visible_to_agent: false
  structures_visible_to_filter: TODO
  score_once_after_freeze: true
future_audit:
  required_after_certification_feedback: true
"""
    hardness = {
        "endpoint": args.endpoint_id,
        "status": "prior_not_assessed",
        "scarcity": {}, "imbalance_or_range": {}, "source_heterogeneity": {},
        "assay_or_label_noise": {}, "distribution_shift": {}, "roughness": {},
        "activity_cliffs": {}, "selection_variance": {},
        "baseline_saturation": {}, "evidence_completeness": {},
    }
    manifest = {
        "bundle_id": args.endpoint_id,
        "version": "0.1.0-draft",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "release_state": "draft_collection",
        "endpoint_id": args.endpoint_id,
        "si_choice": "unconfirmed",
        "raw_files": [],
        "audit": {"status": "not_run", "critical_issues": []},
        "review": {"scientific_reviewer": None, "reviewed_at": None},
    }

    write_if_absent(out / "contracts/endpoint_contract.yaml", endpoint_yaml)
    write_if_absent(out / "contracts/split_contract.yaml", split_yaml)
    write_if_absent(
        out / "hardness/hardness_card.json",
        json.dumps(hardness, indent=2, ensure_ascii=False) + "\n",
    )
    write_if_absent(
        out / "manifest.json",
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
    )
    for report in ["collection_report.md", "access_gap_report.md", "modeling_advice.md"]:
        write_if_absent(out / "reports" / report, f"# {report.removesuffix('.md').replace('_', ' ').title()}\n\nTODO\n")

    print(json.dumps({"status": "initialized", "bundle": str(out), "endpoint_id": args.endpoint_id}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
