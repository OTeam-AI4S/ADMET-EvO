#!/usr/bin/env python3
"""Audit an ADMET collection bundle for evidence, semantics and contamination."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

RECORD_REQUIRED = {
    "record_id", "source_id", "canonical_smiles", "standard_inchikey",
    "inchikey_skeleton", "endpoint_id", "property", "species", "assay_system",
    "readout", "raw_value", "raw_unit", "raw_qualifier", "normalized_value",
    "normalized_unit", "label", "label_definition", "publication_id",
    "evidence_location", "compatibility_grade", "admission_status",
}
SOURCE_REQUIRED = {
    "source_id", "title", "source_type", "doi", "pmid", "url", "oa_status",
    "full_text_status", "data_access_status", "license", "primary_or_secondary",
    "provenance_url",
}
DIRECT_STATUSES = {"accepted_direct", "accepted_converted"}
ALLOWED_STATUSES = DIRECT_STATUSES | {
    "auxiliary_only", "external_validation_only", "quarantined", "rejected",
    "pending_access",
}
NON_OA_HINTS = {
    "metadata_only", "paywalled_fulltext", "institutional_access_available",
    "access_denied", "source_unavailable", "access_status_uncertain",
}
LABEL_LIKE_COLUMNS = {
    "label", "labels", "target", "targets", "y", "value", "raw_value",
    "normalized_value", "outcome", "score", "metric",
}


def read_csv(path: Path) -> tuple[list[dict[str, str]], list[str]]:
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        rows = [{k: (v or "").strip() for k, v in row.items()} for row in reader]
        return rows, list(reader.fieldnames or [])


def issue(bucket: dict[str, list[dict[str, Any]]], severity: str, code: str,
          message: str, **details: Any) -> None:
    bucket[severity].append({"code": code, "message": message, "details": details})


def identity(row: dict[str, str]) -> str:
    return (row.get("inchikey_skeleton") or row.get("standard_inchikey") or
            row.get("canonical_smiles") or row.get("compound_id") or "").upper()


def numeric(value: str) -> float | None:
    try:
        result = float(value)
        return result if math.isfinite(result) else None
    except (TypeError, ValueError):
        return None


def parse_binary(value: str) -> int | None:
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "positive", "active", "substrate", "inhibitor"}:
        return 1
    if normalized in {"0", "false", "no", "negative", "inactive", "non-substrate", "non_inhibitor"}:
        return 0
    return None


def top_level_yaml_value(path: Path, key: str) -> str:
    if not path.exists():
        return ""
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith(f"{key}:"):
            return line.split(":", 1)[1].strip().strip("'\"")
    return ""


def markdown_report(result: dict[str, Any]) -> str:
    lines = ["# Automated ADMET Evidence Audit", "", f"Generated: {result['generated_at']}", ""]
    summary = result["summary"]
    lines += [
        "## Summary", "",
        f"- Records: {summary['records']}",
        f"- Sources: {summary['sources']}",
        f"- Critical issues: {summary['critical_issues']}",
        f"- Warnings: {summary['warnings']}",
        f"- Suggested release state: `{summary['suggested_release_state']}`", "",
    ]
    lines += ["## Hardness signals", "", "```json",
              json.dumps(result["hardness_signals"], indent=2, ensure_ascii=False), "```", ""]
    for severity in ["critical", "warning", "info"]:
        lines += [f"## {severity.title()}", ""]
        if not result["issues"][severity]:
            lines.append("None.")
        else:
            for item in result["issues"][severity]:
                lines.append(f"- **{item['code']}**: {item['message']}")
        lines.append("")
    return "\n".join(lines)


def access_gap_report(sources: list[dict[str, str]]) -> str:
    lines = ["# Access and License Gaps", ""]
    gaps = []
    for source in sources:
        oa = source.get("oa_status", "").lower()
        access = source.get("data_access_status", "").lower()
        license_value = source.get("license", "").lower()
        if oa in NON_OA_HINTS or access in NON_OA_HINTS or not license_value or "unclear" in license_value or "restrict" in license_value:
            gaps.append(source)
    if not gaps:
        lines.append("No access or license gaps were detected from the source registry fields.")
        return "\n".join(lines) + "\n"
    for source in gaps:
        lines += [
            f"## {source.get('source_id') or 'unidentified source'}", "",
            f"- Title: {source.get('title') or 'not recorded'}",
            f"- DOI/PMID: {source.get('doi') or '-'} / {source.get('pmid') or '-'}",
            f"- OA status: {source.get('oa_status') or 'unrecorded'}",
            f"- Data access: {source.get('data_access_status') or 'unrecorded'}",
            f"- License: {source.get('license') or 'unrecorded'}",
            f"- URL: {source.get('url') or 'unrecorded'}",
            f"- Notes: {source.get('access_notes') or 'Scientific content and lawful next route require review.'}", "",
        ]
    return "\n".join(lines)


def near_analogue_audit(records: list[dict[str, str]], heldout: list[dict[str, str]],
                        threshold: float, max_comparisons: int) -> dict[str, Any]:
    pairs = len(records) * len(heldout)
    if pairs > max_comparisons:
        return {"status": "not_run_comparison_limit", "comparisons": pairs, "limit": max_comparisons}
    try:
        from rdkit import Chem, DataStructs  # type: ignore
        from rdkit.Chem import rdFingerprintGenerator  # type: ignore
    except Exception:
        return {"status": "not_run_rdkit_unavailable", "comparisons": pairs}

    generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    heldout_fps: list[tuple[str, Any]] = []
    for row in heldout:
        smiles = row.get("canonical_smiles", "")
        molecule = Chem.MolFromSmiles(smiles) if smiles else None
        if molecule is not None:
            heldout_fps.append((identity(row) or smiles, generator.GetFingerprint(molecule)))
    matches = []
    invalid = 0
    for row in records:
        smiles = row.get("canonical_smiles", "")
        molecule = Chem.MolFromSmiles(smiles) if smiles else None
        if molecule is None:
            invalid += 1
            continue
        fingerprint = generator.GetFingerprint(molecule)
        for heldout_id, heldout_fp in heldout_fps:
            similarity = float(DataStructs.TanimotoSimilarity(fingerprint, heldout_fp))
            if similarity >= threshold:
                matches.append({
                    "record_id": row.get("record_id"), "source_id": row.get("source_id"),
                    "heldout_id": heldout_id, "similarity": round(similarity, 6),
                })
    return {
        "status": "completed", "threshold": threshold,
        "fingerprint": "Morgan radius=2 fpSize=2048", "matches": matches,
        "invalid_record_smiles": invalid, "valid_heldout_smiles": len(heldout_fps),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", required=True, type=Path)
    parser.add_argument("--report-dir", type=Path)
    parser.add_argument("--heldout", type=Path, help="Label-free, evaluator-authorized structures CSV")
    parser.add_argument("--source-overlap-threshold", type=float, default=0.05)
    parser.add_argument("--near-analogue-threshold", type=float, default=0.80)
    parser.add_argument("--max-analog-comparisons", type=int, default=2_000_000)
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args()

    bundle = args.bundle.resolve()
    report_dir = (args.report_dir or bundle / "reports").resolve()
    report_dir.mkdir(parents=True, exist_ok=True)
    records_path = bundle / "normalized/normalized_records.csv"
    sources_path = bundle / "sources/source_registry.csv"
    if not records_path.exists() or not sources_path.exists():
        parser.error("Bundle must contain normalized/normalized_records.csv and sources/source_registry.csv")

    records, record_fields = read_csv(records_path)
    sources, source_fields = read_csv(sources_path)
    issues: dict[str, list[dict[str, Any]]] = {"critical": [], "warning": [], "info": []}

    missing_record_fields = sorted(RECORD_REQUIRED - set(record_fields))
    missing_source_fields = sorted(SOURCE_REQUIRED - set(source_fields))
    if missing_record_fields:
        issue(issues, "critical", "record_schema_missing", "Required record columns are absent", columns=missing_record_fields)
    if missing_source_fields:
        issue(issues, "critical", "source_schema_missing", "Required source columns are absent", columns=missing_source_fields)

    contract_path = bundle / "contracts/endpoint_contract.yaml"
    endpoint_id = top_level_yaml_value(contract_path, "endpoint_id")
    if not endpoint_id or endpoint_id.upper() == "TODO":
        issue(issues, "critical", "endpoint_contract_missing", "EndpointContract has no resolved endpoint_id")

    source_ids = [row.get("source_id", "") for row in sources if row.get("source_id", "")]
    duplicate_source_ids = [key for key, count in Counter(source_ids).items() if count > 1]
    if duplicate_source_ids:
        issue(issues, "critical", "duplicate_source_id", "Source IDs must be unique", source_ids=duplicate_source_ids)
    source_index = {row.get("source_id", ""): row for row in sources if row.get("source_id", "")}

    missing_source_links = sorted({row.get("source_id", "") for row in records if row.get("source_id", "") not in source_index})
    if missing_source_links:
        issue(issues, "critical", "unregistered_sources", "Records refer to unregistered sources", source_ids=missing_source_links)

    endpoint_mismatches = [row.get("record_id") for row in records if endpoint_id and row.get("endpoint_id") != endpoint_id]
    if endpoint_mismatches:
        issue(issues, "critical", "endpoint_mismatch", "Records do not match EndpointContract", count=len(endpoint_mismatches), sample=endpoint_mismatches[:20])

    invalid_status = [row.get("record_id") for row in records if row.get("admission_status") not in ALLOWED_STATUSES]
    if invalid_status:
        issue(issues, "critical", "invalid_admission_status", "Records have missing or invalid admission status", count=len(invalid_status), sample=invalid_status[:20])

    direct_rows = [row for row in records if row.get("admission_status") in DIRECT_STATUSES]
    direct_required = ["source_id", "endpoint_id", "property", "species", "assay_system", "readout", "evidence_location", "compatibility_grade"]
    incomplete_direct = []
    for row in direct_rows:
        missing = [field for field in direct_required if not row.get(field)]
        if not identity(row):
            missing.append("chemical_identity")
        if not row.get("label") and not row.get("normalized_value"):
            missing.append("label_or_normalized_value")
        if missing:
            incomplete_direct.append({"record_id": row.get("record_id"), "missing": missing})
    if incomplete_direct:
        issue(issues, "critical", "incomplete_direct_records", "Direct modeling records lack decisive evidence fields", count=len(incomplete_direct), sample=incomplete_direct[:20])

    record_ids = [row.get("record_id", "") for row in records if row.get("record_id", "")]
    duplicate_record_ids = [key for key, count in Counter(record_ids).items() if count > 1]
    if duplicate_record_ids:
        issue(issues, "critical", "duplicate_record_id", "Record IDs must be unique", record_ids=duplicate_record_ids[:50])

    grouped: dict[tuple[str, ...], list[dict[str, str]]] = defaultdict(list)
    for row in records:
        key = (identity(row), row.get("endpoint_id", ""), row.get("assay_system", ""), row.get("readout", ""), row.get("timepoint", ""))
        if key[0]:
            grouped[key].append(row)
    duplicate_groups = [rows for rows in grouped.values() if len(rows) > 1]
    conflict_groups = []
    for rows in duplicate_groups:
        labels = {row.get("label", "") for row in rows if row.get("label", "")}
        values = [numeric(row.get("normalized_value", "")) for row in rows]
        finite_values = [value for value in values if value is not None]
        if len(labels) > 1 or (len(finite_values) > 1 and max(finite_values) - min(finite_values) > 1e-12):
            conflict_groups.append([row.get("record_id") for row in rows])
    if duplicate_groups:
        issue(issues, "info", "replicate_or_duplicate_groups", "Identity/endpoint/assay groups contain multiple records", count=len(duplicate_groups))
    if conflict_groups:
        issue(issues, "warning", "unresolved_measurement_conflicts", "Duplicate-condition groups contain differing labels or values; adjudicate provenance and replicate meaning", count=len(conflict_groups), sample=conflict_groups[:20])

    access_gaps = []
    license_gaps = []
    for source in sources:
        oa = source.get("oa_status", "").lower()
        access = source.get("data_access_status", "").lower()
        license_value = source.get("license", "").lower()
        if oa in NON_OA_HINTS or access in NON_OA_HINTS:
            access_gaps.append(source.get("source_id"))
        if not license_value or "unclear" in license_value or "restrict" in license_value:
            license_gaps.append(source.get("source_id"))
    if access_gaps:
        issue(issues, "warning", "non_oa_or_inaccessible_sources", "Some candidate evidence is not openly accessible", source_ids=access_gaps)
    if license_gaps:
        issue(issues, "warning", "license_gaps", "Some sources have missing, unclear or restricted licenses", source_ids=license_gaps)

    species_counts = Counter(row.get("species", "") or "unrecorded" for row in direct_rows)
    assay_counts = Counter(row.get("assay_system", "") or "unrecorded" for row in direct_rows)
    unit_counts = Counter(row.get("normalized_unit", "") or "unrecorded" for row in direct_rows)
    source_counts = Counter(row.get("source_id", "") or "unrecorded" for row in direct_rows)

    binary = [parse_binary(row.get("label", "")) for row in direct_rows]
    binary_values = [value for value in binary if value is not None]
    imbalance: dict[str, Any] = {"binary_labels_parsed": len(binary_values)}
    if binary_values:
        positives = sum(binary_values)
        negatives = len(binary_values) - positives
        imbalance.update({
            "positive_rate": positives / len(binary_values), "positive_count": positives,
            "negative_count": negatives, "minority_count": min(positives, negatives),
        })

    numeric_values = [numeric(row.get("normalized_value", "")) for row in direct_rows]
    numeric_values = [value for value in numeric_values if value is not None]
    response_range: dict[str, Any] = {"numeric_values_parsed": len(numeric_values)}
    if numeric_values:
        response_range.update({
            "min": min(numeric_values), "max": max(numeric_values),
            "median": statistics.median(numeric_values),
        })

    contamination: dict[str, Any] = {"status": "not_requested"}
    if args.heldout:
        heldout, heldout_fields = read_csv(args.heldout.resolve())
        exposed = sorted({field.lower() for field in heldout_fields} & LABEL_LIKE_COLUMNS)
        if exposed:
            issue(issues, "critical", "heldout_contains_label_like_columns", "Held-out audit file is not structure-only", columns=exposed)
        heldout_ids = {identity(row) for row in heldout if identity(row)}
        exact_matches = [
            {"record_id": row.get("record_id"), "source_id": row.get("source_id"), "identity": identity(row)}
            for row in records if identity(row) and identity(row) in heldout_ids
        ]
        source_match_counts = Counter(match["source_id"] for match in exact_matches)
        denominator = max(1, len(heldout_ids))
        source_overlap = {
            source_id: {"exact_matches": count, "heldout_fraction": count / denominator,
                        "reject_source": count / denominator > args.source_overlap_threshold}
            for source_id, count in source_match_counts.items()
        }
        near = near_analogue_audit(records, heldout, args.near_analogue_threshold, args.max_analog_comparisons)
        contamination = {
            "status": "completed", "heldout_unique_identities": len(heldout_ids),
            "exact_matches": exact_matches, "source_overlap": source_overlap,
            "source_overlap_threshold": args.source_overlap_threshold,
            "near_analogue": near,
        }
        if exact_matches:
            issue(issues, "critical", "heldout_exact_overlap", "External records overlap held-out structures", count=len(exact_matches), sample=exact_matches[:20])
        rejected_sources = [key for key, value in source_overlap.items() if value["reject_source"]]
        if rejected_sources:
            issue(issues, "critical", "same_source_overlap_threshold", "Sources exceed the predeclared held-out overlap threshold", source_ids=rejected_sources)
        if near.get("status") == "completed" and near.get("matches"):
            issue(issues, "critical", "heldout_near_analogues", "External records exceed the near-analogue similarity threshold", count=len(near["matches"]), sample=near["matches"][:20])
        elif near.get("status") != "completed":
            issue(issues, "warning", "near_analogue_audit_not_completed", "Near-analogue audit did not complete", status=near.get("status"))

    evidence_completeness = {
        "source_count": len(sources), "record_count": len(records),
        "direct_record_count": len(direct_rows), "pending_access_records": sum(row.get("admission_status") == "pending_access" for row in records),
        "access_gap_sources": len(access_gaps), "license_gap_sources": len(license_gaps),
        "records_with_evidence_location": sum(bool(row.get("evidence_location")) for row in records),
    }
    hardness = {
        "status": "automated_partial_requires_scientific_review",
        "scarcity": {"direct_records": len(direct_rows), "unique_identities": len({identity(row) for row in direct_rows if identity(row)})},
        "imbalance_or_range": {**imbalance, **response_range},
        "source_heterogeneity": {"source_distribution": dict(source_counts), "species_distribution": dict(species_counts), "assay_distribution": dict(assay_counts), "unit_distribution": dict(unit_counts)},
        "assay_or_label_noise": {"duplicate_or_replicate_groups": len(duplicate_groups), "conflict_groups": len(conflict_groups)},
        "distribution_shift": {"heldout_audit_status": contamination.get("status"), "note": "Compute scaffold, descriptor and domain-classifier metrics separately."},
        "roughness": {"status": "not_estimated"}, "activity_cliffs": {"status": "not_estimated"},
        "selection_variance": {"status": "requires model-run evidence"},
        "baseline_saturation": {"status": "requires benchmark evidence"},
        "evidence_completeness": evidence_completeness,
    }

    critical_count = len(issues["critical"])
    warning_count = len(issues["warning"])
    if critical_count:
        release_state = "research_use_only"
    elif not records or not direct_rows or access_gaps or license_gaps or conflict_groups:
        release_state = "pending_scientific_review"
    else:
        release_state = "modeling_ready_discovery"

    result = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "bundle": str(bundle), "endpoint_id": endpoint_id,
        "summary": {"records": len(records), "sources": len(sources), "direct_records": len(direct_rows),
                    "critical_issues": critical_count, "warnings": warning_count,
                    "suggested_release_state": release_state},
        "issues": issues, "hardness_signals": hardness, "contamination": contamination,
    }

    (report_dir / "automated_audit.json").write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (report_dir / "automated_audit.md").write_text(markdown_report(result) + "\n", encoding="utf-8")
    (report_dir / "access_gap_report.md").write_text(access_gap_report(sources), encoding="utf-8")
    (bundle / "hardness/hardness_card_auto.json").write_text(json.dumps(hardness, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (bundle / "audits/contamination_audit.json").write_text(json.dumps(contamination, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print(json.dumps(result["summary"], ensure_ascii=False))
    return 2 if args.strict and critical_count else 0


if __name__ == "__main__":
    raise SystemExit(main())
