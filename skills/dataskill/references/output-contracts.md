# Collection Bundle and Output Contracts

## Contents

1. Directory layout
2. Required tables
3. Reports
4. Versioning and release states

## 1. Directory layout

```text
<bundle>/
  contracts/
    endpoint_contract.yaml
    split_contract.yaml
  queries/query_ledger.csv
  sources/
    source_registry.csv
    access_report.csv
  raw/<source_id>/...
  extracted/raw_records.csv
  normalized/normalized_records.csv
  audits/
    exclusion_log.csv
    conflict_log.csv
    contamination_audit.json
  hardness/hardness_card.json
  reports/
    collection_report.md
    access_gap_report.md
    modeling_advice.md
    automated_audit.json
    automated_audit.md
  manifest.json
```

Raw artifacts are immutable. Corrections create a new normalized version and lineage entry.

## 2. Required tables

### `source_registry.csv`

`source_id,title,source_type,doi,pmid,url,publication_year,oa_status,full_text_status,data_access_status,license,license_url,primary_or_secondary,provenance_url,retrieved_at,access_notes`

### `access_report.csv`

`source_id,access_status,license_status,verified_content,unverified_content,importance,next_lawful_route,bias_risk,checked_at`

### `raw_records.csv` and `normalized_records.csv`

`record_id,source_id,source_record_id,compound_name,compound_id,submitted_structure,canonical_smiles,standard_inchikey,inchikey_skeleton,endpoint_id,property,species,matrix_or_tissue,assay_system,assay_protocol,readout,timepoint,raw_value,raw_unit,raw_qualifier,normalized_value,normalized_unit,transform,label,label_definition,censoring,publication_id,evidence_location,evidence_quote,extraction_method,compatibility_grade,admission_status,extraction_date,curator,notes`

Leave unavailable fields empty and explain decisive gaps in `notes`. Do not fabricate placeholders that look like values.

### `query_ledger.csv`

`query_id,endpoint_id,source,query_text_or_parameters,date,filters,result_count,retrieved_count,error,retry,reviewer,stopping_reason`

### `exclusion_log.csv`

`record_id_or_source_id,level,gate,reason,evidence,decision,date,reviewer,reversible`

## 3. Reports

The collection report summarizes scope, contract, searches, stopping, source/record flow, compatibility, transformations, conflicts, exclusions, contamination, shift and limitations.

The access-gap report lists every paywalled, restricted, metadata-only, unavailable or license-unclear source, with verified/unverified content, importance, lawful next action and bias risk.

The HardnessCard stores raw metrics, method/version, uncertainty, split and `not_estimated_reason`, distinguishing prior from data-derived assessment.

Modeling advice specifies each source's role, split, metrics, comparisons, uncertainty, abstention and next evidence action, linking recommendations to audit findings.

The manifest records bundle/version/time; contract hashes; counts by status; raw checksums; script/dependency versions; audit results; unresolved issues; release state; and reviewer sign-off.

## 4. Versioning and release states

Use immutable semantic versions or dated snapshots. Never silently replace a release.

Allowed states:

`draft_collection, pending_access, pending_scientific_review, research_use_only, auxiliary_only, modeling_ready_discovery, external_validation_reserved, rejected`

`modeling_ready_discovery` does not authorize using certification labels or reusing a certification set after feedback influenced development. Move the next version to a new future audit set.
