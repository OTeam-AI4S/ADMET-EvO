# ADMET Evidence Validation and Data Admission

## Contents

1. Evidence model
2. Twelve validation gates
3. Conflict and replicate policy
4. Held-out contamination control
5. Admission decisions
6. Independent QA

## 1. Evidence model

Maintain three linked layers:

- `Source`: article, SI, database release, regulatory document or repository object.
- `Assay`: biological system, protocol, readout, conditions and label definition.
- `Record`: compound identity, measurement, qualifier, unit and evidence location.

A record is not validated merely because its source is reputable. Track bibliographic identity, provenance, access, license, compound identity, endpoint compatibility, measurement integrity, independence, reproducibility and modeling role independently.

## 2. Twelve validation gates

### Gate 1: bibliographic identity

Cross-check title, authors, year, journal, DOI/PMID and version against two structured sources when possible. Resolve corrections, preprint-to-journal links and duplicate datasets.

### Gate 2: provenance and primary evidence

Trace each row to a primary table, SI, assay record, regulatory document or database record. Mark secondary compilations and find their original source. Reject circular provenance where databases cite each other without a primary measurement.

### Gate 3: access and license

Record what was accessible and whether data may be reused or redistributed. Keep `metadata_only` and `paywalled_fulltext` visible. License uncertainty blocks redistribution and may block modeling.

### Gate 4: compound identity

Preserve submitted name/structure, then standardize with a versioned pipeline: parse; apply declared salt/mixture policy; normalize charge and selected tautomers without erasing relevant stereochemistry; generate canonical SMILES, full InChIKey and connectivity key; flag ambiguous names, mixtures, metabolites and stereochemical uncertainty.

Never map an ambiguous synonym to a structure without independent confirmation.

### Gate 5: endpoint and assay semantics

Compare each source with the EndpointContract and assign compatibility A-D. Require species, entity, matrix/cell system, protocol, readout, time and relevant temperature/pH. Missing decisive fields cause quarantine or auxiliary-only status.

### Gate 6: measurement integrity

Validate measurement type/direction, unit and molar-mass assumptions, linear/log scale and base, qualifier, censoring, threshold, replicate aggregation, impossible ranges and transcription errors. Preserve raw and normalized values, units, qualifier, transform and conversion rationale.

### Gate 7: duplicates, replicates and conflicts

Distinguish aggregator duplicates, technical replicates, biological replicates, independent laboratories, different conditions and genuine conflicts. Do not average incompatible conditions. Cluster copied provenance before splitting.

### Gate 8: source independence

Check whether an external file reaggregates the benchmark or a sibling source using publication IDs, lineage, record IDs, compounds and assay provenance. Public does not imply independent.

### Gate 9: held-out contamination

Use evaluator-owned, label-free held-out structures only under a predeclared protocol. Run exact identity, connectivity-key, same-source and near-analogue checks. Never expose labels, metrics or per-record outcomes.

### Gate 10: domain and distribution shift

Compare external data with target training and intended deployment using scaffold/similarity distributions, physicochemical coverage, domain-classifier separability, label/censoring distribution and source/protocol effects. Contamination safety does not prove compatibility.

### Gate 11: reproducibility and extraction QA

Record checksums, parser/tool versions, transformations and row lineage. Re-extract all high-impact sources and a prespecified sample in a second pass.

### Gate 12: modeling-role decision

Assign direct, converted, auxiliary, external-validation, quarantine, reject or pending-access. State the evidence and uncertainty. Do not let modeling convenience determine compatibility.

## 3. Conflict and replicate policy

1. Verify transcription and unit/scale.
2. Separate different assay conditions.
3. Collapse exact reaggregations to one provenance cluster.
4. Retain independent replicates with source/lab identifiers.
5. Quantify within- and between-source variation.
6. Use consensus labels only under a declared rule.
7. Otherwise retain conflicts and model uncertainty or exclude from direct labels.

Store original continuous measurements when available. Prefer censor-aware modeling; do not silently replace every `<x` or `>x` with the threshold.

## 4. Held-out contamination control

```text
candidate source
-> source-lineage comparison
-> standardized exact identity exclusion
-> connectivity-key exclusion
-> whole-source overlap test
-> near-analogue filtering
-> assay compatibility
-> domain-shift audit
-> source-only and leave-one-source-out validation
-> admission decision
```

Predeclare identity standardization, overlap denominator, source rejection threshold, fingerprint and similarity threshold, salt/stereoisomer treatment, grouping policy and access ownership.

The reference paper used standardized identities, whole-source rejection above 5% test-structure overlap and ECFP4 Tanimoto >=0.9. Treat these as precedent, not universal constants. Set thresholds before evaluation according to deployment and benchmark policy.

## 5. Admission decisions

| Status | Meaning | Permitted use |
|---|---|---|
| `accepted_direct` | contract-aligned experimental evidence | direct supervised training |
| `accepted_converted` | valid prespecified conversion/restriction | direct training with metadata |
| `auxiliary_only` | related but non-equivalent evidence | pretraining, multitask, priors |
| `external_validation_only` | independent evidence reserved from development | final external evaluation |
| `quarantined` | unresolved identity, assay, unit, conflict or license issue | no modeling |
| `rejected` | incompatible, predicted-only, circular, contaminated or invalid | excluded but retained |
| `pending_access` | likely relevant but full evidence unavailable | access report only |

## 6. Independent QA

- Review 100% of sources that materially change sample size, domain or headline results.
- Review 100% of conversions, threshold-derived labels, conflicts and held-out matches.
- Review a stratified random sample across endpoint, source, access and admission states.
- Combine automated schema/audit checks with manual scientific review.
- Record reviewer, date, version and unresolved issues.

If one person performs extraction and review, label it self-check rather than independent verification.
