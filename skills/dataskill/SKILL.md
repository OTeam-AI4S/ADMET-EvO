---
name: admet-data-collector
description: Collect, normalize, verify, audit, and prepare evidence-grounded ADMET data for molecular modeling. Use when Codex must identify or disambiguate an ADMET endpoint, search public databases and literature for experimental records, document open-access and paywalled evidence, extract assay-aware data, detect provenance or test-set contamination, grade source compatibility, build an EndpointContract/AssayContract and HardnessCard, or recommend whether data should be pooled, modeled separately, used as an auxiliary task, quarantined, or rejected. Trigger on ADMET data collection, external evidence acquisition, endpoint mapping, assay harmonization, CYP/hERG/clearance/solubility/toxicity datasets, chemical-space expansion, data readiness audits, and modeling advice based on dataset hardness.
---

# ADMET Data Collector

Build an auditable evidence package, not merely a larger CSV. Treat endpoint meaning, assay conditions, provenance, access rights, contamination risk, and modeling compatibility as separate questions.

## Non-negotiable rules

1. Create an `EndpointContract` before searching or merging data. Never infer label meaning from a short dataset column name alone.
2. Preserve raw evidence and every exclusion. Never overwrite raw values or silently discard inconvenient records.
3. Distinguish experimental evidence from database aggregation, literature interpretation, and in-silico prediction. Do not use predicted values as experimental labels.
4. Treat accessibility, licensing, and scientific validity as independent dimensions. A paywalled article can contain relevant evidence; an open dataset can still be incompatible or contaminated.
5. Do not bypass paywalls, DRM, authentication, or license restrictions. Report unavailable evidence explicitly.
6. Keep held-out labels inaccessible to the collector and research agent. Only an evaluator-owned audit may use held-out structures for predeclared contamination filtering.
7. Do not claim completeness from a single database. Search source families until the predeclared saturation or stopping rule is met.
8. Do not pool sources until endpoint, species, assay, unit, direction, threshold, censoring, provenance, and split compatibility have passed review.

## Select the operating mode

- `discover`: identify candidate sources and return a source/access register without extracting records.
- `collect`: retrieve lawful data, preserve raw artifacts, extract records, and run all validation gates.
- `audit`: assess an existing dataset and source manifest without searching for new data.
- `advise`: build a HardnessCard and modeling recommendations from an already audited bundle.
- `update`: add new evidence to an existing versioned bundle without mutating prior releases.

For collection and update, ask whether Supporting Information (SI) should be retrieved before downloading papers. An explicit request for SI or main text only settles this once for the batch.

## Load the relevant references

- Read [endpoint-ontology.md](references/endpoint-ontology.md) for every new endpoint or any ambiguous task name.
- Read [source-routing.md](references/source-routing.md) before discovery or retrieval.
- Read [evidence-validation.md](references/evidence-validation.md) before extracting, merging, or excluding records.
- Read [hardness-modeling.md](references/hardness-modeling.md) when producing hardness or modeling advice.
- Read [output-contracts.md](references/output-contracts.md) before creating deliverables or validating a bundle.
- Read [design-provenance.md](references/design-provenance.md) when explaining how this workflow extends the closed-loop Auto Research paper.

## End-to-end workflow

### 1. Establish collection scope and protocol ownership

Record the target endpoint and intended use; species, tissue/matrix, assay system, readout, units, direction, threshold, time point, and censoring; compounds and chemical domain; discovery, certification, and future-audit ownership; allowed sources, date/language limits, license constraints, and SI choice; and whether the goal is direct labels, auxiliary evidence, pretraining, external validation, or chemical-space mapping.

If any field changes the biological meaning of the task, stop and resolve it before collection.

### 2. Build the EndpointContract and search facets

Use `references/endpoint-ontology.md`. Expand the endpoint into synonyms plus explicit exclusions. For CYP tasks, always distinguish substrate, reversible inhibition, time-dependent inhibition, mechanism-based inhibition, induction, and phenotyping. For permeability, clearance, solubility, PPB, hERG, DILI, and toxicity, apply the corresponding endpoint-specific distinctions.

Assign each candidate source a provisional compatibility grade:

- `A`: same endpoint, species, assay semantics, unit/direction, and intended label use;
- `B`: compatible after a documented conversion or restricted subgroup;
- `C`: biologically related but unsuitable as direct labels; auxiliary/pretraining only;
- `D`: incompatible, circular, predicted-only, contaminated, or unverifiable.

### 3. Estimate the pre-collection hardness prior

From the target dataset and EndpointContract, record expected risks: scarcity, imbalance, source heterogeneity, assay noise, distribution shift, activity cliffs, selection variance, and baseline saturation. Mark these as `prior` until verified from collected data. Use the prior to prioritize sources, not to lower evidence standards.

### 4. Search by independent source family

Follow `references/source-routing.md`: primary regulatory or experimental repositories; endpoint-specific curated databases with record-level provenance; PubMed/CrossRef and publisher metadata for primary articles and SI; broader indexes and repositories for recall; and patents or secondary compilations only as clearly labeled lower-tier evidence.

Use a query ledger. Record every query, source, date, filters, result count, failure, and stopping decision. Deduplicate papers by DOI/PMID/title and datasets by provenance, not file name.

### 5. Resolve access and license status

For every source, assign both an access status and a license status. If full text or data are not openly accessible, retain the bibliographic record and create a visible access-gap entry. Do not infer measurements from an abstract when assay details or tables are unavailable.

When lawful full-text retrieval is needed, use the installed literature-downloader workflow. Prefer publisher APIs or legitimate OA copies; use institution-authorized browser access only with the user's session and consent. Never mislabel HTML, a login page, or metadata as a downloaded PDF.

### 6. Preserve raw artifacts and extract record-level evidence

Store immutable raw files with URL, retrieval time, checksum, MIME type, and source ID. Extract each measurement with its exact evidence location: table/figure/SI page, database record ID, or API endpoint. Include a short evidence quote or field mapping sufficient for audit without copying copyrighted full text unnecessarily.

Never normalize in place. Write extracted raw records first, then generate normalized records with explicit transformation fields.

### 7. Run evidence gates

Apply all gates in `references/evidence-validation.md`: bibliographic identity; primary provenance; access/license; compound identity; endpoint/assay alignment; unit/transform/direction/threshold/censoring; duplicate/conflict handling; source independence; held-out contamination; distribution shift; extraction QA; and modeling-role decision.

Assign one status per record/source: `accepted_direct`, `accepted_converted`, `auxiliary_only`, `external_validation_only`, `quarantined`, `rejected`, or `pending_access`. Never collapse `pending_access` into `rejected`.

### 8. Protect held-out evidence

Enforce standardized identity and InChIKey-skeleton exclusion; whole-source rejection when predeclared held-out overlap exceeds the threshold; and near-analogue exclusion at the predeclared fingerprint and similarity threshold.

Log counts before and after each layer. The collector may receive only blinded structure identifiers or fingerprints needed for filtering, never labels or scores. Do not tune thresholds after observing certification results.

### 9. Build the final HardnessCard

Use audited, split-aware data to quantify sample and effective-scaffold scarcity; class imbalance or response-range sparsity; source/assay heterogeneity and conflicts; scaffold novelty, similarity, and domain separability; landscape roughness and cliffs when estimable; seed/rank instability and baseline headroom when runs exist; and access/provenance incompleteness.

Keep the multidimensional card. Do not replace it with one opaque difficulty score.

### 10. Produce modeling advice

Map evidence and hardness to actions using `references/hardness-modeling.md`. State whether each source can be pooled directly; whether source indicators, hierarchical models, multitask learning, transfer learning, pretraining, calibration, or abstention are appropriate; which split and metric match deployment; which claims remain unsupported; and the most valuable next data or experiment.

Every recommendation must cite observed evidence and a falsification or monitoring criterion.

### 11. Deliver the auditable bundle

Create the directory and schemas with:

```powershell
python scripts/init_collection_bundle.py --out <output> --endpoint-id <id>
```

Populate the bundle, then run:

```powershell
python scripts/audit_admet_records.py --bundle <output> --report-dir <output>/reports
```

Add `--heldout <structures.csv>` only when the file is evaluator-authorized and label-free. Add `--strict` for a nonzero exit when critical evidence or contamination failures remain.

Return the deliverables defined in `references/output-contracts.md`, including the non-OA/access report, exclusion ledger, unresolved conflicts, HardnessCard, and modeling advice. Report partial completion honestly when access, licensing, or endpoint ambiguity remains.

## Completion gate

Do not call a bundle modeling-ready unless the EndpointContract and split ownership are explicit; every included record maps to a registered source and evidence location; direct labels pass semantic and measurement checks; license and access states are visible; conflict and contamination audits are complete; transformations are reproducible; exclusions and negative findings are retained; the HardnessCard and modeling role are based on audited data; and a second-pass spot check covers all high-impact sources plus a sample of ordinary records.

If any condition fails, label the output `research-use only`, `auxiliary only`, `pending evidence`, or `not modeling-ready` rather than weakening the gate.
