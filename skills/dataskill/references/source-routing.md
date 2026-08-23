# Source Routing and Completeness Strategy

## Contents

1. Source hierarchy
2. Endpoint-oriented sources
3. Search execution
4. Access and licensing
5. Completeness and stopping

## 1. Source hierarchy

Search independent source families rather than treating one aggregator as complete.

| Tier | Source family | Intended use | Main risk |
|---|---|---|---|
| E1 | primary experimental tables/SI, regulatory assessment, official assay repository | direct evidence | access or extraction burden |
| E2 | curated database with record-level primary provenance | discovery and direct evidence after verification | transformed values, incomplete conditions |
| E3 | primary article without accessible record-level table | source lead or qualitative support | cannot reconstruct labels |
| E4 | secondary review/compilation | query expansion and provenance tracing | circular citation or lost assay context |
| E5 | predicted/commercial opaque aggregation | hypothesis generation only | not experimental, license/provenance uncertainty |

Evidence tier is not the same as open-access status. Record both.

## 2. Endpoint-oriented sources

Use current official interfaces and verify terms and licensing at collection time.

- **Broad bioactivity:** ChEMBL, PubChem BioAssay and BindingDB. Inspect document, assay, target, qualifier, unit and provenance fields.
- **Toxicology:** EPA CompTox/ToxCast/Tox21, OECD eChemPortal, NTP and regulatory resources. Preserve assay endpoint metadata.
- **Clinical and PK:** FDA labels, clinical-pharmacology reviews, DDI resources, Drugs@FDA, EMA reports and primary clinical studies.
- **Benchmarks:** TDC and MoleculeNet support evaluation but do not prove that an external source is independent.

Endpoint query anchors:

- CYP inhibition: isoform + IC50/Ki + probe substrate + matrix.
- TDI/MBI: isoform + preincubation + NADPH + IC50 shift or kinact/KI.
- Clearance: species + microsome/hepatocyte/in vivo + CLint/half-life + unit.
- Transporter: transporter + substrate/inhibitor + cell line + efflux/Papp/IC50.
- hERG: patch clamp/current or binding + cell system + IC50/percent inhibition.
- DILI: clinical causality dataset + label definition + version.
- Solubility/permeability/PPB: protocol, pH/matrix/species and measurement.

Treat DrugBank, CAS-derived products, subscription databases and commercial collections according to their current licenses. Do not redistribute restricted content merely because it can be viewed.

## 3. Search execution

Use purpose-built academic search connectors when available: PubMed for biomedical literature, CrossRef for DOI metadata, preprint servers when relevant, and Semantic Scholar/Scopus/Web of Science for recall and citation chaining. Use Google Scholar only as a last-resort discovery surface and mark it unstable.

Prefer official database APIs or bulk releases over scraping. Save query parameters, API/release version, retrieval time, count statistics, raw checksum, license and citation requirements.

Query ledger fields:

`query_id,endpoint_id,source,query_text_or_parameters,date,filters,result_count,retrieved_count,error,retry,reviewer,stopping_reason`

Deduplicate papers by DOI, PMID/PMCID, then normalized title/year/first author. Deduplicate datasets by provenance, not file name.

## 4. Access and licensing

Assign one access status:

`oa_data_downloaded, oa_fulltext_downloaded, oa_fulltext_html, repository_copy_available, metadata_only, paywalled_fulltext, institutional_access_available, access_denied, source_unavailable, access_status_uncertain`

Assign license independently:

`redistribution_allowed, research_use_allowed_no_redistribution, citation_required, custom_terms_reviewed, license_unclear, restricted`

For every non-OA or inaccessible source, report the full citation; expected evidence; relevance; what metadata/abstract verified; what remains unverified; lawful next route; and potential selection bias. Never infer row-level values from abstracts or search snippets.

## 5. Completeness and stopping

At minimum:

1. search one primary structured database, one literature index and one endpoint-specific/regulatory family when applicable;
2. use synonym and assay-specific variants;
3. backward-chain compilations to primary sources;
4. forward-chain high-value datasets when tools permit;
5. inspect documentation and SI;
6. repeat search after aliases and source datasets are discovered;
7. stop when two consecutive expansions yield no new eligible primary source, scope is exhausted, or the declared budget is reached.

Report the stopping rule, unsearched sources, API failures, paywalled gaps and likely coverage bias. Say `comprehensive within the declared scope`, not `complete`, unless a closed registry proves completeness.
