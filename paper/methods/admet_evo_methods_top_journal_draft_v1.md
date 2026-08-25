# ADMET-EvO Methods and top-journal figure design

## Figure strategy

Two composite figures should carry the Methods story.

**Figure 1 | Autonomous architecture and workflow**

- **1a, System architecture:** task intake, EndpointContract, research state, hypothesis engine, five-axis action router, matched executor, evidence ledger, action memory and frozen certification.
- **1b, Task-conditioned data expansion:** endpoint-specific source search, query/access ledger, immutable raw evidence, semantic and license gates, structure standardization, conflict resolution, contamination firewall, domain audit, modeling-role decision and matched augmentation.
- **1c, Evidence-gated episode:** supported, refuted and inconclusive outcomes lead respectively to promotion, correction/rollback, replication or stopping, followed by an immutable certification boundary.

**Figure 2 | Multi-axis interventions and feedback**

- **2a, Axis-isolated action matrix:** feature, model and data-filter effects across five sentinel endpoints.
- **2b, Data-expansion feedback:** current hERG, DILI, Caco-2 and BBB discovery-validation effects after external-data expansion.
- **2c, Confirmation and memory revision:** independent confirmation, including the pruning regret that forced policy rollback.
- **2d, Stop/cost feedback:** performance non-inferiority and selected-recipe compute reduction.

This layout keeps the paper focused on autonomous research. Conventional TDC leaderboard plots should move to Extended Data or Supplementary Information.

---

# Methods

## System objective and study design

ADMET-EvO was designed as a protocol-constrained autonomous research agent for molecular absorption, distribution, metabolism, excretion and toxicity prediction. It was not formulated as a single universal predictor. Instead, it optimized a research policy that converted an endpoint-specific scientific question into an auditable sequence of task definition, diagnosis, hypothesis generation, experimental intervention, evidence evaluation, policy revision, stopping and frozen certification (Fig. 1a).

For every endpoint, the system maintained three distinct objects. An **EndpointContract** encoded the scientific and statistical meaning of the task. A **research state** summarized discovery data, multidimensional task hardness, faults, prior action evidence and remaining resources. A **frozen predictive product** serialized the selected representations, learner, preprocessing, hyperparameters, seeds and prediction rule. This separation prevented predictive performance from being used as post hoc justification for every intermediate Agent decision.

All development decisions were restricted to discovery data. Validation data were used for hypothesis tests and candidate selection. Certification or blind data were accessed only after the finalist and execution source had been frozen. Certification outcomes could not mutate a finalist in the same research lineage.

## Endpoint contracts and protocol ownership

Before profiling or fitting models, ADMET-EvO instantiated an EndpointContract containing the endpoint identifier, ADMET category, task type, metric and optimization direction, assay system, species where applicable, label semantics, unit or transform, source family, discovery split, certification split and external-data policy.

The contract was executable. A source could not be admitted merely because its endpoint name matched the target. CYP substrate, reversible inhibition, time-dependent inhibition, mechanism-based inhibition and induction were treated as different scientific endpoints. Permeability, clearance, solubility, protein binding, hERG and toxicity records likewise required assay-, species-, threshold- and unit-specific interpretation.

Protocol ownership was assigned to data partitions. Discovery components accessed only training and permitted validation evidence. Certification labels were inaccessible to the planner, data collector, model selector and memory. An evaluator-authorized structure-only file could be used for contamination filtering, but only standardized structures or fingerprints were visible to the sealed filter; labels, scores and per-record outcomes were never exposed to the planner.

## Task-conditioned data discovery and dataset expansion

### Endpoint-first source search

The new data-expansion module operated independently for each endpoint. It supported source discovery, lawful collection, audit, modeling advice and immutable update of an existing bundle. The planner could request a data action when scarcity, data quality, assay drift or an evidence gap was detected, but ordinary model search never silently downloaded or merged records.

The EndpointContract was expanded into endpoint-specific synonyms, assay terms and explicit exclusions. The Agent searched independent source families: primary experimental or regulatory repositories, endpoint-specific curated databases, primary publications and supporting information, and broader literature indexes. Each query was written to a query ledger containing the source, parameters, date, filters, result and retrieval counts, error, retry decision and stopping reason. Search stopped when the declared source scope was exhausted, the acquisition budget was reached or two consecutive expansions produced no new eligible primary source.

Each candidate source received a provisional compatibility grade. Grade A denoted direct compatibility; B required a documented conversion or restricted subgroup; C was related evidence suitable only for auxiliary learning or pretraining; and D denoted incompatible, predicted-only, circular, contaminated or unverifiable evidence. Scientific compatibility, access and license status were recorded independently.

### Immutable evidence and provenance

Raw files were preserved without in-place modification. Each artifact was associated with a source identifier, origin, release or retrieval time, checksum, access state, license, citation and assay metadata. Record-level extraction retained the submitted structure, source record identifier, raw value, unit, qualifier, censoring, evidence location and every normalization step.

Evidence was represented at linked Source, Assay and Record levels. This prevented a reputable database name from substituting for record-level provenance and prevented measurements from non-equivalent assays from being pooled without a bridge. Missing full text, inaccessible supporting information, API failures and license uncertainty remained visible states.

### Molecular standardization and conflict handling

External structures were parsed with RDKit and processed by molecular cleanup, fragment-parent selection, neutralization, canonical tautomer selection, canonical isomeric SMILES generation and InChIKey generation. Invalid structures were excluded with a recorded reason.

Exact identities were grouped by standardized InChIKey. For classification, identities with conflicting labels were removed from direct-label cohorts unless a prespecified resolution rule existed. For regression, repeated measurements were retained as replicate evidence and could be summarized by a prespecified median when a single-label table was required. Source-copy duplicates, technical replicates, biological replicates and independent-laboratory measurements were separated whenever provenance allowed. Raw measurements were never overwritten.

### Contamination and domain firewall

External records passed a layered firewall: exact identity exclusion, connectivity-key comparison, whole-source overlap assessment and near-analogue removal under a prespecified fingerprint and threshold. Current task-conditioned scaling experiments used 2,048-bit Morgan fingerprints at radius 2 and required maximum Tanimoto similarity below 0.80 to protected validation structures within each fold. The threshold was fixed before model evaluation.

Passing contamination control did not establish scientific compatibility. The Agent additionally assessed assay semantics, species, units, label direction, censoring, source independence, scaffold coverage and distribution shift. A source was assigned to direct supervised training, converted direct training, auxiliary-only use, external validation, quarantine, rejection or pending access.

A source could enter model development only after the versioned bundle reached the release state modeling_ready_discovery, scientific review was recorded and contamination auditing was complete. Expansion was then tested as an intervention: baseline and expanded arms used identical representations, learner, hyperparameters, split and seed, with only training membership changing. Source-held-out and target-validation evidence were stored separately. Positive, negative and inconclusive results all entered the ledger.

## Discovery-only profiling and HardnessCard

The Agent profiled the discovery training and validation partitions without loading certification data. The HardnessCard contained six dimensions: scarcity, imbalance, domain shift, local roughness or activity cliffs, data quality and split-leakage risk.

Scarcity depended on the effective number of training scaffolds. Classification imbalance was derived from minority prevalence. Domain shift combined unseen validation scaffolds, low-similarity validation molecules and mean nearest-neighbour similarity. Local roughness combined nearest-neighbour label disagreement and activity-cliff evidence. Data quality summarized invalid structures, missing or non-finite labels, repeats and conflicting identities. Split-leakage risk used standardized train–validation overlap.

For task \(t\),

\[
H_t=\frac{\sum_k w_k h_{tk}}{\sum_k w_k},
\]

where \(h_{tk}\in[0,1]\) was dimension severity. Implemented weights were 0.18 for scarcity, 0.12 for classification imbalance, 0.25 for domain shift, 0.23 for local roughness, 0.12 for data quality and 0.10 for leakage risk. The full vector, not only the scalar summary, was passed to the planner. Each HardnessCard was content-hashed and linked to its resulting action.

## Research-state construction and fault detection

The research state combined the EndpointContract, task profile, HardnessCard, data-skill route, detected protocol or data faults, prior action outcomes, compute budget and source snapshot. Fault detection checked metric direction, task type, label semantics, split ownership, structure overlap, invalid targets and source compatibility.

Tasks with a contract or data-integrity failure did not launch ordinary model search. They generated a repair or data-stewardship ticket. A healthy task with no defensible intervention could reuse a qualified frozen recipe or stop with zero new research fits.

## Falsifiable hypotheses and five action axes

Each proposal was represented by an action identifier, axis, hypothesis, falsification test, diagnostic evidence, expected effect, evidence confidence, compute cost, scientific risk and treatment specification.

1. **Feature axis:** circular, substructure, pharmacophore, topological, descriptor, pretrained or graph representations.
2. **Model axis:** estimator family, imbalance handling, objective, target transform or fusion rule.
3. **Data axis:** training membership or data role after the new search–clean–expand workflow.
4. **Replicate axis:** additional seeds, resampling, source-held-out evaluation, scaffold/OOD analysis or stability checks.
5. **Stop axis:** reuse, abstention, freeze, rollback, request for evidence or termination of low-information search.

For the deterministic effect-first policy, action utility was

\[
U(a)=p\,\widetilde E_a C_a-(1-p)(0.55K_a+0.45R_a),
\]

where \(p\) was effect priority, \(\widetilde E_a\) was expected effect calibrated by admissible prior evidence, \(C_a\) was evidence confidence, \(K_a\) was normalized compute cost and \(R_a\) was scientific risk. The default effect priority was 0.80. Historical evidence could modify routing only after transport and quality gates. Destructive negative-memory pruning required repeated refutation and independent confirmation.

The default planner was deterministic and local. An optional language-model planner could propose candidates through the same registered schema, but it received only task profiles, registered tools and validation history. It could not add unregistered models, change the protocol, access blind data or participate in certification.

## Molecular representations and predictive learners

The registered two-dimensional representations comprised 2,048-bit Morgan fingerprints at radius 2, 2,048-bit Morgan fingerprints at radius 3, count-based Morgan features, 1,024-bit Avalon fingerprints, 315-dimensional ErG reduced-graph features, 167 MACCS keys, 2,048-bit RDKit path fingerprints, hashed atom-pair fingerprints, hashed topological-torsion fingerprints and finite RDKit 2D descriptors.

A common multi-view baseline concatenated Morgan, Avalon, ErG and RDKit2D features. Alternative feature hypotheses changed one representation family while holding estimator, split and seed fixed. Feature matrices were cached by content hash.

Registered learners included ExtraTrees, Random Forest, balanced ExtraTrees for classification, CatBoost, XGBoost, LightGBM and RBF support-vector machines. The RBF-SVM branch was restricted to at most 5,000 training examples. Optional pretrained 3D or graph representations were enabled only when installed and justified by a feature-action gate. Missing dependencies were recorded; another representation could not be relabeled as the unavailable component.

Weighted blends were allowed only after members were evaluated independently. A blend was compared with its stronger member across matched seeds and promoted only when mean oriented improvement and the positive-seed fraction passed prespecified thresholds.

## Axis-isolated experiment execution

The executor translated an action into a matched control and treatment. A valid pair held endpoint, split, seed, metric, budget and all non-target axes constant. Shared controls could be physically reused, but every logical comparison remained in the resource ledger.

Each execution recorded its full specification, predictions, native validation score, runtime, status, failure reason, data audit and artifact hashes. Candidate wall-time and episode-level budgets were enforced. Pilot runtime could soft-stop a planned scale-up when extrapolated runtime exceeded the remaining budget by the preregistered safety margin.

For treatment \(T\), control \(C\), metric \(m\) and direction \(d\),

\[
\Delta=d[m(T)-m(C)],
\]

where \(d=+1\) for maximized metrics and \(d=-1\) for MAE. This orientation permitted cross-task action analysis while retaining native endpoint metrics.

Unless a protocol specified otherwise, an action was supported when the multi-seed mean oriented effect was at least 0.005 and stable across required seeds, refuted when it was at most −0.005, and inconclusive otherwise. Timeouts, invalid pairs and missing seeds remained explicit failures or incomplete evidence.

## Candidate selection and compute-aware routing

Protocol-valid candidates could be selected by mean score, lower confidence bound or a compute-aware lower confidence bound. Candidate \(i\) received

\[
Q_i=d\,s_i-\lambda_u\sigma_i-\lambda_c
\frac{\log(1+t_i)}{\max_j\log(1+t_j)},
\]

where \(s_i\) was validation score, \(\sigma_i\) the multi-seed standard deviation, \(t_i\) observed runtime, and \(\lambda_u,\lambda_c\) uncertainty and cost penalties. Ties were resolved by lower runtime and deterministic run identifier.

Score, uncertainty-aware and compute-aware policies were prespecified variants, not chosen from certification results. An efficiency policy was promoted only if aggregate compute ratio, nonmaterial-regression rate and endpoint-bootstrap non-inferiority passed the predefined gate.

## Evidence ledger, action memory and self-correction

Plans, outcomes, failures, promotions and freezes were appended to a SHA-256 hash chain. Records referenced the preceding digest, making deletion, reordering and mutation detectable. Negative and inconclusive experiments remained in the chain and follow-up hypotheses referenced their parent actions.

Action memory stored endpoint, task type, ADMET category, hardness context, action, matched effect, status and provenance. Cross-task effects were shrunk toward an uninformative prior and could affect routing only when action definitions and contexts were comparable.

A refuted or inconclusive action could trigger one bounded correction. The correction planner selected an untried axis, created a child hypothesis and inherited the remaining budget and protocol. A scheduled plan did not count as self-correction until the matched experiment completed. Contradictory confirmation caused memory downgrade or rollback. If no valid axis remained, expected gain was below uncertainty or the budget was insufficient, the Agent stopped.

## Freeze lock and blind certification

After selection, the Agent serialized the full recipe, representations, learner, preprocessing, hyperparameters, ensemble weights where applicable, seeds, dependency versions and source snapshot. Recipe, source and prediction files were content-addressed by SHA-256. A finalization directory could not silently replace an existing finalist.

Certification used a separate frozen executor. Models were recreated from the serialized recipe without calling the planner. Certification results were written to the audit chain but not returned to discovery, action memory or candidate generation. Prospective blind predictions were frozen before labels or leaderboard feedback became available.

## Evaluation of predictive and autonomous-research behavior

Predictive utility and research behavior were evaluated separately. Predictive analyses used native endpoint metrics, oriented effects, broad endpoint coverage, calibration, chemical-space error and performance–compute trade-offs. Process analyses included contract coverage, diagnosis completeness, fault identification, unsafe merger count, causal-pair completeness, negative-result retention, correction completion, rollback, stopping, conditional review and ledger integrity.

Policy comparisons used the same registered feature, model and data primitives and the same logical fit budget. Multi-role review shared one execution queue and did not add fits. Ablations removed a single capability such as diagnosis, memory or data acquisition while preserving the remaining execution environment.

## Statistical analysis

The endpoint was the primary unit for cross-task inference. Seeds were paired robustness replicates, not independent endpoints. Pairwise effects were calculated within endpoint and seed before aggregation. Cross-task confidence intervals used endpoint-level bootstrap resampling.

Classification used the contract-specified AUROC or area under the precision–recall curve. Regression used MAE or Spearman correlation. Metric directions were normalized only for cross-task action analyses; native values were retained in endpoint reports.

Minimum action effects, non-inferiority margins, positive-seed requirements, source-overlap thresholds, similarity cutoffs and compute gates were fixed before evaluation. An experiment that completed but failed its gate was reported as a completed negative result.

## Reproducibility and artifact retention

The system recorded dependency versions, hardware route, source snapshot, input/output hashes, seeds, experiment specifications, prediction arrays, runtimes, data audits and failures. Feature matrices were cached by content hash. Evidence bundles contained the EndpointContract, HardnessCard, plan, action outcomes, query ledger, source manifests, exclusions, conflicts, contamination audit, frozen recipe and certification artifacts.

Automated tests covered schemas, contracts, routing, metric direction, molecular curation, source admission, structure firewalls, evidence chains, source hashes, freeze locks, self-correction and policy analysis. Reproduction should begin from a frozen manifest and source snapshot rather than a mutable working directory.

---

# Publication-ready captions

## Figure 1 | Architecture of ADMET-EvO as an autonomous, evidence-gated research agent

**a,** Endpoint-specific intake is formalized as an executable contract before a multidimensional research state is constructed. The hypothesis engine routes a single-axis feature, model, data, replicate or stop intervention to a matched executor. Effects, uncertainty, cost and failures update a hash-chained ledger and action memory. **b,** The task-conditioned data branch searches independent database and literature sources, preserves record-level evidence, audits endpoint and assay compatibility, standardizes molecular identity, resolves conflicts, applies exact and near-neighbour contamination controls, assesses domain shift and assigns a permissible modeling role. Expansion is accepted only after a fixed-recipe matched experiment. **c,** Supported actions are promoted; refuted actions trigger correction or rollback; inconclusive actions trigger bounded replication or stopping. A content-addressed freeze separates discovery from blind certification.

## Figure 2 | Multi-axis experimental effects are returned to the ADMET-EvO research policy

**a,** Oriented validation effects for feature, model and data-quality interventions across five sentinel endpoints; S, R and I indicate supported, refuted and inconclusive outcomes. **b,** Validation changes after task-conditioned external-data search, molecular standardization, conflict removal and exact/near-neighbour filtering. Values are discovery-validation estimates from fixed recipes, not blind-test improvements. **c,** Independent confirmation retained two positive actions, refuted two and left one inconclusive. Maximum positive pruning regret of 0.259 disabled global negative-memory pruning. **d,** The compute-aware stop policy remained within the prespecified practical non-inferiority region while selecting recipes with an aggregate compute ratio of 0.278.

---

# Author notes before submission

1. Describe the complete system as an autonomous research-agent architecture, not a single neural-network architecture.
2. Describe fitted models as an endpoint-specific predictive portfolio.
3. Keep Figure 1 and Figure 2 in the main text; move leaderboard detail to Extended Data.
4. Current data-scaling results are discovery-validation evidence. Preserve source-role and independence qualifications.
5. The 0.80 similarity threshold is protocol-specific, not a universal chemical constant.
6. OpenADMET blind predictions are frozen, but official readout remains future external evidence.
