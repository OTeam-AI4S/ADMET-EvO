# Methods

## Study design and system objective

ADMET-EvO was designed as a protocol-constrained autonomous research agent for molecular absorption, distribution, metabolism, excretion and toxicity (ADMET) prediction. The system was not optimized as a single universal predictive model. Instead, it learned and executed an endpoint-conditioned research policy that transformed a scientific question into a closed sequence of task formalization, data diagnosis, hypothesis generation, controlled intervention, evidence evaluation, policy revision, stopping and frozen certification (Fig. 1).

The architecture separated three objects that are often conflated in automated machine-learning systems. First, an executable **EndpointContract** defined the biological and statistical meaning of the prediction task. Second, a mutable **research state** represented the Agent's current knowledge, including task hardness, diagnosed faults, admissible data, prior experiments, uncertainty and remaining resources. Third, a content-addressed **predictive product** serialized the final endpoint-specific recipe. This separation allowed the Agent to revise its research strategy without silently changing endpoint definitions and prevented certification feedback from altering a finalized model.

ADMET-EvO operated as a closed loop:

\[
\mathcal{S}_{t}
\xrightarrow{\pi}
a_t
\xrightarrow{\mathcal{E}}
e_t
\xrightarrow{\mathcal{U}}
\mathcal{S}_{t+1},
\]

where \(\mathcal{S}_{t}\) is the research state, \(\pi\) the constrained planning policy, \(a_t\) a registered intervention, \(\mathcal{E}\) the matched experimental executor, \(e_t\) a standardized EvidencePacket and \(\mathcal{U}\) the evidence-gated state update. The loop terminated when a prespecified stopping condition was met, the expected value of additional information became insufficient, no protocol-valid action remained or the research budget was exhausted.

## Endpoint contracts and partition ownership

Before inspecting molecular profiles or fitting a model, the Agent instantiated an EndpointContract containing the endpoint identifier, ADMET category, task type, primary metric and optimization direction, assay system, species where applicable, label semantics, unit and transformation, source family, discovery partitions, protected partitions and permissible external-data roles.

The contract was executable rather than descriptive. A data source was not considered compatible solely because its title contained the target endpoint. For example, CYP substrate status, reversible inhibition, time-dependent inhibition, mechanism-based inhibition and induction were represented as distinct endpoint semantics. Similarly, permeability, clearance, solubility, protein binding, hERG activity and toxicity measurements required assay-, species-, threshold- and unit-aware interpretation. Contract violations generated a repair or data-stewardship action instead of ordinary model optimization.

Data access was governed by partition ownership. Training and permitted validation data formed the discovery environment. Protected test labels were inaccessible to the planner, data collector, model selector, action memory and adaptive data module. When contamination filtering required knowledge of protected structures, only an evaluator-authorized structure-only view was made available. The firewall could derive standardized identities and fingerprints from these structures, but it could not access labels, scores or per-record outcomes.

## Autonomous research architecture

### Research-state construction

For each endpoint, the Agent constructed a research state from the EndpointContract, a discovery-only molecular profile, a multidimensional HardnessCard, detected protocol and data faults, the task data registry, admissible prior evidence, action memory, compute budget and source snapshot. Every state-changing artifact was content-hashed so that an action could be traced to the exact evidence available when it was proposed.

The planner did not receive an unrestricted natural-language workspace. It received typed summaries and a registry of allowed actions. Consequently, it could prioritize among registered feature, model, data, replicate and stop interventions, but could not introduce an unregistered learner, alter the evaluation metric, redefine the endpoint, access protected labels or bypass an admission gate.

### Falsifiable hypothesis generation

Every proposal was represented by an action identifier, intervention axis, scientific hypothesis, matched control, treatment, falsification criterion, expected effect, evidence confidence, estimated cost and scientific risk. The five action axes were:

1. **Feature:** molecular fingerprints, descriptors, pharmacophore or topological features, pretrained embeddings and graph or three-dimensional representations.
2. **Model:** estimator family, objective, class-imbalance treatment, target transformation, regularization and fusion rule.
3. **Data:** task-specific source search, cleaning, admission, data-role assignment, core-set construction and controlled expansion.
4. **Replicate:** additional seeds, resampling, source-held-out evaluation, scaffold-stratified analysis and out-of-distribution diagnostics.
5. **Stop:** reuse of a qualified recipe, abstention, rollback, freeze, request for additional evidence or termination of low-information search.

For the deterministic effect-first policy, the utility of action \(a\) was

\[
U(a)=p\,\widetilde{E}_a C_a-(1-p)(0.55K_a+0.45R_a),
\]

where \(p\) is the effect priority, \(\widetilde{E}_a\) the context-calibrated expected effect, \(C_a\) the confidence in the supporting evidence, \(K_a\) normalized compute cost and \(R_a\) scientific risk. The default effect priority was 0.80. Historical evidence could change routing only when its endpoint semantics, intervention definition and evaluation context were transportable to the current task.

### Matched execution and standardized evidence

The executor translated each accepted proposal into a control–treatment pair that held endpoint, split, metric, seed, fit budget and all non-target axes constant. Shared controls could be cached, but each logical comparison remained explicit in the ledger. Execution produced an EvidencePacket containing the native score, oriented effect, between-seed variability, positive-seed fraction, runtime, status, failure reason, data and protocol audits, artifact hashes and, where applicable, scaffold- or source-held-out analyses.

For treatment \(T\), control \(C\), metric \(m\) and metric direction \(d\), the oriented effect was

\[
\Delta=d[m(T)-m(C)],
\]

where \(d=+1\) for maximized metrics and \(d=-1\) for mean absolute error. Thus, positive \(\Delta\) consistently denoted improvement while native endpoint metrics remained available in all reports.

Unless an endpoint-specific protocol superseded the generic rule, an action was supported when its mean oriented effect was at least 0.005 and the required seed-consistency criterion was satisfied, refuted when the effect was at most \(-0.005\), and inconclusive otherwise. Completed negative experiments, invalid pairs, timeouts and missing-seed runs remained explicit ledger entries.

## Task-specific data search, cleaning and adaptive dataset expansion

The updated data module treated external-data acquisition as a scientific intervention rather than a file-concatenation step. For each endpoint, it executed six separable phases: source discovery, immutable evidence capture, endpoint-aware curation, score-blind admission, adaptive training-set construction and independent confirmation. A failure in any mandatory phase caused the module to fail closed without altering the baseline training set.

### Endpoint-specific source discovery

The EndpointContract was expanded into an endpoint-specific search plan containing synonyms, assay terms, species and unit constraints, and explicit exclusions. The Agent searched complementary source families, including primary experimental or regulatory repositories, endpoint-specific curated databases, primary articles and supporting information, and broader literature indexes used to identify primary evidence.

Every acquisition attempt was recorded in a query ledger with the source, query parameters, access time, filters, retrieval count, error state, retry decision and stopping reason. Source discovery stopped when the declared scope was exhausted, the acquisition budget was reached or successive query expansions yielded no new eligible primary source. Search completion therefore denoted exhaustion of a predefined evidence route, not the absence of all possible data.

Candidate sources were assigned a provisional semantic role. Direct sources were endpoint-equivalent under the contract; convertible sources required a prespecified unit or label transformation; auxiliary sources were mechanistically related but not endpoint-equivalent; incompatible, predicted-only, circular, contaminated or unverifiable sources were rejected or quarantined. Scientific compatibility, access state and licensing were recorded independently.

### Immutable evidence and record-level provenance

Raw source files were preserved without in-place modification and associated with a source identifier, origin, version or retrieval time, checksum, access state, license and citation. Record-level extraction retained the submitted structure, source record identifier, raw target value, unit, qualifier, censoring, assay metadata, evidence location and each subsequent normalization operation.

Provenance was represented at linked Source, Assay and Record levels. This design prevented database reputation from substituting for record-level evidence and prevented measurements from distinct assays from being pooled without an explicit semantic bridge. Missing full text, unavailable supporting information, incomplete assay annotation, API failure and license uncertainty remained machine-readable states rather than being silently discarded.

### Molecular identity standardization and target normalization

External molecules were parsed with RDKit and processed through molecular cleanup, fragment-parent selection, charge normalization, canonical tautomer selection, canonical isomeric SMILES generation and InChIKey generation. Invalid structures were excluded with a recorded reason. Standardized identities, rather than raw strings, were used for deduplication, overlap detection and scaffold analysis.

Targets were normalized only when the EndpointContract defined a valid transformation. The curation record retained original values, units, qualifiers and transformed values. Directionality was checked explicitly so that, for example, greater activity could not be interpreted as lower risk after a threshold conversion.

Exact molecular identities were grouped by standardized InChIKey. For classification, identities with irreconcilable labels were excluded from direct-label cohorts unless a prespecified assay-priority rule applied. For regression, repeated measurements remained visible as replicate evidence and could be summarized by a prespecified median only when a single-label modeling table was required. Source-copy duplicates, technical replicates, biological replicates and independent-laboratory measurements were distinguished whenever provenance permitted.

### Score-blind external-data admission

Admission was performed in a dedicated `data_audit_only` run before external records could affect model fitting. The audit explicitly verified that protected test data were not loaded and that model scores were not consulted. It evaluated minimum usable identities, structure validity, within-source conflicts, source lineage, exact overlap and target agreement, target-distribution alignment and chemical-space coverage.

The strict direct-admission thresholds were fixed as follows:

| Criterion | Classification | Regression / shared threshold |
|---|---:|---:|
| Minimum standardized identities | 100 | 30 |
| Maximum invalid-structure fraction | 0.01 | 0.01 |
| Maximum conflicting-row fraction | 0.02 | 0.02 |
| Maximum exact-overlap fraction | 0.10 | 0.10 |
| Maximum exact-label disagreement | 0.10 | — |
| Maximum exact-target difference, scaled by reference IQR | — | 0.25 |
| Maximum target-distribution shift | 0.50 | 0.50 |
| Minimum fraction with nearest-train Tanimoto \(\geq0.4\) | 0.05 | 0.05 |

The admission engine combined automated checks with explicit endpoint review and source-lineage status. **Direct supervised admission** required all automated quality checks, an `approved_direct` endpoint review and cleared independent lineage. **Controlled bridge admission** was reserved for scientifically related but non-equivalent sources that passed a separate bridge protocol, were marked `bridge_only`, and were used only for train–validation development followed by disjoint confirmation. The bridge protocol used a maximum exact-overlap fraction of 0.05, a maximum target-distribution shift of 0.25 and a minimum chemical-space coverage fraction of 0.025, while retaining the same structure, conflict and minimum-identity safeguards. All other cases were assigned `fail_closed` and routed to quarantine, repair or abstention.

The controlled-bridge designation did not convert an auxiliary assay into endpoint-equivalent ground truth. It defined a restricted experimental role and a lower claim tier. The strict direct-admission policy was frozen before all-endpoint fitting; the bridge tier was subsequently isolated as a developmental policy and required independent confirmation. This chronology and the applicable claim tier were retained in the audit artifact.

### Fold-specific contamination firewall

Admitted external records were re-filtered within each discovery fold. The firewall removed exact matches to protected structures and near analogues using 2,048-bit Morgan fingerprints at radius 2. Records with maximum Tanimoto similarity \(\geq0.80\) to the protected set were excluded. The same 0.80 default was used by the data-collection and structure-firewall components. Because filtering was fold-specific, an external molecule eligible in one training fold could remain ineligible in another.

The protected structure view remained label-blind. Passing the firewall established neither assay compatibility nor source independence; it was necessary only after semantic and admission checks had already assigned a permissible modeling role.

### Train-relevance and scaffold-diverse core-set construction

After contamination filtering, the Agent estimated each external molecule's relevance to the current training fold from chemical similarity. It then constructed a task-adaptive external core set rather than using every available external record.

For binary classification, an external budget \(B\) was divided according to the class prevalence \(p_{\mathrm{TDC}}\) in the target training fold:

\[
B_{+}=\operatorname{round}(B p_{\mathrm{TDC}}),
\qquad
B_{-}=B-B_{+}.
\]

Candidates below a minimum train-relevance threshold were removed. Within each class, molecules were ranked by relevance but selected in scaffold rounds: the highest-relevance member of each Bemis–Murcko scaffold was selected before a second member of an already represented scaffold. This preserved the target class balance while preventing a large congeneric series from dominating the expansion.

For regression, the target distribution of the original training fold was divided into five reference quantile bins. The external budget was allocated across these bins to approximate the target-fold distribution. Within each bin, the same relevance-first, scaffold-diverse selection rule was applied. The core-set audit recorded the requested and eligible counts, selected count, unique scaffolds, minimum and mean relevance, and—for regression—the selected and reference median and interquartile range.

If one class or target bin could not supply its requested quota, the selector used a deterministic fill from the remaining eligible candidates and recorded the deficit. This preserved reproducibility while making coverage limitations explicit.

### External sample-mass control

To prevent a numerically large external source from overwhelming endpoint-native observations, ADMET-EvO separated the number of selected external records from their total training weight. Native training rows received unit weight. For \(N_{\mathrm{TDC}}\) native rows, \(N_{\mathrm{ext}}\) selected external rows and external-mass ratio \(\rho\), each external row received

\[
w_{\mathrm{ext}}=\rho\frac{N_{\mathrm{TDC}}}{N_{\mathrm{ext}}}.
\]

Thus, the total external sample mass was \(\rho N_{\mathrm{TDC}}\), regardless of source size. This cap reduced source-size bias and made different external budgets comparable.

### Adaptive data-ratio selection and locked confirmation

Data amount was treated as a tunable intervention rather than fixed after collection. During development, the Agent evaluated external core-set budgets corresponding to \(0.5\times\), \(1\times\), \(2\times\), \(4\times\) and \(8\times\) the number of native training rows, subject to availability and relevance filters. Each ratio used the same baseline representation, learner, partitions and development seeds. Ratio summaries included the mean oriented effect, standard error, positive-seed fraction and core-set audit.

Among ratios that passed the prespecified practical-effect and seed-stability gates, the Agent selected the smallest ratio whose mean effect lay within one standard error of the best-performing ratio:

\[
\rho^{*}=\min\left\{\rho:\bar{\Delta}_{\rho}
\geq \bar{\Delta}_{\mathrm{best}}-\operatorname{SE}(\Delta_{\mathrm{best}})\right\}.
\]

This one-standard-error rule favored the least data-intensive intervention statistically compatible with the best development effect. The selected ratio was then locked and evaluated with seeds disjoint from ratio development. Promotion required a positive mean oriented effect and the prespecified seed-consistency rule. If confirmation failed, the baseline was retained and the negative or unstable result updated contextual action memory. Test labels remained inaccessible throughout development and confirmation.

### Versioning and abstention

Every accepted expansion produced a versioned bundle containing the source manifest, checksums, EndpointContract, query ledger, curation log, exclusion and conflict tables, admission decision, firewall audit, core-set audit, weighting rule, development trajectory and confirmation result. Data acquisition was therefore allowed to end in a useful abstention: a qualified failure to merge external data was recorded as evidence about source compatibility and protected subsequent experiments from unsafe reuse.

## Discovery-only task profiling and HardnessCard

The Agent profiled discovery training and validation partitions without loading certification data. The HardnessCard represented six dimensions: scarcity, class imbalance, domain shift, local roughness or activity cliffs, data quality and split-leakage risk.

Scarcity depended on the effective number of training scaffolds. Classification imbalance was derived from minority prevalence. Domain shift combined unseen validation scaffolds, low-similarity validation molecules and mean nearest-neighbour similarity. Local roughness combined nearest-neighbour target disagreement and activity-cliff evidence. Data quality summarized invalid structures, non-finite labels, repeated identities and conflicts. Split-leakage risk was calculated from standardized train–validation overlap.

For task \(t\),

\[
H_t=\frac{\sum_k w_k h_{tk}}{\sum_k w_k},
\]

where \(h_{tk}\in[0,1]\) is the severity of dimension \(k\). Implemented weights were 0.18 for scarcity, 0.12 for classification imbalance, 0.25 for domain shift, 0.23 for local roughness, 0.12 for data quality and 0.10 for leakage risk. The full severity vector, rather than only the scalar summary, conditioned action routing.

## Molecular representations and predictive portfolio

ADMET-EvO selected an endpoint-specific predictive portfolio from a registered library. Two-dimensional representations included Morgan fingerprints at radii 2 and 3, count-based Morgan features, Avalon fingerprints, ErG reduced-graph features, MACCS keys, RDKit path fingerprints, hashed atom-pair fingerprints, hashed topological-torsion fingerprints and finite RDKit two-dimensional descriptors. A common multiview baseline concatenated Morgan, Avalon, ErG and RDKit2D features. Feature matrices were cached by content hash.

Registered learners included ExtraTrees, Random Forest, balanced ExtraTrees for classification, CatBoost, XGBoost, LightGBM, support-vector machines and a Morgan-feature multilayer perceptron. Optional pretrained graph or three-dimensional representations were activated only when installed and justified by a feature-action hypothesis. Missing dependencies were recorded and could not be silently replaced by a differently named surrogate.

Weighted ensembles were permitted only after each component was evaluated independently. A blend was compared with its strongest member under matched seeds and promoted only if it passed both the mean-effect and seed-consistency gates.

## Candidate selection and compute-aware routing

Protocol-valid candidates were ranked by a prespecified score-only, uncertainty-aware or compute-aware policy. Under compute-aware selection, candidate \(i\) received

\[
Q_i=d s_i-\lambda_u\sigma_i-
\lambda_c\frac{\log(1+t_i)}{\max_j\log(1+t_j)},
\]

where \(s_i\) is the validation score, \(\sigma_i\) its between-seed standard deviation, \(t_i\) observed runtime, and \(\lambda_u\) and \(\lambda_c\) uncertainty and cost penalties. Ties were resolved by lower runtime and deterministic run identifier. Selection-policy variants were fixed before certification and were not chosen from protected outcomes.

## Evidence ledger, contextual memory and self-correction

Plans, executions, failures, promotions, rollbacks and freezes were appended to a SHA-256 hash chain. Each entry referenced the previous digest, making deletion, reordering or mutation detectable. Negative and inconclusive experiments remained first-class evidence, and correction actions referenced their parent hypotheses.

Action memory stored the endpoint, task type, ADMET category, HardnessCard context, intervention specification, matched effect, stability, status and provenance. Cross-task evidence was shrunk toward an uninformative prior and could affect another endpoint only when the action and context were comparable. A single adverse result could suppress local reuse but could not create an unconditional global prohibition. Destructive pruning required repeated refutation and independent confirmation.

A refuted or inconclusive action could trigger one bounded correction. The correction planner selected an untried axis, inherited the remaining budget and generated a new falsifiable hypothesis. Merely scheduling a correction did not count as self-correction; the matched follow-up experiment had to complete. Contradictory confirmation downgraded memory and triggered rollback. If no valid action remained, the expected information gain was too low or the remaining budget was insufficient, the Agent stopped or abstained.

## Freeze lock and sealed certification

After candidate selection, the Agent serialized the complete endpoint-specific recipe, including representations, learner, preprocessing, hyperparameters, ensemble weights where applicable, seeds, dependency versions and source snapshot. Recipes, execution sources and prediction files were content-addressed by SHA-256. A finalization directory could not silently replace a previously frozen finalist.

Certification used a separate executor that reconstructed the model exclusively from the frozen recipe. The planner, data module and action memory were not invoked. Certification outcomes were appended to the audit chain but could not mutate the finalist within the same research lineage. Prospective blind predictions were frozen before labels or external evaluation feedback became available.

## Evaluation of predictive and autonomous-research behavior

Predictive utility and research behavior were assessed separately. Predictive analyses used endpoint-native metrics, paired oriented effects, seed stability, calibration where applicable, chemical-space transfer and performance–compute trade-offs. Autonomous-process analyses included contract coverage, diagnosis completeness, fault detection, unsafe-merger avoidance, matched-pair completeness, negative-result retention, self-correction completion, rollback, stopping, abstention, review routing and ledger integrity.

Leaderboard performance was treated as external evidence that the closed-loop procedure could produce competitive endpoint-specific predictors, not as the definition of autonomy. The primary methodological object was the auditable research trajectory linking a task contract to a frozen predictive product.

## Statistical analysis

The endpoint was the primary unit for cross-task inference. Seeds were paired robustness replicates rather than independent endpoints. Pairwise effects were calculated within endpoint and seed before aggregation, and cross-task confidence intervals were obtained by endpoint-level bootstrap resampling.

Classification used the contract-specified area under the receiver-operating-characteristic curve or area under the precision–recall curve. Regression used mean absolute error or Spearman correlation. Metric direction was normalized only for cross-task action analysis; all endpoint reports retained native values.

Minimum practical effects, non-inferiority margins, positive-seed requirements, overlap thresholds, fingerprint similarity cutoffs, source-admission rules and compute gates were fixed before the corresponding evaluation phase. An experiment that completed but failed its promotion criterion was reported as a completed negative result. Controlled-bridge results were reported at their restricted claim tier and were not described as direct endpoint-equivalent data integration.

## Reproducibility and artifact retention

The system recorded dependency versions, hardware route, source snapshot, input and output hashes, random seeds, experimental specifications, prediction arrays, runtimes, data audits and failures. Evidence bundles contained the EndpointContract, HardnessCard, task data registry, plan, action outcomes, query ledger, source manifests, exclusions, conflicts, admission report, contamination audit, core-set selection, weighting configuration, frozen recipe and certification artifacts.

Automated tests covered schemas, contracts, routing, metric direction, molecular curation, source admission, structure firewalls, adaptive core-set selection, evidence chains, source hashes, freeze locks, self-correction and policy analysis. Reproduction began from a frozen manifest and source snapshot rather than a mutable working directory.

---

## Manuscript positioning note

The main-text narrative should present ADMET-EvO as an autonomous, closed-loop research agent. The OpenADMET/TDC performance should appear as one independent validation of the system's ability to discover competitive endpoint-specific solutions, while the principal methodological contribution remains autonomous task formalization, evidence-guided intervention, score-blind data admission, adaptive dataset expansion, self-correction, stopping and auditable freeze.

The matching Markdown/Mermaid figure source is provided in `paper/methods/admet_evo_architecture_mermaid_v2.md`.
