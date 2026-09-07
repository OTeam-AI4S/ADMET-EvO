# Methods writing structure

The Methods section follows one continuous argument:

> **overall system → endpoint-specific research context → closed-loop scientific intervention → predictive implementation → evaluation and statistical support.**

The main text is organized into five major sections and eleven method subsections.

## Section hierarchy and narrative purpose

### 1. Overall architecture of ADMET-EvO

This section defines ADMET-EvO as an endpoint-conditioned research framework and presents the complete three-layer architecture. It explains what enters the system, which decisions are autonomous, how evidence returns to the research state, and what the system ultimately outputs.

- Three-layer system architecture: scientific task layer, autonomous research layer, and evidence/model-evolution layer.
- Unified state–action–experiment–evidence loop.
- Boundary between generative research planning and deterministic evaluation.
- **Figure 1:** overall architecture and information flow.

### 2. Endpoint-conditioned research-state construction

This section explains how a heterogeneous ADMET endpoint is converted into the scientific and empirical context required for research decisions.

#### 2.1 Endpoint contracts

- Endpoint and assay semantics.
- Metric, data-partition, molecular-identity, and external-evidence rules.
- Access permissions and protected-information boundaries.

#### 2.2 Discovery-side endpoint diagnosis

- Scientific, empirical, and historical components of the ResearchState.
- Scarcity, imbalance, domain shift, local roughness, data quality, and leakage-risk diagnostics.
- Translation from diagnosed limitations to testable candidate hypotheses.

### 3. Multi-axis closed-loop model and evidence evolution

This is the technical core of the Methods. It explains how ADMET-EvO turns endpoint diagnoses into executable scientific interventions and how experimental evidence changes later decisions.

#### 3.1 Falsifiable hypotheses and registered actions

- Hypothesis and ActionSpec definitions.
- Feature, Model, and Data scientific intervention axes.
- Replicate and Stop as loop-control actions.
- Study-specific action-selection policies and budgets.

#### 3.2 Matched execution and evidence feedback

- Control–treatment comparability and oriented intervention effects.
- Complete-pair requirements, failure accounting, and EvidencePacket construction.
- Negative-result retention, bounded correction, and stopping.

#### 3.3 Governed Data-axis evidence evolution

- Source discovery, assay/identity curation, admission, contamination firewall, coreset selection, and evidence-mass control.
- Relationship among broad/adaptive screens, Evidence Packet v5, Evidence Packet v5.4, and endpoint best-seed studies.
- Confirmation, rollback, and abstention for external evidence.

#### 3.4 Research memory and endpoint-specific output

- Evidence lineage and endpoint-specific FrozenRecipe.
- Independent readout, rollback, and claim-state assignment.
- **Figure 2:** multi-axis intervention and evidence-feedback loop.

### 4. Predictive realization and implementation

This section describes how registered research actions are instantiated as molecular representations, learners, experts, ensembles, and reproducible executable configurations.

- Two-dimensional, graph, and three-dimensional molecular representations.
- Tree, boosting, kernel, graph, pretrained, and ensemble learners.
- Configuration binding, dependency gates, seeds, predictions, and runtime records.
- **Table 1:** operational configurations of all evaluated protocols.

### 5. Evaluation design and statistical analysis

This section connects each claimed capability to an explicit evaluation question and comparator.

#### 5.1 Benchmark tasks and predictive metrics

- TDC-22 tasks and endpoint-native metrics.
- Predictive comparison, three-axis ablation, routing controls, applicability, and uncertainty.

#### 5.2 Research-process and safety evaluation

- Action validity, matched execution, budget compliance, negative-evidence use, stopping, and correction.
- Sealed protocol, leakage, data-governance, and freeze-integrity tests.
- Prospective CYP blind-freeze study.

#### 5.3 Statistical analysis

- Endpoint-level inference, paired seeds, endpoint bootstrap, and sign-flip tests.
- Direction-oriented aggregation, selection units, and separation of exploratory, confirmation, and certification estimates.

#### 5.4 Reproducibility and claim states

- Protocols, source dossiers, configurations, predictions, hashes, and frozen manifests.
- Certified, independently confirmed, development-only, exploratory/best-observed, and pending claim states.

---

# Methods

## Overall architecture of ADMET-EvO

ADMET-EvO was designed as an endpoint-conditioned autonomous research framework rather than a single predictive architecture. Given a molecular-property endpoint and its available evidence, the system aims to produce both an endpoint-specific predictive recipe and an auditable research lineage explaining why that recipe was retained, which alternatives were rejected, and what level of evidence supports the resulting claim. The optimization target is therefore not only a validation score. At each step, the system must decide which scientific intervention is most defensible under the remaining experimental budget and whether the resulting evidence is sufficient to continue, revise, or stop the research process.

The framework comprises three interacting layers. The **scientific task layer** converts a heterogeneous ADMET endpoint into an executable contract. The contract fixes the prediction target, assay interpretation, species where applicable, label semantics, task type, primary metric and direction, molecular-identity convention, data provenance, partition ownership, and external-evidence policy. This layer prevents the same modelling assumptions from being applied silently across endpoints that measure different biological or physicochemical processes.

The **autonomous research layer** constructs an endpoint-specific research state from the contract, discovery-side data diagnostics, previously executed interventions, available experts, and remaining budget. A planning component uses this state to formulate falsifiable hypotheses and to select among three scientific intervention axes: molecular representation, predictive model, and empirical training evidence. Proposed interventions must be compiled into registered, executable action specifications before they can be evaluated. Replication and stopping are treated as control actions rather than additional optimization axes.

The **evidence and model-evolution layer** converts completed experimental runs into structured evidence. Each evidence record contains the evaluated effect, uncertainty or stability information, computational cost, execution status, configuration and data provenance. Supported, inconclusive, refuted, and failed interventions are all retained. The resulting evidence updates the endpoint research state and may trigger another intervention, a bounded corrective experiment, retention of the current recipe, rollback to a previously supported configuration, or termination. The final output is an endpoint-specific recipe accompanied by the lineage of observations that support or contradict it.

The resulting research loop can be summarized as

\[
S_t \xrightarrow{\pi} a_t
\xrightarrow{\mathcal{E}} e_t
\xrightarrow{\mathcal{U}} S_{t+1},
\]

where \(S_t\) is the endpoint research state, \(\pi\) selects an admissible registered action \(a_t\), \(\mathcal{E}\) executes and evaluates the intervention, and \(\mathcal{U}\) updates the state from the resulting evidence \(e_t\). The language-model component could assist hypothesis formation and constrained action choice, but it could not assign its own evidence verdict, modify an endpoint metric or protected partition, or promote an incomplete experiment. Data processing, model fitting, metric computation, protocol checks, and evidence serialization were performed by deterministic components. In this separation, the agent chooses what evidence to acquire next, whereas the evaluator determines what the acquired evidence supports.

> **[FIGURE 1 HERE — Overall architecture of ADMET-EvO]**  
> The figure should show the three layers from left to right or top to bottom: (i) scientific task formalization through an EndpointContract; (ii) construction of the ResearchState followed by diagnosis, hypothesis formation and selection among Feature, Model and Data actions; and (iii) executable evaluation, EvidencePacket generation, memory update and endpoint-specific output. A feedback arrow should return measured evidence to the ResearchState. Protected evaluation labels should be drawn outside the planning boundary. The visual emphasis should be on control and information flow, not on Python modules.

## Endpoint-conditioned research-state construction

### Endpoint contracts

Each research episode began with an `EndpointContract`. For the TDC ADMET panel, one contract was instantiated for each of 22 endpoints. Contract fields described task identity and ADMET category; classification or regression type; assay and property semantics; species, units and transformations where defined; source family; primary metric and whether it was maximized or minimized; discovery and certification partitions; and the policy governing admissible external evidence. Contract validation preceded any diagnostic or modelling action. Missing or contradictory scientific fields produced an explicit abstention or a request for endpoint-specific resolution rather than an inferred default.

Partition ownership was included in the contract. Discovery-side procedures could use the training data and the validation information assigned to model development, but protected certification labels were not available to the planner or research memory. When contamination filtering required comparison with protected molecules, only structures were exposed to the evaluator-owned firewall; those structures and their labels were not returned to the planning context. Metric direction and partition definitions were immutable after the contract had been accepted.

### Discovery-side endpoint diagnosis

The research state combined scientific, empirical, and historical context. Scientific context came from the EndpointContract. Empirical context was summarized from discovery-side data only and included six dimensions: data scarcity, class imbalance when applicable, chemical-domain shift, local structure–activity roughness, data quality, and split-leakage risk. Scarcity reflected effective sample and scaffold coverage. Imbalance described minority-class prevalence. Domain shift used scaffold novelty and fingerprint-neighbour similarity between training and development molecules. Local roughness summarized disagreement among structurally similar molecules and the prevalence of activity-cliff-like neighbourhoods. Data quality included invalid structures, missing or non-finite targets, duplicate measurements, and conflicting labels. Leakage risk recorded molecular-identity and group overlap across discovery partitions.

These diagnostics were serialized as a `HardnessCard` containing component severities, diagnostic confidence, supporting statistics, and recommended classes of follow-up action. The complete formulas, component weights, sampling limits, and sensitivity analyses are provided in Supplementary Methods. The card was used to narrow the hypothesis space, not to dictate the final recipe. For example, domain shift could motivate a representation or applicability-domain experiment; local roughness could motivate an alternative learner or ensemble; scarcity could motivate a governed external-evidence action; and data-quality or leakage risk could motivate curation, quarantine, or abstention. Each such recommendation remained a hypothesis until tested through an executable comparison.

Historical research context comprised completed evidence records, including negative results, the current incumbent recipe, available model or representation experts, and remaining compute and action budgets. Thus, two endpoints with similar sample sizes could receive different candidate actions if their assay semantics, chemical neighbourhoods, previous intervention effects, or evidence risks differed. This state construction is the basis of endpoint-conditioned research: the system does not assume that a globally strong architecture is the appropriate next intervention for every ADMET task.

## Multi-axis closed-loop model and evidence evolution

### Falsifiable hypotheses and registered actions

A research proposal was represented as a falsifiable hypothesis and an executable `ActionSpec`. The hypothesis stated why a specific endpoint limitation might be reduced and the expected direction of the effect. The corresponding action specification identified the endpoint, active intervention axis, control and treatment configurations, primary metric, seed schedule, estimated resource requirement, and promotion or falsification rule. Free-text proposals could not be executed until they were resolved to a registered action with a valid configuration and budget.

Scientific interventions were grouped into three axes. **Feature actions** changed the molecular representation while holding the learner and admissible data fixed. **Model actions** changed the estimator, optimization, regularization, ensemble, or endpoint expert while holding representation and data fixed. **Data actions** changed the empirical evidence available for training through curation, weighting, subset selection, or admission of a compatible external source while preserving the designated model and feature configuration for that comparison. Replicate actions acquired additional paired seeds to distinguish stable effects from selection variance, and Stop actions retained the current evidence state when no defensible intervention remained.

Different experimental studies instantiated the action-selection policy differently. The initial effect-first policy ranked admissible actions by expected effect, diagnostic compatibility, uncertainty, cost, and scientific risk. Sequential EC–VOI variants additionally used evidence from completed matched comparisons to estimate the value of acquiring another observation. Data-axis controllers used packet- or configuration-specific development and confirmation gates. These policies were not combined retrospectively into a single score. The planner, candidate set, seed schedule, budget, promotion rule, and certification layer used by each study are listed in the operational-configuration table in Supplementary Methods.

### Matched execution and evidence feedback

Where the intervention design permitted, control and treatment runs shared the same endpoint contract, molecular identities, split, seed, evaluation metric, and all non-intervened settings. A treatment was therefore interpretable as a change along its registered axis rather than as an untracked bundle of representation, model, and data changes. Multi-axis factorial experiments were run separately to measure interactions after the corresponding single-axis configurations had been defined.

For a higher-is-better metric, the oriented paired effect was

\[
\Delta=m_{\mathrm{treatment}}-m_{\mathrm{control}},
\]

and for a lower-is-better metric it was

\[
\Delta=m_{\mathrm{control}}-m_{\mathrm{treatment}}.
\]

Positive \(\Delta\) therefore always denotes improvement. Raw endpoint metrics and their original directions were retained in all artifacts. When multiple seeds were available, the evidence record included the complete paired seed effects, their mean and dispersion, and the fraction of positive pairs. Study-specific material-effect and stability thresholds were fixed in the corresponding protocol rather than inferred after inspecting certification outcomes.

Matched-pair resources were reserved as a unit in the sequential research protocol. A pair was not started if the remaining candidate-fit or fit-time budget was insufficient to complete both arms. Only complete, same-seed, single-axis-verified comparisons could update an effect posterior or promote an action. Timeouts, failed fits, invalid outputs, and incomplete pairs remained in the lineage and consumed their registered resources; they could not be silently discarded. This prevented the research policy from receiving a favourable treatment result without its corresponding control.

Completed runs were converted into an `EvidencePacket` containing the action identity, parent state, control and treatment configurations, oriented effect, stability information, fit cost, status, input and output hashes, and a verdict of supported, inconclusive, refuted, or failed. The planner could use this structured evidence in later decisions but could not replace it with a textual assessment. Evidence records were appended to a hash-linked ledger, so altered ancestry invalidated downstream records. The ledger was used as scientific memory rather than a winner-only leaderboard: negative and failed actions remained available to discourage unsupported repetition and to motivate a more specific child hypothesis.

Correction was bounded. A corrective action had to name the deficiency revealed by the parent evidence, specify a new falsifiable change, and satisfy an independent promotion rule. Unless a separate protocol was registered, repeated open-ended correction was not allowed. An episode stopped when the action or compute budget was exhausted, no admissible action had positive defensible value, all candidate interventions failed their gates, or a safety condition required abstention.

### Governed Data-axis evidence evolution

Data-axis actions treated external measurements as scientific evidence rather than interchangeable training rows. The shared workflow comprised endpoint-specific source discovery, immutable acquisition, endpoint and assay curation, score-blind admission, contamination filtering, relevance- and diversity-aware subset construction, external-mass control, and independent evaluation. Each external evidence source was bound to a named endpoint and retained source-level and record-level provenance.

Molecules were standardized before matching, and duplicate identities, invalid structures, label conflicts, species or assay incompatibilities, unit transformations, and censored measurements were handled according to the endpoint contract. Admission considered source lineage, endpoint compatibility, usable sample size, exact target overlap, disagreement among overlaps, distribution shift, and chemical coverage. A missing required field or failed compatibility condition produced quarantine or rejection rather than an automatic merge.

An evaluator-owned structure firewall removed exact and near-analogue overlaps with protected partitions and could reject an entire source when overlap indicated a shared or derivative benchmark lineage. Protected labels were never used for admission or subset selection. Within an admitted source, classification subsets preserved class composition, regression subsets covered the native target distribution, and scaffold-aware selection limited redundancy. External observations received less total influence than native observations through row-level reliability weights and an external-to-native mass cap. Quantity and weight settings were selected on development evidence under a low-mass preference or one-standard-error rule where specified.

The Data axis was evaluated through several protocol instances that answer different questions. Broad and adaptive screens identified endpoints and data regimes with potential gains but were treated as exploratory. Evidence Packet v5 tested whether small, source-traceable evidence packets transferred across confirmation seeds. Evidence Packet v5.4 introduced a task-specific frozen recipe and separate development, confirmation, and certification gates. Single-endpoint best-seed studies estimated best-observed potential for selected data actions and were not substituted for multi-seed certification. These studies therefore belong to one Data axis but occupy different claim states.

An external-data configuration that failed confirmation or certification was rolled back to the native-data baseline or the previous supported incumbent. The ability to reject, abstain, or roll back is part of the Data-axis method: more external observations were not assumed to be better, and contamination control was treated as necessary but not sufficient for assay transfer.

### Research memory and endpoint-specific output

At termination, the framework produced a `FrozenRecipe` describing the selected representation, model, preprocessing, data state, and ensemble composition, together with the evidence lineage supporting that selection. Recipes were endpoint specific; the framework did not require one representation or learner to dominate all tasks. A recipe entering an independent readout was frozen with its protocol, source packets, selection rule, seeds, and implementation hashes. The protected result could update the claim state but could not reopen selection. Failed independent gates triggered rollback or retention of the baseline. This mechanism is included to preserve the meaning of the research output; it is not treated as a separate optimization contribution.

> **[FIGURE 2 HERE — Multi-axis intervention and evidence-feedback loop]**  
> The figure should start from an endpoint-specific ResearchState and branch into Feature, Model and Data actions. Each branch should show a matched or otherwise protocol-bound experiment returning effect, stability, cost and provenance to a common EvidencePacket. From the evidence node, arrows should lead to retain, revise, replicate, rollback or stop, and then return to the ResearchState. The Data branch should contain a compact inset showing source → assay/identity checks → firewall → weighted evidence. Negative evidence should visibly return to memory rather than disappear.

## Predictive realization and implementation

The registered representation portfolio covered complementary two-dimensional, graph, and three-dimensional molecular views. Fixed two-dimensional representations included Morgan fingerprints at multiple radii, count fingerprints, Avalon, ErG, MACCS keys, RDKit path fingerprints, atom-pair and topological-torsion fingerprints, and normalized RDKit physicochemical descriptors. Multi-view configurations concatenated prespecified representation blocks after block-specific preprocessing. Invalid molecular structures were logged and handled according to the EndpointContract.

Registered tabular learners included ExtraTrees, Random Forest, imbalance-aware tree ensembles, CatBoost, XGBoost, LightGBM, and an RBF support-vector model with a training-size cap. Graph and pretrained three-dimensional experts, including MapLight/GNN and Uni-Mol2-based variants, were activated only when their dependencies were available and the endpoint-specific acquisition or compute gate was satisfied. Candidate ensembles combined molecule-level predictions from compatible members; classification probabilities and regression predictions were retained for paired analysis, calibration, and applicability evaluation.

Model, feature, and data changes were bound to executable configurations rather than reconstructed from natural-language descriptions. Random seeds, split identifiers, preprocessing settings, model hyperparameters, dependency versions, device requirements, prediction shapes, and execution times were stored with each run. The exact predictive portfolio and search spaces differed among the broad benchmark, controlled axis studies, Data-axis evidence protocols, and prospective CYP study. To avoid conflating these experimental generations, their operational settings are reported separately.

> **[TABLE 1 HERE — Operational configurations of the evaluated research protocols]**  
> Recommended columns: study; endpoints; active intervention axis or axes; baseline/frozen recipe; planner or controller; development seeds; confirmation seeds; certification partition/seeds; action or fit budget; promotion rule; protected information; final claim state. Rows should include the TDC predictive benchmark, three-axis ablation, routing controls, Evidence Packet v5, Evidence Packet v5.4, single-endpoint best-seed studies, safety/fault studies, and the prospective CYP study.

## Evaluation design and statistical analysis

### Benchmark tasks and predictive metrics

The principal benchmark comprised 22 TDC ADMET endpoints spanning absorption, distribution, metabolism, excretion, and toxicity. The panel contained 13 classification and 9 regression tasks. Endpoint-native metrics were retained: AUROC for most classification tasks, AUPRC for highly imbalanced CYP tasks, MAE for absolute-error regression tasks, and Spearman correlation for rank-oriented pharmacokinetic tasks. Metric orientation was changed only when effects were aggregated; all endpoint tables retained the raw metric and direction.

Predictive comparisons addressed both absolute utility and endpoint heterogeneity. The source-traceable cross-system analysis compared ADMET-EvO with publicly reported ADMET systems only where task identity and metric direction could be audited. These comparisons were distinguished from uniform retraining experiments because published values could differ in implementation, pretraining, or compute. Within-project intervention experiments used bound splits and executable configurations and were therefore used for paired effect and causal-axis analyses.

The three-axis study evaluated representation, model, and data ladders relative to a fixed baseline and included factorial cells to estimate pairwise and three-way interactions. Routing studies assessed whether endpoint-conditioned action selection improved the performance–cost trade-off relative to score-only choice, matched-budget random selection, expanded AutoML, and deterministic policies. Data-axis studies separately reported exploratory screening, confirmation-stage retention, strict certification, and best-observed endpoint results. This separation prevented a development maximum from being reported as an independently certified mean effect.

Applicability and uncertainty analyses used molecular similarity and ensemble dispersion as distinct signals. Classification calibration was summarized using Brier score and adaptive expected calibration error. For regression tasks with conformal analysis, interval coverage and efficiency were reported on data not used to fit the underlying model. A combined applicability signal was retained only if it improved on both constituent signals under its registered gate.

### Research-process and safety evaluation

Research-process evaluation measured whether the system generated valid actions, completed matched comparisons, retained negative evidence, respected action and compute budgets, performed registered corrections, and stopped or abstained under the appropriate conditions. Final predictive performance alone was not used as evidence that these intermediate research behaviours occurred.

Protocol-safety evaluations used sealed fixtures representing contract mutation, label access, schema failure, data leakage, source-lineage conflict, unsafe data merging, execution failure, and freeze-integrity violations. Evaluated outputs were the decision, evidence status, and prescribed remediation rather than a molecular prediction score. Comparators included conventional and expanded AutoML, ablated policies, and complete deterministic rules. Because prespecified fixtures can favour complete rules, fixture performance was interpreted as verification that controls executed as designed, not as evidence of unrestricted scientific reasoning.

A prospective multi-endpoint CYP study provided a separate protocol-completion test. Four direct-inhibition regression endpoints and two time-dependent-inhibition classification endpoints were assigned chemistry-aware folds, and recipe selection was performed without access to the final external labels. External evidence was routed separately by assay type. Prediction matrices were schema validated, hashed, and frozen before external feedback. Until an immutable external readout was returned, this study was reported as a completed blind-freeze protocol rather than evidence of external predictive superiority.

### Statistical analysis

The endpoint was the primary unit for panel-level inference. Seeds were treated as paired repetitions of an endpoint-level intervention rather than as independent endpoints. For matched comparisons, all prespecified seed pairs were retained. Panel confidence intervals were obtained by bootstrap resampling endpoints, and small paired intervention studies used sign-flip tests where appropriate, with their limited power reported explicitly.

Metrics on different scales were never averaged in raw form. Cross-endpoint summaries used direction-oriented effects, baseline-normalized improvements under the registered formula, or within-task ranks. Selection and reporting units were declared for each protocol. Where a one-standard-error rule was used, configuration selection occurred using development evidence and favoured the lowest-complexity or lowest-external-mass configuration within the prespecified performance range. Certification results were not used to reselect configurations.

The default material-effect threshold and practical-noninferiority margin were defined in each frozen protocol; protocol-specific values take precedence over framework defaults. Timeouts, failed fits, missing artifacts, rejected sources, inconclusive comparisons, and failed confirmation or certification gates remained in all denominators appropriate to their analysis. Exploratory envelopes and best-seed results were labelled separately from multi-seed confirmation and certification estimates.

### Reproducibility and claim states

Reproducibility artifacts included validated EndpointContracts, split manifests, executable configurations, source dossiers, Evidence Packets, molecule-level predictions where redistribution was permitted, paired summaries, compute records, ledger events, protocol hashes, and frozen recipe manifests. Software environments, dependency versions, seeds, devices, and hardware-dependent wall times were reported with their corresponding experiment. Hash validation and reconstruction checks were used to confirm that certification artifacts corresponded to the frozen configuration.

Each result was assigned a claim state: **certified**, **independently confirmed**, **development-only**, **exploratory/best-observed**, or **pending external readout**. This state accompanied the reported value so that a locally promising intervention could be distinguished from an effect that survived independent evaluation. Complete schemas, protocol-specific thresholds, model hyperparameters, source-admission records, and artifact maps are provided in Supplementary Methods.
