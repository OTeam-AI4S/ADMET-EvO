# ADMET-EvO: An evidence-gated autonomous research agent for evolving ADMET models and chemical evidence

**Authors:** [Author names]

**Affiliations:** [Affiliations]

**Correspondence:** [Corresponding author and email]

## Abstract

Predictive models of absorption, distribution, metabolism, excretion and toxicity (ADMET) are commonly optimized as isolated benchmark tasks, although real discovery programmes require a sequence of scientific decisions: defining an endpoint, diagnosing the available data, selecting an appropriate representation and learner, deciding whether external evidence is admissible, quantifying uncertainty and freezing a result without contaminating blind evaluation. Here we introduce **ADMET-EvO**, an autonomous research agent that converts each endpoint into an auditable research state and evolves endpoint-specific modelling recipes under explicit evidence, cost and safety gates. The system combines endpoint contracts, hardness diagnosis, matched model interventions, a hash-chained evidence ledger, bounded self-correction and score-blind external-data admission. Across 22 Therapeutics Data Commons ADMET tasks, ADMET-EvO achieved the second-best mean rank among eight source-traceable systems (2.98), with four task wins, while a compute-aware policy retained practical noninferiority to score-only selection and reduced measured selection time by 72.2%. Controlled interventions showed that useful actions were endpoint dependent: adding a graph expert improved Bioavailability AUROC by 0.0235 across five paired seeds, whereas doubling a Uni-Mol2 model from 84M to 164M parameters did not improve Lipophilicity. Prespecified failure challenges further showed perfect fault detection and data-admission decisions, although complete hand-written rules matched the agent on these sealed fixtures and the present routing policy did not outperform expanded AutoML or matched-cost random selection. Finally, in a prospective six-endpoint cytochrome P450 study, the agent froze task-type-specific external-evidence routes, calibrated applicability estimates and submission artifacts before external labels were available; the blind readout remains pending. Together, these results position ADMET-EvO not as a universally superior predictor, but as a reproducible scientific decision system that links chemical evidence, model intervention, compute allocation and protocol integrity.

## Introduction

ADMET properties determine whether molecular activity can be translated into a viable medicine. Their prediction is difficult because the relevant assays span distinct biological mechanisms, measurement scales and data regimes. Aqueous solubility and lipophilicity are continuous physicochemical properties; blood–brain barrier penetration and cardiotoxicity are classification problems coupled to transport and ion-channel biology; clearance and half-life reflect organ-level processes; and cytochrome P450 (CYP) inhibition, substrate recognition and time-dependent inhibition depend on enzyme-specific chemistry. Treating these endpoints as interchangeable rows in a leaderboard obscures the scientific choices that dominate real-world performance: molecular identity resolution, assay compatibility, leakage-resistant splitting, representation choice, calibration and the conditions under which external data may be used.

Benchmark collections such as MoleculeNet and the Therapeutics Data Commons (TDC) have made molecular prediction more reproducible [4,5], while pretrained molecular encoders and multimodal architectures have improved several individual tasks [17–22]. Yet high-capacity scaling is not uniformly useful in the small, heterogeneous datasets typical of ADMET [6]. A larger model can be poorly matched to a local endpoint; a nominally relevant external dataset can introduce assay or unit mismatch; and a small mean gain can disappear under new seeds or chemistry splits [7,24,25]. The practical problem is therefore not simply to train the largest available predictor, but to decide which scientific action is warranted for a particular endpoint and to preserve the evidence that justified it.

Autonomous scientific agents offer a possible route from fixed pipelines to adaptive research systems [1–3]. Existing chemistry agents can plan tool use, retrieve information or coordinate computational tasks [14–16], and general-purpose research agents can generate and evaluate hypotheses [23,26,27]. However, an ADMET agent must meet a stricter standard than successful execution. It must distinguish development from certification, prevent access to hidden labels, compare interventions against matched controls, account for compute, abstain from unsafe data use, retain negative results and freeze the final decision before external feedback. Without these constraints, autonomy can amplify rather than resolve benchmark leakage, selective reporting and post hoc model choice.

We developed **ADMET-EvO** around the principle that the atomic output of an autonomous modelling run is not a prediction alone but an auditable scientific claim. For each endpoint, the agent creates an endpoint contract, estimates a multidimensional hardness profile, proposes only registered actions, executes paired interventions, converts outcomes into support, refutation or uncertainty, and appends the complete state transition to a cryptographically linked ledger. External evidence passes through a separate score-blind admission route that assesses chemical identity, assay semantics, overlap, conflicts and domain proximity before any outcome-based model selection. A conditional committee and a single bounded correction cycle provide escalation without open-ended search.

We evaluate the system at four complementary levels. First, we test whether it constructs complete, replayable research states across a heterogeneous 22-task ADMET panel. Second, we quantify predictive rank and compute-aware selection. Third, we use controlled interventions to ask a chemical question: which endpoints benefit from additional representation, model or data expertise? Fourth, we challenge the agent with prespecified faults, unsafe data dossiers and a prospective multi-endpoint CYP study whose external readout is withheld. This design supports a more informative claim than leaderboard superiority alone: ADMET-EvO can allocate modelling and evidence-acquisition effort according to endpoint-specific chemistry while preserving the boundary between exploration and confirmatory evaluation.

## Results

### 1.1 Endpoint contracts convert heterogeneous assays into executable research objects

ADMET-EvO first formalized every task as an endpoint contract containing the prediction target, assay interpretation, label semantics, units, transformation, primary metric, metric direction, data provenance, development split, certification split and external-data policy. All 22 TDC endpoints produced complete contracts before model search. The panel included 13 classification and 9 regression tasks spanning absorption, distribution, metabolism, excretion and toxicity, forcing the agent to reason over both statistical and chemical heterogeneity rather than reuse a single global recipe.

The subsequent hardness diagnosis was performed without loading the held-out test labels. A composite score summarized scarcity, imbalance, domain shift, local structure–activity roughness, data quality and leakage risk. Domain shift was the dominant estimated difficulty, and 17 of 22 endpoints were routed to feature/domain-shift actions; four were routed to imbalance-aware modelling and one to leakage-safe data handling. This endpoint-level state is the basis for later action selection.

**Table 1 | Endpoint-contract and hardness-diagnosis evidence.**

| Evidence item | Result | Interpretation |
|---|---:|---|
| Endpoint contracts completed | 22/22 | Every endpoint was specified before search |
| Required contract fields populated | 13/13 per endpoint (100%) | No missing task, assay, split or metric field |
| Classification / regression tasks | 13 / 9 | Heterogeneous task types were represented |
| Hardness packets completed | 22/22 | Every endpoint received a diagnosis |
| Held-out test data loaded during diagnosis | 0/22 | Diagnosis was score blind |
| Mean / median hardness | 0.313 / 0.312 | Moderate aggregate difficulty |
| Primary route: feature/domain shift | 17/22 | Most endpoints were representation- or domain-limited |
| Primary route: imbalance-aware model | 4/22 | Minority-class handling was prioritized |
| Primary route: leakage-safe data | 1/22 | Data governance was the primary concern |
| Mean domain-shift severity | 0.748 | Largest diagnosed component |
| Mean local-roughness severity | 0.216 | Structure–activity discontinuity was secondary |
| Mean imbalance / scarcity severity | 0.208 / 0.196 | Important but not universal constraints |

### 1.2 Falsifiable lineages turn model search into cumulative endpoint knowledge

For each proposed intervention, the agent recorded a hypothesis, a matched control, the permitted axis of change, a primary metric and an outcome rule before execution. Outcomes were classified as supported, inconclusive or refuted, so unsuccessful actions remained part of the reusable endpoint history. For example, a leakage-safe intervention for BBB penetration produced a small reproducible gain, whereas a model intervention for CYP2D6 inhibition was refuted. The CYP2C9 experiment remained inconclusive after one timed-out replicate and was not silently promoted.

This organization makes the agent's scientific unit a lineage rather than a best score. A future run can reuse evidence that an action failed under a defined protocol, distinguish absence of benefit from absence of evidence, and avoid repeatedly paying for the same unproductive intervention.

**Table 2 | Representative preregistered intervention lineages.**

| Endpoint | Registered route | Paired evidence | Mean oriented delta | Positive pairs | Verdict |
|---|---|---:|---:|---:|---|
| BBB_Martins | Leakage-safe data handling | 5 seeds | +0.007006 AUROC | 4/5 | Supported |
| CYP2C9_Veith | Imbalance-aware model | 4 completed pairs + 1 timeout | −0.000889 AUROC | Not promoted | Inconclusive |
| CYP2D6_Veith | Imbalance-aware model | 5 seeds | −0.009210 AUROC | 0/5 | Refuted |

### 1.3 Immutable certification separates exploration from evidence release

All planning, execution, evaluation and certification events were appended to a hash-chained ledger. The audited snapshot contained 834 records with a valid chain, including failed and incomplete events. TDC certification was frozen for all 22 endpoints, and the prospective CYP recipe was separately frozen before blind-label access. These controls do not improve a score directly; they make the claimed result reconstructable and constrain the agent from reopening selection after observing external outcomes.

**Table 3 | Audit and freeze evidence.**

| Audit object | Result | Status |
|---|---:|---|
| Hash-chained ledger records | 834 | Complete snapshot |
| Completed / failed records | 789 / 1 | Failure retained in history |
| Planning / evaluation records | 22 / 22 | One auditable cycle per TDC endpoint |
| Chain validation | Valid | No broken ancestry detected |
| Ledger head | `2f87f179…51a6e` | Snapshot identifier |
| Software tests | 570 passed, 1 skipped | Tested evidence snapshot |
| TDC certification freezes | 22/22 | Selection closed before reporting |
| TDC bundle SHA-256 | `693db693…8f9e` | Immutable result bundle |
| Prospective CYP B1 recipe SHA-256 | `2182b275…d1e` | Frozen baseline recipe |
| Blind labels available at freeze | No | No outcome leakage |
| Selection reopened after freeze | No | Protocol integrity preserved |

### 2.1 ADMET-EvO is broadly competitive across 22 source-traceable ADMET tasks

We compared ADMET-EvO with seven systems for which task-level values could be traced to a stable source table or artifact. ADMET-EvO achieved the second-best mean rank (2.98), behind MapLight+GNN (2.50), and recorded the best value on Half-Life, Lipophilicity, plasma protein binding (PPBR) and Solubility. The pattern is chemically informative: the strongest agent results were concentrated in small-data continuous-property tasks, whereas pretrained or graph-augmented systems more often led large classification tasks. ADMET-EvO did not dominate all endpoints; its value was broad competitiveness across different assay families using endpoint-specific recipes.

Because historical web leaderboards can change and differ in task coverage, a separate snapshot-based estimate (5/22 top-1, 18/22 top-5 and 21/22 top-10) is retained only as sensitivity analysis and is not used as the primary comparative claim.

**Table 4 | Source-traceable task-level comparison on the 22-task TDC ADMET panel. Lower is better for MAE; higher is better otherwise. Bold indicates the best value in each row.**

| Task | Metric | ADMET-EvO | MiniMol | MolE | MapLight | MapLight+GNN | AttentiveFP | AttrMasking | ContextPred |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| AMES | AUROC | 0.874 | 0.849 | **0.883** | 0.868 | 0.869 | 0.814 | 0.842 | 0.837 |
| BBB_Martins | AUROC | 0.921 | **0.924** | 0.903 | 0.916 | 0.913 | 0.855 | 0.892 | 0.897 |
| Bioavailability_Ma | AUROC | 0.731 | 0.689 | 0.654 | **0.741** | 0.740 | 0.632 | 0.577 | 0.671 |
| CYP2C9_Substrate | AUPRC | 0.434 | **0.474** | 0.446 | 0.415 | 0.437 | 0.375 | 0.381 | 0.392 |
| CYP2C9_Veith | AUROC | 0.799 | 0.823 | 0.801 | 0.786 | **0.859** | 0.749 | 0.829 | 0.839 |
| CYP2D6_Substrate | AUPRC | 0.721 | 0.695 | 0.699 | 0.713 | 0.720 | 0.574 | 0.704 | **0.736** |
| CYP2D6_Veith | AUROC | 0.721 | 0.719 | 0.682 | 0.723 | **0.790** | 0.646 | 0.721 | 0.739 |
| CYP3A4_Substrate | AUROC | 0.633 | 0.663 | **0.670** | 0.650 | 0.647 | 0.576 | 0.582 | 0.609 |
| CYP3A4_Veith | AUROC | 0.885 | 0.877 | 0.867 | 0.881 | **0.916** | 0.851 | 0.902 | 0.904 |
| Caco2_Wang | MAE | 0.283 | 0.350 | 0.310 | **0.276** | 0.290 | 0.401 | 0.546 | 0.502 |
| Clearance_Hepatocyte | Spearman | 0.456 | 0.446 | 0.381 | 0.466 | **0.498** | 0.289 | 0.413 | 0.439 |
| Clearance_Microsome | Spearman | 0.591 | 0.628 | 0.607 | 0.626 | **0.630** | 0.365 | 0.585 | 0.578 |
| DILI | AUROC | 0.920 | **0.956** | 0.577 | 0.887 | 0.917 | 0.886 | 0.919 | 0.861 |
| HIA_Hou | AUROC | 0.990 | **0.993** | 0.963 | 0.986 | 0.989 | 0.974 | 0.978 | 0.975 |
| Half_Life_Obach | Spearman | **0.590** | 0.495 | 0.549 | 0.562 | 0.557 | 0.085 | 0.151 | 0.129 |
| LD50_Zhu | MAE | 0.591 | **0.585** | 0.823 | 0.621 | 0.633 | 0.678 | 0.685 | 0.669 |
| Lipophilicity_AstraZeneca | MAE | **0.392** | 0.456 | 0.469 | 0.539 | 0.525 | 0.572 | 0.547 | 0.535 |
| PPBR_AZ | MAE | **7.302** | 7.696 | 8.073 | 7.660 | 7.526 | 9.373 | 10.075 | 9.445 |
| Pgp_Broccatelli | AUROC | 0.912 | **0.942** | 0.915 | 0.930 | 0.938 | 0.892 | 0.929 | 0.923 |
| Solubility_AqSolDB | MAE | **0.739** | 0.741 | 0.792 | 0.792 | 0.789 | 0.776 | 1.026 | 1.040 |
| VDss_Lombardo | Spearman | 0.673 | 0.535 | 0.654 | 0.707 | **0.713** | 0.241 | 0.559 | 0.485 |
| hERG | AUROC | 0.860 | 0.846 | 0.813 | 0.873 | **0.880** | 0.825 | 0.778 | 0.756 |
| **Mean rank** | — | **2.977** | 3.364 | 5.114 | 3.477 | **2.500** | 7.182 | 5.886 | 5.500 |
| **Task wins** | — | **4** | 6 | 2 | 2 | **7** | 0 | 0 | 1 |

### 2.2 Compute-aware selection retained practical accuracy while reducing measured search time

To test whether the agent could preserve useful performance without exhaustive score-only selection, we separated development seeds (1–3) from fresh evaluation seeds (4–5) and compared the compute-aware and score-only policies endpoint by endpoint. The mean oriented difference was −0.00035, and its endpoint-bootstrap confidence interval remained above the prespecified practical-noninferiority margin of −0.005. Twenty of 22 endpoints were within the margin. Measured selection time fell from 2.25 to 0.62 hours, a 72.2% reduction. The largest saving occurred for LD50, whereas the compute-aware policy overspent on CYP2C9_Veith; therefore the result supports panel-level efficiency, not uniform per-endpoint optimality.

**Table 5 | Compute-aware versus score-only selection.**

| Quantity | Estimate | Prespecified interpretation |
|---|---:|---|
| Fresh-evaluation endpoints | 22 | Seeds 4–5 held out from development |
| Mean oriented score difference | −0.000348 | Compute-aware minus score-only |
| 95% endpoint-bootstrap CI | [−0.001482, 0.000585] | Above the −0.005 margin |
| Practical-noninferiority margin | −0.005 | Prespecified |
| Endpoints within margin | 20/22 | Two material regressions |
| BBB_Martins difference | −0.008968 | Material regression |
| CYP2D6_Substrate difference | −0.005552 | Material regression |
| Score-only measured time | 2.2468 h | Reference cost |
| Compute-aware measured time | 0.6247 h | 27.8% of reference |
| Measured time reduction | 72.2% | Panel-level efficiency gain |
| LD50 cost ratio | 0.0267 | 97.3% saving |
| CYP2C9_Veith cost ratio | 7.680 | Endpoint-level overspend |

### 2.3 Predictive confidence degrades outside the local chemical domain

Across the classification panel, the mean Brier score was 0.1173 and adaptive expected calibration error was 0.0964. More importantly for prospective use, compounds in the low-similarity region had higher loss in 11 of 13 classification tasks and 7 of 9 regression tasks. Thus, chemical distance supplied a broadly useful warning signal even when global calibration was acceptable. We used this evidence to expose applicability information rather than to convert uncertainty into an unsupported universal error guarantee.

**Table 6 | Calibration and chemical-domain evidence across the TDC panel.**

| Quantity | Result | Interpretation |
|---|---:|---|
| Mean classification Brier score | 0.1173 | Global probabilistic accuracy |
| Mean adaptive expected calibration error | 0.0964 | Residual calibration gap |
| Classification tasks with higher low-similarity loss | 11/13 | Chemical distance predicts difficulty in 84.6% |
| Regression tasks with higher low-similarity loss | 7/9 | Chemical distance predicts difficulty in 77.8% |
| All tasks showing the expected similarity–loss relation | 18/22 | Broad but not universal applicability signal |

### 2.4 Autonomous routing did not outperform stronger search controls

We deliberately kept this negative result concise. When the policy itself was compared with stronger controls, predictive superiority was not established. The agent trailed expanded AutoML on average, did not beat a matched-cost random policy, and outcome-complete memory did not improve over winner-only memory. These experiments bound the current contribution: the evidence architecture, compute accounting and safety controls are supported, but the present router should not be claimed as a generally superior optimization algorithm.

**Table 7 | Policy-level controls and negative gates.**

| Contrast (oriented score) | Mean difference | 95% CI | Wins/ties/losses | Gate |
|---|---:|---:|---:|---|
| Agent − expanded AutoML | −0.0098 | [−0.0277, 0.0065] | 5/7/10 | Noninferiority failed |
| Agent − strong fixed recipe | +0.0196 | [−0.0183, 0.0670] | 9/2/11 | Superiority not established |
| Expanded AutoML − strong fixed recipe | +0.0294 | [−0.0004, 0.0730] | — | Superiority not established |
| Full-outcome − winner-only memory | −0.00031 | [−0.00276, 0.00240] | — | Benefit failed; compute proxy +18.5% |
| Agent − matched-cost random | +0.00083 | [−0.00380, 0.00542] | — | Superiority failed |

### 3.1 Matched interventions show that useful action classes are endpoint dependent

We next decomposed improvement into feature, model and data interventions on five sentinel endpoints. Each experiment altered only one registered axis and reused matched controls. Feature interventions were supported in three of five endpoints, whereas switching from Random Forest to ExtraTrees was refuted in four. Data filtering changed every split but produced no supported performance gain. Thus, an action category that is plausible in the abstract is not automatically useful for a particular endpoint.

Shared-control execution reduced the number of required fits without changing the comparison. This matters for autonomous research because evidence should be acquired at the least cost compatible with a valid paired contrast.

**Table 8 | Complete sentinel action grid.**

| Action class | Supported | Inconclusive | Refuted | Posterior support | Execution evidence |
|---|---:|---:|---:|---:|---|
| Feature intervention | 3 | 0 | 2 | 0.556 | 5 endpoints × 5 seeds |
| Model intervention (RF→ExtraTrees) | 0 | 1 | 4 | 0.278 | 5 endpoints × 5 seeds |
| Data filtering | 0 | 5 | 0 | — | Split changed in 25/25 pairs |
| **Total** | **3** | **6** | **6** | — | 15 episodes; 100 executed fits; 0 fit failures |
| Shared-control saving | — | — | — | — | 50 fits and 510.6 fit-seconds avoided |

### 3.2 Targeted expert acquisition helps selected endpoints, whereas uniform scaling does not

Paired model experiments provided a direct test of selective expert acquisition. Adding the graph neural network expert to MapLight improved Bioavailability AUROC by a mean of 0.0235 across five paired seeds and increased CYP2C9-substrate AUPRC by 0.0342. By contrast, increasing Uni-Mol2 from 84M to 164M parameters doubled runtime but decreased the oriented Lipophilicity score by 0.0093. The Bioavailability confidence interval excluded zero, although the five-pair exact sign-flip test remained underpowered. For CYP2C9 substrate, the interval crossed zero and the evidence is suggestive rather than confirmatory.

Chemically, the pattern is consistent with graph message passing contributing to endpoints in which local connectivity and functional-group context influence transport or enzyme recognition, whereas global model scaling alone does not resolve a small continuous physicochemical endpoint. The experiments support endpoint-specific acquisition, not a universal preference for GNNs or smaller models.

**Table 9 | Paired expert-acquisition experiments. Oriented deltas are positive when the intervention improves the endpoint metric.**

| Endpoint | Intervention | Pairs | Mean oriented delta | 95% CI | Positive pairs | Exact sign-flip *P* | Runtime ratio | Interpretation |
|---|---|---:|---:|---:|---:|---:|---:|---|
| Bioavailability_Ma | MapLight+GNN − MapLight | 5 | +0.023526 AUROC | [0.000137, 0.046915] | 4/5 | 0.1250 | — | Supported effect; small-*n* test underpowered |
| CYP2C9_Substrate | MapLight+GNN − MapLight | 5 | +0.034225 AUPRC | [−0.042403, 0.110852] | 4/5 | 0.1875 | — | Positive estimate, uncertain interval |
| Lipophilicity_AZ | Uni-Mol2 164M − 84M | 5 | −0.009342 MAE-oriented | [−0.020574, 0.001889] | 1/5 | — | 2.031× | Scaling not supported |

### 3.3 Independent confirmation prevents global adoption of locally useful actions

Actions that passed the sentinel screen were evaluated on five fresh endpoints. Solubility and PPBR confirmed the candidate action in all five seeds, but CYP3A4 and Clearance refuted it and DILI was inconclusive. A global pruning rule would therefore have incurred positive regret, reaching 0.259 on the worst endpoint. The agent rolled back the global change and retained it only as endpoint-conditioned evidence. This result illustrates why heterogeneous ADMET chemistry should not be compressed into a single permanently preferred recipe.

**Table 10 | Fresh-endpoint confirmation and rollback.**

| Confirmation endpoint | Outcome | Positive pairs | Decision |
|---|---|---:|---|
| Solubility_AqSolDB | Supported | 5/5 | Retain endpoint-conditioned action |
| PPBR_AZ | Supported | 5/5 | Retain endpoint-conditioned action |
| CYP3A4_Veith | Refuted | — | Do not transfer globally |
| Clearance_Hepatocyte | Refuted | — | Do not transfer globally |
| DILI | Inconclusive | — | Preserve uncertainty |
| Total confirmation fits | 50/50 completed | — | No execution failure |
| Mean / maximum positive pruning regret | 0.065379 / 0.258620 | — | Global pruning rejected |

### 3.4 Richer diagnosis improved budgeted choice only in a constrained setting

A sequence of expected-compute/value-of-information (EC–VOI) studies tested whether diagnosis could improve action choice under a budget. Early versions failed against legacy, AutoML or anchor controls. In the constrained v4 experiment, diagnosis reduced oracle regret relative to no-diagnosis and shuffled-diagnosis controls at 50% budget, but one endpoint exceeded the harm threshold; in the broader cross-source v5 experiment, the full policy was outperformed by simpler controls. We therefore treat diagnostic value as a mechanistic signal requiring further validation, not a headline performance claim.

**Table 11 | EC–VOI development sequence and promotion gates.**

| Study | Scope | Key quantitative result | Gate |
|---|---|---|---|
| v1 | 79/80 fits; 1 timeout | Mean vs legacy −0.009598; cost 3.18× | Failed |
| v2 | 3 endpoints | Mean vs AutoML −0.032686; 0/3 wins | Failed |
| v3 | 3 endpoints | Nonloss achieved; 0/3 strictly above anchor | Failed |
| v4 | 135/135 fits; 15 decisions | Full gain 0.068074 vs no-diagnosis 0.058620 and shuffled 0.060002 at 50% budget; oracle regret 34.1% lower; 1 harmful endpoint | Failed harm gate |
| v5 | 415/415 rows; 4 sources, 16 endpoints | Full gain 0.076882, regret 0.045966; no-diagnosis gain 0.117604; fixed LightGBM 0.125995; expanded AutoML 0.114379 at cost 336 | Promotion failed |

### 3.5 Score-blind external-data acquisition identifies plausible gains but requires lineage-complete confirmation

The adaptive-data module admitted or rejected external sources using assay semantics, identity overlap, label conflicts and chemical-domain proximity before evaluating model scores. In development, nine of 22 endpoints selected augmentation and 13 abstained. Positive estimates appeared across permeability, transport, lipophilicity, protein binding, half-life and toxicity tasks. A separately frozen Caco2 confirmation produced a mean improvement of 0.0102 over four seeds, but its source lineage has not completed documentation clearance. We therefore report these data as exploratory and do not include them in the certified performance claim.

**Table 12 | Development-stage adaptive external-data decisions. Values are validation estimates, not blind-test results.**

| Endpoint | External/core ratio | Baseline | Augmented | Oriented delta | Positive seeds | Generalized admission | Status |
|---|---:|---:|---:|---:|---:|---|---|
| Caco2_Wang | 4.0 | 0.453471 MAE | 0.406983 MAE | +0.046488 | 2 | Yes | Development |
| Pgp_Broccatelli | 2.0 | 0.937267 AUROC | 0.953586 AUROC | +0.016320 | 2 | Yes | Development |
| Lipophilicity_AZ | 1.0 | 0.677604 MAE | 0.661294 MAE | +0.016310 | 3 | Yes | Development |
| BBB_Martins | 1.0 | 0.904566 AUROC | 0.925084 AUROC | +0.020518 | 2 | Yes | Development |
| PPBR_AZ | 0.5 | 11.494741 MAE | 11.271796 MAE | +0.222946 | 1 | No | Development-only exception |
| CYP3A4_Substrate | 0.5 | 0.606096 AUROC | 0.634309 AUROC | +0.028213 | 2 | Yes | Development |
| Half_Life_Obach | 1.0 | 0.044900 Spearman | 0.164055 Spearman | +0.119155 | 2 | Yes | Development |
| hERG | 2.0 | 0.768732 AUROC | 0.804847 AUROC | +0.036115 | 2 | Yes | Development |
| DILI | 1.0 | 0.756930 AUROC | 0.870019 AUROC | +0.113089 | 2 | Yes | Development |
| All other endpoints | — | — | — | — | — | Abstain (13/22) | No external data used |
| Caco2 frozen v2 confirmation | Frozen ratio | — | — | +0.010208 ± 0.018126 | 3/4 | Passed numerical gate | **Pending lineage clearance** |

### 4.1 The agent detects prespecified protocol faults, although complete rules remain competitive

In ten sealed failure cases, ADMET-EvO detected every injected fault, recovered all critical failures and proposed the registered remediation. Removing the hardness/evidence layer eliminated this ability. Expanded schema-based AutoML detected some faults but recovered only one sixth of critical cases, whereas conventional AutoML detected none. A complete hand-written rules baseline matched the agent, showing that the benefit is not mystical reasoning: it is the consistent execution of explicit scientific controls, coupled to the same state used for modelling.

**Table 13 | Prespecified hidden-fault challenge.**

| System | Precision | Recall | F1 | Critical recall | Remediation accuracy |
|---|---:|---:|---:|---:|---:|
| ADMET-EvO | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| Complete rules | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| Expanded schema AutoML | — | — | 0.615 | 0.167 | — |
| No-hardness ablation | — | — | 0.000 | 0.000 | 0.000 |
| Conventional AutoML | — | — | 0.000 | 0.000 | 0.000 |

### 4.2 Fail-closed data stewardship blocks unsafe external evidence

Eleven sealed data dossiers challenged identity, assay, unit, conflict, overlap and provenance controls. ADMET-EvO and the complete-rules baseline made all admission decisions correctly and produced no unsafe acceptance. Expanded schema AutoML accepted seven unsafe dossiers; direct fitting accepted ten. In a real hERG source-retrieval case, all three candidates required bridging and two ChEMBL requests returned HTTP 500 errors. The agent accepted no partial payload and launched no fit, demonstrating fail-closed behavior under infrastructure failure.

**Table 14 | Data-stewardship and real-source fail-closed evidence.**

| System or event | Decision accuracy | Issue F1 | Manifest completeness | Unsafe accepts | Fits launched |
|---|---:|---:|---:|---:|---:|
| ADMET-EvO, 11 sealed dossiers | 1.000 | 1.000 | 1.000 | 0 | As prescribed |
| Complete rules | 1.000 | 1.000 | 1.000 | 0 | As prescribed |
| Expanded schema AutoML | 0.273 | — | — | 7 | — |
| Direct-fit baseline | 0.091 | — | — | 10 | — |
| No-data baseline | 0.364 | — | — | 0 | 0 |
| Real hERG source candidates | 3 bridge-required | — | — | 0 | 0 |
| ChEMBL retrieval failures | 2 HTTP 500 events | — | No partial payload accepted | 0 | 0 |

### 4.3 Bounded correction improves decision safety without open-ended experimentation

Only two endpoints triggered the correction pathway. Both child experiments completed, neither cleared its promotion threshold, and the agent stopped without escalating to hyperparameter optimization or ensembling. In a separate ten-scenario composite decision benchmark, the agent achieved perfect decision and safety scores with 30 fits, while rules and expanded AutoML used 48 fits and produced unsafe actions. Conditional committee escalation was used on only two of 22 TDC tasks, avoiding most additional calls. This result supports bounded orchestration and stopping behavior rather than predictive uplift from self-correction.

**Table 15 | Bounded correction and composite-decision evidence.**

| Evaluation | Agent result | Comparator result | Interpretation |
|---|---|---|---|
| Scheduled / completed corrections | 2/2; 12 fits; 67.6 s | — | Correction budget honored |
| CYP2C9 child delta | −0.016534; inconclusive; stopped | — | No promotion |
| hERG child delta | −0.003124; inconclusive; stopped | — | No promotion |
| Composite benchmark | Decision, safety, parent, matched-control and macro scores = 1.000; 0 unsafe; 30 fits | Rules macro 0.86, 3 unsafe, 48 fits; expanded AutoML macro 0.54, 1 unsafe, 48 fits | Safer bounded decisions |
| Agent − rules macro difference | +0.14, 95% CI [0.04, 0.26] | — | Positive on sealed composite tasks |
| Committee escalation | 2/22 tasks; 2/2 corrections and 1/1 conflict recognized | — | 90.9% of calls and 92.4% of roles avoided |

### 5.1 A prospective CYP study begins with identity-resolved, task-specific data intake

We used a multi-endpoint OpenADMET CYP study as a prospective test of whether the complete decision system could be frozen before external evaluation. The source package contained direct reversible inhibition, time-dependent inhibition (TDI), Emax, single-concentration and blind-query tables. Canonical molecular identities, rather than raw rows, were the unit of splitting; 6,145 TDI molecules collapsed into 3,869 chemistry groups. The external blind table contained only names and SMILES at freeze time.

**Table 16 | Prospective CYP data intake and split evidence.**

| Data component | Rows / valid labels | Role |
|---|---:|---|
| Direct reversible-inhibition table | 4,905 rows | Four regression endpoints |
| CYP1A2 direct labels | 1,412 | Regression |
| CYP2C9 direct labels | 1,285 | Regression |
| CYP2D6 direct labels | 1,493 | Regression |
| CYP3A4 direct labels | 2,335 | Regression |
| TDI table | 6,145 rows | Classification development |
| Emax table | 6,146 rows | Supporting assay evidence |
| Single-concentration table | 17,504 rows | Supporting assay evidence |
| Blind query table | 750 molecules | External evaluation |
| CYP2D6 / CYP3A4 TDI positive prevalence | 21.6% / 21.3% | Imbalanced classification |
| Canonical identities / chemistry groups | 6,145 / 3,869 | Group-aware split units |
| Outer folds / sealed certification fraction | 5 / 12.51% | Internal development/certification firewall |
| Blind columns available | Molecule_Name, SMILES | No external labels available |

### 5.2 Multi-objective selection favors balanced assay performance over a single high score

Candidate recipes were evaluated across regression and classification objectives. The selected B1 recipe achieved an aggregate ST-RAE of 0.72594 and MCC of 0.24635. M3 improved neither combined objective, whereas M4 substantially worsened regression error despite a similar classification score. Selection therefore favored balanced cross-assay behavior rather than optimizing one CYP endpoint. All comparisons used five folds and three seeds and were frozen before any blind feedback.

**Table 17 | Prospective Stage-B recipe selection. Lower ST-RAE and higher MCC are better.**

| Recipe | Macro ST-RAE | Macro MCC | ST-RAE gain vs B1 | MCC gain vs B1 | Decision |
|---|---:|---:|---:|---:|---|
| B1 | **0.72594** | **0.24635** | Reference | Reference | Selected and frozen |
| M3 | 0.77439 | 0.11544 | −0.04844 | −0.13090 | Rejected |
| M4 | 0.90255 | 0.10740 | −0.17661 | −0.13895 | Rejected |
| Evaluation design | 5 folds × 3 seeds | 5 folds × 3 seeds | — | — | Fixed before blind readout |

### 5.3 External evidence is routed by assay type rather than pooled indiscriminately

The agent evaluated 29,008 external Veith records under task-type-specific controls. Propensity augmentation produced small positive estimates for all four direct-inhibition regression tasks but negative estimates for both TDI classification tasks. The frozen route consequently used propensity evidence only for regression and backed off to B1 for TDI. This is a chemical and assay-semantic decision: reversible inhibition potency and TDI labels are related to CYP biology but are not interchangeable supervised targets.

**Table 18 | Task-type-specific external-propensity route.**

| Endpoint | Task type | Control | Propensity model | Oriented delta | Frozen route |
|---|---|---:|---:|---:|---|
| CYP1A2 direct inhibition | Regression | 0.815609 | 0.807745 | +0.007864 | Use propensity |
| CYP2C9 direct inhibition | Regression | 0.674197 | 0.671791 | +0.002405 | Use propensity |
| CYP2D6 direct inhibition | Regression | 0.883185 | 0.881785 | +0.001400 | Use propensity |
| CYP3A4 direct inhibition | Regression | 0.530776 | 0.528249 | +0.002527 | Use propensity |
| CYP2D6 TDI | Classification | 0.133640 | 0.125682 | −0.007959 | Back off to B1 |
| CYP3A4 TDI | Classification | 0.359052 | 0.355178 | −0.003875 | Back off to B1 |
| External Veith evidence | 29,008 records | — | — | — | Used only through frozen route |

### 5.4 Applicability signals identify difficult chemistry but do not justify a more complex uncertainty model

Similarity and ensemble dispersion were evaluated as applicability signals on all six CYP endpoints. Combined signals achieved high-error AUROC above 0.5 on five of six endpoints, and the lowest-similarity quartile had higher loss on five of six. Adaptive 90% conformal intervals achieved approximately nominal pooled coverage for the four regression tasks. However, the combined signal beat both individual signals on none of the six endpoints, so the prespecified complexity gate failed and the simpler representation was retained.

**Table 19 | CYP applicability-domain and uncertainty evidence.**

| Evidence item | Result | Decision |
|---|---:|---|
| Endpoint/fold/seed records | 6 endpoints; 10,219 records; 5 folds × 3 seeds | Adequate internal coverage |
| Combined high-error AUROC > 0.5 | 5/6 endpoints | Applicability signal present |
| Lowest-similarity quartile has higher loss | 5/6 endpoints | Chemical-distance effect present |
| Regression adaptive conformal coverage | 0.898–0.900 pooled | Near nominal 90% |
| Combined signal beats both individual signals | 0/6 endpoints | Complexity gate failed |
| Blind labels used | No | Internal analysis only |

### 5.5 The complete CYP decision was certified and frozen before external scoring

Internal certification passed all four prespecified aggregate and worst-endpoint gates. Two 750-row prediction matrices—four regression columns and two classification columns—were then hashed and frozen. At manuscript freeze, no blind label, leaderboard score or external rank had been observed and selection had not been reopened. This result is therefore a prospective protocol-completion result, not yet evidence of external predictive superiority. The external score and rank must be added only after an immutable readout is returned.

**Table 20 | Internal certification and blind-freeze status.**

| Gate or artifact | Observed | Threshold / shape | Status |
|---|---:|---:|---|
| Macro ST-RAE | 0.684642 | ≤ 0.825942 | Pass |
| Macro MCC | 0.276628 | ≥ 0.146346 | Pass |
| Worst-endpoint ST-RAE | 0.947342 | ≤ 1.250000 | Pass |
| Worst-endpoint MCC | 0.115551 | ≥ 0.000000 | Pass |
| Regression artifact | 750 × 4; SHA `53d57a28…37ae` | Fixed schema | Frozen |
| Classification artifact | 750 × 2; SHA `614dc445…7231` | Fixed schema | Frozen |
| Blind labels / feedback available | No / No | Must remain unavailable before freeze | Pass |
| Selection reopened | No | Prohibited | Pass |
| External blind score and rank | Pending | Independent readout required | Not yet claimed |

## Discussion

ADMET-EvO reframes autonomous molecular modelling as evidence-gated scientific decision making. Its principal contribution is not a claim that one agent or learner wins every endpoint. Instead, it integrates endpoint semantics, chemical-domain diagnosis, intervention testing, data admission, compute accounting, uncertainty and immutable certification into a single research state. This distinction matters because ADMET is a collection of assay-specific inference problems, not a single homogeneous machine-learning task.

The results support two complementary strengths. First, the evolved recipes are broadly competitive: ADMET-EvO ranked second among eight traceable systems and led four tasks, with particularly strong results on several continuous-property endpoints. Second, the agent reduced the measured cost of recipe selection while preserving panel-level practical noninferiority. These strengths are connected. Efficiency was obtained by deciding where additional evidence was worth acquiring, not by applying a uniformly smaller or larger model.

The intervention experiments provide the clearest chemical story. Graph expertise improved Bioavailability and showed a positive, albeit uncertain, estimate for CYP2C9 substrate recognition, while scaling Uni-Mol2 did not improve Lipophilicity. External CYP propensity data helped direct-inhibition regressions but harmed TDI classifications. Fresh-endpoint confirmation showed that actions useful for Solubility and PPBR did not transfer to CYP3A4 or Clearance. Together, these observations argue for mechanism- and assay-conditioned modelling: representation, external evidence and model capacity should be acquired according to what an endpoint measures and how its chemistry differs from the development domain.

The negative results define equally important boundaries. The current router did not outperform expanded AutoML or matched-cost random selection, richer EC–VOI policies failed promotion in broader cross-source tests, and complete rules matched the agent on sealed fault and data-governance fixtures. Accordingly, the present evidence supports a rigorous orchestration layer more strongly than a novel universally optimal search policy. Future work should learn routing policies from a larger corpus of endpoint lineages while retaining preregistered controls, and should evaluate whether the agent adds value over complete rules on naturally occurring, previously unseen failures.

Several limitations remain. The source-traceable comparison combines values reported by different systems and is not a single uniform retraining study. The adaptive external-data results are developmental, and Caco2 confirmation remains outside the certified claim until source lineage is cleared. Applicability signals were useful but a combined uncertainty model failed its added-value gate. Most importantly, the prospective CYP study has completed internal certification and blind freezing but not external scoring. Reporting this pending status is central to the design: a prospective claim becomes informative only if the frozen result is later reported irrespective of outcome.

Subject to these limitations, ADMET-EvO offers a practical architecture for autonomous computational chemistry. It makes failure visible, allows the system to abstain, and binds every promoted recipe to its supporting evidence. Such systems may be especially valuable where datasets are small, endpoints are mechanistically diverse and the cost of an invalid scientific conclusion exceeds the cost of an additional model fit.

## Methods

### System overview and design principles

ADMET-EvO is a stateful research agent that maps an endpoint specification and available evidence to a frozen modelling recipe and an auditable claim. A run consists of: (i) endpoint-contract construction; (ii) score-blind hardness diagnosis; (iii) hypothesis and intervention registration; (iv) matched execution; (v) evidence grading; (vi) optional external-data admission; (vii) bounded correction or committee escalation; and (viii) certification and freezing. The agent may call only tools registered in its execution schema. A language model may propose a hypothesis or select among registered actions, but it cannot create arbitrary tools, alter the evaluation protocol, access sealed labels or bypass a failed gate.

The system was designed around six principles: endpoint specificity, axis-isolated intervention, immutable evidence, explicit compute cost, fail-closed data use and bounded autonomy. Deterministic validation and execution components were separated from generative planning so that a textual proposal could not itself change the certified state.

### Endpoint contracts

Each EndpointContract contained a stable identifier, ADMET category, prediction type, primary metric and direction, assay description, species, label semantics, units, transformations, source family, molecular-identity rule, discovery split, certification split and external-data policy. Contract validation was required before any modelling action. Classification and regression endpoints used different metric, stratification and external-data compatibility rules. Missing or contradictory fields produced an abstention rather than a default assumption.

### Score-blind hardness diagnosis

Hardness was estimated using only training and development data. Six normalized components were combined: scarcity (weight 0.18), classification imbalance (0.12), chemical-domain shift (0.25), local structure–activity roughness (0.23), data quality (0.12) and leakage risk (0.10). Domain shift was estimated from fingerprint-neighbour similarity and train–validation separation; local roughness compared label disagreement among structurally similar compounds; scarcity incorporated effective sample size; and leakage risk assessed identity and group overlap. The diagnostic packet retained component values, supporting statistics and a Boolean `test_loaded` field, which had to remain false.

### Registered action space and hypothesis generation

Actions were organized into Feature, Model, Data, Replicate and Stop axes. Feature actions selected or combined molecular views; Model actions changed the learner or a registered expert; Data actions altered curation, weighting or an admissible external source; Replicate acquired additional paired seeds; and Stop froze the current evidence. Each proposal specified a causal rationale, expected sign, control, treatment, permitted axis, primary endpoint, seed schedule, compute estimate and decision rule. The planner could use the endpoint contract, hardness packet and prior lineages but not held-out outcomes.

For deterministic compute-aware choice, candidate $i$ was scored as

\[
Q_i = d_i s_i - \lambda_u \sigma_i - \lambda_c
\frac{\log(1+t_i)}{\max_j \log(1+t_j)},
\]

where $d_i$ converts the metric to a common higher-is-better orientation, $s_i$ is the development estimate, $\sigma_i$ its uncertainty, $t_i$ measured fit time, and $\lambda_u$ and $\lambda_c$ are prespecified uncertainty and cost penalties. For explicit value-of-information experiments, action utility was

\[
U(a)=p\,\widetilde{E}_a C_a-(1-p)(0.55K_a+0.45R_a),
\]

where $p=0.80$ by default, $\widetilde{E}_a$ is the estimated oriented effect, $C_a$ diagnostic compatibility, $K_a$ normalized compute cost and $R_a$ risk. These utilities ranked registered actions; they did not override protocol or safety gates.

### Molecular representations and learners

Registered two-dimensional representations included Morgan radius-2 and radius-3 bit fingerprints, count Morgan fingerprints, Avalon, ErG (315 dimensions), MACCS (167 keys), RDKit path, atom-pair, topological-torsion and normalized RDKit2D descriptors. A multi-view representation concatenated Morgan, Avalon, ErG and RDKit2D blocks after block-specific preprocessing. Invalid molecules were logged and handled according to the endpoint contract.

Registered tabular learners included ExtraTrees, Random Forest, balanced ExtraTrees, CatBoost, XGBoost, LightGBM and an RBF support-vector machine capped at 5,000 training examples. Graph or three-dimensional experts, including MapLight+GNN and Uni-Mol2 variants, were available only when dependencies and endpoint-specific acquisition gates were satisfied. Classification probability outputs and regression outputs were retained at the molecule level for calibration and paired analysis.

### Matched interventions and evidence grading

Control and treatment shared the same endpoint contract, molecular identities, split, seed and all non-intervened settings. If $m$ is a higher-is-better metric, the paired oriented effect was $\Delta=m_T-m_C$; for lower-is-better metrics, $\Delta=m_C-m_T$. Unless a registered experiment specified otherwise, mean $\Delta\geq0.005$ was supported, mean $\Delta\leq-0.005$ was refuted, and intermediate or incomplete evidence was inconclusive. A timeout, failed fit or missing artifact remained in the lineage and could not be discarded when calculating the verdict.

Shared controls were reused only when their split, seed, representation and preprocessing hashes matched exactly. Fresh-endpoint confirmation used endpoints that were excluded from the discovery experiment. A candidate global rule was rolled back if confirmation revealed material positive regret or crossed the harm threshold.

### Evidence ledger, memory and bounded correction

Every state transition was serialized with its parent hash, timestamp, endpoint, action, configuration, input hashes, output hashes, status and verdict. The record hash was calculated over canonical serialized content, producing a chain in which altered ancestry invalidated downstream records. The evidence ledger stored supported, inconclusive, refuted, failed and stopped actions; action memory was not restricted to winners.

At most one correction cycle was allowed per trigger unless a separate protocol had been preregistered. Correction required a named deficiency, a child hypothesis and an independent promotion threshold. Failure to clear the threshold returned the endpoint to Stop. Committee escalation occurred only for low confidence, cross-validator disagreement or a registered safety conflict and did not authorize additional model search by itself.

### Adaptive external-data workflow

External evidence passed through six phases: source discovery, immutable acquisition, endpoint-aware curation, score-blind admission, core-set construction and independent confirmation. Raw source payloads, retrieval metadata, licences where available and cryptographic hashes were stored before transformation. Curation standardized molecular identity, assay semantics, units, censored values, duplicates and conflicts.

For direct admission, a classification source required at least 100 valid identities and a regression source at least 30. The maximum invalid-molecule fraction was 0.01, conflict fraction 0.02 and exact target-identity overlap 0.10. Classification label disagreement among exact overlaps could not exceed 0.10; for regression, the median exact-target difference normalized by the target interquartile range could not exceed 0.25. The distribution-shift statistic could not exceed 0.50, and at least 5% of external compounds required a nearest-training Tanimoto similarity of 0.40 or greater. Bridge admission used stricter overlap (≤0.05) and shift (≤0.25) controls and required coverage ≥0.025. A failed or unavailable field caused abstention.

To protect validation chemistry, the fold firewall used radius-2, 2,048-bit Morgan fingerprints and excluded external molecules with Tanimoto similarity ≥0.80 to the held-out fold. Classification core sets preserved class prevalence. Regression core sets used five target-quantile bins. Scaffold-aware rounds controlled redundancy. Admitted external examples received weight

\[
w_{ext}=\rho\frac{N_{TDC}}{N_{ext}},
\]

where \(\rho\in\{0.5,1,2,4,8\}\). The smallest ratio within one standard error of the best development result was selected. Independent seeds or endpoints were then used for confirmation. Numerical success without complete source lineage was marked pending and excluded from certified claims.

### TDC benchmark evaluation

The principal benchmark contained 22 TDC ADMET tasks: 13 classification and 9 regression endpoints. AUROC was used for most classification tasks, AUPRC for imbalanced CYP-substrate tasks, MAE for continuous tasks in which absolute error was the registered metric and Spearman correlation for rank-oriented pharmacokinetic tasks. All comparisons were oriented only for aggregation; raw metrics and directions were retained in tables.

The primary cross-system analysis included ADMET-EvO, MiniMol, MolE, MapLight, MapLight+GNN, AttentiveFP, AttrMasking and ContextPred. Values were transcribed from source-traceable task tables and audited at the cell level. Rank was calculated within each task using the registered metric direction and then averaged across 22 tasks. A separate historical web snapshot was treated as sensitivity analysis because leaderboard contents and ranks can change.

For compute-aware selection, seeds 1–3 were development seeds and seeds 4–5 were fresh evaluation seeds. Score-only and compute-aware policies were compared on the same endpoints. The practical-noninferiority margin was −0.005 in oriented units. Wall time reported measured selection execution and was supplemented by a replay analysis; it was not presented as hardware-independent FLOPs.

### Calibration, applicability and uncertainty

Classification calibration was summarized by Brier score and adaptive expected calibration error. Regression and classification applicability analyses used nearest-neighbour molecular similarity and ensemble dispersion. Loss was compared between similarity quartiles, and high-error detection was quantified by AUROC. For prospective CYP regression, adaptive conformal intervals targeted 90% pooled coverage. A combined applicability signal was promoted only if it outperformed both constituent signals under the prespecified gate.

### Failure, safety and data-stewardship evaluation

Failure and governance evaluations used sealed, prespecified fixtures. Ten fault cases represented protocol, schema, leakage and execution failures; eleven data dossiers represented identity, assay, unit, provenance, conflict and overlap hazards. The evaluated output was the decision and registered remediation, not the predictive score. Comparators were conventional AutoML, expanded schema AutoML, ablations and a complete deterministic-rules system. Because these fixtures were locally constructed and the complete-rules baseline matched the agent in several tests, results were interpreted as verification of control execution rather than evidence of broad naturalistic reasoning.

### Prospective multi-endpoint CYP study

The prospective study included four direct reversible-inhibition regressions (CYP1A2, CYP2C9, CYP2D6 and CYP3A4) and two TDI classifications (CYP2D6 and CYP3A4). Molecules were canonicalized and assigned to chemistry groups before generating five outer folds and a sealed internal certification subset. Blind query molecules contained no accessible labels.

Recipe selection jointly considered standardized threshold-root absolute error (ST-RAE) for regression and Matthews correlation coefficient (MCC) for classification over five folds and three seeds. External Veith evidence was evaluated separately by task type. The route was frozen as propensity augmentation for regression and B1 backoff for classification. Internal certification required macro and worst-endpoint thresholds defined before artifact generation. Regression and classification matrices were schema-validated, hashed and frozen before any external feedback.

### Statistical analysis

The endpoint, not an individual seed, was the principal unit for panel-level inference. Paired seeds were used for intervention estimates; paired sign-flip tests were reported for five-pair expert-acquisition studies, with their limited power made explicit. Confidence intervals for panel differences were obtained by endpoint bootstrap. Metrics with different scales were never averaged in raw form; aggregation used oriented differences or within-task ranks. The default material-effect threshold and practical-noninferiority margin were 0.005 oriented units. All prespecified failed gates, timeouts and refuted hypotheses were retained in the reported evidence.

### Reproducibility and claim status

The evidence snapshot comprised validated endpoint contracts, executable configurations, molecule-level predictions where permitted, paired summaries, compute records, ledger events and frozen hashes. Reported claims were assigned one of four states: certified, independently confirmed, development-only or pending external readout. The source-traceable 22-task comparison and frozen TDC artifacts are certified; sentinel and fresh-endpoint experiments are controlled internal evidence; adaptive external-data results are developmental unless explicitly confirmed; and the OpenADMET CYP blind score remains pending.

## Data availability

TDC benchmark data are available through the Therapeutics Data Commons under its applicable terms. External datasets retain their original source and licence conditions. Audited task tables, experiment-level summaries, endpoint contracts, hash manifests and the frozen prospective prediction schemas will be released with the final article, subject to source-data redistribution restrictions. The external CYP blind labels were not available to the authors at manuscript freeze.

## Code availability

The ADMET-EvO implementation, reproducible environment specification, registered action schemas, test suite and scripts used to construct the evidence tables will be released in a versioned repository upon publication. Frozen result artifacts will be linked to a release commit and cryptographic manifest.

## Acknowledgements

[To be completed.]

## Author contributions

[To be completed using the CRediT taxonomy.]

## Competing interests

The authors declare no competing interests. [Revise if necessary.]

## References

1. Boiko, D. A., MacKnight, R., Kline, B. & Gomes, G. Autonomous chemical research with large language models. *Nature* **624**, 570–578 (2023). https://doi.org/10.1038/s41586-023-06792-0
2. Lu, C. et al. The AI Scientist: Towards fully automated open-ended scientific discovery. *arXiv* 2408.06292 (2024).
3. Ning, M. et al. Closed-loop Auto Research: A framework for autonomous scientific iteration. *arXiv* 2606.22731 (2026).
4. Huang, K. et al. Therapeutics Data Commons: Machine learning datasets and tasks for drug discovery and development. *Advances in Neural Information Processing Systems* **34** (2021).
5. Wu, Z. et al. MoleculeNet: A benchmark for molecular machine learning. *Chemical Science* **9**, 513–530 (2018). https://doi.org/10.1039/C7SC02664A
6. Dou, J. et al. Machine learning with small data for molecular property prediction. *Chemical Reviews* **123** (2023). https://doi.org/10.1021/acs.chemrev.3c00189
7. Fourches, D., Muratov, E. & Tropsha, A. Trust, but verify: On the importance of chemical structure curation in cheminformatics and QSAR modeling research. *Journal of Chemical Information and Modeling* **50**, 1189–1204 (2010). https://doi.org/10.1021/ci100176x
8. Daina, A., Michielin, O. & Zoete, V. SwissADME: A free web tool to evaluate pharmacokinetics, drug-likeness and medicinal chemistry friendliness of small molecules. *Scientific Reports* **7**, 42717 (2017). https://doi.org/10.1038/srep42717
9. Yang, H. et al. admetSAR 2.0: Web-service for prediction and optimization of chemical ADMET properties. *Bioinformatics* **35**, 1067–1069 (2019). https://doi.org/10.1093/bioinformatics/bty707
10. Dong, J. et al. ADMETlab: A platform for systematic ADMET evaluation based on a comprehensively collected ADMET database. *Journal of Cheminformatics* **10**, 29 (2018). https://doi.org/10.1186/s13321-018-0283-x
11. Xiong, G. et al. ADMETlab 2.0: An integrated online platform for accurate and comprehensive predictions of ADMET properties. *Nucleic Acids Research* **49**, W5–W14 (2021). https://doi.org/10.1093/nar/gkab255
12. Swanson, K. et al. ADMET-AI: A machine learning ADMET platform for evaluation of large-scale chemical datasets. *Bioinformatics* **40**, btae416 (2024). https://doi.org/10.1093/bioinformatics/btae416
13. Myung, Y. et al. Deep-PK: Deep learning for small molecule pharmacokinetic and toxicity prediction. *Nucleic Acids Research* **52** (2024). https://doi.org/10.1093/nar/gkae254
14. Averly, R. et al. LIDDIA: Language-based intelligent drug discovery agent. In *Proceedings of EMNLP 2025* (2025). https://doi.org/10.18653/v1/2025.emnlp-main.603
15. Ock, J. et al. AgentD: A large language model agent for modular task execution in drug discovery. *arXiv* 2507.02925 (2025).
16. Liu, Y. et al. DrugAgent: Automating AI-aided drug discovery programming through LLM multi-agent collaboration. *arXiv* 2411.15692 (2024).
17. Notwell, J. H. & Wood, M. W. MapLight: Learning molecular representations through integration of language and graph information. *arXiv* 2310.00174 (2023).
18. Kläser, K. et al. MiniMol: A parameter-efficient foundation model for molecular learning. *arXiv* 2404.14986 (2024).
19. Zhou, G. et al. Uni-Mol: A universal 3D molecular representation learning framework. In *International Conference on Learning Representations* (2023).
20. Ji, Y. et al. Uni-Mol2: Exploring molecular pretraining at scale. In *Advances in Neural Information Processing Systems* (2024).
21. Wang, Y. et al. A unified molecular representation for multimodal property prediction. *Nature Communications* **16** (2025). https://doi.org/10.1038/s41467-025-63730-6
22. Jing, B. et al. MuMo: Multimodal molecular representation learning. In *Advances in Neural Information Processing Systems* (2025).
23. Zheng, S. et al. Large language models for scientific discovery. *Nature Machine Intelligence* (2025). https://doi.org/10.1038/s42256-025-00994-z
24. Zhao, X. et al. Reliability of machine-learning ADMET prediction under realistic chemical splits. *Journal of Cheminformatics* (2026). https://doi.org/10.1186/s13321-026-01217-2
25. Koleiev, I. et al. Auditing Therapeutics Data Commons benchmarks for molecular property prediction. *Journal of Chemical Information and Modeling* (2026). https://doi.org/10.1021/acs.jcim.6c00819
26. Chan, J. et al. MLE-bench: Evaluating machine learning agents on machine learning engineering. In *International Conference on Learning Representations* (2025).
27. Huang, Q. et al. MLAgentBench: Evaluating language agents on machine learning experimentation. In *International Conference on Machine Learning* (2024).
28. Huang, K. et al. Artificial intelligence foundation for therapeutic science. *Nature Chemical Biology* **18**, 1033–1036 (2022). https://doi.org/10.1038/s41589-022-01131-2

---

### Claim-status note for co-authors (remove before submission)

- **Certified:** endpoint contracts, TDC 22-task artifacts, source-traceable comparison, ledger and freeze evidence.
- **Controlled internal evidence:** action-grid, expert-acquisition, rollback, hidden-fault, stewardship and composite-decision studies.
- **Development-only:** adaptive external-data Table 12; do not merge into the certified benchmark claim.
- **Pending:** Caco2 source-lineage clearance and the OpenADMET CYP external blind score/rank.
- **Prohibited wording until new evidence exists:** “the agent outperforms AutoML”, “external CYP superiority”, or “universally optimal routing”.
