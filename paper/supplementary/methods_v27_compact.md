# Supplementary Methods

## S1. Operational specification of the ADMET-EvO research loop

This section records the parameters needed to interpret and reproduce the research loop without repeating the main text or the full audit trail. ADMET-EvO v2.7 denotes the frozen endpoint-specific system and model mapping. Evidence Packet v5 and v5.4 are successive Data-axis evaluation protocols, not software releases.

### S1.1 Contracted research state and diagnostic routing

Each research episode began with an `EndpointContract` defining the endpoint and ADMET category, task type, assay or property semantics, species, units and transformations where applicable, data-source family, primary metric and direction, partition ownership, and rules for external evidence. Validation preceded diagnosis or modelling. Missing or contradictory fields caused abstention rather than an inferred default, and protected certification labels were unavailable to the planner.

The accepted contract, discovery-side diagnostics, previous evidence and remaining budget formed an endpoint-specific research state (Supplementary Fig. S1a). The loop generated a falsifiable hypothesis, compiled one registered intervention on the Feature, Model or Data axis, evaluated it against its control, and retained the result whether supported, refuted, inconclusive or failed. Replicate and Stop were control actions. External evidence followed a separate admission path before it could enter a matched experiment (Supplementary Fig. S1b).

![Supplementary Figure S1. State-controlled research and external-evidence admission in ADMET-EvO.](paper/methods/figures/figure1_nature_autonomous_research_architecture.png)

**Supplementary Figure S1 | State-controlled research and external-evidence admission in ADMET-EvO.** **a,** The `EndpointContract`, internal data and admitted external evidence populate a research state containing diagnostics, prior evidence, budget and provenance. Diagnose–hypothesize–intervene–evaluate–revise transitions are hash-linked; only a frozen recipe enters blind evaluation. **b,** Candidate external records pass endpoint definition, provenance preservation, assay harmonization, structural decontamination and role assignment before a baseline-versus-expanded comparison. Endpoint-name agreement alone does not authorize merging. The displayed maximum Tanimoto value of 0.80 is the preregistered pilot firewall; later Data-axis protocols used the protocol-specific limits in Supplementary Table S3. Blind certification is an architectural terminal state and was not reached by every study.

Discovery-side diagnostics were serialized as a six-component `HardnessCard`. Molecules were represented by largest-fragment canonical isomeric SMILES; scaffolds were chirality-aware Bemis–Murcko scaffolds; and similarities used radius-2, 2,048-bit Morgan fingerprints with the Tanimoto coefficient. Let \(\operatorname{clip}_{01}(x)=\min(1,\max(0,x))\), and let \(L=2{,}000\) be the default sampling limit (minimum permitted value, 100). Definitions and fixed parameters are given in Supplementary Table S1.

**Supplementary Table S1 | Implemented HardnessCard definitions.**

| Component | Severity \(h_k\) | Weight \(w_k\) | Evaluation and confidence \(c_k\) |
|---|---|---:|---|
| Data scarcity | \(\operatorname{clip}_{01}[1-\ln(1+\max(1,S))/\ln(3001)]\), where \(S\) is the number of unique non-null training scaffolds. | 0.18 | Full training set; \(c_k=1\). |
| Class imbalance | Classification: \(\operatorname{clip}_{01}[1-2\min(p,1-p)]\), where \(p\) is the positive fraction among finite training labels. Regression: 0. | 0.12 for classification; 0 for regression | Full training labels; \(c_k=1\) for classification and 0 for regression. |
| Chemical-domain shift | \(\operatorname{clip}_{01}[0.45U+0.35O+0.20(1-\bar{s})]\). \(U\) is the row-wise fraction of non-empty development scaffolds absent from training; \(s_j\) is development molecule \(j\)'s maximum similarity to training; \(\bar{s}=m^{-1}\sum_j s_j\); and \(O=m^{-1}\sum_j\mathbf{1}(s_j<0.4)\). | 0.25 | At most \(L\) training and \(\min(L,1000)\) development rows, sampled without replacement with seeds \(q\) and \(q+1\); \(c_k=\min(1,m/200)\). |
| Local structure–activity roughness | \(\operatorname{clip}_{01}(0.60D+0.40C)\). \(D\) is nearest-neighbour label disagreement for classification or mean \(\operatorname{clip}_{01}[\lvert y-y_{\mathrm{NN}}\rvert/(2\sigma_{\mathrm{tr}})]\) for regression. \(C\) is the fraction of directed nearest-neighbour pairs with similarity at least 0.75 that disagree in class or differ by at least one sampled-target SD. | 0.23 | \(D\) reuses the domain-shift sample. Cliff analysis uses at most \(\min(L,1200)\) training rows and seed \(q+2\); \(c_k=\min(1,m/200)\). |
| Data quality | Classification: \(\operatorname{clip}_{01}[(I+M+F+4K)/N]\). Regression: \(\operatorname{clip}_{01}[(I+M+F)/N+(R/N)\min(1,d)]\). \(I,M,F,K,R\) are the numbers of invalid structures, missing targets, infinite targets, conflicting canonical structures and excess replicate rows; \(d\) is median within-structure target SD divided by global target SD. | 0.12 | Full training audit with \(N=\max(1,N_{\mathrm{tr}})\); \(c_k=1\). |
| Split-leakage risk | \(\operatorname{clip}_{01}[E/\max(1,\min(N_{\mathrm{tr}},N_{\mathrm{dev}}))]\), where \(E\) counts training-row occurrences whose canonical molecule occurs in development. | 0.10 | Full training and development sets; exact molecular identity only; \(c_k=1\). |

If no non-empty development scaffold was available, \(U=0\). If no fingerprint comparison was evaluable, \(\bar{s}=0\), \(O=1\), \(D=1\) and \(c_k=0\); if no eligible cliff pair existed, \(C=0\). Population SD was used for roughness and sample SD for the replicate audit. Zero scales were replaced by 1, and \(d=0\) when no repeated-group SD was available.

The aggregate score was

\[
H=\frac{\sum_k w_k h_k}{\sum_k w_k}.
\]

The denominator was 1.00 for classification and 0.88 for regression because imbalance was disabled for regression. Confidence was coverage metadata and did not attenuate \(H\). Card-level action triggers were leakage \(>0\), quality \(\geq0.05\), imbalance \(\geq0.35\), domain shift \(\geq0.35\), roughness \(\geq0.25\) and scarcity \(\geq0.45\); multi-seed replication was always recommended. The executable planner used lower routing thresholds of 0.03, 0.25, 0.25, 0.20 and 0.35 for quality, imbalance, shift, roughness and scarcity, respectively. These were fixed implementation choices; no completed sensitivity analysis of the weights, thresholds or sampling limits was identified.

### S1.2 Registered interventions and matched evidence

Each hypothesis was compiled into an `ActionSpec` defining the active axis, control and treatment configurations, metric, seeds, resource requirement and decision rule. Control and treatment runs shared the endpoint contract, molecular identities, split, seed, metric and every non-intervened setting whenever the design permitted. Resources were reserved for the pair; incomplete or failed runs remained in the evidence lineage. A completed comparison generated an `EvidencePacket` containing the action and parent state, both configurations, oriented effect, stability, cost, status, hashes and verdict. The hash-linked ledger retained negative evidence and prevented an altered ancestor from supporting downstream promotion.

The planner and promotion rule varied by study (Supplementary Table S2); no retrospective global score was imposed across protocols.

**Supplementary Table S2 | Evaluation programmes, budgets and claim gates.**

| Programme | Compared configurations | Registered evaluation | Budget and promotion rule | Claim boundary |
|---|---|---|---|---|
| TDC-22 frozen benchmark | Two-round Feature/Model search followed by separately frozen MapLight+GNN upgrades for Bioavailability and CYP2C9 substrate. | Search development seeds 1–3; upgrade confirmation seeds 6–10; official-test refits seeds 1–5. | At most six proposals and 60 min per round, for two rounds. Each upgrade required mean improvement over MapLight and at least 3/5 paired wins. | Protected test labels were excluded from selection. The resulting 22-task mapping was frozen and certified. |
| Three-axis factorial ablation v3 | Eight prespecified Model/Feature/Data cells (`000`–`111`) relative to LightGBM, radius-2 Morgan bits and official-only data. | Development seeds 21–24; confirmation 31–34 and certification 41–44 were registered but not completed. | Eight cells per endpoint; promotion required mean normalized gain \(\geq0.005\), positive-seed fraction \(\geq0.75\), mean minus 0.5 SE, and no-regret inheritance. | Development-only. |
| Router controls | Full-outcome versus winner-only memory; score-based and cost-aware selection; matched-cost random control; metric-risk guard. | Candidate seed 1; held-out seeds 2–3; risk-guard development seeds 1–3 and fresh seeds 4–5. | Eight prespecified cost penalties; matched expected cost; risk-guard cost ratio \(\leq0.35\). | Registered claim gates failed; the frozen selection was unchanged. |
| Data-axis protocols | Native-data control versus the same learner/recipe with admitted external evidence. | Protocol-specific schedules in Supplementary Table S3. | Finite quantity/weight grids; failure at a required stage retained or restored the native-data baseline. | Claims ranged from exploratory screening to one certified endpoint and are not interchangeable. |
| Sealed process controls | Agent, ablations, AutoML controls, complete deterministic rules and a seeded random policy. | Ten hidden-fault cases and 11 data-stewardship dossiers; policy seed 20260820; no predictive fits. | Exact decision/remediation scoring against prespecified sealed outputs. | Local verification of implemented controls, not evidence of unrestricted scientific reasoning. |
| Prospective OpenADMET CYP panel | B1, B2, M0–M4 and E0 under chemistry-group cross-validation; the frozen recipe reverted to single-component B1. | Stage A: one seed and three outer folds. Stage B: seeds 1–3 and five outer folds. | All four gates were required: macro MCC \(\geq0.1463464\), macro ST-RAE \(\leq0.8259417\), worst-endpoint MCC \(\geq0\), and worst-endpoint ST-RAE \(\leq1.25\). | Frozen before external evaluation; blind readout pending. |

### S1.3 Governed admission of external evidence

External data were admitted only after endpoint definition, immutable provenance capture, assay harmonization, structural decontamination and role assignment (Supplementary Fig. S1b). Protected labels were never used for admission or subset selection. Classification subsets preserved class composition, regression subsets covered the native target distribution, and failure of a required gate caused abstention or rollback.

**Supplementary Table S3 | Data-axis protocol instances and permissible claims.**

| Protocol | Candidate construction and development | Confirmation or certification | Result and permissible claim |
|---|---|---|---|
| Fixed-controller adaptive screen | Morgan radius-2/LightGBM; external-to-native row ratios 0.5, 1, 2, 4 and 8; seeds 2–5. Development required mean gain \(\geq0.005\), positive fraction \(\geq0.75\), and the distribution gates below; the smallest passing ratio within one SE of the best mean was selected. | Seeds 8–11 with the gain and positive-fraction gates; only hERG has a completed agent-confirmation artefact. | Nine of 22 endpoints selected augmentation. Exploratory screen only. |
| Frozen-v2.7 adaptive screen v3 | Same ratio grid, seeds and development gates, but the endpoint's frozen v2.7 recipe was identical in both arms. | Seeds 8–11 were planned; no completed confirmation artefact was found. | Twenty-one development artefacts: 6 selected, 15 abstained or lacked sufficient data; incomplete and development-only. |
| Evidence Packet v5 | Quantity fraction and weight multiplier each in \(\{0.25,0.50,0.75,1.00\}\); development seeds 11, 23, 42, 67 and 89; all-seed mean with one-SE selection. | Seeds 101, 131 and 151; corrected Half-Life seeds 181, 211 and 241. Retention required positive mean gain and at least 2/3 positive seeds. | Sixteen tasks evaluated, six sources rejected before training and eight confirmations passed. No certification layer; the all-22 experiment remained incomplete. |
| Evidence Packet v5.4 | Frozen v2.7 recipe in both arms; 3–21 candidates per eligible task; external mass \(\leq0.10\) of native mass. Development seeds 271, 307, 347, 389 and 431; mean \(>0\), positive fraction \(\geq0.80\), then one-SE low-mass selection. | Confirmation seeds 467, 503, 547 and 593: mean \(\geq0.005\), positive fraction \(\geq0.75\). Certification seeds 631, 677, 719 and 761: mean \(\geq0\), positive fraction \(\geq0.75\). | Six sources rejected, eight development selections, four confirmation passes and one certified active endpoint, CYP3A4_Veith. |
| DILI best-seed study | Three FDA DILIRank2 configurations on seeds 271, 307, 347, 389 and 431; selected quantity 0.50, weight 0.50 and seed 307 by maximum single-seed development gain. | No independent confirmation; one official-test readout followed the freeze. | Five-seed development mean \(-0.000861\); official-test gain \(+0.012851\). Exploratory best-observed estimate, not certification. |

For both adaptive screens, the development distribution gate required a class-rate difference \(\leq0.20\) for classification. For regression, the absolute median shift divided by the reference IQR had to be \(\leq0.75\), and the selected/reference IQR ratio had to lie within \([1/3,3]\), in every seed. The pilot also required training relevance \(\geq0.20\) and maximum similarity to protected development molecules \(<0.80\). In v5.4, protected-neighbour similarity \(\geq0.90\) caused row rejection and same-source protected-scaffold coverage \(>0.05\) caused source rejection.

### S1.4 Implemented portfolio and verification boundaries

Supplementary Table S4 groups the implemented portfolio rather than repeating every endpoint-specific recipe. Exact overrides were serialized with each run. Invalid structures produced zero feature rows where supported; an all-invalid batch failed. Missing optional dependencies made a candidate unavailable and never triggered silent substitution.

**Supplementary Table S4 | Implemented predictive portfolio and execution gates.**

| Capability | Frozen specification | Use or dependency gate |
|---|---|---|
| Core two-dimensional representations | Morgan bits (radius 2 or 3, 2,048 bits), Morgan counts (radius 2, 2,048), Avalon (1,024), ErG (315), MACCS (167), and 2,048-bit RDKit path, atom-pair and topological-torsion fingerprints. | Core CPU features; concatenated multi-view blocks were `float32`. |
| RDKit physicochemical descriptors | Runtime `RDKit.Chem.Descriptors._descList`; exceptions and non-finite values replaced by zero. | Not normalized by the featurizer; standardization occurred only inside the RBF-SVM pipeline. |
| Tree ensembles | ExtraTrees and Random Forest: 400 trees, minimum leaf size 2. Balanced ExtraTrees: seven balanced members of 300 trees, classification only. | scikit-learn; task-specific class weighting was applied only when registered. |
| Gradient boosting | XGBoost: 600 trees, depth 6, learning rate 0.05. LightGBM: 600 trees, 31 leaves, learning rate 0.04. CatBoost: 600 iterations, depth 8, learning rate 0.05. | Corresponding package required; endpoint recipes could override these registry defaults. |
| RBF-SVM | Standard scaling, \(C=10\), `gamma="scale"`; classification used three-fold calibrated SVC. | Planner eligibility required \(N_{\mathrm{train}}\leq5{,}000\); this was not a fit-time assertion. |
| Uni-Mol2 | Frozen 84M representation, batch size 32. Fine-tuning used learning rate \(10^{-4}\), 20 epochs, five scaffold folds, early stopping 5 and batch size 8 or 16. | Frozen encoding required `unimol_tools`; the frozen fine-tuning executor required CUDA. |
| Half-Life rank experts | Rank-target XGBoost plus a `fixed_rank_top3` pool comprising three 2D heads, three frozen-Uni-Mol2 tabular heads and fine-tuned Uni-Mol2; the metric-oriented top three out-of-fold predictions were rank-averaged. | All seven pool candidates ran before selection, so all dependencies were required. Rank aggregation was disallowed for MAE. |
| MapLight+GNN | Fixed 2,863-dimensional input: Morgan counts 1,024 + Avalon counts 1,024 + ErG 315 + RDKit2D 200 + frozen GIN 300; CatBoost 1.2.8 with Logloss, `random_strength=2`, fixed seed and no file writing. | Binary classification only; isolated pinned Torch/DGL/DGL-LifeSci/MolFeat runtime. Fresh seeds 6–10 required mean improvement and at least 3/5 wins over MapLight. |

MiniMol and family-transfer experts were protocol-specific predecessors or later local experiments and were not part of the final v2.7 endpoint mapping; they are therefore omitted from the primary portfolio table.

Sealed fixtures represented contract mutation, protected-label access, schema failure, leakage, source-lineage conflict, unsafe merging, execution failure and freeze-integrity violation. Their outputs were decisions and prescribed remediation, not molecular predictions. Agreement with prespecified rules was interpreted as verification that controls executed as intended, not as evidence that the system possessed unrestricted scientific reasoning.

Core v2.7 implementations were cross-checked against the public repository snapshot at commit [`837880b0326e9aa4c5d013e1a6bc2b5d37418162`](https://github.com/YilinWang33/ADMET-EvO/tree/837880b0326e9aa4c5d013e1a6bc2b5d37418162). The later v5/v5.4 and frozen-recipe Data-axis protocols were reconstructed from their frozen local code and result artefacts because those files were not present in the public snapshot. These artefacts should accompany the archived reproducibility package.
