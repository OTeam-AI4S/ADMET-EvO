# ADMET-EvO: Cost-Aware Sequential Evidence Acquisition with Freeze-Locked Evaluation for ADMET Prediction

Research Article

Yilin Wang^1,*^ (`ORCID_REQUIRED`)

^1^ Valhalla Technology  
* Correspondence: `CORRESPONDING_EMAIL_REQUIRED`

## Abstract

### Background

ADMET endpoints differ in label type, sample size, chemical-space shift, and
the value of topology, geometry, and target-distribution evidence. This makes
always-on use of every molecular representation costly and scientifically
fragile.

### Results

We introduce ADMET-EvO, a protocol-safe research agent that evaluates bounded
candidates on scaffold-split train/validation data, records successful and
failed acquisitions, and content-addresses a finalist before one-shot official
evaluation. On a dated snapshot of 22 Therapeutics Data Commons ADMET tasks,
the frozen v2.7 system has 18 estimated Top-5 and 21 Top-10 placements. A
preregistered MapLight+GNN expert improved CYP2C9 substrate from 0.340 to 0.434
+/- 0.009 AUPRC and Bioavailability from 0.697 to 0.731 +/- 0.007 AUROC. The
remaining displayed Bioavailability deficit is reported conservatively because
its public page duplicates the P-glycoprotein table and lacks an authoritative
correction. A held-out-seed router analysis retained 18.4% of score-selected
aggregate compute without improving mean accuracy. Fresh-seed safety, learned
context, full-memory, and matched-cost random controls also failed their
accuracy gates. Low-similarity molecules had worse error on 11/13
classification and 7/9 regression endpoints.

### Contribution

The contribution is a reproducible experiment-level acquisition architecture:
registered actions, measured acquisition cost, optional evidence rejection, a
full negative-result ledger, and an immutable discovery-to-evaluation
boundary. It is not a claim that routing or multimodal fusion universally
improves prediction.

### Conclusions

Protocol-safe evidence acquisition can reduce aggregate experimental compute
while preserving an auditable account of uncertainty, failures, and chemical-
space risk. The negative controls delimit the supported claims and identify
candidate-level risk modeling, rather than metric-name rules, as the next
prospective target.

## 1. Introduction

ADMET prediction is usually presented as selection of a single architecture or
representation. The TDC ADMET group instead exposes 22 heterogeneous endpoints
[1]:
classification metrics include AUROC and AUPRC, regression metrics include MAE
and Spearman correlation, dataset sizes vary substantially, and default
validation partitions enforce scaffold shift. This heterogeneity makes
always-on multimodal computation scientifically questionable and expensive.

ADMET-EvO frames model construction as sequential evidence acquisition. Cheap
2D topology and descriptor experts are evaluated first. Expensive 3D or
metric-aligned evidence is acquired only when repeated validation suggests that
its uncertainty-adjusted gain justifies its cost. Acquired evidence can be used
alone, combined through a preregistered raw/rank recipe, or discarded in favor
of a conservative 2D backoff.

The paper makes four bounded contributions:

1. A protocol-safe research-agent architecture with a deterministic executor,
   registered tools, an auditable evidence ledger, and freeze-locked one-shot
   official evaluation.
2. A cost-aware lower-confidence-bound routing objective that treats compute as
   part of experimental utility and can decline expensive evidence.
3. Heterogeneous frozen recipes combining classical 2D experts, a pinned
   pretrained-GNN/descriptor expert, an 84M fine-tuned Uni-Mol2 expert, and
   metric-aligned rank evidence where supported by validation.
4. An evidence-complete evaluation that reports successful upgrades, negative
   acquisitions, compute-performance trade-offs, statistical uncertainty, and
   tail failures without using leaderboard feedback as a selection signal.

The contribution is not that mixture-of-experts or 2D/3D fusion is intrinsically
novel. The narrower claim is that experiment-level acquisition, optional
backoff, measured cost, and a verifiable discovery-to-freeze boundary form a
useful scientific-agent abstraction for heterogeneous ADMET endpoints.

## 2. Related work and claim boundary

The method is positioned against TDC [1], task-specific ADMET
systems [2, 3], Uni-Mol and Uni-Mol2
[4, 5], parameter-efficient and scaled molecular
foundation models [6, 7], task-routed molecular
experts [8, 9], multimodal fusion
[10], and LLM-assisted scientific discovery
[11]. Recent reliability and end-user audits establish why
chemical-space shift, leakage control, and reproducibility must be first-class
evaluation targets [12, 13]. General ML
agent benchmarks provide adjacent evidence on iterative experimentation but do
not address the chemistry-specific split and freeze boundary
[14, 15].

OmniMol and ASE-Mol therefore already establish routed molecular experts,
while MuMo studies structured 2D/3D fusion and LLM4SD uses an LLM for
literature- and data-derived molecular-rule discovery. ADMET-EvO does not claim
generic MoE, generic multimodal fusion, or autonomous chemical discovery. Its
differentiators are protocol-safe experiment routing, explicit acquisition
cost, optional evidence rejection, a negative-result ledger, and freeze-locked
official evaluation.

## 3. Method

### 3.1 Protocol-safe research loop

For task `t`, the profiler derives train/validation-only metadata. A bounded
planner proposes registered representation/model configurations. The guard
rejects unknown tools, arbitrary code, test-related fields, invalid
hyperparameters, and budget violations. The executor stores specifications,
predictions, runtimes, errors, and dependency information. Robust candidates
are repeated across scaffold seeds and can enter a preregistered ensemble. The
complete finalist is serialized and hashed before the official evaluator is
called.

The official test split is not available to the planner, router, ensemble
selector, or validation-analysis scripts. Official results cannot mutate a
finalist in the same freeze lineage.

### 3.2 Evidence actions

The bounded action vocabulary is:

- `stop_2d`: retain a cheap topology/descriptor expert;
- `acquire_3d`: fine-tune or evaluate the preregistered 3D expert;
- `acquire_target_view`: fit metric-aligned target or rank evidence;
- `fuse`: combine validated complementary evidence;
- `backoff`: reject an acquisition and restore the cheap champion.

For candidate action `a`, the validated router computes

```text
U(a,t) = s(a,t) - beta * sigma(a,t) - lambda * log(1 + c(a,t)) / C_t,
```

where `s` is oriented validation score, `sigma` is cross-seed uncertainty, `c`
is measured cumulative runtime, and `C_t` normalizes log cost within a task.
Ties favor lower measured cost and then a stable run identifier.

### 3.3 What the router does and does not learn

The current validated router uses metric direction plus candidate score,
uncertainty, and cost. Although the planner receives a richer `TaskProfile`, the
router does not learn a task-profile-conditioned policy. A leave-one-task-out
nearest-task prototype using structural or full context failed to improve
held-out score and erased the global cost advantage; it is rejected in v2.4.
A preregistered winner-only-memory control showed that retaining every outcome
reduced aggregate selected compute to 18.5% of the winner-only policy, but did
not improve held-out accuracy. An exactly matched-cost random-action control
also failed to establish selection-intelligence gains. The ledger is therefore
presented as an auditability and cost-control mechanism, not as a learned
accuracy-improving memory.

### 3.4 Freeze and provenance

The current full 22-task v2.7 freeze has SHA-256:

```text
693db6936485d59791eacfd9005d18efca00056274c11c1cc80efd962058c84b
```

It derives from the verified v2.6 mapping and replaces only Bioavailability and
CYP2C9 substrate with a preregistered MapLight+GNN expert after fresh scaffold
seeds 6-10 passed a fixed paired promotion gate. The selection phase never
loaded the official test set and had zero canonical-molecule identity overlap
between train and validation. The public GNN checkpoint, upstream source, full
feature matrices, predictions, and isolated environment are content-addressed.
An earlier seed-propagation audit found that unimol-tools 0.1.5 silently
kept default seeds; the executor now writes and verifies both `seed` and
`split_seed` in each saved runtime configuration. Official verification fails
closed unless result/freeze hashes, five audit records, exact finalist mapping,
and saved seeds 1-5 all agree.

![Protocol-safe evidence acquisition](paper/figures/figure1_architecture.png)

**Figure 1 | Protocol-safe evidence acquisition.** The planner can only propose
registered candidates, the deterministic executor records successes and
failures, and the router may acquire, fuse, or reject evidence. A
content-addressed freeze separates train/validation discovery from one-shot
official evaluation; official outcomes cannot mutate the finalist in the same
lineage.

## 4. Experimental design

### 4.1 Benchmark and metrics

Experiments use the 22 TDC ADMET Group tasks and their official metrics. Search
and ablations use default scaffold train/validation splits. Five-seed mean and
standard deviation are reported for frozen official recipes. Rank estimates
are comparisons with the immutable snapshot
`reports/leaderboard_snapshot_2026-08-17.json`; they are not claims about a live
leaderboard.

### 4.2 Selection and statistical protocol

Candidate choice uses only training/validation evidence. Paired seed
comparisons are preferred for model ablations. Across-task router differences
use two-sided Wilcoxon tests, task bootstrap intervals, improved/tied/worse
counts, action frequencies, and measured cost ratios. Because AUROC, AUPRC,
MAE, and Spearman use different native scales, cross-metric oriented gains are
descriptive and are not interpreted as a universal effect size.

### 4.3 Required baselines

- cheap 2D champion;
- frozen 3D representation plus classical learner;
- fine-tuned 84M Uni-Mol2;
- raw and percentile-rank fusion;
- score, LCB, and cost-aware LCB routing;
- validation oracle;
- winner-only outcome memory and maximum-entropy random action at exactly
  matched expected training cost.

## 5. Results

### 5.1 Dated 22-task leaderboard scorecard

The verified v2.4 scorecard yielded 17/22 estimated Top-5 and 20/22 Top-10
placements. The complete frozen v2.7 scorecard is:

| Task | Metric | Frozen finalist | Official mean +/- SD | Estimated rank | Band |
|---|---|---|---:|---:|---|
| AMES | AUROC | ExtraTrees + XGBoost | 0.874 +/- 0.001 | 1 | Top-5 |
| BBB Martins | AUROC | RF + RF | 0.921 +/- 0.002 | 2 | Top-5 |
| Bioavailability Ma | AUROC | MapLight+GNN | 0.731 +/- 0.007 | 21 | below Top-10 |
| CYP2C9 substrate | AUPRC | MapLight+GNN | 0.434 +/- 0.009 | 5 | Top-5 |
| CYP2C9 Veith | AUPRC | LightGBM + XGBoost | 0.799 +/- 0.001 | 5 | Top-5 |
| CYP2D6 substrate | AUPRC | LightGBM + LightGBM | 0.721 +/- 0.004 | 3 | Top-5 |
| CYP2D6 Veith | AUPRC | LightGBM + LightGBM | 0.721 +/- 0.001 | 4 | Top-5 |
| CYP3A4 substrate | AUROC | RBF-SVM + RBF-SVM | 0.633 +/- 0.000 | 9 | Top-10 |
| CYP3A4 Veith | AUPRC | XGBoost + XGBoost | 0.885 +/- 0.001 | 4 | Top-5 |
| Caco2 Wang | MAE | LightGBM | 0.283 +/- 0.007 | 4 | Top-5 |
| Clearance hepatocyte | Spearman | XGBoost + LightGBM | 0.456 +/- 0.022 | 4 | Top-5 |
| Clearance microsome | Spearman | XGBoost + XGBoost | 0.591 +/- 0.007 | 8 | Top-10 |
| DILI | AUROC | 2D LightGBM + 2D ExtraTrees + Uni-Mol2 | 0.920 +/- 0.004 | 3 | Top-5 |
| HIA Hou | AUROC | ExtraTrees + ExtraTrees | 0.990 +/- 0.001 | 2 | Top-5 |
| Half-life Obach | Spearman | Uni-Mol2 + fixed-rank top-3 + rank-target XGBoost | 0.590 +/- 0.030 | 1 | Top-5 |
| LD50 Zhu | MAE | RF + RF | 0.591 +/- 0.001 | 5 | Top-5 |
| Lipophilicity AZ | MAE | Uni-Mol2 | 0.392 +/- 0.006 | 1 | Top-5 |
| PPBR AZ | MAE | LightGBM + LightGBM | 7.302 +/- 0.042 | 1 | Top-5 |
| Pgp Broccatelli | AUROC | RF | 0.912 +/- 0.003 | 8 | Top-10 |
| Solubility AqSolDB | MAE | LightGBM + LightGBM | 0.739 +/- 0.002 | 1 | Top-5 |
| VDss Lombardo | Spearman | ExtraTrees | 0.673 +/- 0.001 | 3 | Top-5 |
| hERG | AUROC | 2D LightGBM + Uni-Mol2 | 0.860 +/- 0.009 | 5 | Top-5 |

Ranks use the immutable 17 August 2026 snapshot and pessimistic tie handling;
MAE is lower-is-better. Bioavailability retains the displayed-page rank even
though that page fails the separately reported integrity audit.

The v2.7 upgrade passes the same fail-closed verifier for five official seeds:

| Task | Metric | Previous | v2.7 | Estimated snapshot rank | Verification |
|---|---|---:|---:|---:|---|
| CYP2C9 substrate | AUPRC | 0.340 | 0.434 +/- 0.009 | 5 | hashes, finalist, audit, seeds 1-5 |
| Bioavailability | AUROC | 0.697 | 0.731 +/- 0.007 | 21 displayed | hashes, finalist, audit, seeds 1-5 |

The conservative v2.7 scorecard is 18/22 estimated Top-5 and 21/22 Top-10.
CYP2C9 substrate is resolved. Bioavailability is the only displayed Top-10
deficit, but its reference table fails an independent consistency audit: all 16
Pgp rows occur verbatim on the Bio page, MiniMol's primary 2024 report assigns
the current Bio MiniMol value to Pgp rather than Bio, and the peer-reviewed
2026 audit independently obtains the same 0.731 +/- 0.007 as our reproduction.
This discrepancy is reported but does not alter the conservative count.

![Conservative 22-task leaderboard coverage](paper/figures/figure2_leaderboard_ranks.png)

**Figure 2 | Conservative coverage of the dated 22-task TDC ADMET snapshot.**
The frozen v2.7 system has 18/22 estimated Top-5 and 21/22 Top-10 results.
Bioavailability is retained at its displayed conservative rank pending
authoritative correction of the disputed page.

### 5.2 Model and evidence ablations

Fine-tuned Uni-Mol2 is endpoint-dependent. On Lipophilicity, the 84M model
reached validation MAE 0.4036 +/- 0.0208 versus 0.4129 +/- 0.0196 for 164M,
while the larger model used 2.03 times the GPU time and won only one of five
paired seeds. Scaling was also unfavorable on DILI and hERG, so 310M was not
run. This is evidence for selective acquisition, not monotonic scaling.

Metric-aligned rank evidence was complementary on Half-Life. The frozen
three-way rank recipe combines fine-tuned Uni-Mol2, a fixed rank ensemble, and
rank-target XGBoost; its official 0.590 +/- 0.030 result substantially exceeds
the v2.1 score without post-freeze weight changes.

A standardized post-hoc paired summary of the completed validation experiments
quantifies the primary model effects without altering selection. MapLight+GNN
minus MapLight had mean paired gain 0.02353 on Bioavailability (paired-t 95% CI
[0.00014, 0.04692], small-sample-corrected paired dz=1.00, exact two-sided
sign-flip p=0.125) and 0.03422 on CYP2C9 substrate (95% CI
[-0.04240, 0.11085], corrected dz=0.44, p=0.1875). Uni-Mol2 164M minus 84M on
Lipophilicity had oriented gain -0.00934 MAE units (95% CI
[-0.02057, 0.00189], corrected dz=-0.83, p=0.125) while using 2.03 times the GPU
time. With only five paired split seeds, these intervals and standardized
effects describe computational replicates rather than independent chemical
tasks; none is presented as a confirmatory significance claim.

### 5.3 Router cost-performance trade-off

In the 22-task held-out-seed LOOTO analysis, the cost-aware policy retained
18.4% of score-selected cumulative recipe time and 18.5% of LCB-selected time.
Mean oriented gain versus score was -0.0020 (Wilcoxon p=0.249); versus LCB it
was -0.0003 (p=0.721). Thus the result supports aggregate compute efficiency,
not improved prediction.

Task-bootstrap analysis gave a mean gain 95% interval of [-0.0065, 0.0024].
Median per-task cost ratio was 0.750 [0.545, 1.000], demonstrating that the much
smaller aggregate ratio is dominated by a few expensive recipes. At an
absolute native-metric gain threshold of 0.005, the policy produced two
material improvements and four regressions. Spearman endpoints had mean gain
-0.0119, with zero improved, two tied, and two worse tasks.

### 5.4 Rejected context conditioning

The global LOOTO penalty policy obtained aggregate cost ratio 0.184 and mean
gain -0.0020. Five-neighbor structural and full-context variants produced cost
ratios 0.972 and 0.966 with mean gains -0.0024 and -0.0020. They are retained as
negative evidence and are not part of the deployed v2.4 router.

### 5.5 Memory and matched-cost random controls

The preregistered retrospective validation-only control used candidate
generation seed 1, held-out validation seeds 2 and 3, and leave-one-task-out
policy selection over all 22 tasks. Official test data, official results, and
leaderboard values were not loaded. Full-outcome memory versus winner-only
memory had mean oriented gain -0.00031, task-bootstrap 95% CI
[-0.00276, 0.00240], 4/12/6 improved/tied/worse tasks, and Wilcoxon p=0.7213.
It failed the preregistered positive-mean and 12-task-win gate, while retaining
only 18.5% of winner-only memory's aggregate selected-compute proxy.

Against a maximum-entropy random policy constrained to exactly the agent's
expected training cost on every held-out task/seed, the agent had mean oriented
gain 0.00083, 95% CI [-0.00380, 0.00542], 7/5/10 improved/tied/worse tasks, and
Wilcoxon p=0.8684. Maximum relative cost-matching error was 3.96e-16. This gate
also failed. These controls support neither predictive superiority of full
outcome memory nor selection intelligence beyond a cost-matched random policy;
they instead sharpen the contribution boundary around auditability,
reproducible backoff, and aggregate cost control.

### 5.6 Preregistered fresh-seed metric-risk guard

The post-hoc error analysis motivated a simple frozen rule: use the score
policy on Spearman endpoints and the cost-aware policy on AUROC, AUPRC, and MAE.
The rule, 22 locked candidate sets, seeds 4-5, a 0.005 material-regression
threshold, and all promotion criteria were preregistered before fresh labels
were evaluated. These TDC default scaffold splits had zero canonical-molecule
identity overlap, and no official test data, official results, or leaderboard
values were loaded.

The claim gate failed. When candidates were reselected from development seeds
1-3, score and cost-aware routing already converged to the same candidate on
all four Spearman tasks. Consequently, guarded minus unguarded gain was exactly
zero on all 22 tasks (Wilcoxon p=1.0), and both policies had two material
regressions versus score. Guarded versus score mean oriented gain was -0.00035
(task-bootstrap 95% CI [-0.00148, 0.00059]; 5/13/4 improved/tied/worse;
Wilcoxon p=0.859). The actual fresh-run fit time was 0.625 hours for either
cost-aware policy versus 2.247 hours for score, a 0.278 ratio. The experiment
therefore confirms compute reduction in a new split lineage but provides no
evidence that a metric-name guard reduces tail risk or improves accuracy. It
cannot alter the v2.7 official freeze.

![Fresh-seed router compute-accuracy trade-off](paper/figures/figure3_router_tradeoff.png)

**Figure 3 | Fresh-seed compute--accuracy trade-off.** Candidate choices were
locked using development seeds 1--3 and evaluated on default scaffold seeds
4--5. The guarded policy used 0.625 measured fit hours versus 2.247 for score
selection (ratio 0.278), but guard-minus-unguarded gain was 0.000 and the
preregistered safety gate failed. Point size reflects score-policy fit time;
dashed horizontal lines mark +/-0.005 native-metric change.

### 5.7 Weak-task rescue ledger

Exact-overlap-excluded cross-endpoint teachers improved Bioavailability
validation AUROC from 0.6898 to 0.7175, far short of the dated 0.922 Top-10
threshold. A subsequently locked 50/50 MiniMol-cosine/Morgan-Tanimoto kernel
reproduced at 0.7538 +/- 0.0630 on fresh scaffold seeds 6-10, but its one-shot
official AUROC was 0.675 versus the retained 0.697 baseline. This failure is
recorded without post-test reweighting. Cross-endpoint teachers reduced
CYP2C9 substrate AUPRC. On CYP2C9, fresh-seed rank
fusion, balanced bagging, a fixed MiniMol head, equal-rank MiniMol/LightGBM
fusion, and an inner-scaffold MiniMol ensemble all failed to exceed the 0.4801
LightGBM validation mean. A separately pinned MapLight+GNN candidate then
passed a preregistered fresh-seed gate on both weak tasks: it improved four of
five paired seeds on each task, with CYP2C9 validation AUPRC 0.4367 and Bio
validation AUROC 0.7093. The one-shot official results were 0.434 +/- 0.009 and
0.731 +/- 0.007, respectively, with no post-test tuning.

### 5.8 Calibration and chemical-space residual audit

We reconstructed one saved default scaffold-validation split for every frozen
lineage: seed 1 for v2.1/v2.4 finalists and preregistered fresh seed 6 for the
two v2.7 MapLight+GNN upgrades. All 22 prediction vectors matched the validation
row counts. Canonical train-validation molecule identity overlap and
train-validation Bemis--Murcko scaffold overlap were both zero on every task.
The analysis did not load official test data, official results, or leaderboard
values, and could not alter a finalist.

Across the 13 classification tasks, mean Brier score was 0.1173 and mean
ten-bin adaptive expected calibration error was 0.0964. Low-nearest-neighbor-
similarity molecules had worse Brier error than high-similarity molecules on
11 tasks and better error on two. The largest adaptive calibration errors were
on CYP3A4 substrate (0.1909), DILI (0.1750), and hERG (0.1506), identifying
concrete targets for future train-only calibration rather than post-hoc blind
test adjustment.

For the nine regression tasks, MAE endpoints were analyzed in native target
units and Spearman endpoints in percentile-rank residual space. Low-similarity
molecules had larger mean absolute residuals on seven tasks and smaller
residuals on two. The effect was especially large for Lipophilicity
(low-minus-high similarity MAE 0.2723) and PPBR (9.4631 in native units).
Because finalists were selected using the same development evidence, these
figures are diagnostic rather than unbiased prospective error estimates.

![Chemical-space error inflation](paper/figures/figure4_chemical_space_risk.png)

**Figure 4 | Chemical-space error inflation.** In a post-selection diagnostic
on one saved scaffold split per task, low-similarity molecules had worse error
on 11/13 classification and 7/9 regression endpoints. Error ratios are shown
on a log2 scale to compare endpoints with different native units. Identity and
scaffold overlap were both zero; the analysis is diagnostic rather than a
prospective performance estimate.

### 5.9 Compute-accounting boundary

The candidate-stage router replay records 0.256 selected recipe-hours across
22 tasks and three validation seeds, versus 2.488 hours for score-only
selection, a matched recorded-cost ratio of 0.1028. Per-expert wall times are
also available for the DILI/hERG pilots, the Lipophilicity 84M/164M scale
comparison, and the MapLight+GNN confirmation campaign. These measurements
come from different phases and environments, so only matched within-protocol
ratios are interpreted.

Historical runs did not capture process or CUDA peak memory: measured coverage
is 0/31 appendix records. We report this missingness explicitly and make no
absolute peak-memory or hardware-normalized throughput claim. Such a claim
requires a fresh instrumented rerun rather than reconstruction from elapsed
time.

## 6. Discussion

The strongest evidence supports three conclusions. First, ADMET endpoints do
not justify always-on 3D or larger foundation models. Second, a scientific
agent can reduce aggregate acquisition cost while preserving a transparent
record of the accuracy trade-off. Third, protocol engineering is part of the
method: the caught seed bug, immutable freeze, and rejected rescue attempts are
necessary evidence that the leaderboard results are reproducible rather than
post-hoc tuned.

The evidence does not support universal router gains, median-task compute
savings of 40%, naive nearest-task context transfer, or universal multimodal
benefit. The preregistered metric-name guard also failed because it did not
change candidate selection on the targeted Spearman tasks. A future safety
mechanism must operate on candidate disagreement, uncertainty, or
chemical-space coverage and must be preregistered on another untouched split
lineage before promotion.

## 7. Limitations

The public test interface requires procedural isolation rather than a hidden
evaluation server. Rank estimates depend on a dated leaderboard snapshot. The
full displayed Top-10 objective is not yet achieved: Bioavailability is the only
remaining deficit, and its page has a confirmed material discrepancy but no
authoritative correction. The 22-task
router replay contains only the candidate family generated by v2.1 and does not
prove generalization to new tools or datasets. Task-context learning,
accuracy-improving negative memory, and selection intelligence beyond
cost-matched random action are not validated contributions. The calibration
and residual audit is post-selection and uses one split per task; prospective
coverage requires a new frozen protocol. GPU peak memory was not recorded, so
absolute memory and hardware-normalized throughput claims remain unavailable.
The fresh metric-risk guard is also a negative result: it preserved compute
savings but provided no safety or accuracy gain.

## 8. Reproducibility and artifact map

| Claim | Source of truth |
|---|---|
| v2.1 22-task scorecard | `reports/goal_scorecard.json` |
| Dated leaderboard values | `reports/leaderboard_snapshot_2026-08-17.json` |
| Full v2.7 freeze | `runs/v27-maplight-gnn-freeze/full-22-task.json` |
| v2.7 official upgraded results | `runs/v27-official/<task>/final/results.json` |
| MapLight+GNN fresh confirmation | `runs/v27-maplight-gnn-confirmation/result.json` |
| Bio leaderboard discrepancy | `reports/bio_leaderboard_discrepancy_v27.json` |
| Conservative v2.7 scorecard | `reports/goal_scorecard_v27.json` |
| Router replay and LOOTO | `runs/v24-router-replay/report.json`; `runs/v24-router-loso/report.json` |
| Context negative result | `runs/v24-router-context/report.json` |
| Router tail-risk analysis | `runs/v24-router-error-analysis/report.json` |
| Memory and matched-cost random controls | `runs/v27-router-controls/report.json` |
| Fresh metric-risk guard preregistration and result | `runs/v27-router-risk-guard/preregistration.json`; `runs/v27-router-risk-guard/report.json` |
| Primary paired model effects | `reports/primary_model_effects_v27.json` |
| Calibration and chemical-space residual audit | `reports/validation_calibration_residuals_v27.json` |
| Compute and peak-memory availability appendix | `reports/compute_memory_appendix_v27.json` |
| Normalized bibliography and citation audit | `paper/references.bib`; `reports/bibliography_audit_v27.json` |
| Reproducible main-text figures and captions | `paper/figures/manifest.json`; `paper/figures/captions.md` |
| Venue decision and submission gaps | `reports/venue_strategy_v27.md` |
| Weak-task negative ledger | `runs/v24-rescue-ledger/report.json` |
| Paper claim audit | `reports/paper_readiness_v27.md` |
| Hashed paper evidence bundle | `reports/paper_evidence_manifest_v27.json` |

All generated reports record whether official test or leaderboard data were
loaded. Source scripts and tests are included in the repository. The v2.7
scorecard verifies both new experts across seeds 1-5, their source/checkpoint
hashes, row counts, feature manifests, prediction hashes, and test-label
firewall. The discrepancy audit explicitly forbids using historical Bio values
as current rank evidence.

## Abbreviations

ADMET, absorption, distribution, metabolism, excretion, and toxicity; AUPRC,
area under the precision--recall curve; AUROC, area under the receiver
operating-characteristic curve; ECE, expected calibration error; GNN, graph
neural network; LCB, lower confidence bound; MAE, mean absolute error; TDC,
Therapeutics Data Commons.

## Declarations

### Ethics approval and consent to participate

Not applicable. This computational study uses public benchmark datasets and no
new human-participant or animal experiments.

### Consent for publication

Not applicable.

### Availability of data and materials

The public benchmark data are distributed through PyTDC. Source code, frozen
recipes, dependency specifications, derived validation evidence, prediction
hashes, and figure-generation scripts are prepared in the ADMET-EvO repository
under an OSI-approved MIT license. Before submission, the exact reviewed
version must be deposited in a registration-free archival repository and the
placeholder `ARCHIVE_DOI_REQUIRED` replaced with its persistent DOI. Official
test labels are neither redistributed nor exposed to model-selection code.

### Competing interests

`AUTHOR_CONFIRMATION_REQUIRED` -- the author must confirm or replace the draft
statement: The author declares no competing interests.

### Funding

`FUNDING_INFORMATION_REQUIRED` -- name funders and grant numbers, or state that
the research received no specific funding.

### Authors' contributions

Y.W. conceived the study, developed the software, ran the experiments,
analyzed the results, and drafted the manuscript. `AUTHOR_CONFIRMATION_REQUIRED`.

### Acknowledgements

`ACKNOWLEDGEMENTS_OPTIONAL`.

## References

1. Huang, Kexin; Fu, Tianfan; Gao, Wenhao; Zhao, Yue; Roohani, Yusuf; Leskovec, Jure; Coley, Connor W.; Xiao, Cao; Sun, Jimeng; Zitnik, Marinka. Therapeutics Data Commons: Machine Learning Datasets and Tasks for Drug Discovery and Development. *Advances in Neural Information Processing Systems*. 2021;34. https://arxiv.org/abs/2102.09548

2. Swanson, Kyle; Walther, Parker; Leitz, Jeremy; Mukherjee, Souhrid; Wu, Joseph C.; Shivnaraine, Rabindra V.; Zou, James. ADMET-AI: A Machine Learning ADMET Platform for Evaluation of Large-Scale Chemical Libraries. *Bioinformatics*. 2024;40. https://doi.org/10.1093/bioinformatics/btae416

3. Notwell, James H.; Wood, Michael W.. ADMET Property Prediction through Combinations of Molecular Fingerprints. *Preprint*. 2023. https://arxiv.org/abs/2310.00174

4. Zhou, Gengmo; Gao, Zhifeng; Ding, Qiankun; Zheng, Hang; Xu, Hongteng; Wei, Zhewei; Zhang, Linfeng; Ke, Guolin. Uni-Mol: A Universal 3D Molecular Representation Learning Framework. *International Conference on Learning Representations*. 2023. https://openreview.net/forum?id=6K2RM6wVqKu

5. Ji, Xiaohong; Wang, Zhen; Gao, Zhifeng; Zheng, Hang; Zhang, Linfeng; Ke, Guolin; E, Weinan. Uni-Mol2: Exploring Molecular Pretraining Model at Scale. *Advances in Neural Information Processing Systems*. 2024;37;46956--46978. https://arxiv.org/abs/2406.14969

6. Kläser, Kerstin; Banaszewski, Błażej; Maddrell-Mander, Samuel; McLean, Callum; Müller, Luis; Parviz, Ali; Huang, Shenyang; Fitzgibbon, Andrew W.. MiniMol: A Parameter-Efficient Foundation Model for Molecular Learning. *Preprint*. 2024. https://arxiv.org/abs/2404.14986

7. Cai, Fancheng; Zacour, Kelsey; Zhu, Tian; others. ChemFM as a Scaling Law Guided Foundation Model Pre-Trained on Informative Chemicals. *Communications Chemistry*. 2026;9;3. https://doi.org/10.1038/s42004-025-01793-8

8. Wang, Bowen; Li, Junyou; Zhou, Donghao; Li, Lanqing; Li, Jinpeng; Wang, Ercheng; Hao, Jianye; Shi, Liang; Lu, Chengqiang; Qiu, Jiezhong; Hou, Tingjun; Cao, Dongsheng; Chen, Guangyong; others. Unified and Explainable Molecular Representation Learning for Imperfectly Annotated Data from the Hypergraph View. *Nature Communications*. 2025;16;8717. https://doi.org/10.1038/s41467-025-63730-6

9. Jiang, Tianyi; Wang, Zeyu; Yu, Shanqing; Xuan, Qi. Adaptive Substructure-Aware Expert Model for Molecular Property Prediction. *Preprint*. 2025. https://arxiv.org/abs/2504.05844

10. Jing, Zihao; Sun, Yan; Li, Yan Yi; Janarthanan, Sugitha; Deng, Alana; Hu, Pingzhao. Structure-Aware Fusion with Progressive Injection for Multimodal Molecular Representation Learning. *Advances in Neural Information Processing Systems*. 2025. https://openreview.net/forum?id=EYC29OdxgA

11. Zheng, Yizhen; Koh, Huan Yee; Ju, Jiaxin; Nguyen, Anh T. N.; May, Lauren T.; Webb, Geoffrey I.; others. Large Language Models for Scientific Discovery in Molecular Property Prediction. *Nature Machine Intelligence*. 2025;7;437--447. https://doi.org/10.1038/s42256-025-00994-z

12. Zhao, Donghai; Zhu, Yuchen; Wu, Zhenxing; Wan, Yue; Liu, Xiaonan; Li, Shuaiyan; Xu, Hongxia; Hou, Tingjun; Hsieh, Chang-Yu. Revisiting ADMET Prediction Reliability under Real-World Challenges in the Foundation Model Era. *Journal of Cheminformatics*. 2026;18;95. https://doi.org/10.1186/s13321-026-01217-2

13. Koleiev, Ihor; Stratiichuk, Roman; Shevchuk, Nazar; others. An End-User Audit of Reproducibility, Data Leakage, and Overfitting of the Top-Ranked ADMET Prediction Models in TDC Leaderboards. *Journal of Chemical Information and Modeling*. 2026. https://doi.org/10.1021/acs.jcim.6c00819

14. Chan, Jun Shern; Chowdhury, Neil; Jaffe, Oliver; Aung, James; Sherburn, Dane; Mays, Evan; Starace, Giulio; Liu, Kevin; Maksin, Leon; Patwardhan, Tejal; Madry, Aleksander; Weng, Lilian. MLE-bench: Evaluating Machine Learning Agents on Machine Learning Engineering. *International Conference on Learning Representations*. 2025. https://proceedings.iclr.cc/paper_files/paper/2025/hash/7e3767db483c942b883eb4f8cfb74e31-Abstract-Conference.html

15. Huang, Qian; Vora, Jian; Liang, Percy; Leskovec, Jure. MLAgentBench: Evaluating Language Agents on Machine Learning Experimentation. *Proceedings of the 41st International Conference on Machine Learning*. 2024;235. https://proceedings.mlr.press/v235/huang24y.html
