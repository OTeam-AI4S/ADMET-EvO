# ADMET-EvO v2.4 paper-readiness matrix

Snapshot date: 2026-08-13. This matrix separates completed evidence from the
claims that the current artifacts do not yet prove. All validation studies
listed below declare that official test labels were not loaded. Official test
results are used only in the frozen leaderboard-result section, never as model
selection inputs.

## Headline result status

| Requirement | Current evidence | Status |
|---|---|---|
| At least 15 of 22 tasks in Top-5 | The verified final v2.4 scorecard is 17/22 Top-5. | achieved |
| All 22 tasks in Top-10 | The same verified results move the count from 19/22 to 20/22. CYP2C9 substrate and Bioavailability remain below Top-10. | not achieved |
| Reproducible one-shot evaluation | Full 22-task freeze SHA-256 is `2265b92ce20168a779d36f673fcb6f583639891083c71370c9f353a9b18f81de`. DILI, hERG, Half-Life, and Lipophilicity pass result/freeze, audit, finalist, and saved training/split-seed checks for seeds 1-5. The later v2.5 Bioavailability rescue has a separate preregistration and execution-lock audit and was rejected after scoring 0.675 versus 0.697. | supported |

Rank estimates use only `reports/leaderboard_snapshot_2026-08-13.json`, so they
are tied to a dated snapshot rather than claimed as live leaderboard positions.

## Falsifiable-hypothesis audit

| Hypothesis | Strongest artifact | Evidence | Verdict |
|---|---|---|---|
| H1: sequential acquisition matches always-on selection while reducing median task cost by at least 40% | `runs/v24-router-replay/report.json`; `runs/v24-router-loso/report.json` | Replay reduced total selected-recipe cost to 10.3%, but median per-task ratio was 1.0. Held-out LOOTO cost was 18.4% of score selection with mean oriented gain -0.0020 (Wilcoxon p=0.249). | aggregate cost claim supported; preregistered median criterion failed |
| H2: LCB routing improves mean validation performance and reduces regressions versus mean-gain routing | `runs/v24-router-loso/report.json` | LCB minus score mean oriented gain was -0.0017; improved/tied/worse was 4/14/4; p=0.484. | not supported; retain as a calibrated safety mechanism, not an accuracy claim |
| H3: retaining failed acquisitions improves leave-one-task-out routing versus winner-only memory | `runs/v24-router-context/report.json`; rescue ledger | The negative-result ledger is complete and a context-conditioned penalty prototype is evaluated, but there is no matched winner-only comparison. The contextual prototype did not improve held-out score. | missing direct evidence; adjacent prototype rejected |
| H4: optional fusion outperforms unconditional fusion | `runs/v23-ablation-study/summary.md`; `runs/unimol2-pilot-v22/router_ablation.json` | Fine-tuned 3D and unconditional fusion help some endpoints and hurt others; the six-task LOSO study selects both fusion and 2D backoff. A direct 22-task held-out optional-versus-always-fuse statistic is absent. | partially supported |
| H5: metric-aligned rank evidence improves Spearman tasks without degrading MAE tasks | `runs/half-life-transform-ablation/results.json`; `runs/half-life-fusion-ablation/results.json`; frozen Half-Life official result | Half-Life rank evidence produced a frozen official Spearman of 0.590 +/- 0.030, estimated rank 1. A matched multi-endpoint Spearman/MAE interaction analysis is absent. | strongly supported on Half-Life only |

## Required-ablation coverage

| Ablation family | Available evidence | Missing for a paper-grade claim |
|---|---|---|
| Fixed compute: cheap 2D, frozen 3D, fine-tuned 3D | Six-task v2.3 study; paired Lipo 84M/164M scale comparison | Uniform summary table across all 22 tasks |
| Fixed fusion: raw, rank, selected stack | Six-task candidate/LOSO artifacts and Half-Life fusion ablation | One matched cross-task statistical summary |
| Routing: mean, LCB, cost-aware LCB | Completed 22-task replay and held-out-seed LOOTO evaluation | External-task or fresh-candidate-set generalization |
| Memory: winners only versus successes plus failures | Auditable seven-entry rescue ledger exists | Matched winner-only router ablation |
| Context: none, structural, structural plus cost | `runs/v24-router-context/report.json` provides matched 22-task LOOTO runs | Structural/full context cost ratios were 0.972/0.966 versus 0.184 globally, with no score gain; reject context conditioning in v2.4 |
| Controls: random matched cost and validation oracle | Validation oracle values exist in router artifacts | Random-action matched-cost control |

## Statistical and error-analysis coverage

- Completed: five-seed means/standard deviations for frozen recipes; paired
  validation comparisons for key 3D/fusion candidates; across-task Wilcoxon
  tests for router variants; explicit negative-result ledger; train/validation
  scaffold and molecule-overlap audit; task-bootstrap intervals and metric-wise
  tail-risk analysis in `runs/v24-router-error-analysis/report.json`.
- Still required: confidence intervals and effect sizes for the primary model
  ablations, calibration or
  uncertainty analysis for classification, residual/scaffold analysis for
  regression, and a single paper table that reports GPU-hours and peak memory
  for every promoted expert.

The router error analysis found mean oriented gain -0.0020 with task-bootstrap
95% CI [-0.0065, 0.0024] and median per-task cost ratio 0.750 [0.545, 1.000].
Spearman endpoints had mean gain -0.0119 and account for two of four material
regressions. This tail risk must appear alongside the much smaller aggregate
cost ratio, which is dominated by a few expensive recipes.

## Defensible paper story today

The defensible contribution is a protocol-safe, cost-aware evidence-acquisition
agent under scaffold shift. It decides whether to acquire expensive 3D or
target-aligned evidence, optionally fuses it, and can back off to a cheap 2D
expert. Its scientific ledger preserves failures and its freeze lock separates
validation-driven discovery from one-shot official evaluation. The evidence
supports large aggregate compute savings and endpoint-dependent value of
evidence; it does **not** support generic claims that LCB routing improves
accuracy, that the median task saves compute, that task-profile meta-learning
helps, or that multimodal/MoE fusion is itself novel.

## Next evidence with highest value per cost

1. Complete the preregistered CYP2C9 80/20 LightGBM/focal-MiniMol confirmation
   on fresh scaffold seeds 6-10; do not use the official CYP2C9 result unless
   it passes its fixed promotion gate and a new full freeze is built.
2. Treat the failed Bioavailability official rescue as distribution-shift
   evidence; do not retune that recipe against the official result.
3. If compute/usage remains available, prioritize the cheap missing router
   controls (winner-only memory, context removal, matched-cost random action)
   over another broad foundation-model sweep; these directly test the paper's
   novelty claim.
