# ADMET-EvO v2.4 research and leaderboard plan

## Success criteria and current evidence

The operational target is at least 15 Top-5 placements and no placement below
Top-10 on the 22 TDC ADMET Group tasks. The frozen v2.1 baseline, compared with
the leaderboard snapshot captured on 13 August 2026, yielded 13/22 Top-5 and
19/22 Top-10 placements. The baseline machine-readable source of truth is
`reports/goal_scorecard.json`. Three of the four frozen v2.4 official upgrades
are now complete and independently verified: DILI is 0.920 +/- 0.004 AUROC
(estimated rank 3), hERG is 0.860 +/- 0.009 AUROC (estimated rank 5), and
Half-Life is 0.590 +/- 0.030 Spearman (estimated rank 1). Holding the unfinished
Lipophilicity result at its v2.1 value, this is already an interim 16/22 Top-5
and 20/22 Top-10. The final v2.4 scorecard is intentionally fail-closed until
all five frozen Lipophilicity seeds finish.

The Top-5 count target is therefore already exceeded. CYP2C9 substrate and
Bioavailability remain the two unresolved Top-10 rescues. A leaderboard
threshold is a planning aid, not a model-selection signal: finalist selection
used only training and validation evidence, and no recipe was changed after
the one-shot official results were opened.

## Proposed contribution: sequential evidence acquisition

The paper should not claim that multimodal fusion or mixture-of-experts routing
is itself novel. Recent work already includes task-routed experts in
[OmniMol](https://www.nature.com/articles/s41467-025-63730-6), adaptive
substructure experts in [ASE-Mol](https://arxiv.org/abs/2504.05844), and robust
2D/3D structured fusion in
[MuMo](https://openreview.net/forum?id=EYC29OdxgA). The narrower, testable claim
for ADMET-EvO is:

> ADMET-EvO treats molecular modalities as evidence with different acquisition
> costs. A protocol-safe research agent first evaluates cheap 2D and
> metric-aligned experts, predicts the lower-confidence-bound value of acquiring
> expensive 3D evidence under scaffold shift, and then chooses to acquire,
> fuse, or back off. An auditable ledger preserves both successful and failed
> validation experiments, while a freeze lock prevents test feedback from
> entering subsequent recipe decisions.

This differs from a conventional gated network in four ways:

1. Routing is an experiment-level decision made before expensive evidence is
   computed, rather than a molecule-level latent gate after every modality is
   available.
2. The action objective includes measured wall-clock/GPU cost and uncertainty,
   not predictive score alone.
3. Fusion is optional and can be rejected in favor of a single expert or a
   conservative 2D backoff.
4. The agent's evidence ledger and freeze lock make the scientific search
   trajectory and negative results reproducible.

The agent aspect must also be distinguished from
[LLM4SD](https://www.nature.com/articles/s42256-025-00994-z), which uses an LLM
to synthesize literature and data-derived molecular rules. ADMET-EvO's claim is
not autonomous chemical knowledge discovery; it is protocol-safe experimental
decision-making over bounded molecular tools.

## Method specification

For task `t` and candidate action `a`, estimate the validation utility

```text
U(a, t) = mean_gain(a, t) - beta * uncertainty(a, t) - lambda * cost(a, t)
```

where gains are computed against the current cheap champion on repeated outer
scaffold splits. Uncertainty is the cross-seed standard error or a bootstrap
bound; cost is normalized GPU-seconds plus peak memory. The router acquires an
expensive modality only when its lower-confidence-bound utility is positive.

Actions are bounded and auditable:

- `stop_2d`: freeze the robust 2D champion;
- `acquire_3d`: fine-tune the preregistered Uni-Mol2 scale;
- `acquire_target_view`: fit a metric-aligned target transform or rank expert;
- `fuse`: combine experts from inner-scaffold OOF predictions;
- `backoff`: discard unreliable acquired evidence and retain the cheap champion.

The deterministic core is implemented in `admet_agent/evidence_router.py` and
is exposed by `admet-agent search --selection-policy cost-aware-lcb`. Search
artifacts record every candidate's oriented score, cross-seed uncertainty,
observed cumulative time, normalized log cost, LCB utility, and selected action.
The legacy `score` policy remains the default so existing frozen results retain
their original selection semantics.

The planner receives endpoint type, metric, sample count, imbalance or target
skew, scaffold shift, and measured costs. The validated `EvidenceRouter`
currently uses the metric plus candidate score, uncertainty, and cost; it does
not yet use the remaining `TaskProfile` fields. A preregistered leave-one-task-
out nearest-task prototype tested structural and full-profile conditioning.
Both contextual variants erased the global policy's aggregate cost advantage
(cost ratios 0.972 and 0.966 versus 0.184) without improving held-out score.
They are therefore retained as negative results rather than promoted. No
component may consume official test scores, test labels, or leaderboard rank.

## Falsifiable hypotheses

1. Sequential acquisition matches or exceeds always-on 2D+3D fusion while
   reducing median GPU cost by at least 40% across the 22 tasks.
2. Lower-confidence-bound routing improves mean validation rank and reduces the
   number of material regressions versus routing by mean gain alone.
3. Retaining negative acquisitions improves leave-one-task-out router decisions
   versus a winner-only memory.
4. Optional fusion outperforms unconditional fusion because some endpoints
   receive a stable 2D backoff.
5. Metric-aligned target/rank evidence improves Spearman endpoints without
   degrading MAE endpoints.

The completed 22-task router replay refines hypothesis 1. With a global
`lambda=0.005`, held-out-seed score was statistically indistinguishable from
LCB selection (mean oriented gain +0.0003, Wilcoxon p=0.953) while the aggregate
selected-recipe time proxy fell to 19.1%. A stricter leave-one-task-out lambda
calibration retained 18.5% of the LCB cost with mean gain -0.0003 (p=0.721).
This supports aggregate cost efficiency, but not the original median-task 40%
criterion: many tasks selected the same recipe and therefore had a cost ratio
of one. The paper must report this distinction rather than replacing the
original hypothesis post hoc.

A task-bootstrap error analysis further limits the claim. Relative to score
selection, LOOTO cost-aware routing had mean oriented gain -0.0020 with 95% CI
[-0.0065, 0.0024], while the median per-task cost ratio was 0.750 with 95% CI
[0.545, 1.000]. At an absolute native-metric threshold of 0.005 there were two
material improvements and four regressions. Spearman endpoints were the clear
tail-risk subgroup (mean -0.0119; zero improved, two tied, two worse). These are
descriptive post-hoc findings in
`runs/v24-router-error-analysis/report.json`; they must not be used to alter the
v2.4 freeze. A metric-specific safety guard is a prospective hypothesis that
requires fresh validation seeds.

Failure of hypotheses 1-4 would reduce the method to a conventional model
selection system and invalidate the main novelty claim.

Current evidence rejects a simple nearest-task implementation of task context:
full context produced mean held-out oriented gain -0.0020 (Wilcoxon p=0.110)
and aggregate cost ratio 0.966, while the non-contextual global penalty policy
gave -0.0020 (p=0.249) at cost ratio 0.184. The artifact is
`runs/v24-router-context/report.json`. This result narrows, rather than expands,
the paper claim: task-profile meta-learning and winner-only versus full-memory
learning cannot be presented as validated contributions in v2.4.

## Required comparisons and ablations

| Family | Required condition |
|---|---|
| Fixed compute | best cheap 2D expert; frozen 3D expert; fine-tuned 3D expert |
| Fixed fusion | raw mean; percentile-rank mean; validation-selected stack |
| Routing | mean-gain router; LCB router; cost-aware LCB router |
| Memory | winner-only ledger; successes and failures ledger |
| Context | no task profile; structural profile; full profile plus cost |
| Controls | random action at matched cost; validation oracle upper bound |

Report mean and standard deviation over five outer scaffold seeds, paired seed
differences with confidence intervals, action frequency, GPU-hours, peak memory,
and regret relative to the validation oracle. Across-task comparisons should use
a paired non-parametric test and report effect sizes rather than relying only on
average rank.

## Staged execution and freeze gates

### Gate A: reuse completed evidence

- The completed Lipophilicity scale comparison selected 84M: validation MAE was
  0.4036 +/- 0.0208 for 84M and 0.4129 +/- 0.0196 for 164M. The larger model
  used 2.03x the GPU time and improved only one of five paired seeds.
- Promote Lipophilicity 84M and hERG 2D/3D blend only if their completed
  five-seed validation records and artifacts pass integrity checks.
- Promote DILI fusion and Half-Life rank fusion only after reconstructing their
  exact recipes from saved OOF predictions.

### Gate B: repair missing evidence

- Run the router's cheap experts on all 22 tasks using cached features.
- For Bioavailability and CYP2C9 substrate, add external pretrained embeddings
  or multitask transfer only after a train/validation-only provenance audit.
- Do not launch a broader foundation-model sweep unless the cheap stage predicts
  positive LCB utility.

### Gate C: preregister finalists

- Select recipes by mean score subject to a variance/LCB guard.
- Write one serialized recipe per task, artifact hashes, data hashes, dependency
  versions, and expected resource use.
- Run a dry reconstruction check without loading the official test split.

### Gate D: one-shot official evaluation

- Evaluate only frozen recipes with seeds 1-5.
- Never use test metrics to change weights, members, transforms, or routing.
- Regenerate `reports/goal_scorecard.json` from the new frozen results and the
  same dated leaderboard snapshot.

Current status: the first four candidates (DILI, hERG, Lipophilicity, and
Half-Life) are serialized in `runs/v24-freeze/finalists.mapping.json`. Their
five-seed validation artifacts and runtime sources are SHA-256 locked. The
heterogeneous executor is implemented and synthetic leakage-safe dry-runs pass;
real A100 smoke tests have passed for all four upgrades. A seed-propagation
audit caught that unimol-tools 0.1.5 silently retained its default training and
split seeds; the executor now writes both seeds directly into the runtime
configuration, and the smoke gate verifies the saved values. A full 22-task
freeze reuses 18 verified v2.1 locks and replaces only these four upgrades. All
four fail-closed gates passed under freeze SHA-256
`2265b92ce20168a779d36f673fcb6f583639891083c71370c9f353a9b18f81de`, and the
four one-shot seeds 1-5 official jobs were submitted without changing recipes.

The DILI, hERG, and Half-Life jobs have since completed. Their result/freeze
hashes, complete audit records, serialized finalists, and saved Uni-Mol
training/split seeds all pass `verify_upgrade_result`. Their official readouts
were recorded only after the freeze and are not inputs to any subsequent model
choice. The Lipophilicity job remains in progress; until it finishes,
`scripts/build_v24_goal_scorecard.py` rejects construction of a nominally final
22-task report.

## Near-term leaderboard path

The validation evidence supports this order, without assuming that validation
gains will transfer exactly to the official test:

1. Lipophilicity: 84M fine-tuned Uni-Mol2 reduced five-seed validation MAE from
   the 2D baseline near 0.519 to 0.4036; the Top-5 test threshold gap is only
   0.007. Scaling to 164M was worse and approximately doubled runtime.
2. hERG: routed 2D/3D fusion reached 0.8938 validation AUROC versus 0.8867 for
   the best 2D expert; the Top-5 test threshold gap is 0.022.
3. DILI: routed fusion reached 0.8703 validation AUROC versus the prior champion
   at 0.8670, but the test Top-10 gap is 0.028, so it remains a risk rather than
   a guaranteed rescue.
4. Half-Life: preregistered heterogeneous rank fusion is the next promotion
   candidate, but its Top-5 test gap is 0.056 and the fixed blend still needs a
   fresh-split confirmation.
5. Bioavailability and CYP2C9 substrate require new evidence sources; current
   validation improvements are too small or unstable to justify a test run.

## Weak-task rescue ledger

Exact-overlap-excluded cross-endpoint teachers improved Bioavailability mean
AUROC from 0.6898 to 0.7175, but this is far below the 0.922 Top-10 threshold
and is retained only as an ablation. On CYP2C9 substrate, the same transfer
reduced mean AUPRC by 0.0064. A fresh-seed (6-10) fixed rank ensemble gained
only 0.0069 over LightGBM and remained high variance; balanced bagging gave no
gain. A separately pinned MiniMol 1.3.5 environment was then evaluated without
test access: the fixed official MLP head scored 0.4484, equal-rank fusion with
LightGBM scored 0.4676, and an inner-scaffold five-head ensemble scored 0.4628,
all below the 0.4801 LightGBM validation mean. These negative results are preserved in
`runs/v24-rescue-ledger/report.json` and are not silently promoted.

## Resource discipline

The active Goal may consume at most 60% of the user's weekly Codex usage. The
Goal API does not expose the weekly quota as a numeric token budget, so the
operational policy is conservative: reuse cached features and completed runs,
avoid duplicate literature scans, prefer deterministic local analysis, and
require a validation LCB gate before any new expensive experiment or broad
agent iteration. If remaining weekly usage cannot be measured reliably, stop
before a large exploratory branch and request an explicit numeric allowance.

## Protocol warning

[ChemFM](https://www.nature.com/articles/s42004-025-01793-8) explicitly notes
that some reported ADMET leaderboard results used invalid evaluation practices,
including test-set hyperparameter optimization or combining train and validation
data improperly. ADMET-EvO should optimize reproducible rank under the official
protocol, not chase scores that cannot survive an audit.
