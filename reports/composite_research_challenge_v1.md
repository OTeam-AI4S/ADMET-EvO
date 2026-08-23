# Sealed composite research-decision challenge

Local commit → submit → reveal benchmark; 10 composite episodes, no model training,
no frozen TDC test feedback. Every arm has at most one follow-up and six estimated fits
per episode.

| Arm | Decision accuracy | Safe action | Parent link | Matched pair | Macro process | Unsafe | Est. fits |
|---|---:|---:|---:|---:|---:|---:|---:|
| Agent production feedback policy | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0 | 30 |
| Locked static rules | 0.600 | 0.700 | 1.000 | 1.000 | 0.860 | 3 | 48 |
| Expanded AutoML scalar search | 0.400 | 0.900 | 0.200 | 0.200 | 0.540 | 1 | 48 |
| Conventional AutoML scalar search | 0.400 | 0.900 | 0.200 | 0.200 | 0.540 | 1 | 48 |
| Seeded random action | 0.200 | 0.900 | 0.200 | 0.200 | 0.480 | 1 | 32 |

## Paired episode comparisons

| Control | Mean Agent − control case score | Bootstrap 95% CI | W/T/L |
|---|---:|---:|---:|
| Locked static rules | 0.140 | [0.040, 0.260] | 4/6/0 |
| Expanded AutoML scalar search | 0.460 | [0.300, 0.600] | 8/2/0 |
| Conventional AutoML scalar search | 0.460 | [0.300, 0.600] | 8/2/0 |
| Seeded random action | 0.520 | [0.360, 0.660] | 9/1/0 |

## What the composite cases add

The cases require one policy to combine: supported/refuted/inconclusive feedback,
paired-seed sufficiency, remaining action budget, a 0.60 compute gate, quarantined
versus admitted external data, cross-axis switching, explicit parent linkage and
bounded stopping. This is harder than scoring each capability in isolation.

The Agent safely skipped quarantined data and over-budget Uni-Mol2-style actions,
repaired incomplete evidence, switched after resolved null/refuted results, and
stopped when no safe executable action remained. It used 30 estimated fits versus
48 for static rules and scalar AutoML controls under the same capacity ceiling.

## Claim boundary

- The challenge is a same-team local sealed holdout, not independently authored external validation.
- The Agent arm executes the production validation self-correction planner; it is not a hand-filled answer table.
- The locked static rules arm is a generic one-step transition policy, not an oracle containing per-episode answers.
- A positive local bootstrap interval supports benchmark conformance only and must not be promoted to universal autonomous-reasoning superiority.
- No predictive models were trained and no frozen TDC test feedback was used.

Therefore this artifact upgrades the evidence from three isolated fixtures to one
integrated policy-conformance benchmark. It still does not satisfy the requirement
for an independently authored unseen challenge or prove learned reasoning beyond all
possible deterministic rules.
