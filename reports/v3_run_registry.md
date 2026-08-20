# v3 run registry

| Run directory | Status | Permitted claim |
|---|---|---|
| `runs/v3-priority-pilot` | completed, 4 tasks, 3 seeds, protocol-unbound | development pilot only |
| `runs/v3-priority-pilot-v2` | completed, CYP2D6 substrate, 3 seeds, protocol-unbound | development pilot only |
| `runs/v3-agent-tdc22` | interrupted during Caco2 | no aggregate claim |
| `runs/v3-agent-tdc22-v2` | interrupted during Caco2; protocol-bound but no source snapshot | no formal comparison claim |
| `runs/v3-agent-tdc22-v3` | interrupted during Lipophilicity after detecting seed-pseudoreplication in cross-task memory | strategy-fault discovery only |
| `runs/v3-agent-tdc22-v4` | interrupted after 9 task summaries when robust promotion was found to discard a stronger tuned recipe solely because its model family matched a mandatory causal-control pair | strategy-fault discovery only; excluded from formal aggregate claims |
| `runs/v3-agent-tdc22-v5` | active formal agent arm under promotion/timeout protocol v3 | eligible only after 22-task completion and audit |

Formal v3 arm requirements:

- benchmark protocol SHA-256: `99c391550bccfeff6e3c5fc90240e1a5f87b99b1770a9ff7e818695e64408d2e`
- production source SHA-256: `b71287f4cef7494132ed0ff0d7d818710986565a7e700fecddbb8d05adaf47f5`
- discovery seeds: `1,2,3,4,5`
- selection: matched causal pair + score/runtime Pareto candidates + model diversity, followed by effect-first multi-seed score
- candidate timeout: hard per-candidate wall-time deadline
- test feedback: forbidden

Only `runs/v3-agent-tdc22-v5` and later arms with both v3 hashes embedded in every task summary may enter the formal same-budget comparison.

## Active v5 strategy-fault ledger

- `CYP2C9_Substrate_CarbonMangels`: the routed primary action was
  `model-imbalance-aware`, but robust promotion retained the
  `balanced_extra_trees` treatment without its ordinary `extra_trees` matched
  control. The task correctly reported the action evidence as `inconclusive`
  (zero completed paired seeds). Keep the v5 run protocol frozen, retain the
  predictive result, and exclude this task-action from successful autonomous
  research claims. After the arm completes, extend mandatory-pair discovery to
  model/imbalance interventions and add a regression test before any successor
  formal arm.
- `CYP3A4_Substrate_CarbonMangels`: the feature action was labelled
  `supported` from a mean oriented AUROC delta of `+0.01351`, but only three of
  five seeds were positive, only two exceeded the `0.005` materiality gate,
  and one `+0.08333` seed dominated the mean (`delta_std=0.03755`). Preserve
  the protocol label for v5, but grade this as low-confidence support in the
  evidence package. A successor protocol must require a direction-consistency
  or interval-based stability gate in addition to the mean-effect threshold.
