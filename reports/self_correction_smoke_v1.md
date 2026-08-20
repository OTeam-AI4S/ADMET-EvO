# Prospective validation-only self-correction smoke

- Scheduled/completed actions: 2/2
- Linked behavioral self-correction: 1/1
- Completed evidence repairs: 1
- Supported corrective hypotheses: 0/2
- Resolved corrective hypotheses: 0/2
- Fits: 12; fit-seconds: 67.6
- Uni-Mol2: 0; tuning trials: 0; ensembles: 0; test loaded: false

| Endpoint | Parent outcome | Corrective action | Child evidence | Mean oriented delta | Positive seeds | Next |
|---|---|---|---|---:|---:|---|
| CYP2C9_Substrate_CarbonMangels | inconclusive `model-imbalance-aware` | repair_evidence | inconclusive | -0.016534 | 0.333 | stop |
| hERG | refuted `data-leakage-safe` | switch_axis | inconclusive | -0.003124 | 0.333 | stop |

Interpretation: the system executed both scheduled matched follow-ups and
linked them to the action that created the opportunity. CYP2C9 repaired a
missing model-pair evidence gap; hERG changed from the refuted data axis to
the model axis. Both child hypotheses remained inconclusive, so the bounded
planner stopped instead of escalating to high-cost feature/Uni-Mol2 search.

This is positive evidence for *completed corrective behavior and stopping*,
not for discovery of a better predictive recipe. It also lacks a matched rules
or AutoML feedback control, so no comparative autonomy claim is made yet.
