# Independent post-memory pruning confirmation

Status: **GLOBAL NEGATIVE-MEMORY GATE REFUTED — DEPLOYMENT FORBIDDEN**

- Fresh validation fits: 50/50; failures=0
- Outcomes: `{'inconclusive': 1, 'refuted': 2, 'supported': 2}`
- Supported endpoints: `['Solubility_AqSolDB', 'PPBR_AZ']`
- Mean positive pruning regret: 0.065379
- Maximum positive pruning regret: 0.258620
- Test loaded: false

The global memory prior predicted that RandomForest-vs-ExtraTrees should be
pruned. Independent seeds contradicted that policy: Solubility and PPBR showed
material, 5/5-direction gains. CYP3A4 and Clearance were refuted; DILI was
inconclusive. The action effect is heterogeneous, not globally negative.

The apparent saving of 25 treatment fits is therefore not an approved efficiency
gain: it would have discarded two useful experiments. The correct autonomous
response is to disable global pruning, retain this negative policy result, and
require task-type/hardness-stratified independent confirmation.
