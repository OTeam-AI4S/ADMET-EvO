# Cross-task action-memory readiness audit

Status: **NOT YET ELIGIBLE to influence routing**

- TDC endpoints audited: 22/22
- Single-axis evidence complete: 22/22
- Selected actions: `{'data-leakage-safe': 1, 'feature-domain-shift': 18, 'model-imbalance-aware': 3}`
- Outcomes: `{'inconclusive': 2, 'refuted': 2, 'supported': 18}`
- Largest-action share / normalized action entropy: 0.818 / 0.525
- Transport-eligible records: 0/22
- Test loaded: false

## What this means

The historical results are valid evidence that the Agent can execute matched,
single-axis experiments and retain supported/refuted/inconclusive outcomes. They
are not a causal training set for cross-task routing: the existing policy selected
the action, 18/22 endpoints used the same feature action, and counterfactual actions
are missing. Treating those rows as priors would create selection/positivity bias.

The runtime now enforces this boundary. Observational memory remains auditable but
uses a neutral effect multiplier, cannot reorder actions, and cannot enable UniMol2.
Therefore `agent_no_memory` parity is expected until eligible evidence exists; it
must not be marketed as a demonstrated memory advantage.

## Minimum next experiment

Run a preregistered complete action grid on at least 5 sentinel endpoints across 3 currently observed action families (15 matched episodes minimum).
Use identical budgets, one shared protocol hash, train/validation only, and retain
all negative outcomes. This experiment estimates transport; it does not reopen or
optimize the frozen application test models.
