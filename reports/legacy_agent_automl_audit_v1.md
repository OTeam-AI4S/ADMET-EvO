# Legacy Agent-vs-AutoML audit

Status: **NOT ELIGIBLE for a formal same-budget claim**

- TDC endpoints: 22/22
- Observed action parity: 21/22; Agent same-or-lower actions: 22/22
- Declared max actions: Agent=8, AutoML=8; both compliance flags=false
- Mean train-normalized oriented Agent−AutoML delta: -0.011923
- Task-bootstrap 95% CI: [-0.023226, -0.003536]
- Wins/ties/losses: 5/2/15
- Tasks within −0.01: 14/22
- Worst endpoint: Half_Life_Obach (-0.111457)
- Aggregate Agent/AutoML fit-time ratio: 1.363
- Execution-kernel hashes identical: True
- Full source snapshot identical: False
- Test loaded: false

The exploratory result is negative: the legacy Agent was worse on normalized
macro validation performance and consumed more fit time. This does not challenge
the separate application-performance qualification; it blocks predictive or compute
superiority claims against AutoML. Protocol violations and source mismatch make the
comparison ineligible for formal noninferiority certification in either direction.
