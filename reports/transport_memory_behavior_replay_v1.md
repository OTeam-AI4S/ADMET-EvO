# Transport-memory behavioral replay

Status: **NO BEHAVIORAL ADVANTAGE DEMONSTRATED**

- Historical hardness cards replayed: 22/22
- Selected actions changed: 0/22
- Full action rankings changed: 2/22
- Selected utilities changed: 17/22
- Test loaded: false

The prospective grid created valid feature/model priors, but the current planner
does not yet turn them into a different action choice on the 22 stored hardness
cards. Feature/model evidence can change utilities or lower ranks without changing
the top action; the data prior correctly stays neutral.

Therefore the framework may claim protocol-bound cross-task evidence retention,
not a memory-driven behavioral or predictive advantage. The independent cohort
has now refuted global negative pruning; the next valid upgrade requires a new
task-type/hardness-stratified cohort while destructive pruning remains disabled.
