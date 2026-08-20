# Transport-memory behavioral replay

Status: **NO BEHAVIORAL ADVANTAGE DEMONSTRATED**

- Historical hardness cards replayed: 22/22
- Selected actions changed: 0/22
- Full action rankings changed: 14/22
- Selected utilities changed: 17/22
- Test loaded: false

The prospective grid created valid feature/model priors, but the current planner
does not yet turn them into a different action choice on the 22 stored hardness
cards. Feature/model evidence can change utilities or lower ranks without changing
the top action; the data prior correctly stays neutral.

Therefore the framework may claim protocol-bound cross-task evidence retention,
not a memory-driven behavioral or predictive advantage. The next valid upgrade is
an independent post-memory endpoint cohort with choices frozen before execution.
