# Script index

Scripts are kept at stable paths because frozen protocol manifests hash or
reference many filenames. The prefixes describe their role:

- `init_*`, `freeze_*`, `prepare_*`: create immutable protocols, cohorts, or
  execution locks;
- `run_*`: execute model searches, confirmations, ablations, or held-out runs;
- `audit_*`, `preflight_*`, `promote_*`, `rollback_*`: enforce evidence and
  data-admission gates;
- `build_*`, `summarize_*`, `analyze_*`, `export_*`: produce compact derived
  evidence and manuscript tables;
- `slurm_*.sh` and `slurm/*.sbatch`: cluster launchers;
- `runtime_shims/`: narrowly scoped compatibility hooks for frozen environments.

New reusable logic should live in `admet_agent/`; a script should primarily
parse arguments and orchestrate package APIs. Do not rename a script referenced
by a frozen manifest without issuing a new protocol version.
