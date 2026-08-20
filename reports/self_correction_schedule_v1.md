# Minimal validation-only self-correction schedule

Source: `runs/v3-agent-tdc22-v5`

- Tasks audited: 22/22
- Scheduled follow-ups: 2
- Estimated additional fits: 12
- Maximum normalized action compute cost: 0.60
- Test loaded: false; training started: false

| Endpoint | Trigger | Decision | Next axis | Est. fits | Status |
|---|---|---|---|---:|---|
| Bioavailability_Ma | refuted `model-imbalance-aware` | stop | — | 0 | stopped |
| CYP2C9_Substrate_CarbonMangels | inconclusive `model-imbalance-aware` | repair_evidence | model | 6 | scheduled |
| HIA_Hou | inconclusive `model-imbalance-aware` | stop | — | 0 | stopped |
| hERG | refuted `data-leakage-safe` | switch_axis | model | 6 | scheduled |

Only scheduled follow-ups are candidates for a prospective smoke run. The
18 supported endpoints stop immediately; high-cost follow-ups above the 0.60
normalized compute gate are not launched merely to create more activity.

This artifact is a queue, not evidence of successful self-correction. A task
counts only after its linked matched experiment is executed and retained.
