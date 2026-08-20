# TDC-22 discovery-only search report

- Tasks completed: 4/22
- Mean finalist validation std: 0.07466246369655781
- Stability-risk tasks (>0.05): 3
- Frozen test loaded: no

| Task | Metric | Hardness | Selected action | Action evidence | Action Δ | Finalist | Validation mean ± std |
|---|---|---:|---|---|---:|---|---:|
| Bioavailability_Ma | auroc | 0.377 | model-imbalance-aware | supported | +0.0096 | ensemble | 0.7966 ± 0.0707 |
| CYP2C9_Substrate_CarbonMangels | auprc | 0.360 | model-imbalance-aware | refuted | -0.0165 | lightgbm | 0.4988 ± 0.1476 |
| CYP2D6_Substrate_CarbonMangels | auprc | 0.383 | model-imbalance-aware | not_evaluated | — | ensemble | 0.6088 ± 0.0792 |
| HIA_Hou | auroc | 0.373 | model-imbalance-aware | inconclusive | -0.0029 | ensemble | 0.9947 ± 0.0011 |

> This report contains validation evidence only. It must not be described as frozen-test performance.
