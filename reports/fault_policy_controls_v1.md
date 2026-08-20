# Hidden-fault policy controls

Locked replay over the existing 10-case deterministic fixture; no test data loaded.

| Arm | Micro F1 | Critical recall | Remediation validity | Safety gate | Agent−control case-F1 CI |
|---|---:|---:|---:|---|---:|
| Agent contract guard | 1.000 | 1.000 | 1.000 | PASS | — |
| Complete rules guard | 1.000 | 1.000 | 1.000 | PASS | [0.000, 0.000] |
| Expanded AutoML schema guard | 0.615 | 0.167 | 0.500 | FAIL | [0.200, 0.800] |
| Conventional AutoML | 0.000 | 0.000 | 0.100 | FAIL | [0.700, 1.000] |
| Seeded random | 0.211 | 0.167 | 0.200 | FAIL | [0.500, 1.000] |

## Interpretation

The Agent beats AutoML variants because they lack endpoint-semantic guards. However,
the complete deterministic rules guard exactly ties the Agent. Therefore current fault
discovery is a useful system capability but is **not** evidence of learned scientific
reasoning. A stronger claim requires independently authored unseen composite faults and
remediation planning that exceeds the complete-rules baseline.
