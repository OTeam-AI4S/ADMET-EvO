# EC-VOI v5 public cross-source policy gate v1

Date: 2026-08-23
Decision: **FAIL — no promotion, no fresh seeds, TDC-19 remains sealed**

## Frozen evidence

- Action-screen protocol self-hash: `e6911d3879ded1429baed7ee32a96ac378d566af1da01d265aa733f49dae41ef`.
- Action registry: 415/415 endpoint-fold/action rows, zero failures; SHA-256 `9101d34c242ae2b20d6cfcd9e352b5e9b1f1d2e94b461d642d56a4d380b80208`.
- Policy-analysis artifact SHA-256: `7a67af28b45939fde2b63088f5199ca2c01f8c54d937d96da5e5d862b17a1447`.
- Test/blind/leaderboard access: 0. Analysis code and promotion checks were frozen before action outcomes existed.

## Primary result

| Policy/control | Total gain | Mean gain | Mean oracle regret | Harmful | Fit cost | Gain/fit |
|---|---:|---:|---:|---:|---:|---:|
| Full v5 | 1.230112 | 0.076882 | 0.045966 | 0 | 45 | 0.027336 |
| No diagnosis | 1.881659 | 0.117604 | 0.005245 | 0 | 91 | 0.020678 |
| Shuffled diagnosis | 1.771212 | 0.110701 | 0.012148 | 0 | 85 | 0.020838 |
| Frozen rules | 1.820159 | 0.113760 | 0.009089 | 0 | 66 | 0.027578 |
| Fixed rich-2D LightGBM | 2.015914 | 0.125995 | control-specific | 0 | 16 | 0.125995 |
| Expanded nested AutoML | 1.830066 | 0.114379 | control-specific | 0 | 336 | 0.005447 |
| Full without harm gate | 1.320908 | 0.082557 | 0.040292 | 0 | 50 | 0.026418 |

Full reduced neither comparator's regret: the preregistered reductions were -7.764 versus no-diagnosis and -2.784 versus shuffled, rather than >=0.20. Its total gain was lower by 0.651547 and 0.541100, respectively. Source-cluster bootstrap positive-direction probabilities were 0.056 and 0.3585, below the frozen 0.90 threshold.

Safety and integrity checks did pass: zero harmful interventions, three endpoint-level wins over the fixed boosting anchor, matched maximum fit budget, and zero forbidden evaluation access. These do not rescue the failed identifiability/generalization gate.

## Failure interpretation

When Tox21 was held out as a source, the state-conditioned posterior had only a few non-Tox21 endpoint/source units from which to learn action heterogeneity. Full selected feature-rich on several Tox21 endpoints where nested imbalance was the better observed intervention. In contrast, the intercept-only policy selected nested imbalance on 15/16 endpoints and approached the action oracle. Thus the apparent state encoder added unstable cross-source extrapolation rather than useful diagnosis.

This is not evidence that diagnostic state can never work. It is evidence that the current low-dimensional state representation plus ridge posterior is not identifiable under the available number and diversity of source clusters. Any successor must be treated as a new mechanism and validated on new endpoint/source evidence; retuning this cohort and relabeling it prospective is prohibited.

## Claim boundary and early stop

- Do not claim that v5 diagnosis improves action selection, regret, gain or efficiency.
- It is valid to claim that harm gating produced zero harmful interventions in this cohort, but not that it caused a performance advantage.
- It is valid to report that fixed LightGBM was both strongest and most fit-efficient here, while expanded AutoML added little for 21x fit cost.
- Seeds 29/43 are not run; TDC-19 remains sealed; no confirmation protocol is generated.
- Paper options are: retain this as a rigorous negative result, or develop v6 on this cohort and validate only on a genuinely fresh, source-diverse cohort.
