# Prospective cross-task transport grid

Status: **COMPLETE, PARTIALLY ROUTING-ELIGIBLE**

- Fits: 100/100; failures=0
- Matched episodes: 15/15
- Test loaded: false
- UniMol2 fits: 0
- Routing-eligible actions: `['feature-domain-shift', 'model-estimator-family']`
- Routing-blocked actions: `['data-quality-filter']`

## Scientific outcomes

- `data-quality-filter`: {'inconclusive': 5}; posterior support=0.500; routing_eligible=false
- `feature-domain-shift`: {'refuted': 2, 'supported': 3}; posterior support=0.556; routing_eligible=true
- `model-estimator-family`: {'inconclusive': 1, 'refuted': 4}; posterior support=0.278; routing_eligible=true

Feature transfer was heterogeneous, so it now has a legitimate global prior
rather than the old observational 18/22 success impression. RandomForest versus
ExtraTrees was mostly refuted and can be downweighted. Quality filtering changed
training membership in 25/25 splits but was inconclusive on all five endpoints;
it therefore remains neutral and cannot influence routing.

## Efficiency and evidence boundary

Shared controls avoided 50 duplicate fits and an estimated 510.6 fit-seconds.
Measured fit time / naive repeated-control estimate: 1234.4 / 1745.0 seconds.

This supports prospective cross-task memory for the eligible action priors and
the ability to retain/use negative results. It does not establish contextual
routing superiority, predictive superiority over AutoML, or universal benefit
from data cleaning. The memory package is not automatically installed into a
production run; deployment requires a separate protocol-bound import.
