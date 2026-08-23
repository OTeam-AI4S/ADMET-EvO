# HardnessCard and Modeling Advice

## Contents

1. HardnessCard dimensions
2. Computation guidance
3. Evidence-to-model routing
4. Data integration choices
5. Split, uncertainty and abstention
6. Advice format

## 1. HardnessCard dimensions

| Dimension | Core observations |
|---|---|
| Scarcity | usable rows, unique compounds, effective scaffolds, learning-curve slope |
| Imbalance/range | positive rate, minority count, label entropy, response coverage, censoring |
| Source heterogeneity | source count, source-label association, leave-one-source-out change |
| Assay/label noise | replicate variance, conflict rate, lab/protocol effects, uncertain labels |
| Distribution shift | nearest-neighbor similarity, novel scaffolds, domain-classifier AUC |
| Roughness | MODI/SARI/ROGI or justified alternatives, local discontinuity |
| Activity cliffs | cliff prevalence, baseline cliff recall/error |
| Selection variance | seed variance, rank stability, bootstrap uncertainty |
| Baseline saturation | baseline level, headroom, gain relative to uncertainty |
| Evidence completeness | pending-access fraction, provenance gaps, license restrictions |

Use severity labels only alongside raw metrics and reference distributions. State when a metric is not estimable.

## 2. Computation guidance

- Compute hardness on permitted discovery data, not held-out labels.
- Fit preprocessing inside folds.
- Cluster duplicated provenance before cross-validation.
- Report bootstrap or repeated-split uncertainty for small endpoints.
- Evaluate shift against deployment or authorized certification structures.
- Interpret domain-classifier AUC as separability, not proof of harmful shift.
- Qualify roughness comparisons made with different representations.

## 3. Evidence-to-model routing

| Dominant evidence | Recommended first actions | Avoid by default |
|---|---|---|
| small n/scaffolds | regularized fingerprints/descriptors; pretrained comparator; repeated CV | large model from scratch |
| extreme imbalance | PR-AUC, balanced weights/sampling, threshold and calibration analysis | ROC-AUC only, accuracy |
| scaffold/OOD shift | scaffold/source/temporal splits, 2D/3D comparison, AD and abstention | random-split claims |
| source heterogeneity | source indicators, hierarchical/multitask models, leave-one-source-out | blind pooling |
| label conflict/noise | adjudication, probabilistic labels, replicate-aware or robust losses | unexamined majority label |
| censoring | interval/Tobit/survival-style objective or sensitivity analysis | threshold substitution |
| roughness/cliffs | local or matched-pair evaluation, cliff-aware loss, neighborhood uncertainty | global metric alone |
| selection variance | repeat candidates, LCB, seed ensemble, fixed budget | more trials on one split |
| saturated baseline | stop, seek evidence or deployment metric | tiny validation gains |
| incomplete access | bound claims and prioritize retrieval/new measurement | call data comprehensive |

## 4. Data integration choices

- **Pool directly:** require Grade A compatibility, independent provenance, harmonized measurement, contamination clearance and acceptable shift. Retain source indicators.
- **Convert/weight:** use Grade B with prespecified conversions and sensitivity analysis. Prevent duplicated aggregators from dominating.
- **Multitask/transfer:** use Grade C when labels differ but biology is related, such as animal-to-human clearance or hERG binding-to-function.
- **Pretraining only:** use structures or weak labels without calling them target labels.
- **External validation:** reserve independent sources and freeze analysis before evaluation.
- **Reject/request experiments:** use when identity, semantics, provenance, license or contamination cannot be resolved, or key chemistry lacks compatible evidence.

## 5. Split, uncertainty and abstention

Match deployment with scaffold, temporal, source/lab, leave-one-source-out, grouped, nested/repeated CV and evaluator-owned certification splits. Report raw metric, uncertainty, subgroups, applicability-domain coverage, error versus similarity and calibration.

Define abstention or request-new-data rules for low similarity, high ensemble variance, conformal failure, unsupported assay domains or inconsistent evidence. Tune only on discovery data.

## 6. Advice format

```yaml
finding: source heterogeneity is high
evidence:
  - domain_classifier_auc: 0.81
  - leave_one_source_out_delta: -0.07
recommendation: train a source-aware model and report leave-one-source-out performance
expected_benefit: reduce reliance on source-specific shortcuts
falsification: no improvement in source-held-out calibration or metric
risk: reduced effective sample size
claim_boundary: does not establish prospective generalization
```

Rank advice as `required before modeling`, `recommended comparison`, or `exploratory`. Separate data defects from model opportunities.
