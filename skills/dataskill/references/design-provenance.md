# Design Provenance

This skill adapts two local design sources:

1. `D:\oagent\admet\literature\2026-Closed-loop Auto Research for Molecular Property.pdf`
2. `D:\oagent\admet\admet_evo_research_agent_architecture.md`

## Mechanisms retained from the paper

- Separate feature, model and external-evidence actions.
- Let data agents locate named external sources and submit them through a controlled interface.
- Keep evaluator, metric, leakage filter and held-out labels outside the editable agent surface.
- Freeze selected configurations and score the held-out test once.
- Preserve an auditable lineage of accepted and rejected evidence actions.
- Use standardized identity, same-source and near-analogue controls before external data enter training.

The paper reports 18 admitted sources and 1,666 external rows across three suites. It also shows why volume and contamination control are insufficient: the larger Polaris merge passed structural filtering but produced a negative held-out result. This motivates assay compatibility and distribution-shift audits in addition to leakage checks.

## Extensions introduced here

- EndpointContract and AssayContract before collection.
- Explicit access/OA and license reporting, including `pending_access` rather than silent omission.
- Primary-source and record-level evidence verification.
- Unit, transform, threshold, direction and censoring validation.
- Provenance clusters, replicate/conflict policies and independent QA.
- Multidimensional HardnessCard linked to modeling choices.
- Direct, converted, auxiliary, pretraining and external-validation roles.
- Discovery, certification and future-audit split ownership.

## Hardness alignment

The HardnessCard follows the architecture document: scarcity, imbalance, distribution shift, landscape roughness, activity cliffs, assay/label noise, selection variance and baseline saturation. This skill adds evidence completeness/access as a collection-specific dimension. Hardness guides the next action but never relaxes scientific validation.
