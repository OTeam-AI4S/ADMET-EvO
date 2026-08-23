# OpenADMET CYP assay-conditioned multi-fidelity model contract v1

Status: **training-only design contract**. No blind-label or leaderboard
feedback exists. Architecture selection uses only identity-locked chemistry
group inner folds; one sealed chemistry-group certification partition is opened
once after the recipe and stack are frozen.

## Scientific hypothesis

The same molecule is observed through related but non-exchangeable CYP assay
views. Direct-inhibition pIC50, pIC50 after active preincubation, Emax,
single-concentration response and time-dependent-inhibition class share chemical
signal, but differ by isoform, assay arm, readout and fidelity. A model should
share molecular evidence while conditioning every observation on those assay
semantics and should enforce only biochemically declared cross-head relations.

The novelty claim is the joint mechanism
`assay semantics + sparse multi-fidelity evidence + interval-aware uncertainty + executable Agent selection`,
not the use of a particular molecular embedding.

## Observation representation

All four labeled training tables are converted to sparse long form after
standardization. Each row contains:

- canonical molecular identity and shared chemistry-group/fold assignment;
- molecule features (frozen representation or preregistered 2D view);
- CYP isoform token;
- assay-arm token (`direct`, `TDI_preincubation`, `single_concentration`);
- readout token (`pIC50`, `Emax`, `log2fc`, `is_TDI`);
- fidelity/source token;
- scalar condition metadata when available;
- target, task type, credible interval and observation mask.

Alternate raw SMILES and modalities for one identity cannot cross partitions.
Missing labels remain masked; they are not imputed as negatives or duplicated
as independent samples.

Immediately before every inner selection fit and outer refit, the runner
independently verifies zero InChIKey overlap, zero chemistry-group overlap,
one-group-per-identity consistency and absence of certification partitions.
This runtime check occurs after any external bridge rows are appended and fails
closed even when an external record carries an incorrect chemistry-group ID.

## Model family

`M0` uses a frozen 512-dimensional MiniMol molecular embedding plus a shared
trainable projection trunk, learned isoform/assay/readout/fidelity
embeddings, a gated fusion block and separate regression/classification output
heads. The gate can modulate molecular features from assay context but cannot
access split identity or blind-set summaries.

`M1` adds interval-aware regression. A prediction inside the fitted credible
interval has zero location error; outside it is penalized by distance to the
nearest bound. A bounded log-variance head supplies heteroscedastic uncertainty
and is evaluated for calibration, not only point score.

`M2` admits the TDI table's within-experiment paired direct/TDI-condition pIC50
and Emax observations as separate auxiliary fidelities. `M3` admits
single-concentration log2fc through its own conditioned
head. Auxiliary loss weights are selected from a small frozen grid in inner
folds and are shared across outer folds.

`M4` adds a biochemical consistency term. Only the TDI table's explicitly paired
direct and TDI-condition pIC50 observations define the predicted shift
`pIC50_TDI - pIC50_direct` is mapped to a TDI probability around the declared
two-fold IC50-shift threshold `log10(2)`. Consistency is evaluated only where
the Endpoint Contract declares the relation and cannot fabricate a missing
supervised label.

## Baselines and final stack

- `B1`: rich 2D fingerprints/descriptors with endpoint-specific LightGBM;
- `B2`: the same frozen 512-dimensional MiniMol embedding with
  endpoint-specific LightGBM;
- `M0`–`M4`: the cumulative mechanism sequence above;
- `E0`: non-negative OOF stack of B1, B2 and the single frozen M-family winner.

Stack weights are learned only from out-of-fold development predictions and
then frozen. The sealed certification partition cannot alter members or weights.
E0 uses non-negative weights on a fixed 0.05 grid, separately for regression
and classification. Its reported development estimate is nested: each outer
fold is predicted with weights learned on the other four folds. Exact metric
ties prefer the least-concentrated convex combination. Only after that estimate
is complete are final weights learned from all OOF development predictions for
future frozen certification/refit.

MiniMol extraction receives a label-free manifest of training SMILES and emits
an embedding matrix plus input, package, checkpoint/config/base-shape, package
archive and MIT-license hashes. Neural heads run
in the existing isolated CPU PyTorch environment; tree baselines run in the
main frozen ADMET environment. No dependency or checkpoint is selected from
validation performance.

## Required paired ablations

1. M0 vs B1/B2: assay-conditioned sharing;
2. M1 vs M0: credible-interval-aware loss and uncertainty head;
3. M2 vs M1: paired TDI-pIC50/Emax auxiliary evidence;
4. M3 vs M2: single-concentration evidence;
5. M4 vs M3: biochemical consistency;
6. M4 without isoform token;
7. M4 with shuffled assay tokens;
8. shared model vs endpoint-isolated copies with matched parameter/fit budget;
9. E0 vs its best component;
10. each external-data route vs the identical no-external-data recipe.

An ablated arm reuses the same folds, seeds, training steps and tuning grid. A
mechanism is credited only if its paired task-macro bootstrap estimate improves
the preregistered primary point metric or a declared UQ endpoint without a
material primary-metric regression.

## Metrics and uncertainty

Direct inhibition uses official macro soft-threshold RAE as primary; MAE,
Spearman and interval coverage are secondary. TDI classification uses macro MCC
as primary; PR-AUC, balanced accuracy, Brier and calibration error are
secondary. Report cluster-bootstrap intervals, fold/seed dispersion,
chemistry-distance performance, conformal coverage, risk-coverage and
abstention utility.

The statistical unit for cross-endpoint claims is an endpoint/isoform, not a
row or seed. Within-endpoint resampling quantifies uncertainty but does not
inflate the number of independent tasks.

## External evidence routes

Exact compatible measurements may enter an explicitly source-conditioned
supervised head. Veith/TDC CYP activity labels are bridge-only because they do
not reproduce the exact OpenADMET pIC50/TDI assay contract. Lower-quality or
ambiguous related data are pretraining-only; missing provenance, license,
species, units or relation operators causes quarantine/rejection.

Every route is compared against no external data and evaluated with
leave-one-source-out/source-held-out tests. External records use the fixed
record-level blind-structure firewall, but aggregate blind removal counts cannot
select a route or model.

## Compute-aware promotion

Stage A runs B1, B2, M0, M1 and M2 on three outer folds and one seed. A candidate
with worse task-macro primary score than B1 and no prespecified calibration/UQ
gain stops. Stage B runs survivors plus M3/M4 on all five outer folds and three
seeds. Stage C freezes one component/stack and opens the sealed internal holdout
once. Stage D retrains the identical recipe on all admitted labeled training
data and produces one content-addressed blind inference package.

The machine-enforced Stage-A margins are fixed before execution: ST-RAE may not
worsen by more than 0.05 and MCC may not worsen by more than 0.03 versus B1.
Promotion additionally requires a positive mean change in at least one primary
metric, or a macro Brier improvement of at least 0.01 or ECE improvement of at
least 0.02. A UQ improvement cannot rescue violation of either primary
noninferiority margin.

Before Stage C, `scripts/freeze_openadmet_cyp_recipe.py` must bind the selected
component results, nested-stack weights, split and long-form manifests, MiniMol
weight/license hashes, Endpoint Contract, frozen blind-firewall implementation
and every execution source file into one recipe SHA-256. It fails closed while
the firewall implementation hash is pending or if any component lacks five-fold,
three-seed Stage-B coverage. Certification and blind inference may consume only
that frozen recipe.

Failures, stopped hypotheses and negative transfer remain in the ledger. The
blind leaderboard cannot reopen any architecture, source, weight or loss choice.
