# ADMET Endpoint Ontology and Contract

## Contents

1. Contract rules
2. EndpointContract schema
3. ADMET endpoint distinctions
4. Compatibility grading
5. Search and exclusion terms

## 1. Contract rules

Treat an endpoint as a tuple, not a name:

`property + biological entity + species + matrix/tissue + assay system + readout + time + unit/scale + threshold + direction + censoring + provenance`

Two datasets are directly compatible only when differences in this tuple are absent or scientifically convertible without outcome-dependent tuning. Preserve the original tuple after conversion.

Never conflate substrate, inhibitor, inducer and victim/perpetrator roles; binding, functional inhibition, trafficking effects and clinical outcomes; intrinsic clearance, apparent clearance, extraction ratio, half-life and total clearance; thermodynamic, kinetic, intrinsic and formulation solubility; permeability, efflux, absorption and bioavailability; total and unbound concentration; acute cytotoxicity, organ toxicity, clinical DILI and animal lethality; or experimental labels and predictions.

## 2. EndpointContract schema

```yaml
endpoint_id: cyp3a4_reversible_inhibition_ic50_human_microsomes
property: CYP3A4 reversible inhibition
admet_domain: metabolism
task_type: regression
biological_entity: CYP3A4
species: human
matrix_or_tissue: human liver microsomes
assay_system: probe-substrate enzyme inhibition
readout: IC50
timepoint: null
preincubation:
  present: false
  nadph: null
measurement:
  original_unit: micromolar
  canonical_unit: micromolar
  scale: linear
  direction: lower_is_stronger_inhibition
label_definition:
  type: continuous
  threshold: null
censoring_allowed: true
intended_role: direct_supervised_label
split_contract:
  discovery: scaffold_multiseed
  certification: evaluator_owned_holdout
  future_audit: new_source_or_temporal_set
explicit_exclusions:
  - substrate status
  - time-dependent inhibition without a reversible IC50 arm
  - predicted values
```

Required fields are `endpoint_id`, `property`, `admet_domain`, `task_type`, `species`, `assay_system`, `readout`, `measurement`, `label_definition`, `intended_role`, and `split_contract`. Use `unknown`, not a guessed value, when the source is incomplete.

## 3. ADMET endpoint distinctions

### Absorption

| Family | Required distinctions | Common non-equivalences |
|---|---|---|
| Caco-2/MDCK permeability | cell line/clone, direction, Papp unit, pH, time, transporter conditions | A-to-B Papp is not efflux ratio; MDCK is not Caco-2 |
| PAMPA | membrane composition, pH, donor/acceptor conditions | passive permeability is not cellular absorption |
| P-gp/BCRP transport | substrate vs inhibitor, cell line, direction, efflux ratio or IC50 | substrate and inhibitor labels are different tasks |
| HIA | human/animal, experimental vs curated clinical label, threshold | HIA is not oral bioavailability |
| Oral bioavailability | species, dose/route, formulation, AUC definition | F% is dose- and formulation-dependent |
| Solubility | kinetic/thermodynamic/intrinsic, pH, temperature, medium, unit, log transform | aggregations may mix protocols; logS units vary |

### Distribution

| Family | Required distinctions | Common non-equivalences |
|---|---|---|
| Plasma protein binding | species, plasma/serum, total vs unbound, fraction vs percent | `fu=0.1` equals 90% bound, not 10% bound |
| Volume of distribution | species, steady-state vs apparent Vd, normalization unit | Vdss and Vz are not interchangeable |
| BBB penetration | Kp, Kp,uu, logBB, binary CNS label, species | total brain/plasma ratio is not unbound ratio |
| Tissue distribution | tissue, time, dose, total/unbound | different tissues are separate endpoints |

### Metabolism

| Family | Required distinctions | Common non-equivalences |
|---|---|---|
| CYP inhibition | isoform, reversible IC50/Ki, probe substrate, matrix, species | inhibition is not substrate status |
| Time-dependent inhibition | preincubation, NADPH, IC50 shift, kinact/KI, dilution protocol | TDI positive is not automatically irreversible MBI |
| CYP substrate | isoform, evidence type, fraction metabolized or categorical definition | metabolism does not prove major substrate status |
| CYP induction | mRNA/activity, fold change, concentration, hepatocyte donor, receptor | induction is not inhibition |
| Metabolic stability | microsome/hepatocyte/S9, species, CLint vs half-life, protein/cell density | microsomal and hepatocyte clearance differ |
| Clearance | intrinsic/apparent/in vivo, matrix, species, unit, binding correction | CLint, hepatic and total clearance differ |
| Half-life | in vitro/in vivo, species, route, terminal/effective half-life | half-life is not clearance without assumptions |

### Excretion

Require renal, biliary, fecal or total route; species; intact drug vs metabolite; clearance or fraction excreted; dose/route; and collection window. Do not merge renal clearance with fraction excreted unchanged.

### Toxicity

| Family | Required distinctions | Common non-equivalences |
|---|---|---|
| hERG | binding vs patch clamp/current inhibition, cell system, temperature, IC50/percent | binding affinity is not functional block |
| Cardiotoxicity | QT prolongation, arrhythmia, hERG, multichannel or clinical label | hERG is only one mechanism |
| Ames | strain, S9 activation, protocol, categorical call | strain results and compound consensus differ |
| DILI | clinical causality source, severity, exposure, label scheme | in-vitro hepatotoxicity is not clinical DILI |
| Cytotoxicity | cell line, exposure time, viability readout, CC50/IC50 | results across cell lines and times differ |
| Acute toxicity | species, route, duration, LD50 unit | oral rat LD50 is not general human toxicity |
| Carcinogenicity | species, duration, organ/site, evidence category | genotoxicity is not carcinogenicity |
| Developmental/reproductive | species, life stage, endpoint, dose | adverse outcomes require separate labels |

## 4. Compatibility grading

- **Grade A, direct:** defining fields align; laboratory differences are recorded and may be modeled as source effects.
- **Grade B, convertible/restricted:** a prespecified physical conversion or subgroup restriction resolves the difference. Preserve original values and transformations.
- **Grade C, related:** biologically relevant but not label-equivalent. Use for pretraining, multitask learning, priors or hypothesis generation.
- **Grade D, incompatible:** predicted-only, unclear meaning, unverifiable provenance, label reversal, incompatible assay/species, benchmark reaggregation or held-out contamination.

## 5. Search and exclusion terms

Build each query from:

`entity/property synonyms AND assay/readout synonyms AND species/matrix AND (dataset OR supplementary OR table OR experimental)`

Add negative terms when ambiguity is likely. For CYP inhibition exclude substrate and induction unless separately collected. For TDI require preincubation, IC50 shift, kinact or KI. For clearance specify matrix and species. For hERG specify patch clamp or binding. Record which contract fields a query covers and what still requires full text or SI.
