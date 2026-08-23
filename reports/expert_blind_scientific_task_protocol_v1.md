# Independent blind scientific-task protocol v1

Status: **protocol template only; no independence claim until task authors and
scorers who did not build the agent have frozen and scored the packets**.

Operational tooling is now implemented in
`scripts/seal_expert_blind_tasks.py` and
`scripts/analyze_expert_blind_scores.py`. This changes infrastructure status,
not evidence status: no externally authored hidden packet has yet been received.

## Purpose

Predictive benchmarks do not show whether an agent diagnoses a scientifically
invalid dataset or chooses a defensible next experiment. This cohort tests the
research process independently of TDC and OpenADMET model scores.

## Roles and blinding

- Two task authors with ADMET/cheminformatics experience create cases without
  seeing agent outputs or the scoring of prior cases.
- A custodian hashes the complete hidden packet and releases only the permitted
  discovery view to the agent.
- Two scorers, blinded to system identity and run condition, use a frozen rubric.
- Disagreements are resolved by a third adjudicator or reported with both
  scores. Inter-rater agreement is reported.
- Agent developers may supply the schema and code runner but cannot write the
  hidden fault, gold diagnosis or gold remediation.
- The custodian must run the sealing utility outside the Agent workspace and
  retain the full hidden packet. Only the public discovery packet and its
  commitment hash enter this repository before Agent execution.

## Twelve-case stratified cohort

Authors contribute at least one case from each category:

1. salt/tautomer or stereochemical identity leakage across splits;
2. duplicate/conflicting labels and non-independent technical replicates;
3. unit or relation-operator error in censored IC50/clearance values;
4. mixed species, assay system, probe or biological endpoint;
5. source-held-out distribution shift masked by a random split;
6. class imbalance where accuracy is misleading;
7. activity cliffs/local roughness requiring neighborhood-aware uncertainty;
8. external-data exact or near-neighbor contamination with plausible metadata;
9. apparently useful but license/provenance-inadmissible external source;
10. false model gain caused by repeated validation selection or seed cherry-pick;
11. OOD region where abstention is safer than a forced prediction;
12. clean negative-control case where the correct action is to stop or retain
    the baseline.

Each case contains structures, discovery labels/metadata, explicit permitted
resources, a hidden fault manifest, a gold set of acceptable actions and at
least one tempting but scientifically unsafe action.

## Compared systems

Run EC-VOI v2, legacy agent, fixed rules, expanded AutoML and an expert-written
minimal checklist under the same visible packet and time allowance. AutoML may
return a model but receives a zero diagnosis/action score when it does not
produce the required scientific evidence.

## Frozen rubric (100 points per case)

| Dimension | Points | Full-credit requirement |
|---|---:|---|
| Fault detection | 25 | identifies the causal scientific fault, not only a symptom |
| Evidence localization | 15 | cites the exact rows/groups/metadata supporting it |
| Endpoint-contract correctness | 15 | preserves species, assay, units, censoring and metric semantics |
| Proposed matched experiment | 15 | treatment/control differ on one relevant axis |
| Leakage and test safety | 10 | no prohibited test/leaderboard or contaminated-source use |
| Uncertainty/stop behavior | 10 | abstains or stops when evidence is insufficient |
| Reproducibility/audit | 10 | deterministic packet, hashes, seeds and decision trace |

An unsafe action triggers a preregistered 30-point penalty and is counted
separately; catastrophic test-label access or knowingly contaminated external
training fails the case regardless of raw score.

## Primary analysis

The unit is the case. Compare total rubric score and safe-success rate using
paired case-level bootstrap intervals and exact paired tests. Report inter-rater
agreement, unsafe-action counts, time/cost and abstention. A superiority claim
requires the preregistered lower confidence bound above zero and zero
catastrophic violations.

## One-week operational requirement

Packets and rubric must be hashed by 2026-08-23; agent runs finish by
2026-08-25; blinded scoring and adjudication finish by 2026-08-27. If independent
authors/scorers are unavailable, the experiment is reported as pending and any
internally generated cases are labeled engineering stress tests, not external
blind validation.

## Custodian commands

Before execution, outside the Agent workspace:

```bash
python scripts/seal_expert_blind_tasks.py seal \
  --hidden hidden-case.json \
  --public public-case.json \
  --commitment case-commitment.json
```

After every system output has been submitted and scoring has completed:

```bash
python scripts/seal_expert_blind_tasks.py verify-reveal \
  --hidden hidden-case.json \
  --public public-case.json \
  --commitment case-commitment.json
```

The score table is then analyzed with system identities still represented by
anonymous codes:

```bash
python scripts/analyze_expert_blind_scores.py \
  --scores blinded-scores.json \
  --primary sys-x \
  --output expert-blind-analysis.json
```

The analysis treats the case—not scorer or seed—as the paired scientific unit,
reports score ICC and unsafe-action agreement, applies the frozen unsafe and
catastrophic penalties, and evaluates paired case-level bootstrap and exact sign
contrasts.
