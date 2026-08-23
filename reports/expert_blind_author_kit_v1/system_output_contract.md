# Anonymous system output contract

Every compared system receives the same public packet, resource permissions,
maximum action budget and wall-clock limit. It returns one JSON-compatible
record with these fields:

| Field | Required content |
|---|---|
| `case_id` | Exact public case ID |
| `diagnosis` | Causal fault or explicit clean/insufficient-evidence conclusion |
| `evidence_ids` | Exact visible rows, groups or metadata supporting the diagnosis |
| `endpoint_contract_interpretation` | Species, assay, units, censoring and metric semantics |
| `proposed_action` | One authorized data, feature, model, replicate or stop action |
| `matched_control` | Control differing on only the declared action axis |
| `falsification_criterion` | Outcome that would refute the hypothesis |
| `uncertainty_and_stop` | Confidence, missing evidence and stop/abstain condition |
| `resources_accessed` | Files, tools, web sources and model calls actually used |
| `audit` | Seed, configuration, elapsed time, cost and artifact hashes |

Systems cannot receive rubric scores or gold feedback between cases. A model
fit without a diagnosis, localized evidence and next-action decision is not a
complete scientific-task response.
