# GPT-5.6 integration contract

V2 supports `OpenAIPlanner` through the Responses API and strict Pydantic structured outputs.
The configured deployment is `gpt-5.6-sol`.

GPT-5.6 may:

1. read a train/validation-only `TaskProfile`;
2. read completed `RunResult` records from earlier rounds;
3. read aggregate cross-task recipes from `experiments.sqlite`;
4. inspect the registered representation/model catalog;
5. return one to three schema-valid `ExperimentSpec` candidates;
6. select one available finalist by exact `run_id`.

GPT-5.6 may not:

- receive test labels or per-example test performance;
- change the TDC split, evaluator or metric;
- execute arbitrary code or shell commands;
- invent unregistered tools or hyperparameters;
- expand its compute budget;
- replace a finalist after the test freeze lock is written.

Every proposal is validated by `validate_experiment` before compute. Invalid proposals are
rejected and recorded without weakening the protocol.

The default credential file is outside the repository:

```text
~/.config/admet-agent/openai.env
```

```dotenv
OPENAI_API_KEY=...
# Optional for an OpenAI-compatible endpoint:
# OPENAI_BASE_URL=https://example.com/v1
```

The key is read at runtime and never copied into runs. `api_usage.jsonl` stores only response IDs,
model names and token counts. When private molecules are introduced, aggregate task profiles should
be sent by default rather than raw structures.
