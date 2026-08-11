# ADMET-EvO

ADMET-EvO is an evolutionary research agent for protocol-safe model discovery on the 22-task [Therapeutics Data Commons (TDC) ADMET benchmark group](https://tdcommons.ai/benchmark/admet_group/overview/).

The agent profiles each endpoint, proposes bounded experiments, evaluates candidates only on training/validation splits, promotes robust finalists across validation seeds, searches simple prediction ensembles, and freezes the selected recipe before the TDC test split is evaluated. The language-model planner is optional; model training, evaluation, protocol enforcement, and finalist freezing are deterministic Python components.

## Initial TDC submission

The first public submission covers the three endpoints for which ADMET-EvO v0.2.1 is estimated to place first against the TDC leaderboards captured on 10 August 2026.

| Dataset | Metric | ADMET-EvO | Runs | Estimated rank |
|---|---|---:|---:|---:|
| AMES | AUROC, higher is better | 0.874 +/- 0.001 | 5 | 1 |
| PPBR_AZ | MAE, lower is better | 7.302 +/- 0.042 | 5 | 1 |
| Solubility_AqSolDB | MAE, lower is better | 0.739 +/- 0.002 | 5 | 1 |

These are test-set results from five independent TDC runs using seeds 1, 2, 3, 4, and 5. The finalist configuration was fixed before test evaluation. The full non-peer-reviewed technical report is available in [`MANUSCRIPT.md`](MANUSCRIPT.md). Machine-readable scores and finalist recipes are in [`results/`](results/), and the submission-ready description is in [`docs/tdc_submission.md`](docs/tdc_submission.md).

Leaderboard positions are estimates until the TDC maintainers accept and publish the submission.

## Method

ADMET-EvO uses a bounded model and molecular-representation registry:

- molecular fingerprints and descriptors, including Morgan, Avalon, ErG, MACCS, atom-pair, torsion, and RDKit 2D features;
- classical and gradient-boosted learners, including Extra Trees, Random Forest, SVM, XGBoost, LightGBM, and CatBoost;
- validation-only sequential planning, robust multi-seed finalist selection, and prediction-level ensembling;
- a test firewall that prevents test access during search and records a SHA-256 freeze lock before final evaluation;
- SQLite/JSONL experiment memory for reproducible provenance and cross-task recipe reuse.

See [`docs/v2_architecture.md`](docs/v2_architecture.md) for the protocol and component boundaries.

## Exact score reproduction

The reported scores do not require an OpenAI API key or any other external API. Install the exact finalization environment and run the three frozen finalists:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-repro.txt
python -m pip install -e . --no-deps
DEVICE=cuda WORKERS=8 ./scripts/reproduce_top1.sh
```

This runs seeds 1-5 separately for each endpoint, saves per-seed predictions, calls TDC's official `evaluate_many`, and verifies the submitted scores and task-specific SHA-256 freeze hashes. A CUDA-capable GPU is required to match the submitted AMES XGBoost configuration; `DEVICE=cpu` is available for portability.

## Development installation

Python 3.10-3.12 is supported.

```bash
conda env create -f environment-cpu.yml
conda activate admet-agent
pip install -e '.[ranking,dev]'
```

For GPU-capable XGBoost outside the exact reproduction environment, use `environment-gpu.yml` and install the `gpu` extra as appropriate for the CUDA environment.

## Usage

List the canonical TDC endpoints and registered tools:

```bash
admet-agent list
admet-agent tools
```

Run validation-only search:

```bash
admet-agent search \
  --task AMES \
  --rounds 2 \
  --validation-seeds 1,2,3,4,5 \
  --robust-top-k 3 \
  --run-dir runs/ames-search
```

Evaluate a frozen finalist under the five-run TDC protocol:

```bash
admet-agent finalize \
  --task AMES \
  --seeds 1,2,3,4,5 \
  --finalists results/top1_finalists.json \
  --run-dir runs/ames-final
```

The TDC data are downloaded through PyTDC and are not included in this repository.

## Reproducibility and test policy

Search code can access only TDC `train_val` data and validation metrics. Finalization requires at least five unique seeds and a serialized finalist mapping. The finalist mapping is hashed before test evaluation, and the same hash is recorded for every final run. Tests cover task metadata, experiment contracts, test-access guards, finalist freezing, robust selection, and ensemble behavior.

The optional language-model planner only proposes bounded validation experiments. It is not used during frozen finalist training or test evaluation, it never receives test labels or test metrics, and it is not required to reproduce the submitted results. API credentials are intentionally excluded from the repository; users who repeat LLM-guided search must provide their own credentials. API-free rule-based search is also supported.

Run the checks with:

```bash
python -m pytest -q
```

## Contact

- Yilin Wang
- Valhalla Technology

## License

ADMET-EvO is released under the [MIT License](LICENSE).
