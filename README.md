# LLM Health Disclosure Employability Audit

This repository is a clean workspace for rerunning an audit of how large language models evaluate standardized job applications that differ only in disclosed health history.

The project is set up to keep prompts, raw model outputs, processed analysis data, scripts, figures, and manuscript materials separate. Result folders are intentionally empty until the real model runs and analyses are completed.

## What This Repository Does

- Stores prompt templates and application stimuli used for the LLM audit.
- Keeps raw API/model outputs separate from processed analysis-ready data.
- Provides a place for analysis scripts that generate tables and figures.
- Keeps final outputs in `results/` without committing placeholder result files.

## Structure

```text
config/                Experiment configuration
analysis/              Reproducible analysis scripts
data/stimuli/          Committed job and disclosure stimuli
data/raw/              Raw model responses
data/processed/        Analysis-ready datasets
manuscript/            Manuscript draft materials
prompts/               Prompt templates and disclosure text
results/figures/       Generated figures
results/tables/        Generated result tables
```

## Running Experiments

Install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Set whichever provider API keys are being used:

```bash
export OPENAI_API_KEY="..."
export ANTHROPIC_API_KEY="..."
export GOOGLE_API_KEY="..."
```

Preview the full task list without calling APIs:

```bash
python analysis/run_experiment.py --dry-run
```

Run a small pilot:

```bash
python analysis/run_experiment.py --providers openai --limit 3
```

Run the configured experiment:

```bash
python analysis/run_experiment.py
```

Run only selected providers:

```bash
python analysis/run_experiment.py --providers openai,anthropic
```

Raw model responses are written to `data/raw/model_responses.jsonl`. The flattened ratings dataset is written to `data/processed/ratings.csv`. These outputs are ignored by git by default.
