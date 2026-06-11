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
analysis/              Reproducible analysis scripts
data/raw/              Raw model responses
data/processed/        Analysis-ready datasets
manuscript/            Manuscript draft materials
prompts/               Prompt templates and disclosure text
results/figures/       Generated figures
results/tables/        Generated result tables
```
