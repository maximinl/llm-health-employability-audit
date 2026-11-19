# Revision analyses

These scripts reproduce every number, table and figure in the revised manuscript.

Run from the repository root:

```bash
pip install -r requirements.txt
bash analysis/revision/run_all.sh
```

Outputs are written to `results/revision/`.

## Scoring

The ten ratings were requested as single-word JSON fields. Models consistently treated `elimination` as agreement that the applicant should be eliminated (mean 1.1 to 1.5, versus 7.6 to 8.5 for `recommendation`). It is therefore reverse-scored (`10 - x`) before the ten items are averaged. The submitted version of the manuscript averaged it unreversed; the revision corrects this.

## Scripts

| Script | Output | Content |
|---|---|---|
| `freq.py` | `scored.csv`, `contrasts.csv` | Composite scoring; Welch contrasts with Welch-t confidence intervals; Bonferroni (30 contrasts); job-blocked sensitivity model; non-inferiority and TOST at ±0.5, ±0.25, ±0.1; JZS Bayes factors (r = 0.707); nine-item and decision composites |
| `icc.py` | `icc.csv` | ICC(1,1) and ICC(1,k) with F-based 95% CIs; within-stimulus SD |
| `bayes2.py` | `posterior.npz` (not committed) | Bayesian cell-means model in PyMC: Normal(5, 2) cell means, Normal(log 0.5, 1) log residual SD by model × prompt; 4 chains × 4,000 iterations (2,000 warm-up); R-hat, ESS, posterior predictive check |
| `derived.py` | `derived.csv` | Prompt differences and 90 job-specific contrasts from the posterior |
| `tables.py` | `tables.json` | Inputs for manuscript Tables 2 and 3 |
| `desc.py` | `table1.csv`, `items.csv` | Table 1, Table 4 (item level), descriptive four-way ANOVA |
| `figs.py` | `figure1.png`, `figure2.png` | Manuscript Figures 1 and 2 |

Random seed for the Bayesian model: 2026. Re-running may change posterior summaries in the third decimal place.
