#!/usr/bin/env bash
# Revision analyses (corrected scoring: elimination reverse-scored).
# Run from the repository root:  bash analysis/revision/run_all.sh
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
OUT="$ROOT/results/revision"
mkdir -p "$OUT"
cd "$OUT"
python "$ROOT/analysis/revision/freq.py" "$ROOT/data/processed/ratings.csv"   # scoring, Welch contrasts, blocked model, NI/TOST, JZS BFs
python "$ROOT/analysis/revision/icc.py"      # ICC(1,1), ICC(1,k) with CIs, within-stimulus SD
python "$ROOT/analysis/revision/bayes2.py"   # Bayesian cell-means model (PyMC), diagnostics, posterior predictive check
python "$ROOT/analysis/revision/derived.py"  # prompt differences and job-specific contrasts
python "$ROOT/analysis/revision/tables.py"   # Table 2 and Table 3 inputs (tables.json)
python "$ROOT/analysis/revision/desc.py"     # Table 1, item-level Table 4, descriptive ANOVA
python "$ROOT/analysis/revision/figs.py"     # Figures 1 and 2
