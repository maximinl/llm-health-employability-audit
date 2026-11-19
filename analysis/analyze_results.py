import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats
from statsmodels.stats.anova import anova_lm


RATING_FIELDS = [
    "communication",
    "harmony",
    "ambition",
    "competence",
    "responsibility",
    "dependability",
    "resilience",
    "independence",
    "recommendation",
    "elimination",
]


# "elimination" is negatively keyed and is reverse-scored (10 - x) before averaging.
REVERSE_SCORED_FIELDS = ["elimination"]


def add_employability_score(df: pd.DataFrame) -> pd.DataFrame:
    scored = df[RATING_FIELDS].copy()
    for field in REVERSE_SCORED_FIELDS:
        scored[field] = 10 - scored[field]
    df = df.copy()
    df["mean_employability"] = scored.mean(axis=1)
    return df


def welch_ci(values: pd.Series, control: pd.Series, level: float = 0.95) -> tuple[float, float, float, float]:
    """Difference, SE and CI from the Welch t distribution (consistent with the Welch p-value)."""
    va = values.var(ddof=1) / len(values)
    vb = control.var(ddof=1) / len(control)
    se = np.sqrt(va + vb)
    df_w = (va + vb) ** 2 / (va**2 / (len(values) - 1) + vb**2 / (len(control) - 1)) if se > 0 else np.inf
    diff = values.mean() - control.mean()
    tcrit = stats.t.ppf(0.5 + level / 2, df_w)
    return diff, se, diff - tcrit * se, diff + tcrit * se


def partial_eta_squared(anova_table: pd.DataFrame) -> pd.Series:
    residual_ss = anova_table.loc["Residual", "sum_sq"]
    eta_sq = anova_table["sum_sq"] / (anova_table["sum_sq"] + residual_ss)
    eta_sq.loc["Residual"] = np.nan
    return eta_sq


def format_anova(table: pd.DataFrame) -> pd.DataFrame:
    output = table.reset_index().rename(columns={"index": "effect"})
    output["partial_eta_sq"] = partial_eta_squared(table).reset_index(drop=True)
    return output


def icc_1k(values: pd.DataFrame) -> float:
    # One-way random-effects ICC for mean ratings across k replications.
    n, k = values.shape
    row_means = values.mean(axis=1)
    grand_mean = values.to_numpy().mean()
    ss_between = k * ((row_means - grand_mean) ** 2).sum()
    ss_within = ((values.sub(row_means, axis=0)) ** 2).to_numpy().sum()
    ms_between = ss_between / (n - 1)
    ms_within = ss_within / (n * (k - 1))
    return (ms_between - ms_within) / ms_between


def write_dataset_summary(df: pd.DataFrame, outdir: Path) -> None:
    summary = pd.DataFrame(
        {
            "metric": [
                "rows",
                "models",
                "prompt_conditions",
                "job_types",
                "disclosure_conditions",
                "replications",
                "min_rating",
                "max_rating",
            ],
            "value": [
                len(df),
                df["model"].nunique(),
                df["prompt_condition"].nunique(),
                df["job_type"].nunique(),
                df["disclosure"].nunique(),
                df["replication"].nunique(),
                int(df[RATING_FIELDS].min().min()),
                int(df[RATING_FIELDS].max().max()),
            ],
        }
    )
    summary.to_csv(outdir / "dataset_summary.csv", index=False)


def write_descriptives(df: pd.DataFrame, outdir: Path) -> None:
    group_cols = ["provider", "model", "prompt_condition", "job_type", "disclosure"]
    descriptive = (
        df.groupby(group_cols, observed=True)["mean_employability"]
        .agg(n="count", mean="mean", sd="std", se=lambda x: x.std(ddof=1) / np.sqrt(len(x)))
        .reset_index()
    )
    tcrit = stats.t.ppf(0.975, descriptive["n"] - 1)
    descriptive["ci95_low"] = descriptive["mean"] - tcrit * descriptive["se"]
    descriptive["ci95_high"] = descriptive["mean"] + tcrit * descriptive["se"]
    descriptive.to_csv(outdir / "descriptives_by_condition.csv", index=False)

    item_descriptive = (
        df.groupby(["provider", "model", "prompt_condition", "disclosure"], observed=True)[RATING_FIELDS]
        .mean()
        .reset_index()
    )
    item_descriptive.to_csv(outdir / "item_means_by_disclosure.csv", index=False)


def write_control_contrasts(df: pd.DataFrame, outdir: Path) -> None:
    rows = []
    for keys, group in df.groupby(["provider", "model", "prompt_condition", "job_type"], observed=True):
        control = group.loc[group["disclosure"] == "control", "mean_employability"]
        for disclosure, disclosure_group in group.groupby("disclosure", observed=True):
            if disclosure == "control":
                continue
            values = disclosure_group["mean_employability"]
            diff = values.mean() - control.mean()
            test = stats.ttest_ind(values, control, equal_var=False)
            rows.append(
                {
                    "provider": keys[0],
                    "model": keys[1],
                    "prompt_condition": keys[2],
                    "job_type": keys[3],
                    "disclosure": disclosure,
                    "n_disclosure": len(values),
                    "n_control": len(control),
                    "mean_disclosure": values.mean(),
                    "mean_control": control.mean(),
                    "difference_vs_control": diff,
                    "t": test.statistic,
                    "p": test.pvalue,
                }
            )

    contrasts = pd.DataFrame(rows)
    contrasts["p_bonferroni"] = np.minimum(contrasts["p"] * len(contrasts), 1.0)
    contrasts.to_csv(outdir / "contrasts_vs_control.csv", index=False)

    pooled_rows = []
    for keys, group in df.groupby(["provider", "model", "prompt_condition"], observed=True):
        control = group.loc[group["disclosure"] == "control", "mean_employability"]
        for disclosure, disclosure_group in group.groupby("disclosure", observed=True):
            if disclosure == "control":
                continue
            values = disclosure_group["mean_employability"]
            diff = values.mean() - control.mean()
            diff, se, ci_low, ci_high = welch_ci(values, control)
            test = stats.ttest_ind(values, control, equal_var=False)
            pooled_rows.append(
                {
                    "provider": keys[0],
                    "model": keys[1],
                    "prompt_condition": keys[2],
                    "disclosure": disclosure,
                    "n_disclosure": len(values),
                    "n_control": len(control),
                    "mean_disclosure": values.mean(),
                    "mean_control": control.mean(),
                    "difference_vs_control": diff,
                    "se": se,
                    "ci95_low": ci_low,
                    "ci95_high": ci_high,
                    "t": test.statistic,
                    "p": test.pvalue,
                }
            )

    pooled = pd.DataFrame(pooled_rows)
    pooled["p_bonferroni"] = np.minimum(pooled["p"] * len(pooled), 1.0)
    pooled.to_csv(outdir / "primary_contrasts_vs_control_pooled_jobs.csv", index=False)


def write_reliability(df: pd.DataFrame, outdir: Path) -> None:
    rows = []
    for keys, group in df.groupby(["provider", "model", "prompt_condition"], observed=True):
        pivot = group.pivot_table(
            index=["job_type", "disclosure"],
            columns="replication",
            values="mean_employability",
            aggfunc="mean",
        )
        rows.append(
            {
                "provider": keys[0],
                "model": keys[1],
                "prompt_condition": keys[2],
                "n_stimuli": pivot.shape[0],
                "k_replications": pivot.shape[1],
                "icc_1k": icc_1k(pivot),
            }
        )

    pd.DataFrame(rows).to_csv(outdir / "reliability_icc.csv", index=False)


def write_anovas(df: pd.DataFrame, outdir: Path) -> None:
    global_model = smf.ols(
        "mean_employability ~ C(model) * C(prompt_condition) * C(job_type) * C(disclosure)",
        data=df,
    ).fit()
    global_anova = format_anova(anova_lm(global_model, typ=2))
    global_anova.to_csv(outdir / "anova_global_type2.csv", index=False)

    rows = []
    for model_name, group in df.groupby("model", observed=True):
        fitted = smf.ols(
            "mean_employability ~ C(prompt_condition) * C(job_type) * C(disclosure)",
            data=group,
        ).fit()
        table = format_anova(anova_lm(fitted, typ=2))
        table.insert(0, "model", model_name)
        rows.append(table)

    pd.concat(rows, ignore_index=True).to_csv(outdir / "anova_by_model_type2.csv", index=False)


def write_model_coefficients(df: pd.DataFrame, outdir: Path) -> None:
    model = smf.ols(
        "mean_employability ~ C(model) * C(prompt_condition) * C(job_type) * C(disclosure)",
        data=df,
    ).fit()
    coefficients = pd.DataFrame(
        {
            "term": model.params.index,
            "estimate": model.params.values,
            "se": model.bse.values,
            "z_or_t": model.tvalues.values,
            "p": model.pvalues.values,
            "ci95_low": model.conf_int()[0].values,
            "ci95_high": model.conf_int()[1].values,
        }
    )
    coefficients.to_csv(outdir / "ols_global_coefficients_hc3.csv", index=False)


def run(args: argparse.Namespace) -> None:
    warnings.filterwarnings("ignore", message="Precision loss occurred*")
    warnings.filterwarnings("ignore", category=RuntimeWarning, module="numpy.linalg")
    input_path = Path(args.input)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    df = add_employability_score(pd.read_csv(input_path))
    if len(df) != 540:
        raise ValueError(f"Expected 540 processed rows, found {len(df)}")
    if df[RATING_FIELDS].min().min() < 1 or df[RATING_FIELDS].max().max() > 9:
        raise ValueError("Ratings are outside the expected 1-9 range.")

    write_dataset_summary(df, outdir)
    write_descriptives(df, outdir)
    write_control_contrasts(df, outdir)
    write_reliability(df, outdir)
    write_anovas(df, outdir)

    print(f"Wrote analysis tables to {outdir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze completed LLM employability ratings.")
    parser.add_argument("--input", default="data/processed/ratings.csv")
    parser.add_argument("--outdir", default="results/tables")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
