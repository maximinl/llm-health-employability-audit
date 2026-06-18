import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


MODEL_LABELS = {
    "claude-opus-4-8": "Claude Opus 4.8",
    "gemini-3.1-flash-lite": "Gemini 3.1 Flash-Lite",
    "gpt-5.5": "GPT-5.5",
}

DISCLOSURE_LABELS = {
    "control": "Control",
    "anxiety_disorder": "Anxiety",
    "depression": "Depression",
    "eating_disorder": "Eating disorder",
    "hyperthyroidism": "Hyperthyroidism",
    "schizophrenia": "Schizophrenia",
}

PROMPT_LABELS = {
    "fairness_aware": "Fairness-aware",
    "realistic_recruiter": "Realistic recruiter",
}

DISCLOSURE_ORDER = [
    "control",
    "anxiety_disorder",
    "depression",
    "eating_disorder",
    "schizophrenia",
    "hyperthyroidism",
]


def savefig(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["model_label"] = df["model"].map(MODEL_LABELS)
    df["disclosure_label"] = df["disclosure"].map(DISCLOSURE_LABELS)
    df["prompt_label"] = df["prompt_condition"].map(PROMPT_LABELS)
    df["disclosure_label"] = pd.Categorical(
        df["disclosure_label"],
        categories=[DISCLOSURE_LABELS[item] for item in DISCLOSURE_ORDER],
        ordered=True,
    )
    return df


def plot_mean_by_disclosure(df: pd.DataFrame, outdir: Path) -> None:
    plt.figure(figsize=(11, 6))
    sns.pointplot(
        data=df,
        x="disclosure_label",
        y="mean_employability",
        hue="model_label",
        errorbar=("ci", 95),
        dodge=0.35,
        markers="o",
        linestyles="none",
    )
    plt.axhline(df["mean_employability"].mean(), color="0.85", linewidth=1, zorder=0)
    plt.xlabel("Disclosure condition")
    plt.ylabel("Mean employability rating")
    plt.title("Mean Employability Ratings by Disclosure Condition and Model")
    plt.xticks(rotation=25, ha="right")
    plt.legend(title="Model", frameon=False)
    savefig(outdir / "figure_1_mean_employability_by_disclosure.png")


def plot_primary_contrasts(contrasts: pd.DataFrame, outdir: Path) -> None:
    contrasts = contrasts.copy()
    contrasts["model_label"] = contrasts["model"].map(MODEL_LABELS)
    contrasts["prompt_label"] = contrasts["prompt_condition"].map(PROMPT_LABELS)
    contrasts["disclosure_label"] = contrasts["disclosure"].map(DISCLOSURE_LABELS)
    contrasts["disclosure_label"] = pd.Categorical(
        contrasts["disclosure_label"],
        categories=[DISCLOSURE_LABELS[item] for item in DISCLOSURE_ORDER if item != "control"],
        ordered=True,
    )

    grid = sns.FacetGrid(
        contrasts,
        row="prompt_label",
        col="model_label",
        sharex=True,
        sharey=True,
        height=3.3,
        aspect=1.15,
        margin_titles=True,
    )

    def draw(data, **kwargs):
        data = data.sort_values("disclosure_label")
        y = range(len(data))
        plt.axvline(0, color="0.45", linewidth=1)
        plt.errorbar(
            data["difference_vs_control"],
            y,
            xerr=[
                data["difference_vs_control"] - data["ci95_low"],
                data["ci95_high"] - data["difference_vs_control"],
            ],
            fmt="o",
            color="#2f5d8c",
            ecolor="#2f5d8c",
            capsize=3,
        )
        plt.yticks(y, data["disclosure_label"])

    grid.map_dataframe(draw)
    grid.set_axis_labels("Difference vs control", "Disclosure")
    grid.set_titles(row_template="{row_name}", col_template="{col_name}")
    grid.fig.suptitle("Disclosure Effects on Employability Ratings vs Control", y=1.03)
    savefig(outdir / "figure_2_primary_contrasts_vs_control.png")


def plot_job_type_heatmap(df: pd.DataFrame, outdir: Path) -> None:
    summary = (
        df.groupby(["model_label", "job_type", "disclosure"], observed=True)["mean_employability"]
        .mean()
        .reset_index()
    )
    controls = summary.loc[summary["disclosure"] == "control", ["model_label", "job_type", "mean_employability"]]
    controls = controls.rename(columns={"mean_employability": "control_mean"})
    summary = summary.merge(controls, on=["model_label", "job_type"], how="left")
    summary["delta_vs_control"] = summary["mean_employability"] - summary["control_mean"]
    summary = summary[summary["disclosure"] != "control"]
    summary["disclosure_label"] = summary["disclosure"].map(DISCLOSURE_LABELS)

    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.5), sharey=True)
    for ax, model_label in zip(axes, MODEL_LABELS.values()):
        subset = summary[summary["model_label"] == model_label]
        pivot = subset.pivot(index="disclosure_label", columns="job_type", values="delta_vs_control")
        pivot = pivot.reindex([DISCLOSURE_LABELS[item] for item in DISCLOSURE_ORDER if item != "control"])
        sns.heatmap(
            pivot,
            ax=ax,
            center=0,
            cmap="vlag",
            annot=True,
            fmt=".2f",
            cbar=model_label == list(MODEL_LABELS.values())[-1],
            linewidths=0.5,
            linecolor="white",
        )
        ax.set_title(model_label)
        ax.set_xlabel("Job type")
        ax.set_ylabel("")
    axes[0].set_ylabel("Disclosure condition")
    fig.suptitle("Mean Employability Difference vs Control by Job Type", y=1.02)
    savefig(outdir / "figure_3_job_type_delta_heatmap.png")


def plot_prompt_comparison(df: pd.DataFrame, outdir: Path) -> None:
    summary = (
        df.groupby(["model_label", "prompt_label", "disclosure"], observed=True)["mean_employability"]
        .mean()
        .reset_index()
    )
    controls = summary.loc[summary["disclosure"] == "control", ["model_label", "prompt_label", "mean_employability"]]
    controls = controls.rename(columns={"mean_employability": "control_mean"})
    summary = summary.merge(controls, on=["model_label", "prompt_label"], how="left")
    summary = summary[summary["disclosure"] != "control"]
    summary["delta_vs_control"] = summary["mean_employability"] - summary["control_mean"]
    summary["disclosure_label"] = summary["disclosure"].map(DISCLOSURE_LABELS)

    plt.figure(figsize=(11, 6))
    sns.barplot(
        data=summary,
        x="disclosure_label",
        y="delta_vs_control",
        hue="prompt_label",
        errorbar=None,
        palette="Set2",
    )
    plt.axhline(0, color="0.35", linewidth=1)
    plt.xlabel("Disclosure condition")
    plt.ylabel("Mean difference vs control")
    plt.title("Prompt Framing Differences Averaged Across Models and Jobs")
    plt.xticks(rotation=25, ha="right")
    plt.legend(title="Prompt", frameon=False)
    savefig(outdir / "figure_4_prompt_framing_deltas.png")


def run(args: argparse.Namespace) -> None:
    df = prepare(pd.read_csv(args.input))
    contrasts = pd.read_csv(args.contrasts)
    outdir = Path(args.outdir)

    sns.set_theme(style="whitegrid", context="paper", font_scale=1.15)
    plot_mean_by_disclosure(df, outdir)
    plot_primary_contrasts(contrasts, outdir)
    plot_job_type_heatmap(df, outdir)
    plot_prompt_comparison(df, outdir)
    print(f"Wrote figures to {outdir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create manuscript figures for the LLM audit.")
    parser.add_argument("--input", default="data/processed/ratings.csv")
    parser.add_argument("--contrasts", default="results/tables/primary_contrasts_vs_control_pooled_jobs.csv")
    parser.add_argument("--outdir", default="results/figures")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
