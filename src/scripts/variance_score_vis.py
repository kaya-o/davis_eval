from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = PROJECT_ROOT / "data" / "davis_other_data_models.csv"
VIS_DIR = PROJECT_ROOT / "data" / "vis"
DEFAULT_PLOT_PATH = VIS_DIR / "variance_scores.png"
DEFAULT_SORTED_PLOT_PATH = VIS_DIR / "variance_scores_sorted.png"
DEFAULT_SORTED_SCORE_RESIDUAL_PLOT_PATH = VIS_DIR / "variance_sorted_scores_and_residuals.png"
THRESHOLD_LINES = (0.3, 0.15, 0.1)


def variance_score(muhat_1, muhat_2, muhat_3):
    muhat_1 = np.asarray(muhat_1)
    muhat_2 = np.asarray(muhat_2)
    muhat_3 = np.asarray(muhat_3)

    if not (muhat_1.shape == muhat_2.shape == muhat_3.shape):
        raise ValueError("muhat_1, muhat_2, and muhat_3 must have the same shape.")

    return np.var(np.stack([muhat_1, muhat_2, muhat_3], axis=0), axis=0)


def load_variance_scores(data_path=DATA_PATH):
    data_df = pd.read_csv(data_path)
    return variance_score(
        data_df["muhat_1"].to_numpy(),
        data_df["muhat_2"].to_numpy(),
        data_df["muhat_3"].to_numpy(),
    )


def load_variance_scores_and_residuals(data_path=DATA_PATH):
    data_df = pd.read_csv(data_path)
    scores = variance_score(
        data_df["muhat_1"].to_numpy(),
        data_df["muhat_2"].to_numpy(),
        data_df["muhat_3"].to_numpy(),
    )
    residuals = np.abs(
        data_df["Label"].to_numpy(dtype=float)
        - data_df["muhat_1"].to_numpy(dtype=float)
    )
    return scores, residuals


def add_threshold_lines(ax, thresholds=THRESHOLD_LINES):
    colors = ("tab:green", "tab:purple", "tab:orange")
    for threshold, color in zip(thresholds, colors):
        ax.axhline(
            threshold,
            color=color,
            linestyle=":",
            linewidth=2,
            label=f"Threshold = {threshold:g}",
        )


def format_stats_text(scores, thresholds=THRESHOLD_LINES):
    mean_score = np.mean(scores)
    median_score = np.median(scores)
    std_score = np.std(scores)
    variance_of_scores = np.var(scores)
    threshold_text = "    ".join(
        f">= {threshold:g}: {np.sum(scores >= threshold)} ({np.mean(scores >= threshold):.2%})"
        for threshold in thresholds
    )

    return (
        f"Mean: {mean_score:.6f}    "
        f"Median: {median_score:.6f}    "
        f"Std. dev.: {std_score:.6f}    "
        f"Variance: {variance_of_scores:.6f}\n"
        f"{threshold_text}"
    )


def plot_variance_scores(
    scores,
    save_path=DEFAULT_PLOT_PATH,
    show=True,
    point_alpha=0.35,
    thresholds=THRESHOLD_LINES,
):
    scores = np.asarray(scores)
    mean_score = np.mean(scores)
    median_score = np.median(scores)

    fig, ax = plt.subplots(figsize=(14, 7))
    ax.scatter(
        np.arange(scores.size),
        scores,
        alpha=point_alpha,
        s=12,
        edgecolors="none",
    )
    ax.axhline(mean_score, color="tab:red", linewidth=2, label="Mean")
    ax.axhline(median_score, color="black", linestyle="--", linewidth=1.5, label="Median")
    add_threshold_lines(ax, thresholds)

    ax.set_title("Variance Scores Across DAVIS Datapoints")
    ax.set_xlabel("Datapoint index")
    ax.set_ylabel("Variance score")
    ax.grid(alpha=0.25)
    ax.legend()

    stats_text = format_stats_text(scores, thresholds)
    fig.text(0.5, 0.02, stats_text, ha="center", va="bottom")
    fig.tight_layout(rect=(0, 0.09, 1, 1))

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches="tight")

    if show:
        plt.show()

    return fig, ax


def plot_sorted_variance_scores(
    scores,
    save_path=DEFAULT_SORTED_PLOT_PATH,
    show=True,
    thresholds=THRESHOLD_LINES,
):
    scores = np.asarray(scores)
    sorted_scores = np.sort(scores)
    mean_score = np.mean(scores)
    median_score = np.median(scores)

    fig, ax = plt.subplots(figsize=(14, 7))
    ax.plot(sorted_scores, color="tab:blue", linewidth=2)
    ax.axhline(mean_score, color="tab:red", linewidth=2, label="Mean")
    ax.axhline(median_score, color="black", linestyle="--", linewidth=1.5, label="Median")
    add_threshold_lines(ax, thresholds)

    ax.set_title("Sorted Variance Scores Across DAVIS Datapoints")
    ax.set_xlabel("Datapoints sorted by variance score")
    ax.set_ylabel("Variance score")
    ax.grid(alpha=0.25)
    ax.legend()

    stats_text = format_stats_text(scores, thresholds)
    fig.text(0.5, 0.02, stats_text, ha="center", va="bottom")
    fig.tight_layout(rect=(0, 0.09, 1, 1))

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches="tight")

    if show:
        plt.show()

    return fig, ax


def plot_sorted_variance_scores_and_residuals(
    scores,
    residuals,
    save_path=DEFAULT_SORTED_SCORE_RESIDUAL_PLOT_PATH,
    show=True,
):
    scores = np.asarray(scores)
    residuals = np.asarray(residuals)
    if scores.shape != residuals.shape:
        raise ValueError("scores and residuals must have the same shape.")

    sort_idx = np.argsort(scores)
    sorted_scores = scores[sort_idx]
    sorted_residuals = residuals[sort_idx]
    sorted_index = np.arange(scores.size)

    fig, axes = plt.subplots(
        2,
        1,
        figsize=(14, 8),
        sharex=True,
        gridspec_kw={"height_ratios": [1, 1]},
    )

    axes[0].plot(sorted_index, sorted_scores, color="tab:blue", linewidth=2)
    axes[0].scatter(
        sorted_index,
        sorted_scores,
        s=6,
        alpha=0.28,
        color="tab:blue",
        edgecolors="none",
    )
    axes[0].set_title("Ensemble Variance")
    axes[0].set_ylabel("Ensemble-variance score")
    axes[0].grid(alpha=0.25)

    axes[1].scatter(
        sorted_index,
        sorted_residuals,
        s=8,
        alpha=0.35,
        color="tab:orange",
        edgecolors="none",
    )
    axes[1].set_title("Residuals In The Same Order")
    axes[1].set_xlabel("Datapoints sorted by ensemble variance")
    axes[1].set_ylabel(r"$|Y - \hat{\mu}_1|$")
    axes[1].grid(alpha=0.25)

    fig.tight_layout()

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches="tight")

    if show:
        plt.show()

    return fig, axes


if __name__ == "__main__":
    scores, residuals = load_variance_scores_and_residuals()
    plot_variance_scores(scores, show=False)
    plot_sorted_variance_scores(scores, show=False)
    plot_sorted_variance_scores_and_residuals(scores, residuals, show=False)
