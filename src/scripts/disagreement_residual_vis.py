import argparse
import os
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
EXP_DIR = PROJECT_ROOT / "data" / "exp"
VIS_DIR = PROJECT_ROOT / "data" / "vis"
PIPELINE_THRESHOLD = 0.15

FONT_CACHE_DIR = Path(tempfile.gettempdir()) / "davis_eval_matplotlib_cache"
MPL_CACHE_DIR = FONT_CACHE_DIR / "matplotlib"
XDG_CACHE_DIR = FONT_CACHE_DIR / "xdg"
MPL_CACHE_DIR.mkdir(parents=True, exist_ok=True)
XDG_CACHE_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPL_CACHE_DIR))
os.environ.setdefault("XDG_CACHE_HOME", str(XDG_CACHE_DIR))

import matplotlib.pyplot as plt


def load_tables(exp_dir=EXP_DIR):
    exp_dir = Path(exp_dir)
    return {
        "correlations": pd.read_csv(exp_dir / "disagreement_residual_correlations.csv"),
        "deciles": pd.read_csv(exp_dir / "disagreement_residual_deciles.csv"),
        "thresholds": pd.read_csv(exp_dir / "disagreement_residual_thresholds.csv"),
    }


def plot_variance_correlation(correlations, output_dir=VIS_DIR):
    row = correlations[
        correlations["disagreement"].eq("variance")
        & correlations["residual"].eq("abs_resid_muhat_1")
    ].iloc[0]

    labels = ["Pearson r", "Spearman rho"]
    values = [row["pearson_r"], row["spearman_rho"]]
    colors = ["tab:blue", "tab:orange"]

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(labels, values, color=colors, width=0.55)
    ax.axhline(0, color="black", linewidth=1)
    ax.set_ylim(0, 0.35)
    ax.set_ylabel("Correlation")
    ax.set_title("Ensemble Variance vs Absolute Residual")
    ax.grid(axis="y", alpha=0.25)

    for bar, value in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + 0.012,
            f"{value:.3f}",
            ha="center",
            va="bottom",
            fontsize=12,
        )

    fig.tight_layout()

    return save_figure(fig, output_dir, "variance_residual_correlation.png")


def plot_decile_trend(deciles, output_dir=VIS_DIR):
    x = deciles["decile"].to_numpy()
    mean_residual = deciles["abs_resid_muhat_1_mean"].to_numpy()

    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.plot(
        x,
        mean_residual,
        marker="o",
        linewidth=2.5,
        color="tab:blue",
    )

    ax.set_title("Residuals Increase in Higher Variance Deciles")
    ax.set_xlabel("Ensemble-variance decile")
    ax.set_ylabel("Mean |Y - muhat_1|")
    ax.set_xticks(x)
    ax.grid(alpha=0.25)

    fig.tight_layout()

    return save_figure(fig, output_dir, "variance_residual_deciles.png")


def plot_threshold_check(thresholds, output_dir=VIS_DIR):
    row = thresholds[np.isclose(thresholds["threshold"], PIPELINE_THRESHOLD)].iloc[0]
    selected_mean = row["abs_resid_muhat_1_selected_mean"]
    unselected_mean = row["abs_resid_muhat_1_unselected_mean"]
    labels = ["Not selected", "Selected\nvariance > 0.15"]
    values = [unselected_mean, selected_mean]
    colors = ["tab:gray", "tab:blue"]

    fig, ax = plt.subplots(figsize=(8, 5.5))
    bars = ax.bar(labels, values, color=colors, width=0.5)
    ax.set_ylim(0, max(values) * 1.35)
    ax.set_ylabel("Mean |Y - muhat_1|")
    ax.set_title("Selection Rule Picks Higher-Residual Points")
    ax.grid(axis="y", alpha=0.25)

    for bar, value in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + 0.03,
            f"{value:.3f}",
            ha="center",
            va="bottom",
            fontsize=12,
        )

    fig.tight_layout()

    return save_figure(fig, output_dir, "variance_selection_rule_check.png")


def save_figure(fig, output_dir, filename):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / filename
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return path


def write_plots(exp_dir=EXP_DIR, output_dir=VIS_DIR):
    tables = load_tables(exp_dir)
    return {
        "correlation": plot_variance_correlation(tables["correlations"], output_dir),
        "deciles": plot_decile_trend(tables["deciles"], output_dir),
        "selection_rule": plot_threshold_check(tables["thresholds"], output_dir),
    }


def parse_args():
    parser = argparse.ArgumentParser(
        description="Create simple DAVIS variance-vs-residual plots."
    )
    parser.add_argument("--exp-dir", type=Path, default=EXP_DIR)
    parser.add_argument("--output-dir", type=Path, default=VIS_DIR)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    paths = write_plots(exp_dir=args.exp_dir, output_dir=args.output_dir)
    for name, path in paths.items():
        print(f"{name}: {path}")
