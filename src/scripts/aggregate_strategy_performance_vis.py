from argparse import ArgumentParser
import os
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/davis_eval_matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/private/tmp/davis_eval_cache")

import matplotlib.pyplot as plt


RESULTS_DIR = PROJECT_ROOT / "results"
DEFAULT_OUTPUT_DIR_NAME = "vis"
DEFAULT_OUTPUT_NAME = "strategy_performance_3panels.png"
DEFAULT_SUMMARY_NAME = "strategy_performance_3panels.csv"
STRATEGY_ORDER = [
    "EXPRESS",
    "RELAXED-EXPRESS",
    "WEIGHTED-EXPRESS",
    "ADAPTIVE-WEIGHTED-EXPRESS",
]
STRATEGY_LABELS = {
    "EXPRESS": "EXPRESS",
    "RELAXED-EXPRESS": "Relaxed\nEXPRESS",
    "WEIGHTED-EXPRESS": "Weighted\nEXPRESS",
    "ADAPTIVE-WEIGHTED-EXPRESS": "Adaptive weighted\nEXPRESS",
}
COLORS = ["#4C78A8", "#F58518", "#54A24B", "#B279A2"]


def latest_results_dir(results_dir=RESULTS_DIR):
    candidates = [
        path
        for path in Path(results_dir).glob("*_runs")
        if (path / "aggregate_results.csv").exists()
    ]
    if not candidates:
        raise FileNotFoundError(f"No aggregate_results.csv found under {results_dir}")
    return max(candidates, key=lambda path: path.stat().st_mtime)


def load_aggregate(result_dir):
    aggregate_path = Path(result_dir) / "aggregate_results.csv"
    if not aggregate_path.exists():
        raise FileNotFoundError(f"Missing {aggregate_path}")

    aggregate = pd.read_csv(aggregate_path)
    missing = [strategy for strategy in STRATEGY_ORDER if strategy not in set(aggregate["strategy"])]
    if missing:
        raise ValueError(f"Missing strategies in {aggregate_path}: {missing}")

    aggregate = aggregate.set_index("strategy").loc[STRATEGY_ORDER].reset_index()
    for column in ["miscoverage", "median_interval_length", "infinite_fraction"]:
        aggregate[column] = pd.to_numeric(aggregate[column], errors="coerce")
    return aggregate


def positive_axis_top(values, fallback=1.0):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return fallback
    top = np.nanmax(values) * 1.18
    return top if np.isfinite(top) and top > 0 else fallback


def plot_bar_panel(ax, aggregate, metric, title, ylabel, ylim_top=None, percent=False):
    labels = [STRATEGY_LABELS[strategy] for strategy in aggregate["strategy"]]
    values = aggregate[metric].to_numpy(dtype=float)
    x = np.arange(len(labels))
    bars = ax.bar(x, values, color=COLORS, width=0.68)

    ax.set_title(title, loc="left", fontsize=13, fontweight="bold")
    ax.set_ylabel(ylabel)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.grid(axis="y", alpha=0.25)
    ax.set_axisbelow(True)
    ax.set_ylim(bottom=0, top=ylim_top or positive_axis_top(values))

    for bar, value in zip(bars, values):
        if not np.isfinite(value):
            continue
        label = f"{value * 100:.1f}%" if percent else f"{value:.3f}"
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            label,
            ha="center",
            va="bottom",
            fontsize=9,
            color="#333333",
        )


def plot_aggregate(aggregate, output_path, result_dir):
    fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.8))
    fig.suptitle(
        f"Strategy performance - {Path(result_dir).name}",
        fontsize=16,
        fontweight="bold",
    )

    plot_bar_panel(
        axes[0],
        aggregate,
        "miscoverage",
        "Miscoverage",
        "miscoverage",
        ylim_top=positive_axis_top(aggregate["miscoverage"]),
        percent=True,
    )
    plot_bar_panel(
        axes[1],
        aggregate,
        "median_interval_length",
        "Median interval length",
        "median interval length",
    )
    plot_bar_panel(
        axes[2],
        aggregate,
        "infinite_fraction",
        "Infinite interval fraction",
        "infinite fraction",
        ylim_top=max(0.01, positive_axis_top(aggregate["infinite_fraction"])),
        percent=True,
    )

    fig.tight_layout(rect=(0, 0.02, 1, 0.92))
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def build_outputs(result_dir=None, output_dir=None):
    result_dir = latest_results_dir() if result_dir is None else Path(result_dir)
    output_dir = result_dir / DEFAULT_OUTPUT_DIR_NAME if output_dir is None else Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    aggregate = load_aggregate(result_dir)
    output_path = output_dir / DEFAULT_OUTPUT_NAME
    summary_path = output_dir / DEFAULT_SUMMARY_NAME
    aggregate.to_csv(summary_path, index=False)
    plot_aggregate(aggregate, output_path, result_dir)
    return output_path, summary_path, aggregate


def parse_args():
    parser = ArgumentParser(
        description="Plot aggregate performance metrics for the four EXPRESS variants."
    )
    parser.add_argument("--result-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    output_path, summary_path, aggregate = build_outputs(args.result_dir, args.output_dir)
    print(f"Wrote plot to {output_path}")
    print(f"Wrote summary CSV to {summary_path}")
    print(aggregate[["strategy", "miscoverage", "median_interval_length", "infinite_fraction"]].to_string(index=False))


if __name__ == "__main__":
    main()
