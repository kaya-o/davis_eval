from argparse import ArgumentParser
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/davis_eval_matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/private/tmp/davis_eval_cache")

import matplotlib.pyplot as plt


DEFAULT_SUITE_DIR = (
    PROJECT_ROOT
    / "results"
    / "suite_20260606_180347_weighted_neighborhood_expr_max_rank_pct_sweep"
)
DEFAULT_OUTPUT_NAME = "max_rank_pct_aggregate_metrics.png"
DEFAULT_SUMMARY_NAME = "max_rank_pct_aggregate_metrics.csv"
DEFAULT_STRATEGIES = [
    "RELAXED-EXPRESS",
    "WEIGHTED-EXPRESS",
    "WEIGHTED-NEIGHBORHOOD-EXPRESS",
    "EXPRESS",
]
METRICS = [
    ("miscoverage", "Miscoverage"),
    ("median_interval_length", "Median interval length"),
    ("infinite_fraction", "Infinite interval fraction"),
]
STRATEGY_STYLES = {
    "RELAXED-EXPRESS": {
        "label": "Relaxed EXPRESS",
        "color": "#1f77b4",
        "linestyle": "-",
        "marker": "o",
    },
    "WEIGHTED-EXPRESS": {
        "label": "Weighted EXPRESS",
        "color": "#2ca02c",
        "linestyle": "-",
        "marker": "s",
    },
    "WEIGHTED-NEIGHBORHOOD-EXPRESS": {
        "label": "Weighted Neighborhood EXPRESS",
        "color": "#9467bd",
        "linestyle": "-",
        "marker": "^",
    },
    "EXPRESS": {
        "label": "EXPRESS baseline",
        "color": "#d62728",
        "linestyle": "--",
        "marker": None,
    },
}


def result_dirs(suite_dir):
    suite_dir = Path(suite_dir)
    dirs = [
        path
        for path in suite_dir.iterdir()
        if path.is_dir() and (path / "aggregate_results.csv").exists()
    ]
    if not dirs:
        raise FileNotFoundError(f"No result directories with aggregate_results.csv under {suite_dir}")
    return sorted(dirs)


def load_result_config(run_dir):
    for filename in ("resolved_config.json", "config.json"):
        config_path = Path(run_dir) / filename
        if config_path.exists():
            with config_path.open() as f:
                return json.load(f)
    raise FileNotFoundError(f"Missing config.json or resolved_config.json under {run_dir}")


def max_rank_pct_from_config(config):
    conformal = config.get("conformal", {})
    pct = conformal.get("weighted_neighborhood_express_max_rank_pct")
    if pct is None:
        pct = conformal.get("weighted_express_max_rank_pct")
    if pct is None:
        raise KeyError(
            "Expected conformal.weighted_neighborhood_express_max_rank_pct "
            "or conformal.weighted_express_max_rank_pct in run config"
        )
    return float(pct)


def summarize_suite(suite_dir, strategies=None):
    rows = []
    for run_dir in result_dirs(suite_dir):
        config = load_result_config(run_dir)
        aggregate = pd.read_csv(run_dir / "aggregate_results.csv")
        aggregate["max_rank_pct"] = max_rank_pct_from_config(config)
        aggregate["run_dir"] = run_dir.name
        rows.append(aggregate)

    summary = pd.concat(rows, ignore_index=True)
    if strategies:
        summary = summary[summary["strategy"].isin(strategies)].copy()

    metric_cols = [metric for metric, _ in METRICS]
    for col in metric_cols:
        summary[col] = pd.to_numeric(summary[col], errors="coerce")
    return summary.sort_values(["max_rank_pct", "strategy"]).reset_index(drop=True)


def plot_summary(summary, output_path, strategies=None):
    strategies = DEFAULT_STRATEGIES if strategies is None else strategies
    max_rank_pcts = sorted(summary["max_rank_pct"].unique())
    x = np.arange(len(max_rank_pcts))
    pct_to_x = {value: idx for idx, value in enumerate(max_rank_pcts)}

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8), sharex=True)
    for ax, (metric, ylabel) in zip(axes, METRICS):
        for strategy in strategies:
            strategy_df = summary[summary["strategy"] == strategy].sort_values("max_rank_pct")
            if strategy_df.empty:
                continue

            style = STRATEGY_STYLES.get(strategy, {})
            ax.plot(
                [pct_to_x[value] for value in strategy_df["max_rank_pct"]],
                strategy_df[metric].replace([np.inf, -np.inf], np.nan).to_numpy(dtype=float),
                marker=style.get("marker", "o"),
                markersize=5,
                linewidth=1.7,
                linestyle=style.get("linestyle", "-"),
                color=style.get("color"),
                label=style.get("label", strategy),
            )

        ax.set_ylabel(ylabel)
        ax.set_xlabel("Maximum rank pct")
        ax.set_xticks(x)
        ax.set_xticklabels([f"{value:g}" for value in max_rank_pcts])
        ax.set_ylim(bottom=0)
        ax.grid(alpha=0.25)

    axes[-1].legend(frameon=False, fontsize=8, loc="best")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def parse_args():
    parser = ArgumentParser(
        description="Plot aggregate metrics over a weighted-neighborhood max-rank-pct sweep.",
    )
    parser.add_argument(
        "--suite-dir",
        type=Path,
        default=DEFAULT_SUITE_DIR,
        help="Suite directory containing max-rank-pct experiment result subdirectories.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional output image path. Defaults under suite-dir/vis.",
    )
    parser.add_argument(
        "--summary-csv",
        type=Path,
        default=None,
        help="Optional CSV path for the plotted suite summary. Defaults under suite-dir/vis.",
    )
    parser.add_argument(
        "--strategies",
        nargs="+",
        default=DEFAULT_STRATEGIES,
        help="Strategies to include in the plot and summary CSV.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    suite_dir = Path(args.suite_dir)
    vis_dir = suite_dir / "vis"
    output_path = args.output or vis_dir / DEFAULT_OUTPUT_NAME
    summary_csv = args.summary_csv or vis_dir / DEFAULT_SUMMARY_NAME

    summary = summarize_suite(suite_dir, strategies=args.strategies)
    summary_csv.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(summary_csv, index=False)
    plot_summary(summary, output_path, strategies=args.strategies)

    print(f"Wrote plot to {output_path}")
    print(f"Wrote summary CSV to {summary_csv}")


if __name__ == "__main__":
    main()
