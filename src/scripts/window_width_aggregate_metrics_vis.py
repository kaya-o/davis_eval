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


DEFAULT_OUTPUT_NAME = "window_width_aggregate_metrics.png"
DEFAULT_SUMMARY_NAME = "window_width_aggregate_metrics.csv"
DEFAULT_STRATEGIES = ["EXPRESS", "RELAXED-EXPRESS", "WEIGHTED-EXPRESS"]


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


def summarize_suite(suite_dir, strategies=None):
    rows = []
    for run_dir in result_dirs(suite_dir):
        config = load_result_config(run_dir)
        aggregate = pd.read_csv(run_dir / "aggregate_results.csv")
        aggregate["window_width"] = float(config["selection"]["window_width"])
        aggregate["run_dir"] = run_dir.name
        rows.append(aggregate)

    summary = pd.concat(rows, ignore_index=True)
    if strategies:
        summary = summary[summary["strategy"].isin(strategies)].copy()
    metric_cols = ["miscoverage", "median_interval_length", "infinite_fraction"]
    for col in metric_cols:
        summary[col] = pd.to_numeric(summary[col], errors="coerce")
    return summary.sort_values(["window_width", "strategy"]).reset_index(drop=True)


def plot_summary(summary, output_path, strategies=None):
    metrics = [
        ("miscoverage", "miscoverage"),
        ("median_interval_length", "median interval length"),
        ("infinite_fraction", "infinite interval fraction"),
    ]
    window_widths = sorted(summary["window_width"].unique())
    x = np.arange(len(window_widths))
    width_to_x = {width: idx for idx, width in enumerate(window_widths)}

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8), sharex=True)

    if strategies is None:
        strategies = list(dict.fromkeys(summary["strategy"]))
    else:
        strategies = [
            strategy
            for strategy in dict.fromkeys(strategies)
            if strategy in set(summary["strategy"])
        ]
    for ax, (metric, ylabel) in zip(axes, metrics):
        for strategy in strategies:
            strategy_df = summary[summary["strategy"] == strategy].sort_values("window_width")
            y = strategy_df[metric].replace([np.inf, -np.inf], np.nan).to_numpy(dtype=float)
            ax.plot(
                [width_to_x[width] for width in strategy_df["window_width"]],
                y,
                marker="o",
                linewidth=1.5,
                label=strategy,
            )

        ax.set_ylabel(ylabel)
        ax.set_xlabel("window width")
        ax.set_xticks(x)
        ax.set_xticklabels([f"{width:g}" for width in window_widths])
        ax.grid(alpha=0.25)

    axes[-1].legend(frameon=False, fontsize=8, loc="best")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def parse_args():
    parser = ArgumentParser(description="Plot aggregate metrics over a window-width sweep.")
    parser.add_argument(
        "--suite-dir",
        type=Path,
        required=True,
        help="Suite directory containing window-width experiment result subdirectories.",
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
