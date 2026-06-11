import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/davis_eval_matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/private/tmp/davis_eval_cache")

import matplotlib.pyplot as plt


DEFAULT_SUITE_DIR = PROJECT_ROOT / "results" / "suite_20260528_101421_k_sweep"
DEFAULT_OUTPUT_NAME = "k_sweep_binned_performance_timeseries_3x4.png"
DEFAULT_SUMMARY_NAME = "k_sweep_binned_performance_timeseries_3x4.csv"
STRATEGY = "K-EXPRESS"
MISCOVERAGE_COLOR = "#1f77b4"
INTERVAL_COLOR = "#d62728"


def load_config(run_dir):
    for filename in ("resolved_config.json", "config.json"):
        config_path = Path(run_dir) / filename
        if config_path.exists():
            with config_path.open() as f:
                return json.load(f)
    return {}


def discover_k_run_dirs(suite_dir):
    suite_dir = Path(suite_dir)
    run_dirs = []
    for path in suite_dir.iterdir():
        if not path.is_dir() or not (path / "raw_selected_events.csv").exists():
            continue
        aggregate_path = path / "aggregate_results.csv"
        if not aggregate_path.exists():
            continue
        aggregate = pd.read_csv(aggregate_path, usecols=["strategy"])
        if STRATEGY not in set(aggregate["strategy"]):
            continue
        config = load_config(path)
        k_value = config.get("conformal", {}).get("k_express")
        if k_value is None:
            continue
        run_dirs.append((int(k_value), path))
    if not run_dirs:
        raise FileNotFoundError(f"No k-sweep raw_selected_events.csv files found under {suite_dir}")
    return [path for _, path in sorted(run_dirs, key=lambda item: item[0])]


def summarize_run_dir(run_dir, bin_width):
    config = load_config(run_dir)
    k_value = config.get("conformal", {}).get("k_express")
    if k_value is None:
        raise ValueError(f"Missing conformal.k_express in {run_dir}")

    raw_path = Path(run_dir) / "raw_selected_events.csv"
    raw_df = pd.read_csv(
        raw_path,
        usecols=["run", "t", "strategy", "miscovered", "interval_length"],
    )
    raw_df = raw_df[raw_df["strategy"].eq(STRATEGY)].copy()
    if raw_df.empty:
        raise ValueError(f"No {STRATEGY} rows found in {raw_path}")

    for column in ["run", "t", "miscovered", "interval_length"]:
        raw_df[column] = pd.to_numeric(raw_df[column], errors="coerce")
    raw_df = raw_df.dropna(subset=["run", "t", "miscovered", "interval_length"])
    raw_df["t_bin"] = (raw_df["t"].astype(int) // int(bin_width)) * int(bin_width)

    summary = (
        raw_df.groupby("t_bin", sort=True)
        .agg(
            miscoverage=("miscovered", "mean"),
            median_interval_length=("interval_length", "median"),
            contributing_runs=("run", "nunique"),
            selected_events=("miscovered", "size"),
        )
        .reset_index()
    )
    summary["k"] = int(k_value)
    summary["run_dir"] = Path(run_dir).name
    return summary


def summarize_suite(suite_dir, bin_width):
    summaries = [
        summarize_run_dir(run_dir, bin_width=bin_width)
        for run_dir in discover_k_run_dirs(suite_dir)
    ]
    return pd.concat(summaries, ignore_index=True)


def finite_max(values, fallback=1.0):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return fallback
    max_value = float(np.nanmax(values))
    return max_value if max_value > 0 else fallback


def plot_summary(summary, output_path):
    k_values = sorted(summary["k"].unique())
    fig, axes = plt.subplots(3, 4, figsize=(17, 9.5), sharex=True)
    axes_flat = list(axes.flat)
    interval_top = finite_max(summary["median_interval_length"], fallback=1.0) * 1.08

    legend_handles = None
    legend_labels = None
    for panel_index, (ax, k_value) in enumerate(zip(axes_flat, k_values)):
        panel = summary[summary["k"].eq(k_value)].sort_values("t_bin")
        ax2 = ax.twinx()

        line_miscoverage, = ax.plot(
            panel["t_bin"],
            panel["miscoverage"],
            color=MISCOVERAGE_COLOR,
            linewidth=1.25,
            marker="o",
            markersize=2.2,
            label="miscoverage",
        )
        finite_interval = panel["median_interval_length"].replace([np.inf, -np.inf], np.nan)
        line_interval, = ax2.plot(
            panel["t_bin"],
            finite_interval,
            color=INTERVAL_COLOR,
            linewidth=1.25,
            marker="o",
            markersize=2.2,
            label="median interval length",
        )

        ax.set_title(f"k={int(k_value)}", fontsize=10)
        ax.set_ylim(0, 1)
        ax2.set_ylim(0, interval_top)
        ax.grid(alpha=0.25)

        row, col = divmod(panel_index, 4)
        if col == 0:
            ax.set_ylabel("miscoverage", color=MISCOVERAGE_COLOR)
        else:
            ax.set_yticklabels([])
        if col == 3:
            ax2.set_ylabel("median interval length", color=INTERVAL_COLOR)
        else:
            ax2.set_yticklabels([])
        if row == 2:
            ax.set_xlabel("t_bin")

        legend_handles = [line_miscoverage, line_interval]
        legend_labels = [handle.get_label() for handle in legend_handles]

    for ax in axes_flat[len(k_values):]:
        ax.axis("off")

    if legend_handles is not None:
        axes_flat[0].legend(legend_handles, legend_labels, loc="best", frameon=True)

    fig.tight_layout()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=250, bbox_inches="tight")
    plt.close(fig)


def build_outputs(suite_dir=DEFAULT_SUITE_DIR, output_path=None, bin_width=250):
    suite_dir = Path(suite_dir)
    output_path = suite_dir / "vis" / DEFAULT_OUTPUT_NAME if output_path is None else Path(output_path)
    summary_path = output_path.parent / DEFAULT_SUMMARY_NAME

    summary = summarize_suite(suite_dir, bin_width=bin_width)
    summary = summary[
        [
            "k",
            "run_dir",
            "t_bin",
            "miscoverage",
            "median_interval_length",
            "contributing_runs",
            "selected_events",
        ]
    ]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(summary_path, index=False)
    plot_summary(summary, output_path)
    return output_path, summary_path, summary


def main():
    parser = argparse.ArgumentParser(
        description="Plot binned miscoverage and median interval length over time for k sweep."
    )
    parser.add_argument("--suite-dir", type=Path, default=DEFAULT_SUITE_DIR)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--bin-width", type=int, default=50)
    args = parser.parse_args()

    output_path, summary_path, summary = build_outputs(
        suite_dir=args.suite_dir,
        output_path=args.output,
        bin_width=args.bin_width,
    )
    print(f"Wrote plot to {output_path}")
    print(f"Wrote summary CSV to {summary_path}")
    print(f"k panels={summary['k'].nunique()}, bin_width={args.bin_width}")


if __name__ == "__main__":
    main()
