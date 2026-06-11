import argparse
import json
import os
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/davis_eval_matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/private/tmp/davis_eval_cache")

import matplotlib.pyplot as plt


DEFAULT_SUITE_DIR = PROJECT_ROOT / "results" / "suite_20260528_101421_k_sweep"
DEFAULT_OUTPUT_NAME = "k_sweep_calibration_timeseries_3x4.png"
DEFAULT_SUMMARY_NAME = "k_sweep_calibration_timeseries_3x4.csv"
STRATEGY = "K-EXPRESS"


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


def summarize_run_dir(run_dir):
    config = load_config(run_dir)
    k_value = config.get("conformal", {}).get("k_express")
    if k_value is None:
        raise ValueError(f"Missing conformal.k_express in {run_dir}")

    raw_path = Path(run_dir) / "raw_selected_events.csv"
    raw_df = pd.read_csv(
        raw_path,
        usecols=["run", "t", "strategy", "n_calibration"],
    )
    raw_df = raw_df[raw_df["strategy"].eq(STRATEGY)].copy()
    if raw_df.empty:
        raise ValueError(f"No {STRATEGY} rows found in {raw_path}")

    raw_df["run"] = pd.to_numeric(raw_df["run"], errors="coerce")
    raw_df["t"] = pd.to_numeric(raw_df["t"], errors="coerce")
    raw_df["n_calibration"] = pd.to_numeric(raw_df["n_calibration"], errors="coerce")
    raw_df = raw_df.dropna(subset=["run", "t", "n_calibration"])

    summary = (
        raw_df.groupby("t", sort=True)
        .agg(
            mean_n_calibration=("n_calibration", "mean"),
            contributing_runs=("run", "nunique"),
            selected_events=("n_calibration", "size"),
        )
        .reset_index()
    )
    summary["k"] = int(k_value)
    summary["run_dir"] = Path(run_dir).name
    return summary


def summarize_suite(suite_dir):
    summaries = [summarize_run_dir(run_dir) for run_dir in discover_k_run_dirs(suite_dir)]
    return pd.concat(summaries, ignore_index=True)


def plot_summary(summary, output_path):
    k_values = sorted(summary["k"].unique())
    fig, axes = plt.subplots(3, 4, figsize=(16, 9), sharex=True, sharey=True)
    axes_flat = list(axes.flat)

    for ax, k_value in zip(axes_flat, k_values):
        panel = summary[summary["k"].eq(k_value)].sort_values("t")
        ax.scatter(
            panel["t"],
            panel["mean_n_calibration"],
            s=3,
            alpha=0.75,
            color="#1f77b4",
            linewidths=0,
        )
        ax.set_title(f"k={int(k_value)}", fontsize=10)
        ax.grid(alpha=0.25)

    for ax in axes_flat[len(k_values):]:
        ax.axis("off")

    for ax in axes[:, 0]:
        ax.set_ylabel("calibration set size")
    for ax in axes[-1, :]:
        if ax.has_data():
            ax.set_xlabel("t")

    fig.tight_layout()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=250, bbox_inches="tight")
    plt.close(fig)


def build_outputs(suite_dir=DEFAULT_SUITE_DIR, output_path=None):
    suite_dir = Path(suite_dir)
    output_path = suite_dir / "vis" / DEFAULT_OUTPUT_NAME if output_path is None else Path(output_path)
    summary_path = output_path.parent / DEFAULT_SUMMARY_NAME

    summary = summarize_suite(suite_dir)
    summary = summary[["k", "run_dir", "t", "mean_n_calibration", "contributing_runs", "selected_events"]]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(summary_path, index=False)
    plot_summary(summary, output_path)
    return output_path, summary_path, summary


def main():
    parser = argparse.ArgumentParser(
        description="Plot K-EXPRESS calibration set size over time for each swept k."
    )
    parser.add_argument("--suite-dir", type=Path, default=DEFAULT_SUITE_DIR)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    output_path, summary_path, summary = build_outputs(
        suite_dir=args.suite_dir,
        output_path=args.output,
    )
    print(f"Wrote plot to {output_path}")
    print(f"Wrote summary CSV to {summary_path}")
    print(f"k panels={summary['k'].nunique()}")


if __name__ == "__main__":
    main()
