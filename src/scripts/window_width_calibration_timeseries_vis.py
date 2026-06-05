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


DEFAULT_OUTPUT_NAME = "express_mean_calibration_size_by_window_width_timeseries.png"
DEFAULT_SUMMARY_NAME = "express_mean_calibration_size_by_window_width_timeseries.csv"


def result_dirs(suite_dir):
    suite_dir = Path(suite_dir)
    dirs = [
        path
        for path in suite_dir.iterdir()
        if path.is_dir() and (path / "raw_selected_events.csv").exists()
    ]
    if not dirs:
        raise FileNotFoundError(f"No result directories with raw_selected_events.csv under {suite_dir}")
    return sorted(dirs)


def load_result_config(run_dir):
    for filename in ("resolved_config.json", "config.json"):
        config_path = Path(run_dir) / filename
        if config_path.exists():
            with config_path.open() as f:
                return json.load(f)
    raise FileNotFoundError(f"Missing config.json or resolved_config.json under {run_dir}")


def aggregate_mean_calibration_by_time(raw_path, strategy, chunksize):
    sum_by_t = pd.Series(dtype=float)
    count_by_t = pd.Series(dtype=float)

    for chunk in pd.read_csv(
        raw_path,
        usecols=["t", "strategy", "n_calibration"],
        chunksize=chunksize,
    ):
        chunk = chunk[chunk["strategy"] == strategy]
        if chunk.empty:
            continue

        grouped = chunk.groupby("t")["n_calibration"].agg(["sum", "count"])
        sum_by_t = sum_by_t.add(grouped["sum"], fill_value=0.0)
        count_by_t = count_by_t.add(grouped["count"], fill_value=0.0)

    if count_by_t.empty:
        raise ValueError(f"No rows for strategy {strategy!r} in {raw_path}")

    mean_by_t = (sum_by_t / count_by_t).sort_index()
    return pd.DataFrame({
        "t": mean_by_t.index.astype(int),
        "mean_n_calibration": mean_by_t.to_numpy(dtype=float),
        "n_events": count_by_t.loc[mean_by_t.index].to_numpy(dtype=int),
    })


def summarize_suite(suite_dir, strategy, chunksize):
    rows = []
    for run_dir in result_dirs(suite_dir):
        config = load_result_config(run_dir)
        window_width = float(config["selection"]["window_width"])
        time_df = aggregate_mean_calibration_by_time(
            run_dir / "raw_selected_events.csv",
            strategy,
            chunksize,
        )
        time_df["window_width"] = window_width
        time_df["run_dir"] = run_dir.name
        rows.append(time_df)

    summary_df = pd.concat(rows, ignore_index=True)
    return summary_df.sort_values(["window_width", "t"]).reset_index(drop=True)


def plot_window_width_timeseries(summary_df, output_path, strategy):
    window_widths = sorted(summary_df["window_width"].unique())
    n_panels = len(window_widths)
    n_cols = 3
    n_rows = int(np.ceil(n_panels / n_cols))

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4.2 * n_rows), sharex=True, sharey=True)
    axes = np.asarray(axes).reshape(-1)

    y_max = summary_df["mean_n_calibration"].max()
    y_top = y_max * 1.08 if np.isfinite(y_max) and y_max > 0 else 1.0

    for ax, window_width in zip(axes, window_widths):
        width_df = summary_df[summary_df["window_width"] == window_width]
        ax.scatter(
            width_df["t"],
            width_df["mean_n_calibration"],
            s=8,
            alpha=0.6,
            color="tab:red",
            edgecolors="none",
        )
        ax.set_title(f"w = {window_width:g}")
        ax.set_ylim(bottom=0, top=y_top)
        ax.grid(alpha=0.25)

    for ax in axes[n_panels:]:
        ax.axis("off")

    for ax in axes:
        if ax.has_data():
            ax.set_xlabel("Time t")
            ax.set_ylabel("Calibration size")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def parse_args():
    parser = ArgumentParser(description="Plot calibration-size time series across window widths.")
    parser.add_argument(
        "--suite-dir",
        type=Path,
        required=True,
        help="Suite directory containing window-width experiment result subdirectories.",
    )
    parser.add_argument(
        "--strategy",
        default="EXPRESS",
        help="Strategy to plot. Defaults to EXPRESS.",
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
        help="Optional CSV path for the computed time-series summary. Defaults under suite-dir/vis.",
    )
    parser.add_argument(
        "--chunksize",
        type=int,
        default=500_000,
        help="CSV chunk size for raw_selected_events.csv reads.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    suite_dir = Path(args.suite_dir)
    vis_dir = suite_dir / "vis"
    output_path = args.output or vis_dir / DEFAULT_OUTPUT_NAME
    summary_csv = args.summary_csv or vis_dir / DEFAULT_SUMMARY_NAME

    summary_df = summarize_suite(
        suite_dir=suite_dir,
        strategy=args.strategy,
        chunksize=args.chunksize,
    )
    summary_csv.parent.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(summary_csv, index=False)
    plot_window_width_timeseries(summary_df, output_path, args.strategy)

    print(f"Wrote plot to {output_path}")
    print(f"Wrote summary CSV to {summary_csv}")


if __name__ == "__main__":
    main()
