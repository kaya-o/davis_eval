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
DEFAULT_OUTPUT_NAME = "selected_event_adaptive_diagnostics_timeseries.png"
DEFAULT_SVG_NAME = "selected_event_adaptive_diagnostics_timeseries.svg"
DEFAULT_SUMMARY_NAME = "selected_event_adaptive_diagnostics_timeseries.csv"
EXPRESS_STRATEGY = "EXPRESS"
ADAPTIVE_STRATEGY = "ADAPTIVE-WEIGHTED-EXPRESS"


def latest_results_dir(results_dir=RESULTS_DIR):
    candidates = [
        path
        for path in Path(results_dir).glob("*_runs")
        if (path / "raw_selected_events.csv").exists()
    ]
    if not candidates:
        raise FileNotFoundError(f"No raw_selected_events.csv found under {results_dir}")
    return max(candidates, key=lambda path: path.stat().st_mtime)


def summarize_metric(raw_df, strategy, source_column, output_prefix):
    strategy_df = raw_df[raw_df["strategy"] == strategy].dropna(
        subset=["t", "run", source_column],
    )
    if strategy_df.empty:
        raise ValueError(f"No rows found for {strategy} / {source_column}")

    summary = (
        strategy_df.groupby("t", sort=True)
        .agg(
            **{
                f"{output_prefix}_mean": (source_column, "mean"),
                f"{output_prefix}_n_runs": ("run", "nunique"),
            }
        )
        .reset_index()
    )
    if summary["t"].duplicated().any():
        raise ValueError(f"Expected one summary row per t for {output_prefix}")
    return summary


def summarize_selected_events(result_dir):
    raw_path = Path(result_dir) / "raw_selected_events.csv"
    if not raw_path.exists():
        raise FileNotFoundError(f"Missing {raw_path}")

    usecols = [
        "run",
        "t",
        "strategy",
        "n_calibration",
        "adaptive_weighted_express_stress",
        "adaptive_weighted_express_lambda_t",
    ]
    raw_df = pd.read_csv(raw_path, usecols=usecols)
    for column in usecols:
        if column != "strategy":
            raw_df[column] = pd.to_numeric(raw_df[column], errors="coerce")

    express_n = summarize_metric(
        raw_df,
        EXPRESS_STRATEGY,
        "n_calibration",
        "express_n_calibration",
    )
    adaptive_stress = summarize_metric(
        raw_df,
        ADAPTIVE_STRATEGY,
        "adaptive_weighted_express_stress",
        "adaptive_weighted_express_stress",
    )
    adaptive_lambda = summarize_metric(
        raw_df,
        ADAPTIVE_STRATEGY,
        "adaptive_weighted_express_lambda_t",
        "adaptive_weighted_express_lambda_t",
    )

    summary = express_n.merge(adaptive_stress, on="t", how="outer").merge(
        adaptive_lambda,
        on="t",
        how="outer",
    )
    return summary.sort_values("t").reset_index(drop=True)


def positive_axis_top(values, fallback=1.0):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return fallback
    top = np.nanmax(values) * 1.08
    return top if np.isfinite(top) and top > 0 else fallback


def plot_panel(ax, summary, prefix, color, title, ylabel):
    x = summary["t"].to_numpy(dtype=float)
    mean = summary[f"{prefix}_mean"].to_numpy(dtype=float)
    valid = np.isfinite(x) & np.isfinite(mean)

    ax.scatter(
        x[valid],
        mean[valid],
        color=color,
        s=3,
        alpha=0.9,
        marker=".",
        linewidths=0,
    )
    ax.set_title(title, fontsize=10)
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.25)


def plot_summary(summary, output_path, result_dir):
    fig, axes = plt.subplots(3, 1, figsize=(12.5, 9), sharex=True)
    fig.suptitle(
        f"Selected-Event Diagnostics Over Time - {Path(result_dir).name}",
        fontsize=13,
    )

    plot_panel(
        axes[0],
        summary,
        "express_n_calibration",
        "#1f77b4",
        "EXPRESS calibration set size n",
        "Mean n",
    )
    axes[0].set_ylim(
        bottom=0,
        top=positive_axis_top(summary["express_n_calibration_mean"]),
    )

    plot_panel(
        axes[1],
        summary,
        "adaptive_weighted_express_stress",
        "#d62728",
        "Adaptive Weighted EXPRESS stress",
        "Mean stress",
    )
    axes[1].set_ylim(
        bottom=0,
        top=max(1.02, positive_axis_top(summary["adaptive_weighted_express_stress_mean"])),
    )

    plot_panel(
        axes[2],
        summary,
        "adaptive_weighted_express_lambda_t",
        "#2ca02c",
        "Adaptive Weighted EXPRESS lambda_t",
        r"Mean $\lambda_t$",
    )
    axes[2].set_ylim(
        bottom=0,
        top=positive_axis_top(summary["adaptive_weighted_express_lambda_t_mean"]),
    )
    axes[2].set_xlabel("Time t")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(rect=(0, 0.02, 1, 0.96))
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    fig.savefig(output_path.with_suffix(".svg"), bbox_inches="tight")
    plt.close(fig)


def parse_args():
    parser = ArgumentParser(
        description=(
            "Plot one selected-event mean point per observed timestep for "
            "EXPRESS n_calibration and adaptive weighted EXPRESS diagnostics."
        )
    )
    parser.add_argument("--result-dir", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--summary-csv", type=Path, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    result_dir = latest_results_dir() if args.result_dir is None else Path(args.result_dir)
    output_path = args.output or result_dir / DEFAULT_OUTPUT_NAME
    summary_csv = args.summary_csv or result_dir / DEFAULT_SUMMARY_NAME

    summary = summarize_selected_events(result_dir)
    summary_csv.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(summary_csv, index=False)
    plot_summary(summary, output_path, result_dir)

    n_run_columns = [column for column in summary.columns if column.endswith("_n_runs")]
    min_runs = int(summary[n_run_columns].min().min())
    max_runs = int(summary[n_run_columns].max().max())
    print(f"Wrote summary CSV to {summary_csv}")
    print(f"Wrote plot to {output_path}")
    print(
        f"Observed timesteps={len(summary)}, "
        f"runs contributing per t range={min_runs}-{max_runs}"
    )


if __name__ == "__main__":
    main()
