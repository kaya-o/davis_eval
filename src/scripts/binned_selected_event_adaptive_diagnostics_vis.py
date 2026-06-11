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


RESULTS_DIR = PROJECT_ROOT / "results"
DEFAULT_OUTPUT_DIR_NAME = "vis"
DEFAULT_OUTPUT_NAME = "adaptive_weighted_express_timeseries_4panels_run_binned_clipped_nlow.png"
DEFAULT_SUMMARY_NAME = "adaptive_weighted_express_timeseries_4panels_run_binned_clipped_nlow.csv"
DEFAULT_RUN_BIN_MEANS_NAME = "adaptive_weighted_express_timeseries_4panels_run_bin_means_clipped_nlow.csv"
EXPRESS_STRATEGY = "EXPRESS"
ADAPTIVE_STRATEGY = "ADAPTIVE-WEIGHTED-EXPRESS"


PLOT_SPECS = [
    {
        "metric": "express_n_calibration",
        "title": "EXPRESS calibration size",
        "ylabel": "mean n",
        "color": "#1f77b4",
    },
    {
        "metric": "adaptive_lambda_t",
        "title": "Adaptive Weighted EXPRESS lambda_t",
        "ylabel": "lambda_t",
        "color": "#1f77b4",
    },
    {
        "metric": "adaptive_stress",
        "title": "Adaptive stress",
        "ylabel": "stress",
        "color": "#1f77b4",
    },
    {
        "metric": "adaptive_stress_input_count",
        "title": "Stress input count",
        "ylabel": "stress input count",
        "color": "#1f77b4",
    },
]


def latest_results_dir(results_dir=RESULTS_DIR):
    candidates = [
        path
        for path in Path(results_dir).glob("*_runs")
        if (path / "raw_selected_events.csv").exists()
    ]
    if not candidates:
        raise FileNotFoundError(f"No raw_selected_events.csv found under {results_dir}")
    return max(candidates, key=lambda path: path.stat().st_mtime)


def load_result_config(result_dir):
    result_dir = Path(result_dir)
    for filename in ("resolved_config.json", "config.json"):
        path = result_dir / filename
        if path.exists():
            with path.open() as f:
                return json.load(f)
    return {}


def target_low_distance_count(config, raw_df):
    column = "adaptive_weighted_express_target_low_distance_count"
    values = pd.to_numeric(raw_df.get(column), errors="coerce")
    if values is not None and values.notna().any():
        return float(values.dropna().iloc[0])
    return float(
        config.get("conformal", {}).get(
            "adaptive_weighted_express_target_low_distance_count",
            np.nan,
        )
    )


def stress_midpoint_count(config):
    conformal_config = config.get("conformal", {})
    if conformal_config.get("adaptive_weighted_express_stress_mode") != "sigmoid":
        return np.nan
    return float(
        conformal_config.get(
            "adaptive_weighted_express_stress_midpoint_count",
            np.nan,
        )
    )


def stress_slope(config):
    conformal_config = config.get("conformal", {})
    if conformal_config.get("adaptive_weighted_express_stress_mode") != "sigmoid":
        return np.nan
    return float(
        conformal_config.get(
            "adaptive_weighted_express_stress_slope",
            np.nan,
        )
    )


def sigmoid_stress_from_count(count, midpoint_count, slope):
    count = np.asarray(count, dtype=float)
    z = slope * (count - midpoint_count)
    return np.where(
        z >= 0,
        np.exp(-z) / (1.0 + np.exp(-z)),
        1.0 / (1.0 + np.exp(z)),
    )


def load_raw_events(result_dir):
    raw_path = Path(result_dir) / "raw_selected_events.csv"
    if not raw_path.exists():
        raise FileNotFoundError(f"Missing {raw_path}")

    raw_columns = set(pd.read_csv(raw_path, nrows=0).columns)
    usecols = [
        "run",
        "t",
        "strategy",
        "n_calibration",
        "adaptive_weighted_express_lambda_t",
        "adaptive_weighted_express_stress",
        "adaptive_weighted_express_n_low_distance",
        "adaptive_weighted_express_target_low_distance_count",
    ]
    for column in [
        "adaptive_weighted_express_stress_count_source",
        "adaptive_weighted_express_stress_count",
        "adaptive_weighted_express_express_n_calibration_for_stress",
    ]:
        if column in raw_columns:
            usecols.append(column)

    raw_df = pd.read_csv(raw_path, usecols=usecols)
    for column in usecols:
        if column not in {"strategy", "adaptive_weighted_express_stress_count_source"}:
            raw_df[column] = pd.to_numeric(raw_df[column], errors="coerce")
    return raw_df


def stress_input_count(adaptive):
    if (
        "adaptive_weighted_express_stress_count" in adaptive.columns
        and adaptive["adaptive_weighted_express_stress_count"].notna().any()
    ):
        return adaptive["adaptive_weighted_express_stress_count"]
    return adaptive["adaptive_weighted_express_n_low_distance"]


def metric_run_bin_means(raw_df, bin_width, target_count):
    rows = []

    express = raw_df[raw_df["strategy"] == EXPRESS_STRATEGY].copy()
    express["metric"] = "express_n_calibration"
    express["value"] = express["n_calibration"]
    rows.append(express[["run", "t", "metric", "value"]])

    adaptive = raw_df[raw_df["strategy"] == ADAPTIVE_STRATEGY].copy()
    for metric, column in [
        ("adaptive_lambda_t", "adaptive_weighted_express_lambda_t"),
        ("adaptive_stress", "adaptive_weighted_express_stress"),
    ]:
        metric_df = adaptive[["run", "t"]].copy()
        metric_df["metric"] = metric
        metric_df["value"] = adaptive[column]
        rows.append(metric_df)

    stress_input_df = adaptive[["run", "t"]].copy()
    stress_input_df["metric"] = "adaptive_stress_input_count"
    stress_input_df["value"] = stress_input_count(adaptive)
    rows.append(stress_input_df)

    long_df = pd.concat(rows, ignore_index=True).dropna(subset=["run", "t", "value"])
    long_df["t_bin"] = (long_df["t"].astype(int) // int(bin_width)) * int(bin_width)

    return (
        long_df.groupby(["metric", "run", "t_bin"], sort=True)["value"]
        .mean()
        .rename("run_bin_mean")
        .reset_index()
    )


def summarize_run_bins(run_bin_means):
    return (
        run_bin_means.groupby(["metric", "t_bin"], sort=True)
        .agg(
            run_bins=("run", "nunique"),
            mean=("run_bin_mean", "mean"),
            median=("run_bin_mean", "median"),
            q10=("run_bin_mean", lambda values: values.quantile(0.10)),
            q90=("run_bin_mean", lambda values: values.quantile(0.90)),
        )
        .reset_index()
    )


def positive_axis_top(values, fallback=1.0):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return fallback
    top = np.nanmax(values) * 1.08
    return top if np.isfinite(top) and top > 0 else fallback


def plot_metric(ax, summary, spec, target_count=None, midpoint_count=None, slope=None):
    metric_df = summary[summary["metric"] == spec["metric"]].sort_values("t_bin")
    x = metric_df["t_bin"].to_numpy(dtype=float)
    mean = metric_df["mean"].to_numpy(dtype=float)
    q10 = metric_df["q10"].to_numpy(dtype=float)
    q90 = metric_df["q90"].to_numpy(dtype=float)

    line_label = "mean actual stress across runs" if spec["metric"] == "adaptive_stress" else "mean across runs"
    ax.plot(x, mean, color=spec["color"], linewidth=1.8, label=line_label)
    ax.fill_between(
        x,
        q10,
        q90,
        color=spec["color"],
        alpha=0.16,
        linewidth=0,
        label="10-90% across runs",
    )

    if spec["metric"] == "adaptive_stress" and np.isfinite(midpoint_count) and np.isfinite(slope):
        input_df = summary[summary["metric"] == "adaptive_stress_input_count"].sort_values("t_bin")
        if not input_df.empty:
            derived = sigmoid_stress_from_count(
                input_df["mean"].to_numpy(dtype=float),
                midpoint_count,
                slope,
            )
            ax.plot(
                input_df["t_bin"].to_numpy(dtype=float),
                derived,
                color="black",
                linestyle="--",
                linewidth=1.4,
                label="sigmoid(mean stress input count)",
            )

    if spec["metric"] == "adaptive_stress_input_count" and np.isfinite(target_count):
        ax.axhline(target_count, color="red", linestyle="--", linewidth=1.0, label="target")
    if spec["metric"] == "adaptive_stress_input_count" and np.isfinite(midpoint_count):
        ax.axhline(
            midpoint_count,
            color="black",
            linestyle=":",
            linewidth=1.2,
            label="sigmoid midpoint",
        )

    ax.set_title(spec["title"], loc="left", fontsize=13, fontweight="bold")
    ax.set_ylabel(spec["ylabel"])
    ax.grid(alpha=0.25)

    y_top = positive_axis_top(q90)
    if spec["metric"] == "adaptive_stress":
        y_top = max(1.03, y_top)
    if spec["metric"] == "adaptive_stress_input_count" and np.isfinite(target_count):
        y_top = max(target_count * 1.03, y_top)
    ax.set_ylim(bottom=min(0.0, np.nanmin(q10) * 0.98), top=y_top)
    ax.legend(loc="upper right" if spec["metric"] != "adaptive_stress" else "upper left")


def plot_summary(summary, output_path, result_dir, target_count, midpoint_count, slope):
    fig, axes = plt.subplots(4, 1, figsize=(14, 13), sharex=True)
    fig.suptitle(
        "Adaptive Weighted EXPRESS diagnostics over time",
        fontsize=17,
        fontweight="bold",
    )

    for ax, spec in zip(axes, PLOT_SPECS):
        plot_metric(
            ax,
            summary,
            spec,
            target_count=target_count,
            midpoint_count=midpoint_count,
            slope=slope,
        )

    axes[-1].set_xlabel("t")
    fig.tight_layout(rect=(0, 0.02, 1, 0.97))

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def build_outputs(result_dir, output_dir=None, bin_width=250):
    result_dir = latest_results_dir() if result_dir is None else Path(result_dir)
    output_dir = result_dir / DEFAULT_OUTPUT_DIR_NAME if output_dir is None else Path(output_dir)
    config = load_result_config(result_dir)
    raw_df = load_raw_events(result_dir)
    target_count = target_low_distance_count(config, raw_df)
    midpoint_count = stress_midpoint_count(config)
    slope = stress_slope(config)

    run_bin_means = metric_run_bin_means(
        raw_df,
        bin_width=bin_width,
        target_count=target_count,
    )
    summary = summarize_run_bins(run_bin_means)

    output_dir.mkdir(parents=True, exist_ok=True)
    run_bin_means_path = output_dir / DEFAULT_RUN_BIN_MEANS_NAME
    summary_path = output_dir / DEFAULT_SUMMARY_NAME
    output_path = output_dir / DEFAULT_OUTPUT_NAME

    run_bin_means.to_csv(run_bin_means_path, index=False)
    summary.to_csv(summary_path, index=False)
    plot_summary(
        summary,
        output_path,
        result_dir,
        target_count=target_count,
        midpoint_count=midpoint_count,
        slope=slope,
    )

    return output_path, summary_path, run_bin_means_path, summary


def parse_args():
    parser = ArgumentParser(
        description="Plot binned selected-event Adaptive Weighted EXPRESS diagnostics."
    )
    parser.add_argument("--result-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--bin-width", type=int, default=250)
    return parser.parse_args()


def main():
    args = parse_args()
    output_path, summary_path, run_bin_means_path, summary = build_outputs(
        args.result_dir,
        args.output_dir,
        bin_width=args.bin_width,
    )
    print(f"Wrote plot to {output_path}")
    print(f"Wrote summary CSV to {summary_path}")
    print(f"Wrote run-bin means CSV to {run_bin_means_path}")
    print(
        f"Bins per metric={summary.groupby('metric')['t_bin'].nunique().to_dict()}"
    )


if __name__ == "__main__":
    main()
