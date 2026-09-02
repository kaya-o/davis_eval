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
EXPRESS_COMPARISON_OUTPUT_NAME = "k_sweep_calibration_timeseries_3x4_with_express.png"
EXPRESS_COMPARISON_4X3_OUTPUT_NAME = "k_sweep_calibration_timeseries_4x3_with_express.png"
THESIS_SMALL_K_OUTPUT_NAME = "k_sweep_calibration_timeseries_small_k_2x3.png"
THESIS_LARGE_K_OUTPUT_NAME = "k_sweep_calibration_timeseries_large_k_2x3.png"
THESIS_SMALL_K_VALUES = [100, 250, 500, 750, 1000, 2500]
THESIS_LARGE_K_VALUES = [5000, 7500, 10000, 15000, 20000]
STRATEGY = "K-EXPRESS"
EXPRESS_STRATEGY = "EXPRESS"


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


def discover_express_run_dir(suite_dir):
    express_run_dirs = []
    for path in Path(suite_dir).iterdir():
        raw_path = path / "raw_selected_events.csv"
        aggregate_path = path / "aggregate_results.csv"
        if not path.is_dir() or not raw_path.exists() or not aggregate_path.exists():
            continue
        aggregate = pd.read_csv(aggregate_path, usecols=["strategy"])
        if EXPRESS_STRATEGY in set(aggregate["strategy"]):
            express_run_dirs.append(path)

    if len(express_run_dirs) != 1:
        raise ValueError(
            f"Expected exactly one EXPRESS run under {suite_dir}, "
            f"found {len(express_run_dirs)}"
        )
    return express_run_dirs[0]


def summarize_express_run_dir(run_dir):
    raw_path = Path(run_dir) / "raw_selected_events.csv"
    raw_df = pd.read_csv(
        raw_path,
        usecols=["run", "t", "strategy", "n_calibration"],
    )
    raw_df = raw_df[raw_df["strategy"].eq(EXPRESS_STRATEGY)].copy()
    if raw_df.empty:
        raise ValueError(f"No {EXPRESS_STRATEGY} rows found in {raw_path}")

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
    summary["k"] = pd.NA
    summary["run_dir"] = Path(run_dir).name
    return summary


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


def plot_summary_with_express(
    k_summary,
    express_summary,
    output_path,
    nrows=3,
    ncols=4,
    figsize=(16, 9),
    ylabel="calibration set size",
    figure_title=None,
):
    k_values = sorted(k_summary["k"].unique())
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=figsize,
        sharex=True,
        sharey=True,
    )
    axes_flat = list(axes.flat)

    for ax, k_value in zip(axes_flat, k_values):
        panel = k_summary[k_summary["k"].eq(k_value)].sort_values("t")
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

    express_ax = axes_flat[len(k_values)]
    express_panel = express_summary.sort_values("t")
    express_ax.scatter(
        express_panel["t"],
        express_panel["mean_n_calibration"],
        s=3,
        alpha=0.75,
        color="#d62728",
        linewidths=0,
    )
    express_ax.set_title(EXPRESS_STRATEGY, fontsize=10)
    express_ax.grid(alpha=0.25)

    for ax in axes_flat[len(k_values) + 1:]:
        ax.axis("off")

    for ax in axes[:, 0]:
        ax.set_ylabel(ylabel)
    for ax in axes[-1, :]:
        if ax.has_data():
            ax.set_xlabel("t")

    if figure_title is not None:
        fig.suptitle(figure_title, fontsize=16, y=0.995)
        fig.tight_layout(rect=(0, 0, 1, 0.965))
    else:
        fig.tight_layout()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=250, bbox_inches="tight")
    plt.close(fig)


def calibration_axis_limits(k_summary, express_summary):
    combined = pd.concat([k_summary, express_summary], ignore_index=True)
    t_min = float(combined["t"].min())
    t_max = float(combined["t"].max())
    calibration_min = float(combined["mean_n_calibration"].min())
    calibration_max = float(combined["mean_n_calibration"].max())
    t_padding = 0.02 * (t_max - t_min)
    calibration_padding = 0.05 * (calibration_max - calibration_min)
    return (
        (t_min - t_padding, t_max + t_padding),
        (calibration_min - calibration_padding, calibration_max + calibration_padding),
    )


def plot_thesis_grid(panels, output_path, x_limits, y_limits):
    fig, axes = plt.subplots(2, 3, figsize=(12, 7), sharex=True, sharey=True)

    for ax, (title, panel, color) in zip(axes.flat, panels):
        panel = panel.sort_values("t")
        ax.scatter(
            panel["t"],
            panel["mean_n_calibration"],
            s=3,
            alpha=0.75,
            color=color,
            linewidths=0,
        )
        ax.set_title(title, fontsize=12)
        ax.set_xlim(*x_limits)
        ax.set_ylim(*y_limits)
        ax.tick_params(axis="both", labelsize=10)
        ax.grid(alpha=0.25)

    for ax in axes[:, 0]:
        ax.set_ylabel("calibration set size", fontsize=12)
    for ax in axes[-1, :]:
        ax.set_xlabel("t", fontsize=12)

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


def build_outputs_with_express(
    suite_dir=DEFAULT_SUITE_DIR,
    output_path=None,
    nrows=3,
    ncols=4,
    figsize=(16, 9),
    ylabel="calibration set size",
    figure_title=None,
):
    suite_dir = Path(suite_dir)
    output_path = (
        suite_dir / "vis" / EXPRESS_COMPARISON_OUTPUT_NAME
        if output_path is None
        else Path(output_path)
    )
    summary_path = output_path.with_suffix(".csv")

    k_summary = summarize_suite(suite_dir)
    express_summary = summarize_express_run_dir(discover_express_run_dir(suite_dir))

    k_summary["panel"] = k_summary["k"].map(lambda value: f"k={int(value)}")
    k_summary["strategy"] = STRATEGY
    express_summary["panel"] = EXPRESS_STRATEGY
    express_summary["strategy"] = EXPRESS_STRATEGY
    combined_summary = pd.concat([k_summary, express_summary], ignore_index=True)
    combined_summary = combined_summary[
        [
            "panel",
            "strategy",
            "k",
            "run_dir",
            "t",
            "mean_n_calibration",
            "contributing_runs",
            "selected_events",
        ]
    ]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    combined_summary.to_csv(summary_path, index=False)
    plot_summary_with_express(
        k_summary,
        express_summary,
        output_path,
        nrows=nrows,
        ncols=ncols,
        figsize=figsize,
        ylabel=ylabel,
        figure_title=figure_title,
    )
    return output_path, summary_path, combined_summary


def build_thesis_split_outputs(suite_dir=DEFAULT_SUITE_DIR):
    suite_dir = Path(suite_dir)
    vis_dir = suite_dir / "vis"
    small_output_path = vis_dir / THESIS_SMALL_K_OUTPUT_NAME
    large_output_path = vis_dir / THESIS_LARGE_K_OUTPUT_NAME

    k_summary = summarize_suite(suite_dir)
    express_summary = summarize_express_run_dir(discover_express_run_dir(suite_dir))
    available_k_values = set(k_summary["k"].astype(int))
    expected_k_values = set(THESIS_SMALL_K_VALUES + THESIS_LARGE_K_VALUES)
    if available_k_values != expected_k_values:
        raise ValueError(
            "Thesis split expects k values "
            f"{sorted(expected_k_values)}, found {sorted(available_k_values)}"
        )

    x_limits, y_limits = calibration_axis_limits(k_summary, express_summary)
    small_panels = [
        (
            f"k={k_value}",
            k_summary[k_summary["k"].eq(k_value)],
            "#1f77b4",
        )
        for k_value in THESIS_SMALL_K_VALUES
    ]
    large_panels = [
        (
            f"k={k_value}",
            k_summary[k_summary["k"].eq(k_value)],
            "#1f77b4",
        )
        for k_value in THESIS_LARGE_K_VALUES
    ]
    large_panels.append((EXPRESS_STRATEGY, express_summary, "#d62728"))

    plot_thesis_grid(
        small_panels,
        small_output_path,
        x_limits,
        y_limits,
    )
    plot_thesis_grid(
        large_panels,
        large_output_path,
        x_limits,
        y_limits,
    )

    k_summary = k_summary.copy()
    express_summary = express_summary.copy()
    k_summary["panel"] = k_summary["k"].map(lambda value: f"k={int(value)}")
    k_summary["strategy"] = STRATEGY
    express_summary["panel"] = EXPRESS_STRATEGY
    express_summary["strategy"] = EXPRESS_STRATEGY
    combined_summary = pd.concat([k_summary, express_summary], ignore_index=True)
    output_columns = [
        "panel",
        "strategy",
        "k",
        "run_dir",
        "t",
        "mean_n_calibration",
        "contributing_runs",
        "selected_events",
    ]
    small_summary = combined_summary[
        combined_summary["k"].isin(THESIS_SMALL_K_VALUES)
    ][output_columns]
    large_summary = combined_summary[
        combined_summary["k"].isin(THESIS_LARGE_K_VALUES)
        | combined_summary["strategy"].eq(EXPRESS_STRATEGY)
    ][output_columns]

    vis_dir.mkdir(parents=True, exist_ok=True)
    small_summary_path = small_output_path.with_suffix(".csv")
    large_summary_path = large_output_path.with_suffix(".csv")
    small_summary.to_csv(small_summary_path, index=False)
    large_summary.to_csv(large_summary_path, index=False)
    return (
        small_output_path,
        small_summary_path,
        large_output_path,
        large_summary_path,
    )


def main():
    parser = argparse.ArgumentParser(
        description="Plot K-EXPRESS calibration set size over time for each swept k."
    )
    parser.add_argument("--suite-dir", type=Path, default=DEFAULT_SUITE_DIR)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument(
        "--include-express",
        action="store_true",
        help="Use the final panel for the standalone EXPRESS run.",
    )
    parser.add_argument(
        "--thesis-split",
        action="store_true",
        help="Write two 2x3 figures split into small-k and large-k/EXPRESS panels.",
    )
    parser.add_argument(
        "--four-by-three",
        action="store_true",
        help="Write the K-EXPRESS/EXPRESS comparison as 4 rows by 3 columns.",
    )
    args = parser.parse_args()

    if sum([args.include_express, args.thesis_split, args.four_by_three]) > 1:
        parser.error(
            "--include-express, --thesis-split, and --four-by-three "
            "cannot be combined"
        )
    if args.four_by_three:
        output_path = (
            args.suite_dir / "vis" / EXPRESS_COMPARISON_4X3_OUTPUT_NAME
            if args.output is None
            else args.output
        )
        output_path, summary_path, summary = build_outputs_with_express(
            suite_dir=args.suite_dir,
            output_path=output_path,
            nrows=4,
            ncols=3,
            figsize=(12, 12),
            ylabel="Calibration Set Size",
            figure_title=(
                r"Calibration Set Size Over Time for Different $k$ Values"
            ),
        )
        print(f"Wrote 4x3 comparison plot to {output_path}")
        print(f"Wrote summary CSV to {summary_path}")
        print(f"panels={summary['panel'].nunique()}")
        return
    if args.thesis_split:
        if args.output is not None:
            parser.error("--output cannot be used with --thesis-split")
        (
            small_output_path,
            small_summary_path,
            large_output_path,
            large_summary_path,
        ) = build_thesis_split_outputs(suite_dir=args.suite_dir)
        print(f"Wrote small-k thesis plot to {small_output_path}")
        print(f"Wrote small-k summary CSV to {small_summary_path}")
        print(f"Wrote large-k thesis plot to {large_output_path}")
        print(f"Wrote large-k summary CSV to {large_summary_path}")
        return
    if args.include_express:
        output_path, summary_path, summary = build_outputs_with_express(
            suite_dir=args.suite_dir,
            output_path=args.output,
        )
        panel_summary = f"panels={summary['panel'].nunique()}"
    else:
        output_path, summary_path, summary = build_outputs(
            suite_dir=args.suite_dir,
            output_path=args.output,
        )
        panel_summary = f"k panels={summary['k'].nunique()}"
    print(f"Wrote plot to {output_path}")
    print(f"Wrote summary CSV to {summary_path}")
    print(panel_summary)


if __name__ == "__main__":
    main()
