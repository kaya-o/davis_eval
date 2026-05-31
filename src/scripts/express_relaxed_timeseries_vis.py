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
STRATEGIES = ["RELAXED-EXPRESS", "EXPRESS"]
PLOT_SPECS = [
    ("RELAXED-EXPRESS", "Relaxed EXPRESS", "#1f77b4"),
    ("EXPRESS", "EXPRESS", "#d62728"),
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
    for filename in ("resolved_config.json", "config.json"):
        config_path = Path(result_dir) / filename
        if config_path.exists():
            with config_path.open() as f:
                return json.load(f)
    return {}


def total_runs_from_config_or_data(config, raw_df):
    if "n_runs" in config:
        return int(config["n_runs"])
    return int(raw_df["run"].nunique())


def read_raw_events(result_dir):
    raw_path = Path(result_dir) / "raw_selected_events.csv"
    if not raw_path.exists():
        raise FileNotFoundError(f"Missing {raw_path}")

    usecols = [
        "run",
        "t",
        "strategy",
        "interval_length",
        "n_calibration",
        "relaxed_express_added_nonexact",
    ]
    raw_df = pd.read_csv(raw_path, usecols=usecols)
    raw_df = raw_df[raw_df["strategy"].isin(STRATEGIES)].copy()
    raw_df["interval_length"] = pd.to_numeric(raw_df["interval_length"], errors="coerce")
    raw_df["n_calibration"] = pd.to_numeric(raw_df["n_calibration"], errors="coerce")
    return raw_df


def bool_series(series):
    if series.dtype == bool:
        return series
    return series.astype(str).str.lower().isin(["true", "1", "1.0"])


def focus_events(raw_df, subset):
    if subset == "all":
        return raw_df

    if subset != "relaxed-added-nonexact":
        raise ValueError(f"Unknown subset: {subset}")

    relaxed_df = raw_df[raw_df["strategy"] == "RELAXED-EXPRESS"].copy()
    added_mask = bool_series(relaxed_df["relaxed_express_added_nonexact"])
    added_events = relaxed_df.loc[added_mask, ["run", "t"]].drop_duplicates()
    return raw_df.merge(added_events, on=["run", "t"], how="inner")


def interval_mean(values, total_runs, inf_rule):
    values = values.to_numpy()
    inf_count = int(np.isinf(values).sum())
    finite_values = values[np.isfinite(values)]

    if inf_rule == "any":
        if inf_count > 0:
            return np.inf
        return finite_values.mean() if finite_values.size else np.nan

    if inf_rule == "all-runs":
        if inf_count == total_runs:
            return np.inf
        return finite_values.mean() if finite_values.size else np.nan

    if inf_rule == "ignore":
        return finite_values.mean() if finite_values.size else np.nan

    raise ValueError(f"Unknown infinite interval rule: {inf_rule}")


def aggregate_by_time(raw_df, total_runs, inf_rule):
    return (
        raw_df.groupby(["strategy", "t"], sort=True)
        .agg(
            mean_interval_length=(
                "interval_length",
                lambda values: interval_mean(values, total_runs, inf_rule),
            ),
            mean_n_calibration=("n_calibration", "mean"),
            n_events=("run", "count"),
            infinite_count=(
                "interval_length",
                lambda values: int(np.isinf(values.to_numpy()).sum()),
            ),
            finite_count=(
                "interval_length",
                lambda values: int(np.isfinite(values.to_numpy()).sum()),
            ),
        )
        .reset_index()
    )


def default_output_path(result_dir, subset):
    if subset == "relaxed-added-nonexact":
        return Path(result_dir) / "relaxed_express_added_nonexact_timeseries.png"
    return Path(result_dir) / "express_relaxed_express_timeseries_by_t.png"


def plot_timeseries(agg_df, raw_df, result_dir, output_path, subset, total_runs, inf_rule):
    finite_interval_means = agg_df.loc[
        np.isfinite(agg_df["mean_interval_length"]),
        "mean_interval_length",
    ].to_numpy()
    interval_top = (
        np.nanpercentile(finite_interval_means, 99.5) * 1.15
        if finite_interval_means.size
        else 1.0
    )
    if not np.isfinite(interval_top) or interval_top <= 0:
        interval_top = 1.0
    inf_marker_y = interval_top * 0.98

    calibration_top = agg_df["mean_n_calibration"].max() * 1.08
    if not np.isfinite(calibration_top) or calibration_top <= 0:
        calibration_top = 1.0

    fig, axes = plt.subplots(2, 2, figsize=(14, 8), sharex=True)
    if subset == "relaxed-added-nonexact":
        title = "RELAXED-EXPRESS Nonexact-Addition Events, Averaged by Time Across Runs"
    else:
        title = "EXPRESS vs RELAXED-EXPRESS, Selected Events Averaged by Time Across Runs"
    fig.suptitle(title, fontsize=13)

    for row, (strategy, label, color) in enumerate(PLOT_SPECS):
        strategy_df = agg_df[agg_df["strategy"] == strategy].sort_values("t")
        x = strategy_df["t"].to_numpy()

        ax_len = axes[row, 0]
        y = strategy_df["mean_interval_length"].to_numpy()
        finite = np.isfinite(y)
        ax_len.scatter(x[finite], y[finite], s=8, alpha=0.6, color=color, edgecolors="none")
        if np.any(np.isposinf(y)):
            marker_label = (
                f"all {total_runs} runs infinite"
                if inf_rule == "all-runs"
                else "mean is infinite"
            )
            ax_len.scatter(
                x[np.isposinf(y)],
                np.full(np.sum(np.isposinf(y)), inf_marker_y),
                s=16,
                alpha=0.75,
                marker="^",
                color="crimson",
                edgecolors="none",
                label=marker_label,
            )
            ax_len.legend(loc="upper right", fontsize=8)
        ax_len.set_title(f"{label}: Mean Interval Length")
        ax_len.set_ylabel("Interval length")
        ax_len.set_ylim(bottom=0, top=interval_top)
        ax_len.grid(alpha=0.25)

        ax_cal = axes[row, 1]
        ax_cal.scatter(
            x,
            strategy_df["mean_n_calibration"].to_numpy(),
            s=8,
            alpha=0.6,
            color=color,
            edgecolors="none",
        )
        ax_cal.set_title(f"{label}: Mean Calibration Set Size")
        ax_cal.set_ylabel("Calibration size")
        ax_cal.grid(alpha=0.25)

        if strategy == "RELAXED-EXPRESS":
            ax_cal.set_ylim(bottom=1.8, top=calibration_top)
            high_ticks = ax_cal.get_yticks()
            high_ticks = high_ticks[high_ticks >= 100]
            ax_cal.set_yticks(np.concatenate(([2.0], high_ticks)))
            ax_cal.axhline(2.0, color="black", linewidth=0.8, alpha=0.35)
        else:
            ax_cal.set_ylim(bottom=0, top=calibration_top)

    for ax in axes[-1, :]:
        ax.set_xlabel("Time t")

    fig.tight_layout(rect=(0, 0.02, 1, 0.95))

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return output_path


def parse_args():
    parser = ArgumentParser(description="Plot EXPRESS and RELAXED-EXPRESS metrics over time.")
    parser.add_argument(
        "--result-dir",
        type=Path,
        default=None,
        help="Result directory containing raw_selected_events.csv. Defaults to latest.",
    )
    parser.add_argument(
        "--subset",
        choices=["all", "relaxed-added-nonexact"],
        default="all",
        help="Which selected events to include.",
    )
    parser.add_argument(
        "--express-inf-rule",
        choices=["all-runs", "any", "ignore"],
        default="all-runs",
        help=(
            "How to aggregate infinite EXPRESS interval lengths at a fixed t. "
            "'all-runs' marks inf only if all configured runs are inf; otherwise "
            "finite values are averaged."
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional output image path. Defaults under the result directory.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    result_dir = latest_results_dir() if args.result_dir is None else Path(args.result_dir)
    raw_df = read_raw_events(result_dir)
    config = load_result_config(result_dir)
    total_runs = total_runs_from_config_or_data(config, raw_df)
    raw_df = focus_events(raw_df, args.subset)
    if raw_df.empty:
        raise ValueError(f"No rows remain after applying subset={args.subset!r}")

    agg_df = aggregate_by_time(raw_df, total_runs, args.express_inf_rule)
    output_path = args.output or default_output_path(result_dir, args.subset)
    output_path = plot_timeseries(
        agg_df,
        raw_df,
        result_dir,
        output_path,
        args.subset,
        total_runs,
        args.express_inf_rule,
    )

    express_agg = agg_df[agg_df["strategy"] == "EXPRESS"]
    print(f"Wrote plot to {output_path}")
    print(f"EXPRESS inf aggregated t: {int(np.isposinf(express_agg['mean_interval_length']).sum())}")
    print(f"EXPRESS finite aggregated t: {int(np.isfinite(express_agg['mean_interval_length']).sum())}")
    print(f"EXPRESS missing aggregated t: {int(express_agg['mean_interval_length'].isna().sum())}")


if __name__ == "__main__":
    main()
