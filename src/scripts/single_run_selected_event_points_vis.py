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


def load_result_config(result_dir):
    result_dir = Path(result_dir)
    for filename in ("resolved_config.json", "config.json"):
        path = result_dir / filename
        if path.exists():
            with path.open() as f:
                return json.load(f)
    return {}


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
    ]
    for column in [
        "adaptive_weighted_express_stress_count",
        "adaptive_weighted_express_express_n_calibration_for_stress",
    ]:
        if column in raw_columns:
            usecols.append(column)

    raw_df = pd.read_csv(raw_path, usecols=usecols)
    for column in usecols:
        if column != "strategy":
            raw_df[column] = pd.to_numeric(raw_df[column], errors="coerce")
    return raw_df


def stress_input_count(adaptive):
    if (
        "adaptive_weighted_express_stress_count" in adaptive.columns
        and adaptive["adaptive_weighted_express_stress_count"].notna().any()
    ):
        return adaptive["adaptive_weighted_express_stress_count"]
    if (
        "adaptive_weighted_express_express_n_calibration_for_stress" in adaptive.columns
        and adaptive["adaptive_weighted_express_express_n_calibration_for_stress"].notna().any()
    ):
        return adaptive["adaptive_weighted_express_express_n_calibration_for_stress"]
    return adaptive["n_calibration"]


def config_value(config, key, fallback=np.nan):
    return config.get("conformal", {}).get(key, fallback)


def positive_axis_top(values, fallback=1.0):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return fallback
    top = np.nanmax(values) * 1.08
    return top if np.isfinite(top) and top > 0 else fallback


def scatter_panel(ax, frame, column, title, ylabel, color):
    ax.scatter(
        frame["t"],
        frame[column],
        s=7,
        alpha=0.78,
        linewidths=0,
        color=color,
    )
    ax.set_title(title, loc="left", fontsize=12, fontweight="bold")
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.25)


def build_run_frame(raw_df, run):
    express = raw_df[
        (raw_df["run"] == run) & (raw_df["strategy"] == EXPRESS_STRATEGY)
    ][["t", "n_calibration"]].copy()
    express = express.rename(columns={"n_calibration": "express_n_calibration"})

    adaptive = raw_df[
        (raw_df["run"] == run) & (raw_df["strategy"] == ADAPTIVE_STRATEGY)
    ].copy()
    if adaptive.empty:
        raise ValueError(f"No {ADAPTIVE_STRATEGY} rows for run={run}")

    adaptive["adaptive_stress_input_count"] = stress_input_count(adaptive)
    adaptive = adaptive[
        [
            "t",
            "adaptive_weighted_express_lambda_t",
            "adaptive_weighted_express_stress",
            "adaptive_stress_input_count",
        ]
    ]

    frame = express.merge(adaptive, on="t", how="inner").sort_values("t")
    if frame.empty:
        raise ValueError(f"No matched EXPRESS/adaptive selected events for run={run}")
    return frame


def plot_run_points(frame, run, result_dir, output_path, config, x_max=None):
    target_count = float(
        config_value(config, "adaptive_weighted_express_target_low_distance_count", np.nan)
    )
    midpoint_count = float(
        config_value(config, "adaptive_weighted_express_stress_midpoint_count", np.nan)
    )

    fig, axes = plt.subplots(4, 1, figsize=(13.5, 11), sharex=True)
    fig.suptitle(
        f"Selected-event diagnostics without binning - {Path(result_dir).name}, run {int(run)}",
        fontsize=15,
        fontweight="bold",
    )

    scatter_panel(
        axes[0],
        frame,
        "express_n_calibration",
        "EXPRESS calibration size",
        "n",
        "#1f77b4",
    )
    axes[0].set_ylim(bottom=0, top=positive_axis_top(frame["express_n_calibration"]))

    scatter_panel(
        axes[1],
        frame,
        "adaptive_weighted_express_lambda_t",
        "Adaptive Weighted EXPRESS lambda_t",
        "lambda_t",
        "#2ca02c",
    )
    axes[1].set_ylim(bottom=0, top=positive_axis_top(frame["adaptive_weighted_express_lambda_t"]))

    scatter_panel(
        axes[2],
        frame,
        "adaptive_weighted_express_stress",
        "Adaptive stress",
        "stress",
        "#d62728",
    )
    axes[2].set_ylim(bottom=0, top=1.03)

    scatter_panel(
        axes[3],
        frame,
        "adaptive_stress_input_count",
        "Stress input count",
        "stress input count",
        "#9467bd",
    )
    if np.isfinite(target_count):
        axes[3].axhline(target_count, color="red", linestyle="--", linewidth=1.0, label="target")
    if np.isfinite(midpoint_count):
        axes[3].axhline(
            midpoint_count,
            color="black",
            linestyle=":",
            linewidth=1.2,
            label="sigmoid midpoint",
        )
    axes[3].set_ylim(
        bottom=0,
        top=max(
            positive_axis_top(frame["adaptive_stress_input_count"]),
            target_count * 1.03 if np.isfinite(target_count) else 0.0,
        ),
    )
    axes[3].legend(loc="upper right")
    axes[3].set_xlabel("t")

    for ax in axes:
        ax.set_xlim(left=0, right=x_max)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(rect=(0, 0.02, 1, 0.96))
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def build_outputs(result_dir=None, output_dir=None, run=0, x_max=None):
    result_dir = latest_results_dir() if result_dir is None else Path(result_dir)
    output_dir = result_dir / DEFAULT_OUTPUT_DIR_NAME if output_dir is None else Path(output_dir)
    raw_df = load_raw_events(result_dir)
    config = load_result_config(result_dir)
    frame = build_run_frame(raw_df, run)

    suffix = f"_xmax_{int(x_max)}" if x_max is not None else ""
    output_path = output_dir / f"run_{int(run):02d}_selected_event_points_unbinned{suffix}.png"
    csv_path = output_dir / f"run_{int(run):02d}_selected_event_points_unbinned{suffix}.csv"
    output_dir.mkdir(parents=True, exist_ok=True)
    frame.to_csv(csv_path, index=False)
    plot_run_points(frame, run, result_dir, output_path, config, x_max=x_max)
    return output_path, csv_path, frame


def parse_args():
    parser = ArgumentParser(
        description="Plot one run's selected-event diagnostics as unbinned points."
    )
    parser.add_argument("--result-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--run", type=int, default=0)
    parser.add_argument("--x-max", type=float, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    output_path, csv_path, frame = build_outputs(
        args.result_dir,
        args.output_dir,
        args.run,
        x_max=args.x_max,
    )
    print(f"Wrote plot to {output_path}")
    print(f"Wrote selected-event CSV to {csv_path}")
    print(
        f"Run {args.run}: selected events={len(frame)}, "
        f"t range={int(frame['t'].min())}-{int(frame['t'].max())}"
    )


if __name__ == "__main__":
    main()
