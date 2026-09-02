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
DEFAULT_OUTPUT_NAME = "adaptive_weighted_express_stress_vs_nlow.png"
DEFAULT_SUMMARY_NAME = "adaptive_weighted_express_stress_vs_nlow.csv"
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


def load_adaptive_events(result_dir):
    raw_path = Path(result_dir) / "raw_selected_events.csv"
    if not raw_path.exists():
        raise FileNotFoundError(f"Missing {raw_path}")

    raw_columns = set(pd.read_csv(raw_path, nrows=0).columns)
    usecols = [
        "run",
        "t",
        "strategy",
        "adaptive_weighted_express_stress",
        "adaptive_weighted_express_n_low_distance",
    ]
    for column in [
        "adaptive_weighted_express_stress_count_source",
        "adaptive_weighted_express_stress_count",
        "adaptive_weighted_express_express_n_calibration_for_stress",
    ]:
        if column in raw_columns:
            usecols.append(column)

    raw_df = pd.read_csv(raw_path, usecols=usecols)
    adaptive = raw_df[raw_df["strategy"] == ADAPTIVE_STRATEGY].copy()
    for column in [
        "run",
        "t",
        "adaptive_weighted_express_stress",
        "adaptive_weighted_express_n_low_distance",
        "adaptive_weighted_express_stress_count",
        "adaptive_weighted_express_express_n_calibration_for_stress",
    ]:
        if column in adaptive.columns:
            adaptive[column] = pd.to_numeric(adaptive[column], errors="coerce")

    return adaptive.dropna(
        subset=[
            "adaptive_weighted_express_stress",
            "adaptive_weighted_express_n_low_distance",
        ]
    )


def sigmoid_stress(n_low_distance, midpoint_count, slope):
    n_low_distance = np.asarray(n_low_distance, dtype=float)
    z = slope * (n_low_distance - midpoint_count)
    return np.where(
        z >= 0,
        np.exp(-z) / (1.0 + np.exp(-z)),
        1.0 / (1.0 + np.exp(z)),
    )


def stress_input_column(adaptive):
    if (
        "adaptive_weighted_express_stress_count" in adaptive.columns
        and adaptive["adaptive_weighted_express_stress_count"].notna().any()
    ):
        return "adaptive_weighted_express_stress_count"
    return "adaptive_weighted_express_n_low_distance"


def summarize_by_stress_input(adaptive, input_column):
    return (
        adaptive.groupby(input_column, sort=True)
        .agg(
            events=("adaptive_weighted_express_stress", "size"),
            runs=("run", "nunique"),
            mean_stress=("adaptive_weighted_express_stress", "mean"),
            median_stress=("adaptive_weighted_express_stress", "median"),
            q10_stress=("adaptive_weighted_express_stress", lambda values: values.quantile(0.10)),
            q90_stress=("adaptive_weighted_express_stress", lambda values: values.quantile(0.90)),
        )
        .reset_index()
        .rename(columns={input_column: "stress_input_count"})
    )


def plot_stress_vs_nlow(adaptive, summary, output_path, config):
    conformal_config = config.get("conformal", {})
    stress_mode = conformal_config.get("adaptive_weighted_express_stress_mode", "linear")
    midpoint_count = float(
        conformal_config.get("adaptive_weighted_express_stress_midpoint_count", np.nan)
    )
    slope = float(conformal_config.get("adaptive_weighted_express_stress_slope", np.nan))
    input_column = stress_input_column(adaptive)
    source = conformal_config.get("adaptive_weighted_express_stress_count_source", "low_distance")

    x = adaptive[input_column].to_numpy(dtype=float)
    y = adaptive["adaptive_weighted_express_stress"].to_numpy(dtype=float)
    x_max = max(float(np.nanmax(x)), midpoint_count if np.isfinite(midpoint_count) else 0.0)
    x_curve = np.linspace(0.0, x_max, 600)

    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.scatter(
        x,
        y,
        s=8,
        alpha=0.10,
        linewidths=0,
        color="#1f77b4",
        label="selected events",
    )
    ax.plot(
        summary["stress_input_count"],
        summary["mean_stress"],
        color="#1f77b4",
        linewidth=2.0,
        label="mean by stress input count",
    )
    if stress_mode == "sigmoid" and np.isfinite(midpoint_count) and np.isfinite(slope):
        ax.plot(
            x_curve,
            sigmoid_stress(x_curve, midpoint_count, slope),
            color="black",
            linestyle="--",
            linewidth=1.5,
            label=f"configured sigmoid (midpoint={midpoint_count:g}, slope={slope:g})",
        )
        ax.axvline(
            midpoint_count,
            color="black",
            linestyle=":",
            linewidth=1.2,
            label="sigmoid midpoint",
        )

    ax.set_title("Adaptive Weighted EXPRESS stress vs stress input count", fontweight="bold")
    ax.set_xlabel(f"{input_column} (source={source})")
    ax.set_ylabel("adaptive_weighted_express_stress")
    ax.set_ylim(-0.03, 1.03)
    ax.grid(alpha=0.25)
    ax.legend(loc="best")
    fig.tight_layout()

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def build_outputs(result_dir, output_dir=None):
    result_dir = latest_results_dir() if result_dir is None else Path(result_dir)
    output_dir = result_dir / DEFAULT_OUTPUT_DIR_NAME if output_dir is None else Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    config = load_result_config(result_dir)
    adaptive = load_adaptive_events(result_dir)
    summary = summarize_by_stress_input(adaptive, stress_input_column(adaptive))

    output_path = output_dir / DEFAULT_OUTPUT_NAME
    summary_path = output_dir / DEFAULT_SUMMARY_NAME
    summary.to_csv(summary_path, index=False)
    plot_stress_vs_nlow(adaptive, summary, output_path, config)

    return output_path, summary_path, adaptive, summary


def parse_args():
    parser = ArgumentParser(
        description="Plot Adaptive Weighted EXPRESS stress against its low-distance-count input."
    )
    parser.add_argument("--result-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    output_path, summary_path, adaptive, summary = build_outputs(
        args.result_dir,
        args.output_dir,
    )
    print(f"Wrote plot to {output_path}")
    print(f"Wrote summary CSV to {summary_path}")
    print(f"Selected adaptive events={len(adaptive)}")
    print(f"Distinct n_low_distance values={len(summary)}")


if __name__ == "__main__":
    main()
