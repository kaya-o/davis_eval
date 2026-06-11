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


def load_raw_events(result_dir):
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
    return raw_df


def strategy_run_values(raw_df, run, strategy, value_column):
    values = raw_df[
        (raw_df["run"] == run)
        & (raw_df["strategy"] == strategy)
    ][["t", value_column]].dropna()

    if values.empty:
        raise ValueError(f"No rows for run={run}, strategy={strategy}, column={value_column}")
    return values.sort_values("t")


def positive_axis_top(values, fallback=1.0):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return fallback
    top = np.nanmax(values) * 1.08
    return top if np.isfinite(top) and top > 0 else fallback


def plot_panel(ax, frame, value_column, color, title, ylabel):
    ax.plot(
        frame["t"],
        frame[value_column],
        color=color,
        linewidth=0.85,
        alpha=0.95,
    )
    ax.set_title(title, fontsize=10)
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.25)


def plot_run(raw_df, run, result_dir, output_dir, axis_tops):
    express_n = strategy_run_values(raw_df, run, EXPRESS_STRATEGY, "n_calibration")
    adaptive_stress = strategy_run_values(
        raw_df,
        run,
        ADAPTIVE_STRATEGY,
        "adaptive_weighted_express_stress",
    )
    adaptive_lambda = strategy_run_values(
        raw_df,
        run,
        ADAPTIVE_STRATEGY,
        "adaptive_weighted_express_lambda_t",
    )

    fig, axes = plt.subplots(3, 1, figsize=(12.5, 9), sharex=True)
    fig.suptitle(
        f"Selected-Event Diagnostics - {Path(result_dir).name}, Run {int(run)}",
        fontsize=13,
    )

    plot_panel(
        axes[0],
        express_n,
        "n_calibration",
        "#1f77b4",
        "EXPRESS calibration set size n",
        "n",
    )
    axes[0].set_ylim(bottom=0, top=axis_tops["n_calibration"])

    plot_panel(
        axes[1],
        adaptive_stress,
        "adaptive_weighted_express_stress",
        "#d62728",
        "Adaptive Weighted EXPRESS stress",
        "stress",
    )
    axes[1].set_ylim(
        bottom=0,
        top=axis_tops["adaptive_weighted_express_stress"],
    )

    plot_panel(
        axes[2],
        adaptive_lambda,
        "adaptive_weighted_express_lambda_t",
        "#2ca02c",
        "Adaptive Weighted EXPRESS lambda_t",
        r"$\lambda_t$",
    )
    axes[2].set_ylim(
        bottom=0,
        top=axis_tops["adaptive_weighted_express_lambda_t"],
    )
    axes[2].set_xlabel("Time t")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"run_{int(run):02d}_selected_event_adaptive_diagnostics.png"
    fig.tight_layout(rect=(0, 0.02, 1, 0.96))
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_all_runs(result_dir=None, output_dir=None):
    result_dir = latest_results_dir() if result_dir is None else Path(result_dir)
    output_dir = result_dir / DEFAULT_OUTPUT_DIR_NAME if output_dir is None else Path(output_dir)
    raw_df = load_raw_events(result_dir)
    axis_tops = {
        "n_calibration": positive_axis_top(
            raw_df.loc[raw_df["strategy"] == EXPRESS_STRATEGY, "n_calibration"],
        ),
        "adaptive_weighted_express_stress": max(
            1.02,
            positive_axis_top(
                raw_df.loc[
                    raw_df["strategy"] == ADAPTIVE_STRATEGY,
                    "adaptive_weighted_express_stress",
                ],
            ),
        ),
        "adaptive_weighted_express_lambda_t": positive_axis_top(
            raw_df.loc[
                raw_df["strategy"] == ADAPTIVE_STRATEGY,
                "adaptive_weighted_express_lambda_t",
            ],
        ),
    }

    output_paths = []
    for run in sorted(raw_df["run"].dropna().unique()):
        output_paths.append(plot_run(raw_df, run, result_dir, output_dir, axis_tops))
    return output_paths


def parse_args():
    parser = ArgumentParser(
        description=(
            "Write one three-panel selected-event diagnostics figure per run, "
            "without aggregating across runs."
        )
    )
    parser.add_argument("--result-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    output_paths = plot_all_runs(args.result_dir, args.output_dir)
    if not output_paths:
        raise ValueError("No run plots were written.")

    print(f"Wrote {len(output_paths)} run plots to {output_paths[0].parent}")


if __name__ == "__main__":
    main()
