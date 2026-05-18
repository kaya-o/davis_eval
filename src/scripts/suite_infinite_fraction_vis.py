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


DEFAULT_SUITE_DIR = PROJECT_ROOT / "results" / "suite_20260527_231925_window_width_sweep"
DEFAULT_STRATEGIES = ["EXPRESS", "K-EXPRESS", "EXPRESS-M"]
STRATEGY_COLORS = {
    "EXPRESS": "#d62728",
    "K-EXPRESS": "#2ca02c",
    "EXPRESS-M": "#9467bd",
}


def load_config(run_dir):
    for filename in ("resolved_config.json", "config.json"):
        config_path = Path(run_dir) / filename
        if config_path.exists():
            with config_path.open() as f:
                return json.load(f)
    raise FileNotFoundError(f"Missing config.json/resolved_config.json in {run_dir}")


def window_width_from_config(config):
    try:
        return float(config["selection"]["window_width"])
    except KeyError as exc:
        raise KeyError("Expected selection.window_width in run config") from exc


def discover_experiments(suite_dir):
    suite_dir = Path(suite_dir)
    run_dirs = [
        path
        for path in suite_dir.iterdir()
        if path.is_dir() and (path / "raw_selected_events.csv").exists()
    ]
    if not run_dirs:
        raise FileNotFoundError(f"No experiments with raw_selected_events.csv found in {suite_dir}")

    experiments = []
    for run_dir in run_dirs:
        config = load_config(run_dir)
        experiments.append({
            "run_dir": run_dir,
            "config": config,
            "window_width": window_width_from_config(config),
        })
    return sorted(experiments, key=lambda item: (item["window_width"], item["run_dir"].name))


def label_for_strategy(strategy, config):
    if strategy != "K-EXPRESS":
        return strategy

    k_express = config.get("conformal", {}).get("k_express")
    if k_express is None:
        return strategy
    if isinstance(k_express, float) and k_express.is_integer():
        k_express = int(k_express)
    return f"{k_express}-EXPRESS"


def summarize_experiment(experiment, strategies):
    raw_path = experiment["run_dir"] / "raw_selected_events.csv"
    raw_df = pd.read_csv(raw_path, usecols=["run", "strategy", "miscovered", "interval_length"])
    raw_df = raw_df[raw_df["strategy"].isin(strategies)].copy()
    raw_df["interval_length"] = pd.to_numeric(raw_df["interval_length"], errors="coerce")
    raw_df["is_infinite"] = np.isinf(raw_df["interval_length"].to_numpy())

    run_metrics = (
        raw_df.groupby(["run", "strategy"], as_index=False)
        .agg(
            infinite_fraction=("is_infinite", "mean"),
            miscoverage=("miscovered", "mean"),
            median_interval_length=("interval_length", "median"),
        )
    )

    summary = (
        run_metrics.groupby("strategy", as_index=False)
        [["infinite_fraction", "miscoverage", "median_interval_length"]]
        .mean()
    )
    summary["window_width"] = experiment["window_width"]
    summary["experiment"] = experiment["run_dir"].name
    return summary


def suite_summary(suite_dir, strategies):
    experiments = discover_experiments(suite_dir)
    summaries = [
        summarize_experiment(experiment, strategies)
        for experiment in experiments
    ]
    return pd.concat(summaries, ignore_index=True), experiments


def plot_infinite_fraction_by_window_width(suite_dir=DEFAULT_SUITE_DIR, output_path=None, strategies=None):
    suite_dir = Path(suite_dir)
    strategies = DEFAULT_STRATEGIES if strategies is None else strategies
    summary_df, experiments = suite_summary(suite_dir, strategies)

    if output_path is None:
        output_path = suite_dir / "vis" / "infinite_fraction_by_window_width.png"
    else:
        output_path = Path(output_path)

    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    first_config = experiments[0]["config"]

    for strategy in strategies:
        strategy_df = summary_df[summary_df["strategy"] == strategy].sort_values("window_width")
        if strategy_df.empty:
            continue
        ax.plot(
            strategy_df["window_width"],
            strategy_df["infinite_fraction"],
            marker="o",
            markersize=5,
            linewidth=1.8,
            color=STRATEGY_COLORS.get(strategy),
            label=label_for_strategy(strategy, first_config),
        )

    ax.set_xlabel("Window width")
    ax.set_ylabel("Infinite interval fraction")
    ax.set_ylim(bottom=0)
    ax.set_xticks(sorted(summary_df["window_width"].unique()))
    ax.grid(alpha=0.25)
    ax.legend(frameon=True)
    fig.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=250, bbox_inches="tight")
    plt.close(fig)

    csv_path = output_path.with_suffix(".csv")
    summary_df.sort_values(["window_width", "strategy"]).to_csv(csv_path, index=False)
    return output_path, csv_path


def plot_metrics_by_window_width(suite_dir=DEFAULT_SUITE_DIR, output_path=None, strategies=None):
    suite_dir = Path(suite_dir)
    strategies = DEFAULT_STRATEGIES if strategies is None else strategies
    summary_df, experiments = suite_summary(suite_dir, strategies)

    if output_path is None:
        output_path = suite_dir / "vis" / "suite_metrics_by_window_width.png"
    else:
        output_path = Path(output_path)

    metrics = [
        ("infinite_fraction", "Infinite interval fraction"),
        ("miscoverage", "Miscoverage"),
        ("median_interval_length", "Median interval length"),
    ]
    first_config = experiments[0]["config"]

    fig, axes = plt.subplots(3, 1, figsize=(8.8, 9.2), sharex=True)
    for ax, (metric, ylabel) in zip(axes, metrics):
        for strategy in strategies:
            strategy_df = summary_df[summary_df["strategy"] == strategy].sort_values("window_width")
            if strategy_df.empty:
                continue
            ax.plot(
                strategy_df["window_width"],
                strategy_df[metric],
                marker="o",
                markersize=5,
                linewidth=1.8,
                color=STRATEGY_COLORS.get(strategy),
                label=label_for_strategy(strategy, first_config),
            )
        ax.set_ylabel(ylabel)
        ax.set_ylim(bottom=0)
        ax.grid(alpha=0.25)

    axes[-1].set_xlabel("Window width")
    axes[-1].set_xticks(sorted(summary_df["window_width"].unique()))
    axes[0].legend(frameon=True, loc="upper right")
    fig.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=250, bbox_inches="tight")
    plt.close(fig)

    csv_path = output_path.with_suffix(".csv")
    summary_df.sort_values(["window_width", "strategy"]).to_csv(csv_path, index=False)
    return output_path, csv_path


def main():
    parser = argparse.ArgumentParser(
        description="Plot suite-level infinite interval fraction by window width."
    )
    parser.add_argument("--suite-dir", type=Path, default=DEFAULT_SUITE_DIR)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--metrics-output", type=Path, default=None)
    parser.add_argument("--strategies", nargs="+", default=DEFAULT_STRATEGIES)
    args = parser.parse_args()

    output_path, csv_path = plot_infinite_fraction_by_window_width(
        suite_dir=args.suite_dir,
        output_path=args.output,
        strategies=args.strategies,
    )
    print(f"Wrote plot to {output_path}")
    print(f"Wrote summary to {csv_path}")

    metrics_output_path, metrics_csv_path = plot_metrics_by_window_width(
        suite_dir=args.suite_dir,
        output_path=args.metrics_output,
        strategies=args.strategies,
    )
    print(f"Wrote metrics plot to {metrics_output_path}")
    print(f"Wrote metrics summary to {metrics_csv_path}")


if __name__ == "__main__":
    main()
