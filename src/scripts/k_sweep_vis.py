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


DEFAULT_SUITE_DIR = PROJECT_ROOT / "results" / "suite_20260528_101421_k_sweep"
METRICS = [
    ("infinite_fraction", "Infinite interval fraction"),
    ("median_interval_length", "Median interval length"),
    ("miscoverage", "Miscoverage"),
]


def load_config(run_dir):
    for filename in ("resolved_config.json", "config.json"):
        config_path = Path(run_dir) / filename
        if config_path.exists():
            with config_path.open() as f:
                return json.load(f)
    return {}


def discover_run_dirs(suite_dir):
    suite_dir = Path(suite_dir)
    run_dirs = [
        path
        for path in suite_dir.iterdir()
        if path.is_dir() and (path / "aggregate_results.csv").exists()
    ]
    if not run_dirs:
        raise FileNotFoundError(f"No aggregate_results.csv files found under {suite_dir}")
    return sorted(run_dirs)


def read_strategy_row(run_dir, strategy):
    aggregate_path = Path(run_dir) / "aggregate_results.csv"
    aggregate = pd.read_csv(aggregate_path)
    rows = aggregate[aggregate["strategy"].eq(strategy)]
    if rows.empty:
        return None
    return rows.iloc[0].to_dict()


def summarize_k_sweep(suite_dir):
    k_rows = []
    express_rows = []

    for run_dir in discover_run_dirs(suite_dir):
        config = load_config(run_dir)
        k_value = config.get("conformal", {}).get("k_express")

        k_row = read_strategy_row(run_dir, "K-EXPRESS")
        if k_row is not None:
            if k_value is None:
                raise ValueError(f"Missing conformal.k_express in {run_dir}")
            k_row.update({
                "run_dir": run_dir.name,
                "k": int(k_value),
                "series": "K-EXPRESS",
            })
            k_rows.append(k_row)

        express_row = read_strategy_row(run_dir, "EXPRESS")
        if express_row is not None:
            express_row.update({
                "run_dir": run_dir.name,
                "k": None,
                "series": "EXPRESS baseline",
            })
            express_rows.append(express_row)

    if not k_rows:
        raise ValueError(f"No K-EXPRESS rows found under {suite_dir}")
    if len(express_rows) != 1:
        raise ValueError(
            f"Expected exactly one EXPRESS baseline row under {suite_dir}, found {len(express_rows)}"
        )

    k_df = pd.DataFrame(k_rows).sort_values("k")
    baseline = express_rows[0]
    return k_df, baseline


def metric_axis_layout(k_values, baseline_value):
    k_values = np.asarray(k_values, dtype=float)
    baseline_value = float(baseline_value)
    all_values = np.concatenate([k_values, [baseline_value]])
    finite_values = all_values[np.isfinite(all_values)]

    if finite_values.size:
        y_min = float(np.min(finite_values))
        y_max = float(np.max(finite_values))
    else:
        y_min, y_max = 0.0, 1.0

    padding = (
        0.12 * (y_max - y_min)
        if y_max > y_min
        else max(0.01, 0.12 * abs(y_max))
    )
    y_bottom = max(0.0, y_min - padding)
    infinity_level = None
    y_top = y_max + padding

    if np.any(np.isinf(all_values)):
        infinity_gap = max(2.5 * padding, 0.08 * max(abs(y_max), 1.0), 0.05)
        infinity_level = y_max + infinity_gap
        y_top = infinity_level + 0.6 * infinity_gap

    return y_bottom, y_top, y_max, padding, infinity_level


def add_infinity_axis_row(ax, infinity_level, numeric_top):
    numeric_ticks = [tick for tick in ax.get_yticks() if tick <= numeric_top]
    ax.set_yticks([*numeric_ticks, infinity_level])
    ax.set_yticklabels([*[f"{tick:g}" for tick in numeric_ticks], r"$\infty$"])


def plot_k_sweep(suite_dir=DEFAULT_SUITE_DIR, output_path=None):
    suite_dir = Path(suite_dir)
    k_df, baseline = summarize_k_sweep(suite_dir)

    if output_path is None:
        output_path = suite_dir / "vis" / "k_sweep_metrics.png"
    else:
        output_path = Path(output_path)

    x = range(len(k_df))
    x_labels = [str(k) for k in k_df["k"]]

    fig, axes = plt.subplots(3, 1, figsize=(10, 9.2), sharex=True)
    for ax, (metric, ylabel) in zip(axes, METRICS):
        k_values = k_df[metric].to_numpy(dtype=float)
        baseline_value = float(baseline[metric])
        y_bottom, y_top, finite_y_max, padding, infinity_level = (
            metric_axis_layout(k_values, baseline_value)
        )

        ax.plot(
            x,
            np.where(np.isfinite(k_values), k_values, np.nan),
            marker="o",
            markersize=5,
            linewidth=1.8,
            color="#2ca02c",
            label="K-EXPRESS",
        )
        if infinity_level is not None:
            infinite_k = np.flatnonzero(np.isinf(k_values))
            ax.scatter(
                infinite_k,
                np.full(len(infinite_k), infinity_level),
                marker="^",
                s=38,
                color="#2ca02c",
                zorder=4,
            )
        if np.isfinite(baseline_value):
            baseline_y = baseline_value
        elif np.isinf(baseline_value):
            baseline_y = infinity_level
        else:
            baseline_y = None
        if baseline_y is not None:
            ax.axhline(
                baseline_y,
                color="#d62728",
                linestyle="--",
                linewidth=1.6,
                label="EXPRESS baseline",
            )
        ax.set_ylabel(ylabel)
        ax.set_ylim(y_bottom, y_top)
        if infinity_level is not None:
            add_infinity_axis_row(
                ax,
                infinity_level,
                finite_y_max + padding,
            )
        ax.grid(alpha=0.25)

    axes[-1].set_xlabel("k")
    axes[-1].set_xticks(list(x))
    axes[-1].set_xticklabels(x_labels, rotation=35, ha="right")
    axes[0].legend(frameon=True, loc="best")
    fig.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=250, bbox_inches="tight")
    plt.close(fig)

    summary_path = output_path.with_suffix(".csv")
    baseline_df = pd.DataFrame([baseline])
    pd.concat([k_df, baseline_df], ignore_index=True).to_csv(summary_path, index=False)
    return output_path, summary_path


def main():
    parser = argparse.ArgumentParser(
        description="Plot K-EXPRESS k-sweep metrics with EXPRESS baseline."
    )
    parser.add_argument("--suite-dir", type=Path, default=DEFAULT_SUITE_DIR)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    output_path, summary_path = plot_k_sweep(
        suite_dir=args.suite_dir,
        output_path=args.output,
    )
    print(f"Wrote k-sweep plot to {output_path}")
    print(f"Wrote k-sweep summary to {summary_path}")


if __name__ == "__main__":
    main()
