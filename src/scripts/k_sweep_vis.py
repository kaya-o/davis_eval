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
        metric_values = pd.concat([
            k_df[metric],
            pd.Series([baseline[metric]]),
        ]).astype(float)
        y_min = float(metric_values.min())
        y_max = float(metric_values.max())
        padding = 0.12 * (y_max - y_min) if y_max > y_min else max(0.01, 0.12 * y_max)

        ax.plot(
            x,
            k_df[metric],
            marker="o",
            markersize=5,
            linewidth=1.8,
            color="#2ca02c",
            label="K-EXPRESS",
        )
        ax.axhline(
            baseline[metric],
            color="#d62728",
            linestyle="--",
            linewidth=1.6,
            label="EXPRESS baseline",
        )
        ax.set_ylabel(ylabel)
        ax.set_ylim(max(0.0, y_min - padding), y_max + padding)
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
