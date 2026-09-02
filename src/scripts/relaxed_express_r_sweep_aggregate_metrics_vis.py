from argparse import ArgumentParser
import json
import os
from pathlib import Path
import shlex

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/davis_eval_matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/private/tmp/davis_eval_cache")

import matplotlib.pyplot as plt


DEFAULT_SUITE_DIR = PROJECT_ROOT / "results" / "suite_20260612_022224_relaxed_express_r_sweep_easy"
DEFAULT_OUTPUT_NAME = "relaxed_express_r_sweep_aggregate_metrics_4panel.png"
DEFAULT_SUMMARY_NAME = "relaxed_express_r_sweep_aggregate_metrics_4panel.csv"
DEFAULT_STRATEGY = "RELAXED-EXPRESS"
DEFAULT_BASELINE_STRATEGIES = ("EXPRESS", "FULL")
METRICS = [
    ("miscoverage", "miscoverage"),
    ("median_interval_length", "median interval length"),
    ("avg_n_calibration", "calibration set size"),
    ("infinite_fraction", "infinite interval fraction"),
]


def result_dirs(suite_dir):
    suite_dir = Path(suite_dir)
    dirs = [
        path
        for path in suite_dir.iterdir()
        if path.is_dir() and (path / "aggregate_results.csv").exists()
    ]
    if not dirs:
        raise FileNotFoundError(f"No result directories with aggregate_results.csv under {suite_dir}")
    return sorted(dirs)


def load_result_config(run_dir):
    for filename in ("resolved_config.json", "config.json"):
        config_path = Path(run_dir) / filename
        if config_path.exists():
            with config_path.open() as f:
                return json.load(f)
    raise FileNotFoundError(f"Missing config.json or resolved_config.json under {run_dir}")


def relaxed_express_max_distance_from_config(config):
    value = config.get("conformal", {}).get("relaxed_express_max_distance")
    if value is None:
        raise KeyError("Expected conformal.relaxed_express_max_distance in run config")
    return float(value)


def target_alpha_from_configs(configs):
    values = {
        float(config.get("conformal", {}).get("alpha"))
        for config in configs
        if config.get("conformal", {}).get("alpha") is not None
    }
    return values.pop() if len(values) == 1 else None


def summarize_suite(suite_dir, strategy, baseline_strategies):
    rows = []
    configs = []
    keep_strategies = [strategy, *baseline_strategies]
    for run_dir in result_dirs(suite_dir):
        config = load_result_config(run_dir)
        configs.append(config)
        r_value = relaxed_express_max_distance_from_config(config)
        aggregate = pd.read_csv(run_dir / "aggregate_results.csv")
        strategy_df = aggregate[aggregate["strategy"].isin(keep_strategies)].copy()
        if strategy_df.empty:
            continue

        strategy_df["r"] = r_value
        strategy_df["run_dir"] = run_dir.name
        rows.append(strategy_df)

    if not rows:
        raise ValueError(f"No requested aggregate rows found under {suite_dir}")

    summary = pd.concat(rows, ignore_index=True)
    for metric, _ in METRICS:
        summary[metric] = pd.to_numeric(summary[metric], errors="coerce")
    summary["target_alpha"] = target_alpha_from_configs(configs)
    return summary.sort_values(["r", "strategy"]).reset_index(drop=True)


def representative_baseline_row(summary, baseline_strategy):
    baseline_rows = summary[summary["strategy"] == baseline_strategy]
    if baseline_rows.empty:
        return None
    zero_rows = baseline_rows[baseline_rows["r"] == 0]
    return zero_rows.iloc[0] if not zero_rows.empty else baseline_rows.iloc[0]


def plot_summary(summary, output_path, strategy, baseline_strategies):
    strategy_summary = summary[summary["strategy"] == strategy].copy()
    if strategy_summary.empty:
        raise ValueError(f"No {strategy!r} rows found in summary")

    r_values = strategy_summary["r"].to_numpy(dtype=float)
    x = np.arange(len(r_values))
    x_labels = [f"{value:g}" for value in r_values]

    fig, axes = plt.subplots(2, 2, figsize=(12.5, 8.2), sharex=True)
    axes_flat = list(axes.flat)
    baseline_styles = {
        "EXPRESS": ("#4d4d4d", "--"),
        "FULL": ("#009e73", "--"),
    }

    for ax, (metric, ylabel) in zip(axes_flat, METRICS):
        for baseline_strategy in baseline_strategies:
            baseline = representative_baseline_row(summary, baseline_strategy)
            if baseline is None:
                continue
            baseline_y = baseline[metric]
            baseline_y = np.nan if np.isinf(baseline_y) else float(baseline_y)
            if np.isfinite(baseline_y):
                color, linestyle = baseline_styles.get(baseline_strategy, ("#4d4d4d", "--"))
                ax.axhline(
                    baseline_y,
                    color=color,
                    linestyle=linestyle,
                    linewidth=1.3,
                    label=f"{baseline_strategy} baseline",
                )

        y = strategy_summary[metric].replace([np.inf, -np.inf], np.nan).to_numpy(dtype=float)
        ax.plot(
            x,
            y,
            marker="o",
            markersize=5,
            linewidth=1.7,
            color="#1f77b4",
            label=strategy if metric == "miscoverage" else None,
        )
        if metric == "miscoverage":
            target_alpha = summary["target_alpha"].dropna().unique()
            if len(target_alpha) == 1:
                ax.axhline(
                    float(target_alpha[0]),
                    color="tab:red",
                    linestyle="--",
                    linewidth=1.1,
                    label=f"target ({float(target_alpha[0]):g})",
                )
            ax.legend(frameon=False, fontsize=8, loc="best")
        ax.set_ylabel(ylabel)
        ax.set_ylim(bottom=0)
        ax.grid(alpha=0.25)

    for ax in axes[-1, :]:
        ax.set_xlabel("maximum signature distance r")
        ax.set_xticks(x)
        ax.set_xticklabels(x_labels)
    for ax in axes[0, :]:
        ax.tick_params(labelbottom=True)
        ax.set_xticks(x)
        ax.set_xticklabels(x_labels)

    fig.suptitle(f"{strategy} aggregate metrics over maximum distance", y=0.995, fontsize=13)
    fig.tight_layout()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def relative_to_project(path):
    path = Path(path)
    try:
        return path.resolve().relative_to(PROJECT_ROOT)
    except ValueError:
        return path


def write_recreation_script(output_path, suite_dir, summary_csv, strategy, baseline_strategies):
    output_path = Path(output_path)
    script_path = output_path.with_suffix(".sh")
    script_dir_to_project = Path("../../..")
    command = [
        "python3",
        "src/scripts/relaxed_express_r_sweep_aggregate_metrics_vis.py",
        "--suite-dir",
        str(relative_to_project(suite_dir)),
        "--strategy",
        strategy,
        "--output",
        str(relative_to_project(output_path)),
        "--summary-csv",
        str(relative_to_project(summary_csv)),
    ]
    for baseline_strategy in baseline_strategies:
        command.extend(["--baseline-strategy", baseline_strategy])
    script = (
        "#!/usr/bin/env bash\n"
        "set -euo pipefail\n"
        'cd "$(dirname "$0")/'
        f"{script_dir_to_project}"
        '"\n'
        + " ".join(shlex.quote(part) for part in command)
        + "\n"
    )
    script_path.write_text(script)
    script_path.chmod(0o755)
    return script_path


def parse_args():
    parser = ArgumentParser(
        description="Plot RELAXED-EXPRESS aggregate metrics over a max-distance radius sweep."
    )
    parser.add_argument("--suite-dir", type=Path, default=DEFAULT_SUITE_DIR)
    parser.add_argument("--strategy", default=DEFAULT_STRATEGY)
    parser.add_argument(
        "--baseline-strategy",
        action="append",
        dest="baseline_strategies",
        default=None,
        help="Strategy to draw as a dashed horizontal baseline. Can be passed more than once.",
    )
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--summary-csv", type=Path, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    suite_dir = Path(args.suite_dir)
    vis_dir = suite_dir / "vis"
    output_path = args.output or vis_dir / DEFAULT_OUTPUT_NAME
    summary_csv = args.summary_csv or vis_dir / DEFAULT_SUMMARY_NAME

    baseline_strategies = args.baseline_strategies or list(DEFAULT_BASELINE_STRATEGIES)
    summary = summarize_suite(
        suite_dir,
        strategy=args.strategy,
        baseline_strategies=baseline_strategies,
    )
    summary = summary[
        [
            "r",
            "run_dir",
            "strategy",
            "selected",
            "miscovered",
            "miscoverage",
            "avg_n_calibration",
            "median_interval_length",
            "infinite_fraction",
            "target_alpha",
        ]
    ]
    summary_csv.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(summary_csv, index=False)
    plot_summary(summary, output_path, args.strategy, baseline_strategies)
    script_path = write_recreation_script(
        output_path,
        suite_dir,
        summary_csv,
        args.strategy,
        baseline_strategies,
    )

    print(f"Wrote plot to {output_path}")
    print(f"Wrote summary CSV to {summary_csv}")
    print(f"Wrote recreation script to {script_path}")


if __name__ == "__main__":
    main()
