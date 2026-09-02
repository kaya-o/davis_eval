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
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator


DEFAULT_SUITE_DIR = (
    PROJECT_ROOT / "results" / "suite_20260819_130943_lambda_sweep"
)
DEFAULT_OUTPUT_NAME = "weighted_express_lambda_sweep_performance_metrics_3panel.png"
DEFAULT_SUMMARY_NAME = "weighted_express_lambda_sweep_performance_metrics_3panel.csv"
DEFAULT_STRATEGY = "WEIGHTED-EXPRESS"
DEFAULT_BASELINE_STRATEGIES = ("EXPRESS", "RELAXED-EXPRESS", "FULL")
METRICS = (
    ("miscoverage", "Miscoverage rate"),
    ("median_interval_length", "Median interval length"),
    ("infinite_fraction", "Infinite interval fraction"),
)

SWEEP_STYLE = {
    "color": "#0072B2",
    "linestyle": "-",
    "linewidth": 2.2,
    "marker": "o",
    "markersize": 5.8,
}
BASELINE_STYLES = {
    "EXPRESS": {
        "color": "#D55E00",
        "linestyle": (0, (6, 2)),
        "linewidth": 1.8,
    },
    "RELAXED-EXPRESS": {
        "color": "#CC79A7",
        "linestyle": (0, (4, 1.5, 1, 1.5)),
        "linewidth": 1.8,
    },
    "FULL": {
        "color": "#009E73",
        "linestyle": (0, (1, 1.5)),
        "linewidth": 2.0,
    },
}
TARGET_STYLE = {
    "color": "#D62728",
    "linestyle": (0, (2, 2)),
    "linewidth": 1.6,
}


def result_dirs(suite_dir):
    suite_dir = Path(suite_dir)
    dirs = sorted(
        path
        for path in suite_dir.iterdir()
        if path.is_dir() and (path / "aggregate_results.csv").exists()
    )
    if not dirs:
        raise FileNotFoundError(
            f"No result directories with aggregate_results.csv under {suite_dir}"
        )
    return dirs


def load_result_config(run_dir):
    for filename in ("resolved_config.json", "config.json"):
        config_path = Path(run_dir) / filename
        if config_path.exists():
            with config_path.open() as f:
                return json.load(f)
    raise FileNotFoundError(
        f"Missing resolved_config.json or config.json under {run_dir}"
    )


def numeric_config_value(config, section, key, *, required=False):
    value = config.get(section, {}).get(key)
    if value is None:
        if required:
            raise KeyError(f"Expected {section}.{key} in run config")
        return np.nan
    return float(value)


def summarize_suite(suite_dir, strategy, baseline_strategies):
    requested = [strategy, *baseline_strategies]
    records = []
    alpha_values = set()

    for run_dir in result_dirs(suite_dir):
        config = load_result_config(run_dir)
        alpha = numeric_config_value(config, "conformal", "alpha", required=True)
        alpha_values.add(alpha)
        aggregate = pd.read_csv(run_dir / "aggregate_results.csv")

        for requested_strategy in requested:
            rows = aggregate[aggregate["strategy"] == requested_strategy]
            if rows.empty:
                continue
            if len(rows) != 1:
                raise ValueError(
                    f"Expected one {requested_strategy} aggregate row in {run_dir}, "
                    f"found {len(rows)}"
                )

            record = rows.iloc[0].to_dict()
            record.update(
                {
                    "run_dir": run_dir.name,
                    "kind": (
                        "sweep" if requested_strategy == strategy else "baseline"
                    ),
                    "lambda": (
                        numeric_config_value(
                            config,
                            "conformal",
                            "weighted_express_lambda",
                            required=True,
                        )
                        if requested_strategy == strategy
                        else np.nan
                    ),
                    "relaxed_express_max_distance": (
                        numeric_config_value(
                            config,
                            "conformal",
                            "relaxed_express_max_distance",
                        )
                        if requested_strategy == "RELAXED-EXPRESS"
                        else np.nan
                    ),
                    "n_runs": int(config.get("n_runs", 0)),
                    "target_alpha": alpha,
                }
            )
            records.append(record)

    if len(alpha_values) != 1:
        raise ValueError(
            "Expected one common conformal alpha across sweep and baselines, "
            f"found {sorted(alpha_values)}"
        )
    if not records:
        raise ValueError(f"No requested aggregate rows found under {suite_dir}")

    summary = pd.DataFrame.from_records(records)
    for metric, _ in METRICS:
        summary[metric] = pd.to_numeric(summary[metric], errors="coerce")
        if summary[metric].isna().any():
            bad_dirs = summary.loc[summary[metric].isna(), "run_dir"].tolist()
            raise ValueError(f"Invalid {metric} values in {bad_dirs}")

    sweep = summary[summary["strategy"] == strategy]
    if len(sweep) < 2:
        raise ValueError(f"Expected at least two {strategy} lambda values")
    duplicate_lambdas = sweep.loc[sweep["lambda"].duplicated(), "lambda"].tolist()
    if duplicate_lambdas:
        raise ValueError(f"Duplicate lambda values for {strategy}: {duplicate_lambdas}")

    for baseline_strategy in baseline_strategies:
        count = int((summary["strategy"] == baseline_strategy).sum())
        if count != 1:
            raise ValueError(
                f"Expected exactly one {baseline_strategy} baseline row, found {count}"
            )

    strategy_order = {name: i for i, name in enumerate(requested)}
    summary["strategy_order"] = summary["strategy"].map(strategy_order)
    summary = summary.sort_values(
        ["kind", "strategy_order", "lambda"],
        ascending=[False, True, True],
        na_position="last",
    ).drop(columns="strategy_order")
    return summary.reset_index(drop=True)


def baseline_label(strategy, row):
    if strategy != "RELAXED-EXPRESS":
        return strategy
    radius = float(row["relaxed_express_max_distance"])
    if not np.isfinite(radius):
        return strategy
    return rf"RELAXED-EXPRESS ($r={radius:g}$)"


def nonnegative_ticks_and_limits(values):
    finite_values = np.asarray(values, dtype=float)
    finite_values = finite_values[np.isfinite(finite_values)]
    if len(finite_values) == 0:
        raise ValueError("Cannot determine y-axis limits from non-finite values")
    maximum = max(float(np.max(finite_values)), 1e-12)
    ticks = MaxNLocator(nbins=6, steps=[1, 2, 2.5, 5, 10]).tick_values(
        0.0, maximum * 1.06
    )
    ticks = ticks[ticks >= 0]
    upper = float(ticks[-1])
    return ticks, (-0.025 * upper, upper)


def legend_handles(summary, strategy, baseline_strategies, target_alpha):
    handles = [
        Line2D([], [], label=strategy, **SWEEP_STYLE),
    ]
    for baseline_strategy in baseline_strategies:
        row = summary[summary["strategy"] == baseline_strategy].iloc[0]
        style = BASELINE_STYLES.get(
            baseline_strategy,
            {"color": "#666666", "linestyle": "--", "linewidth": 1.8},
        )
        handles.append(
            Line2D(
                [],
                [],
                label=baseline_label(baseline_strategy, row),
                **style,
            )
        )
    handles.append(
        Line2D(
            [],
            [],
            label=rf"Target ($\alpha={target_alpha:g}$)",
            **TARGET_STYLE,
        )
    )
    return handles


def legend_handles_in_row_order(handles, ncols):
    rows = [handles[i : i + ncols] for i in range(0, len(handles), ncols)]
    return [
        row[column]
        for column in range(ncols)
        for row in rows
        if column < len(row)
    ]


def plot_summary(summary, output_path, strategy, baseline_strategies, dpi):
    sweep = summary[summary["strategy"] == strategy].sort_values("lambda")
    lambda_values = sweep["lambda"].to_numpy(dtype=float)
    x = np.arange(len(lambda_values), dtype=float)
    x_labels = [f"{value:g}" for value in lambda_values]
    target_alpha = float(summary["target_alpha"].iloc[0])

    with plt.rc_context(
        {
            "font.size": 10.5,
            "axes.labelsize": 11.5,
            "xtick.labelsize": 10.5,
            "ytick.labelsize": 10.5,
            "legend.fontsize": 10.0,
            "axes.linewidth": 0.9,
            "lines.solid_capstyle": "round",
            "lines.dash_capstyle": "butt",
        }
    ):
        fig = plt.figure(figsize=(10.8, 7.2))
        grid = fig.add_gridspec(2, 4)
        axes = [
            fig.add_subplot(grid[0, 0:2]),
            fig.add_subplot(grid[0, 2:4]),
            fig.add_subplot(grid[1, 1:3]),
        ]

        for panel_index, (ax, (metric, ylabel)) in enumerate(zip(axes, METRICS)):
            comparison_values = list(sweep[metric].to_numpy(dtype=float))
            for baseline_strategy in baseline_strategies:
                baseline = summary[summary["strategy"] == baseline_strategy].iloc[0]
                baseline_value = float(baseline[metric])
                comparison_values.append(baseline_value)
                style = BASELINE_STYLES.get(
                    baseline_strategy,
                    {
                        "color": "#666666",
                        "linestyle": "--",
                        "linewidth": 1.8,
                    },
                )
                ax.axhline(baseline_value, zorder=2, **style)

            if metric == "miscoverage":
                comparison_values.append(target_alpha)
                ax.axhline(target_alpha, zorder=1.5, **TARGET_STYLE)

            ax.plot(x, sweep[metric].to_numpy(dtype=float), zorder=3, **SWEEP_STYLE)
            ticks, limits = nonnegative_ticks_and_limits(comparison_values)
            ax.set_yticks(ticks)
            ax.set_ylim(*limits)
            ax.set_xlim(-0.25, len(x) - 0.75)
            ax.set_xticks(x, x_labels)
            ax.set_ylabel(ylabel)
            ax.grid(axis="y", color="#B0B0B0", linewidth=0.7, alpha=0.28)
            ax.set_axisbelow(True)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.text(
                0.02,
                0.97,
                f"({chr(ord('a') + panel_index)})",
                transform=ax.transAxes,
                ha="left",
                va="top",
                fontsize=11,
                fontweight="semibold",
            )

        handles = legend_handles(
            summary,
            strategy=strategy,
            baseline_strategies=baseline_strategies,
            target_alpha=target_alpha,
        )
        legend_ncols = 3
        fig.legend(
            handles=legend_handles_in_row_order(handles, legend_ncols),
            loc="upper center",
            bbox_to_anchor=(0.5, 0.995),
            ncol=legend_ncols,
            frameon=False,
            handlelength=2.8,
            columnspacing=1.45,
        )
        fig.supxlabel(r"Weight-decay parameter, $\lambda$", fontsize=11.5, y=0.025)
        fig.subplots_adjust(
            left=0.08,
            right=0.985,
            bottom=0.11,
            top=0.86,
            wspace=0.52,
            hspace=0.45,
        )

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(
            output_path,
            dpi=dpi,
            bbox_inches="tight",
            pad_inches=0.04,
            facecolor="white",
        )
        plt.close(fig)


def relative_to_project(path):
    path = Path(path)
    try:
        return path.resolve().relative_to(PROJECT_ROOT)
    except ValueError:
        return path


def write_recreation_script(
    output_path,
    suite_dir,
    summary_csv,
    strategy,
    baseline_strategies,
    dpi,
):
    output_path = Path(output_path)
    script_path = output_path.with_suffix(".sh")
    project_from_script = os.path.relpath(
        PROJECT_ROOT.resolve(),
        start=script_path.parent.resolve(),
    )
    command = [
        "python",
        "src/scripts/weighted_express_lambda_sweep_performance_metrics_vis.py",
        "--suite-dir",
        str(relative_to_project(suite_dir)),
        "--strategy",
        strategy,
        "--output",
        str(relative_to_project(output_path)),
        "--summary-csv",
        str(relative_to_project(summary_csv)),
        "--dpi",
        str(dpi),
    ]
    for baseline_strategy in baseline_strategies:
        command.extend(["--baseline-strategy", baseline_strategy])
    script = (
        "#!/usr/bin/env bash\n"
        "set -euo pipefail\n"
        f'cd "$(dirname "$0")/{project_from_script}"\n'
        + " ".join(shlex.quote(part) for part in command)
        + "\n"
    )
    script_path.write_text(script)
    script_path.chmod(0o755)
    return script_path


def parse_args():
    parser = ArgumentParser(
        description=(
            "Plot WEIGHTED-EXPRESS performance metrics over lambda with "
            "strategy baselines."
        )
    )
    parser.add_argument("--suite-dir", type=Path, default=DEFAULT_SUITE_DIR)
    parser.add_argument("--strategy", default=DEFAULT_STRATEGY)
    parser.add_argument(
        "--baseline-strategy",
        action="append",
        dest="baseline_strategies",
        default=None,
        help="Horizontal baseline strategy. Can be supplied more than once.",
    )
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--summary-csv", type=Path, default=None)
    parser.add_argument("--dpi", type=int, default=600)
    return parser.parse_args()


def main():
    args = parse_args()
    suite_dir = Path(args.suite_dir)
    vis_dir = suite_dir / "vis"
    output_path = args.output or vis_dir / DEFAULT_OUTPUT_NAME
    summary_csv = args.summary_csv or vis_dir / DEFAULT_SUMMARY_NAME
    baseline_strategies = tuple(
        args.baseline_strategies or DEFAULT_BASELINE_STRATEGIES
    )

    summary = summarize_suite(
        suite_dir,
        strategy=args.strategy,
        baseline_strategies=baseline_strategies,
    )
    output_columns = [
        "kind",
        "strategy",
        "lambda",
        "relaxed_express_max_distance",
        "run_dir",
        "n_runs",
        "selected",
        "miscovered",
        "miscoverage",
        "median_interval_length",
        "infinite_fraction",
        "target_alpha",
    ]
    summary_csv.parent.mkdir(parents=True, exist_ok=True)
    summary[output_columns].to_csv(summary_csv, index=False)
    plot_summary(
        summary,
        output_path=output_path,
        strategy=args.strategy,
        baseline_strategies=baseline_strategies,
        dpi=args.dpi,
    )
    script_path = write_recreation_script(
        output_path,
        suite_dir=suite_dir,
        summary_csv=summary_csv,
        strategy=args.strategy,
        baseline_strategies=baseline_strategies,
        dpi=args.dpi,
    )

    print(f"Wrote plot to {output_path}")
    print(f"Wrote summary CSV to {summary_csv}")
    print(f"Wrote recreation script to {script_path}")


if __name__ == "__main__":
    main()
