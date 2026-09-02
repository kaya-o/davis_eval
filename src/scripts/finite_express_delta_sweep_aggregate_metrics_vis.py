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
from matplotlib.ticker import MaxNLocator, NullLocator


DEFAULT_SUITE_DIR = (
    PROJECT_ROOT
    / "results"
    / "suite_20260824_220339_fin_express_delta_sweep"
)
DEFAULT_OUTPUT_NAME = "finite_express_delta_sweep_aggregate_metrics_3panel.png"
DEFAULT_SUMMARY_NAME = "finite_express_delta_sweep_aggregate_metrics_3panel.csv"
SWEEP_STRATEGY = "FINITE-EXPRESS"
BASELINE_STRATEGY = "EXPRESS"
METRICS = (
    ("miscoverage", "Miscoverage rate"),
    ("median_interval_length", "Median interval length"),
    ("avg_n_calibration", "Mean calibration set size"),
)
SUMMARY_METRICS = (*[metric for metric, _ in METRICS], "infinite_fraction")

SWEEP_STYLE = {
    "color": "#0072B2",
    "linestyle": "-",
    "linewidth": 2.2,
    "marker": "o",
    "markersize": 5.8,
}
BASELINE_STYLE = {
    "color": "#D55E00",
    "linestyle": (0, (6, 2)),
    "linewidth": 1.8,
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
            with config_path.open() as handle:
                return json.load(handle)
    raise FileNotFoundError(
        f"Missing resolved_config.json or config.json under {run_dir}"
    )


def required_numeric_config_value(config, section, key):
    value = config.get(section, {}).get(key)
    if value is None:
        raise KeyError(f"Expected {section}.{key} in run config")
    return float(value)


def one_strategy_row(aggregate, strategy, run_dir):
    rows = aggregate[aggregate["strategy"] == strategy]
    if len(rows) != 1:
        raise ValueError(
            f"Expected exactly one {strategy} row in {run_dir}, found {len(rows)}"
        )
    return rows.iloc[0].to_dict()


def summarize_suite(suite_dir):
    records = []
    alpha_values = set()
    n_run_values = set()

    for run_dir in result_dirs(suite_dir):
        config = load_result_config(run_dir)
        alpha = required_numeric_config_value(config, "conformal", "alpha")
        alpha_values.add(alpha)
        n_run_values.add(int(config.get("n_runs", 0)))

        aggregate = pd.read_csv(run_dir / "aggregate_results.csv")
        present_strategies = set(aggregate["strategy"].astype(str))
        requested_present = present_strategies & {SWEEP_STRATEGY, BASELINE_STRATEGY}
        if not requested_present:
            continue
        if len(requested_present) != 1:
            raise ValueError(
                f"Expected one requested strategy per result directory in {run_dir}, "
                f"found {sorted(requested_present)}"
            )

        strategy = requested_present.pop()
        record = one_strategy_row(aggregate, strategy, run_dir)
        record.update(
            {
                "kind": "sweep" if strategy == SWEEP_STRATEGY else "baseline",
                "delta": (
                    required_numeric_config_value(
                        config,
                        "conformal",
                        "finite_express_rank_delta",
                    )
                    if strategy == SWEEP_STRATEGY
                    else np.nan
                ),
                "run_dir": run_dir.name,
                "n_runs": int(config.get("n_runs", 0)),
                "target_alpha": alpha,
            }
        )
        records.append(record)

    if not records:
        raise ValueError(f"No {SWEEP_STRATEGY} or {BASELINE_STRATEGY} rows found")
    if len(alpha_values) != 1:
        raise ValueError(
            "Expected one common conformal alpha across the sweep and baseline, "
            f"found {sorted(alpha_values)}"
        )
    if len(n_run_values) != 1 or 0 in n_run_values:
        raise ValueError(
            "Expected one common positive n_runs value across the sweep and baseline, "
            f"found {sorted(n_run_values)}"
        )

    summary = pd.DataFrame.from_records(records)
    for metric in SUMMARY_METRICS:
        summary[metric] = pd.to_numeric(summary[metric], errors="coerce")
        invalid = ~np.isfinite(summary[metric].to_numpy(dtype=float))
        if invalid.any():
            bad_dirs = summary.loc[invalid, "run_dir"].tolist()
            raise ValueError(f"Invalid {metric} values in {bad_dirs}")

    sweep = summary[summary["strategy"] == SWEEP_STRATEGY]
    if len(sweep) < 2:
        raise ValueError(f"Expected at least two {SWEEP_STRATEGY} delta values")
    if (sweep["delta"] <= 0).any():
        bad_values = sweep.loc[sweep["delta"] <= 0, "delta"].tolist()
        raise ValueError(f"Delta values must be positive for a logarithmic axis: {bad_values}")
    duplicate_deltas = sweep.loc[sweep["delta"].duplicated(), "delta"].tolist()
    if duplicate_deltas:
        raise ValueError(f"Duplicate delta values for {SWEEP_STRATEGY}: {duplicate_deltas}")

    baseline_count = int((summary["strategy"] == BASELINE_STRATEGY).sum())
    if baseline_count != 1:
        raise ValueError(
            f"Expected exactly one {BASELINE_STRATEGY} baseline row, "
            f"found {baseline_count}"
        )

    summary["kind_order"] = summary["kind"].map({"sweep": 0, "baseline": 1})
    summary = summary.sort_values(
        ["kind_order", "delta"],
        na_position="last",
    ).drop(columns="kind_order")
    return summary.reset_index(drop=True)


def nonnegative_ticks_and_limits(values):
    finite_values = np.asarray(values, dtype=float)
    finite_values = finite_values[np.isfinite(finite_values)]
    if len(finite_values) == 0:
        raise ValueError("Cannot determine y-axis limits from non-finite values")
    maximum = max(float(np.max(finite_values)), 1e-12)
    ticks = MaxNLocator(nbins=6, steps=[1, 2, 2.5, 5, 10]).tick_values(
        0.0,
        maximum * 1.06,
    )
    ticks = ticks[ticks >= 0]
    upper = float(ticks[-1])
    return ticks, (-0.025 * upper, upper)


def plot_summary(summary, output_path, dpi):
    sweep = summary[summary["strategy"] == SWEEP_STRATEGY].sort_values("delta")
    baseline = summary[summary["strategy"] == BASELINE_STRATEGY].iloc[0]
    delta_values = sweep["delta"].to_numpy(dtype=float)
    delta_labels = [f"{value:g}" for value in delta_values]
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

        for panel_index, (ax, (metric, ylabel)) in enumerate(
            zip(axes, METRICS)
        ):
            sweep_values = sweep[metric].to_numpy(dtype=float)
            baseline_value = float(baseline[metric])
            comparison_values = [*sweep_values, baseline_value]

            ax.axhline(baseline_value, zorder=2, **BASELINE_STYLE)
            if metric == "miscoverage":
                comparison_values.append(target_alpha)
                ax.axhline(target_alpha, zorder=1.5, **TARGET_STYLE)

            ax.plot(delta_values, sweep_values, zorder=3, **SWEEP_STYLE)
            if metric == "miscoverage":
                ticks = np.arange(0.075, 0.1251, 0.01)
                limits = (0.075, 0.125)
            else:
                ticks, limits = nonnegative_ticks_and_limits(comparison_values)
            ax.set_yticks(ticks)
            ax.set_ylim(*limits)
            ax.set_xscale("log")
            ax.set_xlim(delta_values.min() / 1.3, delta_values.max() * 1.3)
            ax.set_xticks(delta_values)
            ax.set_xticklabels(delta_labels)
            ax.xaxis.set_minor_locator(NullLocator())
            ax.set_ylabel(ylabel)
            ax.grid(axis="y", color="#B0B0B0", linewidth=0.7, alpha=0.28)
            ax.set_axisbelow(True)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.text(
                0.98,
                0.97,
                f"({chr(ord('a') + panel_index)})",
                transform=ax.transAxes,
                ha="right",
                va="top",
                fontsize=11,
                fontweight="semibold",
            )

        legend_handles = [
            Line2D([], [], label=SWEEP_STRATEGY, **SWEEP_STYLE),
            Line2D([], [], label=BASELINE_STRATEGY, **BASELINE_STYLE),
            Line2D(
                [],
                [],
                label=rf"Target ($\alpha={target_alpha:g}$)",
                **TARGET_STYLE,
            ),
        ]
        fig.legend(
            handles=legend_handles,
            loc="upper center",
            bbox_to_anchor=(0.5, 0.992),
            ncol=3,
            frameon=False,
            handlelength=2.8,
            columnspacing=1.6,
        )
        fig.supxlabel(r"Rank-resolution parameter, $\delta$", fontsize=11.5, y=0.025)
        fig.subplots_adjust(
            left=0.09,
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


def write_recreation_script(output_path, suite_dir, summary_csv, dpi):
    output_path = Path(output_path)
    script_path = output_path.with_suffix(".sh")
    project_from_script = os.path.relpath(
        PROJECT_ROOT.resolve(),
        start=script_path.parent.resolve(),
    )
    command = [
        "python",
        "src/scripts/finite_express_delta_sweep_aggregate_metrics_vis.py",
        "--suite-dir",
        str(relative_to_project(suite_dir)),
        "--output",
        str(relative_to_project(output_path)),
        "--summary-csv",
        str(relative_to_project(summary_csv)),
        "--dpi",
        str(dpi),
    ]
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
            "Plot FINITE-EXPRESS performance metrics over rank-resolution delta "
            "with an EXPRESS benchmark."
        )
    )
    parser.add_argument("--suite-dir", type=Path, default=DEFAULT_SUITE_DIR)
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

    summary = summarize_suite(suite_dir)
    output_columns = [
        "kind",
        "strategy",
        "delta",
        "run_dir",
        "n_runs",
        "selected",
        "miscovered",
        "miscoverage",
        "median_interval_length",
        "infinite_fraction",
        "avg_n_calibration",
        "target_alpha",
    ]
    summary_csv.parent.mkdir(parents=True, exist_ok=True)
    summary[output_columns].to_csv(summary_csv, index=False)
    plot_summary(summary, output_path=output_path, dpi=args.dpi)
    script_path = write_recreation_script(
        output_path,
        suite_dir=suite_dir,
        summary_csv=summary_csv,
        dpi=args.dpi,
    )

    print(f"Wrote plot to {output_path}")
    print(f"Wrote summary CSV to {summary_csv}")
    print(f"Wrote recreation script to {script_path}")


if __name__ == "__main__":
    main()
