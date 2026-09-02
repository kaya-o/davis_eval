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
from matplotlib.patches import Patch
from matplotlib.ticker import MaxNLocator, MultipleLocator, StrMethodFormatter


DEFAULT_SUITE_DIR = (
    PROJECT_ROOT
    / "results"
    / "suite_20260825_010303_adaptive_express"
)
DEFAULT_OUTPUT_NAME = (
    "adaptive_weighted_express_calibration_availability_stress_lambda_3panel.png"
)
DEFAULT_SUMMARY_NAME = (
    "adaptive_weighted_express_calibration_availability_stress_lambda_3panel.csv"
)
DEFAULT_RUN_BIN_MEANS_NAME = (
    "adaptive_weighted_express_calibration_availability_stress_lambda_run_bin_means.csv"
)
ADAPTIVE_STRATEGY = "ADAPTIVE-WEIGHTED-EXPRESS"
AVAILABILITY_COLUMN = (
    "adaptive_weighted_express_express_n_calibration_for_stress"
)
STRESS_COLUMN = "adaptive_weighted_express_stress"
LAMBDA_COLUMN = "adaptive_weighted_express_lambda_t"
STRESS_COUNT_COLUMN = "adaptive_weighted_express_stress_count"
STRESS_SOURCE_COLUMN = "adaptive_weighted_express_stress_count_source"
MEAN_COLOR = "#0072B2"
REFERENCE_COLOR = "#4D4D4D"


def load_result_config(result_dir):
    for filename in ("resolved_config.json", "config.json"):
        path = Path(result_dir) / filename
        if path.exists():
            with path.open() as handle:
                return json.load(handle)
    raise FileNotFoundError(
        f"Missing resolved_config.json or config.json under {result_dir}"
    )


def find_adaptive_result_dir(suite_dir):
    suite_dir = Path(suite_dir)
    matches = []
    for run_dir in sorted(path for path in suite_dir.iterdir() if path.is_dir()):
        aggregate_path = run_dir / "aggregate_results.csv"
        raw_path = run_dir / "raw_selected_events.csv"
        if not aggregate_path.exists() or not raw_path.exists():
            continue
        aggregate = pd.read_csv(aggregate_path, usecols=["strategy"])
        if aggregate["strategy"].astype(str).eq(ADAPTIVE_STRATEGY).any():
            matches.append(run_dir)

    if len(matches) != 1:
        raise ValueError(
            f"Expected exactly one result directory containing {ADAPTIVE_STRATEGY} "
            f"under {suite_dir}, found {len(matches)}: {[path.name for path in matches]}"
        )
    return matches[0]


def load_adaptive_events(result_dir):
    raw_path = Path(result_dir) / "raw_selected_events.csv"
    required_columns = [
        "run",
        "t",
        "strategy",
        AVAILABILITY_COLUMN,
        STRESS_COLUMN,
        LAMBDA_COLUMN,
        STRESS_COUNT_COLUMN,
        STRESS_SOURCE_COLUMN,
    ]
    raw_columns = set(pd.read_csv(raw_path, nrows=0).columns)
    missing_columns = sorted(set(required_columns) - raw_columns)
    if missing_columns:
        raise ValueError(f"Missing required raw-event columns: {missing_columns}")

    raw = pd.read_csv(raw_path, usecols=required_columns)
    adaptive = raw[raw["strategy"].astype(str).eq(ADAPTIVE_STRATEGY)].copy()
    if adaptive.empty:
        raise ValueError(f"No {ADAPTIVE_STRATEGY} events found in {raw_path}")

    numeric_columns = [
        "run",
        "t",
        AVAILABILITY_COLUMN,
        STRESS_COLUMN,
        LAMBDA_COLUMN,
        STRESS_COUNT_COLUMN,
    ]
    for column in numeric_columns:
        adaptive[column] = pd.to_numeric(adaptive[column], errors="coerce")
    if adaptive[numeric_columns].isna().any().any():
        missing = adaptive[numeric_columns].isna().sum()
        raise ValueError(
            "Missing or nonnumeric adaptive diagnostics: "
            f"{missing[missing > 0].to_dict()}"
        )
    if not np.isfinite(adaptive[numeric_columns].to_numpy(dtype=float)).all():
        raise ValueError("Adaptive diagnostics contain nonfinite values")
    if adaptive.duplicated(["run", "t"]).any():
        raise ValueError("Duplicate adaptive (run, t) event rows found")

    sources = set(adaptive[STRESS_SOURCE_COLUMN].dropna().astype(str))
    if sources != {"express_calibration"}:
        raise ValueError(
            "This figure defines availability as exact EXPRESS calibration size; "
            f"expected stress source express_calibration, found {sorted(sources)}"
        )
    if not np.array_equal(
        adaptive[STRESS_COUNT_COLUMN].to_numpy(dtype=float),
        adaptive[AVAILABILITY_COLUMN].to_numpy(dtype=float),
    ):
        raise ValueError("Stress input count does not equal exact EXPRESS availability")

    return adaptive


def build_run_bin_means(adaptive, n_runs, n_on, bin_width):
    if bin_width <= 0:
        raise ValueError(f"bin_width must be positive, got {bin_width}")
    if n_on <= 0:
        raise ValueError(f"n_on must be positive, got {n_on}")
    if n_on % bin_width != 0:
        raise ValueError(
            f"n_on={n_on} must be divisible by bin_width={bin_width} for equal bins"
        )

    expected_runs = set(range(n_runs))
    observed_runs = set(adaptive["run"].astype(int))
    if observed_runs != expected_runs:
        raise ValueError(
            f"Expected run IDs 0 through {n_runs - 1}, found "
            f"{len(observed_runs)} distinct IDs"
        )

    t = adaptive["t"].to_numpy(dtype=float)
    if not np.equal(t, np.floor(t)).all():
        raise ValueError("Timesteps must be integer-valued")
    if np.min(t) < 0 or np.max(t) >= n_on:
        raise ValueError(
            f"Timesteps must lie in [0, {n_on - 1}], found [{np.min(t)}, {np.max(t)}]"
        )

    binned = adaptive.copy()
    binned["run"] = binned["run"].astype(int)
    binned["t"] = binned["t"].astype(int)
    binned["bin_index"] = binned["t"] // bin_width
    binned["bin_start"] = binned["bin_index"] * bin_width
    binned["bin_end_exclusive"] = binned["bin_start"] + bin_width
    binned["bin_center"] = binned["bin_start"] + bin_width / 2.0

    run_bin_means = (
        binned.groupby(
            [
                "run",
                "bin_index",
                "bin_start",
                "bin_end_exclusive",
                "bin_center",
            ],
            sort=True,
            as_index=False,
        )
        .agg(
            selected_events=("t", "size"),
            calibration_availability=(AVAILABILITY_COLUMN, "mean"),
            stress=(STRESS_COLUMN, "mean"),
            lambda_t=(LAMBDA_COLUMN, "mean"),
        )
    )

    n_bins = n_on // bin_width
    expected_cells = pd.MultiIndex.from_product(
        [range(n_runs), range(n_bins)],
        names=["run", "bin_index"],
    )
    observed_cells = pd.MultiIndex.from_frame(run_bin_means[["run", "bin_index"]])
    missing_cells = expected_cells.difference(observed_cells)
    if len(missing_cells) > 0:
        raise ValueError(
            f"Missing {len(missing_cells)} of {n_runs * n_bins} run-bin cells; "
            "cannot average every bin across all runs without imputation"
        )
    if len(run_bin_means) != n_runs * n_bins:
        raise ValueError(
            f"Expected {n_runs * n_bins} run-bin rows, found {len(run_bin_means)}"
        )

    return run_bin_means


def summarize_run_bins(run_bin_means, n_runs, bin_width):
    grouped = run_bin_means.groupby(
        ["bin_index", "bin_start", "bin_end_exclusive", "bin_center"],
        sort=True,
    )
    summary = grouped.agg(
        runs=("run", "nunique"),
        selected_events=("selected_events", "sum"),
        selected_events_per_run_min=("selected_events", "min"),
        selected_events_per_run_max=("selected_events", "max"),
        calibration_availability_mean=("calibration_availability", "mean"),
        calibration_availability_median=("calibration_availability", "median"),
        calibration_availability_q10=(
            "calibration_availability",
            lambda values: values.quantile(0.10),
        ),
        calibration_availability_q90=(
            "calibration_availability",
            lambda values: values.quantile(0.90),
        ),
        stress_mean=("stress", "mean"),
        stress_median=("stress", "median"),
        stress_q10=("stress", lambda values: values.quantile(0.10)),
        stress_q90=("stress", lambda values: values.quantile(0.90)),
        lambda_t_mean=("lambda_t", "mean"),
        lambda_t_median=("lambda_t", "median"),
        lambda_t_q10=("lambda_t", lambda values: values.quantile(0.10)),
        lambda_t_q90=("lambda_t", lambda values: values.quantile(0.90)),
    ).reset_index()
    summary.insert(4, "bin_width", int(bin_width))

    if not summary["runs"].eq(n_runs).all():
        bad = summary.loc[~summary["runs"].eq(n_runs), ["bin_index", "runs"]]
        raise ValueError(f"Not every bin contains all {n_runs} runs: {bad.to_dict('records')}")

    return summary


def availability_axis(summary, midpoint):
    values = np.concatenate(
        [
            summary["calibration_availability_q90"].to_numpy(dtype=float),
            np.asarray([midpoint], dtype=float),
        ]
    )
    maximum = float(np.max(values))
    ticks = MaxNLocator(nbins=7, steps=[1, 2, 2.5, 5, 10]).tick_values(
        0.0,
        maximum * 1.05,
    )
    ticks = ticks[ticks >= 0]
    return ticks, float(ticks[-1])


def plot_summary(
    summary,
    output_path,
    n_on,
    midpoint,
    lambda_min,
    lambda_max,
    dpi,
):
    x = summary["bin_center"].to_numpy(dtype=float)
    panel_specs = [
        {
            "mean": "calibration_availability_mean",
            "q10": "calibration_availability_q10",
            "q90": "calibration_availability_q90",
            "ylabel": r"Exact calibration count, $N_t$",
            "reference": midpoint,
        },
        {
            "mean": "stress_mean",
            "q10": "stress_q10",
            "q90": "stress_q90",
            "ylabel": r"Stress, $\eta_t$",
            "reference": 0.5,
        },
        {
            "mean": "lambda_t_mean",
            "q10": "lambda_t_q10",
            "q90": "lambda_t_q90",
            "ylabel": r"Weight-decay rate, $\lambda_t$",
            "reference": float(np.sqrt(lambda_min * lambda_max)),
        },
    ]

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
        fig, axes = plt.subplots(3, 1, figsize=(8.6, 8.7), sharex=True)

        for panel_index, (ax, spec) in enumerate(zip(axes, panel_specs)):
            mean = summary[spec["mean"]].to_numpy(dtype=float)
            q10 = summary[spec["q10"]].to_numpy(dtype=float)
            q90 = summary[spec["q90"]].to_numpy(dtype=float)
            ax.fill_between(
                x,
                q10,
                q90,
                color=MEAN_COLOR,
                alpha=0.16,
                linewidth=0,
                zorder=1,
            )
            ax.plot(
                x,
                mean,
                color=MEAN_COLOR,
                linewidth=2.0,
                zorder=3,
            )
            ax.axhline(
                spec["reference"],
                color=REFERENCE_COLOR,
                linestyle=(0, (4, 2)),
                linewidth=1.35,
                zorder=2,
            )
            ax.set_ylabel(spec["ylabel"])
            ax.grid(axis="y", color="#B0B0B0", linewidth=0.7, alpha=0.28)
            ax.set_axisbelow(True)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.text(
                0.985,
                0.95,
                f"({chr(ord('a') + panel_index)})",
                transform=ax.transAxes,
                ha="right",
                va="top",
                fontsize=11,
                fontweight="semibold",
            )

        availability_ticks, availability_top = availability_axis(summary, midpoint)
        axes[0].set_yticks(availability_ticks)
        axes[0].set_ylim(0.0, availability_top)
        axes[1].set_yticks(np.linspace(0.0, 1.0, 6))
        axes[1].set_ylim(0.0, 1.03)
        lambda_ticks = MaxNLocator(
            nbins=6,
            steps=[1, 2, 2.5, 5, 10],
        ).tick_values(0.0, lambda_max)
        lambda_ticks = lambda_ticks[
            (lambda_ticks >= 0.0) & (lambda_ticks <= lambda_max)
        ]
        axes[2].set_yticks(lambda_ticks)
        axes[2].set_ylim(0.0, lambda_max * 1.03)
        axes[2].set_xlim(0.0, float(n_on))
        axes[2].xaxis.set_major_locator(MultipleLocator(2500))
        axes[2].xaxis.set_major_formatter(StrMethodFormatter("{x:,.0f}"))
        axes[2].set_xlabel(r"Online timestep, $t$")

        legend_handles = [
            Line2D([], [], color=MEAN_COLOR, linewidth=2.0, label="Mean across runs"),
            Patch(
                facecolor=MEAN_COLOR,
                edgecolor="none",
                alpha=0.16,
                label="10th–90th percentile across runs",
            ),
            Line2D(
                [],
                [],
                color=REFERENCE_COLOR,
                linestyle=(0, (4, 2)),
                linewidth=1.35,
                label="Reference level",
            ),
        ]
        fig.legend(
            handles=legend_handles,
            loc="upper center",
            bbox_to_anchor=(0.5, 0.995),
            ncol=3,
            frameon=False,
            handlelength=2.4,
            columnspacing=1.4,
        )
        fig.subplots_adjust(
            left=0.115,
            right=0.985,
            bottom=0.105,
            top=0.895,
            hspace=0.20,
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
    run_bin_csv,
    bin_width,
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
        "src/scripts/adaptive_weighted_express_availability_stress_vis.py",
        "--suite-dir",
        str(relative_to_project(suite_dir)),
        "--output",
        str(relative_to_project(output_path)),
        "--summary-csv",
        str(relative_to_project(summary_csv)),
        "--run-bin-csv",
        str(relative_to_project(run_bin_csv)),
        "--bin-width",
        str(bin_width),
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
            "Plot equal-run binned calibration availability and stress for "
            "ADAPTIVE-WEIGHTED-EXPRESS, including its realized decay rate."
        )
    )
    parser.add_argument("--suite-dir", type=Path, default=DEFAULT_SUITE_DIR)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--summary-csv", type=Path, default=None)
    parser.add_argument("--run-bin-csv", type=Path, default=None)
    parser.add_argument("--bin-width", type=int, default=200)
    parser.add_argument("--dpi", type=int, default=600)
    return parser.parse_args()


def main():
    args = parse_args()
    suite_dir = Path(args.suite_dir)
    adaptive_result_dir = find_adaptive_result_dir(suite_dir)
    config = load_result_config(adaptive_result_dir)
    n_runs = int(config.get("n_runs", 0))
    n_on = int(config.get("data", {}).get("n_on", 0))
    conformal = config.get("conformal", {})
    stress_mode = conformal.get("adaptive_weighted_express_stress_mode")
    if stress_mode != "sigmoid":
        raise ValueError(f"Expected sigmoid stress mode, found {stress_mode!r}")
    midpoint = float(
        conformal.get("adaptive_weighted_express_stress_midpoint_count")
    )
    lambda_min = float(conformal.get("adaptive_weighted_express_lambda_min"))
    lambda_max = float(conformal.get("adaptive_weighted_express_lambda_max"))
    if lambda_min <= 0 or lambda_max < lambda_min:
        raise ValueError(
            f"Invalid adaptive lambda range [{lambda_min}, {lambda_max}]"
        )

    adaptive = load_adaptive_events(adaptive_result_dir)
    run_bin_means = build_run_bin_means(
        adaptive,
        n_runs=n_runs,
        n_on=n_on,
        bin_width=args.bin_width,
    )
    summary = summarize_run_bins(
        run_bin_means,
        n_runs=n_runs,
        bin_width=args.bin_width,
    )

    vis_dir = suite_dir / "vis"
    output_path = args.output or vis_dir / DEFAULT_OUTPUT_NAME
    summary_csv = args.summary_csv or vis_dir / DEFAULT_SUMMARY_NAME
    run_bin_csv = args.run_bin_csv or vis_dir / DEFAULT_RUN_BIN_MEANS_NAME
    summary_csv.parent.mkdir(parents=True, exist_ok=True)
    run_bin_csv.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(summary_csv, index=False)
    run_bin_means.to_csv(run_bin_csv, index=False)
    plot_summary(
        summary,
        output_path=output_path,
        n_on=n_on,
        midpoint=midpoint,
        lambda_min=lambda_min,
        lambda_max=lambda_max,
        dpi=args.dpi,
    )
    recreation_script = write_recreation_script(
        output_path,
        suite_dir=suite_dir,
        summary_csv=summary_csv,
        run_bin_csv=run_bin_csv,
        bin_width=args.bin_width,
        dpi=args.dpi,
    )

    print(f"Adaptive result: {adaptive_result_dir}")
    print(f"Wrote plot to {output_path}")
    print(f"Wrote summary CSV to {summary_csv}")
    print(f"Wrote run-bin CSV to {run_bin_csv}")
    print(f"Wrote recreation script to {recreation_script}")


if __name__ == "__main__":
    main()
