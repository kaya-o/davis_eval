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


DEFAULT_SUITE_DIR = (
    PROJECT_ROOT
    / "results"
    / "suite_20260707_113502_coverage_gap_bound_relaxed_express_r_sweep"
)
DEFAULT_OUTPUT_NAME = (
    "relaxed_express_r_sweep_coverage_gap_bound_binned_500_run_average_timeseries_8panel.png"
)
DEFAULT_SUMMARY_NAME = (
    "relaxed_express_r_sweep_coverage_gap_bound_binned_500_run_average_timeseries_8panel.csv"
)
DEFAULT_STRATEGY = "RELAXED-EXPRESS"
DEFAULT_BOUND_COL = "relaxed_express_bound_gap_positive"
DEFAULT_BIN_WIDTH = 100


def result_dirs(suite_dir):
    suite_dir = Path(suite_dir)
    dirs = [
        path
        for path in suite_dir.iterdir()
        if path.is_dir() and (path / "raw_selected_events.csv").exists()
    ]
    if not dirs:
        raise FileNotFoundError(f"No result directories with raw_selected_events.csv under {suite_dir}")
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


def aggregate_binned_run_average(raw_path, strategy, bound_col, bin_width, chunksize):
    if bin_width <= 0:
        raise ValueError(f"bin_width must be positive, got {bin_width}")

    empty_run_bin_index = pd.MultiIndex.from_arrays(
        [[], []],
        names=["run", "bin_start"],
    )
    sum_by_run_bin = pd.Series(dtype=float, index=empty_run_bin_index)
    count_by_run_bin = pd.Series(dtype=float, index=empty_run_bin_index)

    for chunk in pd.read_csv(
        raw_path,
        usecols=["run", "strategy", "t", bound_col],
        chunksize=chunksize,
    ):
        chunk = chunk[chunk["strategy"] == strategy]
        if chunk.empty:
            continue

        chunk["run"] = pd.to_numeric(chunk["run"], errors="coerce")
        chunk["t"] = pd.to_numeric(chunk["t"], errors="coerce")
        chunk[bound_col] = pd.to_numeric(chunk[bound_col], errors="coerce")
        chunk = chunk.dropna(subset=["run", "t", bound_col])
        if chunk.empty:
            continue

        chunk["run"] = chunk["run"].astype(int)
        chunk["bin_start"] = (chunk["t"].astype(int) // bin_width) * bin_width
        grouped = chunk.groupby(["run", "bin_start"], sort=True)[bound_col].agg(["sum", "count"])
        sum_by_run_bin = sum_by_run_bin.add(grouped["sum"], fill_value=0.0)
        count_by_run_bin = count_by_run_bin.add(grouped["count"], fill_value=0.0)

    if count_by_run_bin.empty:
        raise ValueError(f"No rows for strategy {strategy!r} in {raw_path}")

    run_bin = (
        (sum_by_run_bin / count_by_run_bin)
        .rename("run_bin_mean")
        .reset_index()
        .merge(
            count_by_run_bin.rename("n_selected").reset_index(),
            on=["run", "bin_start"],
            how="left",
        )
    )

    all_starts = range(
        int(run_bin["bin_start"].min()),
        int(run_bin["bin_start"].max()) + bin_width,
        bin_width,
    )
    full = pd.DataFrame({
        "bin_start": list(all_starts),
        "t": [start + bin_width / 2.0 for start in all_starts],
    })
    summary = (
        full.merge(
            run_bin.groupby("bin_start", sort=True)
            .agg(
                **{bound_col: ("run_bin_mean", "mean")},
                q10=("run_bin_mean", lambda values: values.quantile(0.10)),
                q90=("run_bin_mean", lambda values: values.quantile(0.90)),
                n_runs=("run", "nunique"),
                n_selected=("n_selected", "sum"),
            )
            .reset_index(),
            on="bin_start",
            how="left",
        )
        .drop(columns=["bin_start"])
    )
    summary["n_total_runs"] = int(run_bin["run"].nunique())
    return summary


def summarize_suite(suite_dir, strategy, bound_col, bin_width, chunksize):
    rows = []
    for run_dir in result_dirs(suite_dir):
        config = load_result_config(run_dir)
        r_value = relaxed_express_max_distance_from_config(config)
        summary = aggregate_binned_run_average(
            raw_path=run_dir / "raw_selected_events.csv",
            strategy=strategy,
            bound_col=bound_col,
            bin_width=bin_width,
            chunksize=chunksize,
        )
        summary["r"] = r_value
        summary["run_dir"] = run_dir.name
        rows.append(summary)

    if not rows:
        raise FileNotFoundError(
            f"No {strategy} runs with conformal.relaxed_express_max_distance under {suite_dir}"
        )

    return pd.concat(rows, ignore_index=True).sort_values(["r", "t"]).reset_index(drop=True)


def plot_summary(summary_df, output_path, strategy, bound_col, bin_width, n_cols):
    r_values = sorted(summary_df["r"].unique())
    n_panels = len(r_values)
    n_rows = int(np.ceil(n_panels / n_cols))

    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(4.7 * n_cols, 3.35 * n_rows),
        sharex=True,
        sharey=True,
    )
    axes = np.asarray(axes).reshape(-1)

    x_min = float(summary_df["t"].min())
    x_max = float(summary_df["t"].max())

    for ax, r_value in zip(axes, r_values):
        panel = summary_df[summary_df["r"] == r_value].sort_values("t")
        panel = panel[panel[bound_col].notna()]
        if panel.empty:
            ax.set_title(f"r={r_value:g}", fontsize=10)
            ax.grid(alpha=0.25)
            continue

        ax.fill_between(
            panel["t"].to_numpy(),
            panel["q10"].to_numpy(),
            panel["q90"].to_numpy(),
            color="#93c5fd",
            alpha=0.35,
            linewidth=0,
        )
        ax.plot(
            panel["t"],
            panel[bound_col],
            color="#2563eb",
            linewidth=1.4,
        )
        ax.scatter(
            panel["t"],
            panel[bound_col],
            s=5,
            color="#2563eb",
            edgecolors="none",
            alpha=0.9,
        )
        n_total_runs = int(panel["n_total_runs"].dropna().max())
        ax.set_title(f"r={r_value:g} ({n_total_runs} runs)", fontsize=10)
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(0, 1)
        ax.grid(True, alpha=0.22)

    for ax in axes[n_panels:]:
        ax.axis("off")

    axes_matrix = axes.reshape(n_rows, n_cols)
    for ax in axes_matrix[:, 0]:
        if ax.has_data():
            ax.set_ylabel("run-averaged mean coverage gap bound")
    for ax in axes_matrix[-1, :]:
        if ax.has_data():
            ax.set_xlabel("t")

    fig.suptitle(
        f"{strategy} run-averaged mean coverage gap bound over time "
        f"({bin_width}-timestep bins)",
        y=0.995,
        fontsize=13,
    )
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


def write_recreation_script(
    output_path,
    suite_dir,
    summary_csv,
    strategy,
    bound_col,
    bin_width,
    n_cols,
    chunksize,
):
    output_path = Path(output_path)
    script_path = output_path.with_suffix(".sh")
    command = [
        "python3",
        "src/scripts/relaxed_express_r_sweep_gap_bound_timeseries_vis.py",
        "--suite-dir",
        str(relative_to_project(suite_dir)),
        "--strategy",
        strategy,
        "--bound-col",
        bound_col,
        "--bin-width",
        str(bin_width),
        "--n-cols",
        str(n_cols),
        "--output",
        str(relative_to_project(output_path)),
        "--summary-csv",
        str(relative_to_project(summary_csv)),
        "--chunksize",
        str(chunksize),
    ]
    script = (
        "#!/usr/bin/env bash\n"
        "set -euo pipefail\n"
        'cd "$(dirname "$0")/../../.."\n'
        + " ".join(shlex.quote(part) for part in command)
        + "\n"
    )
    script_path.write_text(script)
    script_path.chmod(0o755)
    return script_path


def parse_args():
    parser = ArgumentParser(
        description=(
            "Plot RELAXED-EXPRESS binned run-averaged coverage gap bounds over time "
            "for each swept radius."
        )
    )
    parser.add_argument("--suite-dir", type=Path, default=DEFAULT_SUITE_DIR)
    parser.add_argument("--strategy", default=DEFAULT_STRATEGY)
    parser.add_argument("--bound-col", default=DEFAULT_BOUND_COL)
    parser.add_argument("--bin-width", type=int, default=DEFAULT_BIN_WIDTH)
    parser.add_argument("--n-cols", type=int, default=4)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--summary-csv", type=Path, default=None)
    parser.add_argument("--chunksize", type=int, default=500_000)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.n_cols <= 0:
        raise ValueError(f"n_cols must be positive, got {args.n_cols}")

    suite_dir = Path(args.suite_dir)
    vis_dir = suite_dir / "vis"
    output_path = args.output or vis_dir / DEFAULT_OUTPUT_NAME
    summary_csv = args.summary_csv or vis_dir / DEFAULT_SUMMARY_NAME

    summary_df = summarize_suite(
        suite_dir=suite_dir,
        strategy=args.strategy,
        bound_col=args.bound_col,
        bin_width=args.bin_width,
        chunksize=args.chunksize,
    )
    summary_df = summary_df[
        [
            "r",
            "run_dir",
            "t",
            args.bound_col,
            "q10",
            "q90",
            "n_runs",
            "n_total_runs",
            "n_selected",
        ]
    ]
    summary_csv.parent.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(summary_csv, index=False)
    plot_summary(
        summary_df=summary_df,
        output_path=output_path,
        strategy=args.strategy,
        bound_col=args.bound_col,
        bin_width=args.bin_width,
        n_cols=args.n_cols,
    )
    script_path = write_recreation_script(
        output_path=output_path,
        suite_dir=suite_dir,
        summary_csv=summary_csv,
        strategy=args.strategy,
        bound_col=args.bound_col,
        bin_width=args.bin_width,
        n_cols=args.n_cols,
        chunksize=args.chunksize,
    )

    print(f"Wrote plot to {output_path}")
    print(f"Wrote summary CSV to {summary_csv}")
    print(f"Wrote recreation script to {script_path}")
    print(
        f"r panels={summary_df['r'].nunique()} "
        f"min_total_runs={int(summary_df['n_total_runs'].min())} "
        f"max_total_runs={int(summary_df['n_total_runs'].max())} "
        f"plotted_points={int(summary_df[args.bound_col].notna().sum())}"
    )


if __name__ == "__main__":
    main()
