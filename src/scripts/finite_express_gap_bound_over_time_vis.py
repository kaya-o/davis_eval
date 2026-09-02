from argparse import ArgumentParser
import os
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/davis_eval_matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/private/tmp/davis_eval_cache")

import matplotlib.pyplot as plt


DEFAULT_RESULT_DIR = PROJECT_ROOT / "results" / "20260706_012458_1_runs"
DEFAULT_OUTPUT_NAME = "finite_express_coverage_gap_bound_timeseries.png"
DEFAULT_Y_MIN = -0.02
DEFAULT_Y_MAX = 1.0


def read_events(result_dir, strategy, bound_col, require_single_run):
    raw_path = Path(result_dir) / "raw_selected_events.csv"
    if not raw_path.exists():
        raise FileNotFoundError(f"Missing {raw_path}")

    raw = pd.read_csv(raw_path)
    required = {"run", "strategy", "t", "miscovered", bound_col}
    missing = sorted(required - set(raw.columns))
    if missing:
        raise ValueError(f"Missing required columns in {raw_path}: {missing}")

    df = raw.loc[
        raw["strategy"].eq(strategy),
        ["run", "t", "miscovered", bound_col],
    ].copy()
    if df.empty:
        raise ValueError(f"No rows found for strategy {strategy!r} in {raw_path}")

    runs = sorted(df["run"].dropna().unique().tolist())
    if require_single_run and len(runs) != 1:
        raise ValueError(f"Expected exactly one run for {strategy}, found {runs}")

    df["t"] = pd.to_numeric(df["t"], errors="raise").astype(int)
    df[bound_col] = pd.to_numeric(df[bound_col], errors="raise")
    df["miscovered_bool"] = bool_series(df["miscovered"])
    df = df.sort_values(["run", "t"]).reset_index(drop=True)

    duplicate_events = df.duplicated(["run", "t"])
    if duplicate_events.any():
        dupes = df.loc[duplicate_events, ["run", "t"]].head().to_dict("records")
        raise ValueError(f"Duplicate selected run/t pairs found; first duplicates: {dupes}")

    return df


def bool_series(series):
    if series.dtype == bool:
        return series
    return series.astype(str).str.strip().str.lower().isin({"1", "1.0", "true", "yes"})


def bin_events(df, bound_col, bin_width):
    if bin_width is None:
        return df
    if bin_width <= 0:
        raise ValueError(f"bin_width must be positive, got {bin_width}")

    binned = df.copy()
    binned["bin_start"] = (binned["t"] // bin_width) * bin_width
    all_starts = range(
        int(binned["bin_start"].min()),
        int(binned["bin_start"].max()) + bin_width,
        bin_width,
    )
    full = pd.DataFrame({
        "bin_start": list(all_starts),
        "t": [start + bin_width / 2.0 for start in all_starts],
    })
    summary = (
        full.merge(
            binned.groupby("bin_start", sort=True)
            .agg(
                **{bound_col: (bound_col, "mean")},
                n_selected=("t", "size"),
            )
            .reset_index(),
            on="bin_start",
            how="left",
        )
        .drop(columns=["bin_start"])
    )
    return summary


def bin_events_by_run_then_average(df, bound_col, bin_width):
    if bin_width is None:
        raise ValueError("--average-runs requires --bin-width")
    if bin_width <= 0:
        raise ValueError(f"bin_width must be positive, got {bin_width}")

    binned = df.copy()
    binned["bin_start"] = (binned["t"] // bin_width) * bin_width
    run_bin = (
        binned.groupby(["run", "bin_start"], sort=True)
        .agg(
            run_bin_mean=(bound_col, "mean"),
            n_selected=("t", "size"),
        )
        .reset_index()
    )

    all_starts = range(
        int(binned["bin_start"].min()),
        int(binned["bin_start"].max()) + bin_width,
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
    return summary


def plot_gap_bound(
    df,
    bound_col,
    output_path,
    strategy,
    connect_line,
    bin_width,
    average_runs,
    y_min,
    y_max,
):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plot_df = (
        bin_events_by_run_then_average(df, bound_col, bin_width)
        if average_runs
        else bin_events(df, bound_col, bin_width)
    )
    label = (
        "Run-averaged mean coverage gap bound"
        if average_runs
        else "Mean coverage gap bound"
        if bin_width is not None
        else "Coverage gap bound"
    )
    title = (
        f"{strategy} Run-Averaged Mean Coverage Gap Bound Over Time ({bin_width}-timestep bins)"
        if average_runs
        else f"{strategy} Mean Coverage Gap Bound Over Time ({bin_width}-timestep bins)"
        if bin_width is not None
        else f"{strategy} Coverage Gap Bound Over Time"
    )

    fig, ax = plt.subplots(figsize=(14, 4.8))
    if average_runs:
        ax.fill_between(
            plot_df["t"].to_numpy(),
            plot_df["q10"].to_numpy(),
            plot_df["q90"].to_numpy(),
            color="#93c5fd",
            alpha=0.35,
            linewidth=0,
        )
        ax.plot(
            plot_df["t"],
            plot_df[bound_col],
            color="#2563eb",
            linewidth=1.8,
        )
        ax.scatter(
            plot_df["t"],
            plot_df[bound_col],
            s=10,
            color="#2563eb",
            edgecolors="none",
            alpha=0.9,
        )
    elif bin_width is not None:
        ax.plot(
            plot_df["t"],
            plot_df[bound_col],
            color="#2563eb",
            linewidth=1.6,
        )
        ax.scatter(
            plot_df["t"],
            plot_df[bound_col],
            s=10,
            color="#2563eb",
            edgecolors="none",
            alpha=0.85,
        )
    elif connect_line:
        ax.plot(
            plot_df["t"],
            plot_df[bound_col],
            color="#2563eb",
            linewidth=0.7,
            alpha=0.45,
        )
    else:
        covered = ~plot_df["miscovered_bool"]
        ax.scatter(
            plot_df.loc[covered, "t"],
            plot_df.loc[covered, bound_col],
            s=6,
            color="#2ca02c",
            edgecolors="none",
            alpha=0.85,
        )
        ax.scatter(
            plot_df.loc[~covered, "t"],
            plot_df.loc[~covered, bound_col],
            s=7,
            color="#d62728",
            edgecolors="none",
            alpha=0.9,
        )

    ax.set_xlim(float(plot_df["t"].min()), float(plot_df["t"].max()))
    ax.set_ylim(y_min, y_max)
    ax.set_xlabel("t")
    ax.set_ylabel(label)
    ax.set_title(title)
    ax.grid(True, alpha=0.22)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def parse_args():
    parser = ArgumentParser(
        description="Plot the single-run FINITE-EXPRESS coverage gap bound over time."
    )
    parser.add_argument("--result-dir", type=Path, default=DEFAULT_RESULT_DIR)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--strategy", default="FINITE-EXPRESS")
    parser.add_argument("--bound-col", default="finite_express_bound_gap_positive")
    parser.add_argument(
        "--bin-width",
        type=int,
        default=None,
        help="Average the bound within fixed-width timestep bins.",
    )
    parser.add_argument(
        "--average-runs",
        action="store_true",
        help="After binning within each run, average the per-run bin means by bin.",
    )
    parser.add_argument(
        "--connect-line",
        action="store_true",
        help="Also connect selected timesteps with a faint line.",
    )
    parser.add_argument(
        "--y-min",
        type=float,
        default=DEFAULT_Y_MIN,
        help="Lower y-axis limit. Defaults below zero so zero-valued markers remain visible.",
    )
    parser.add_argument(
        "--y-max",
        type=float,
        default=DEFAULT_Y_MAX,
        help="Upper y-axis limit.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if args.y_min >= args.y_max:
        raise ValueError(f"--y-min must be less than --y-max, got {args.y_min} >= {args.y_max}")

    result_dir = Path(args.result_dir)
    output_path = args.output or result_dir / DEFAULT_OUTPUT_NAME

    df = read_events(
        result_dir,
        strategy=args.strategy,
        bound_col=args.bound_col,
        require_single_run=not args.average_runs,
    )
    plot_gap_bound(
        df=df,
        bound_col=args.bound_col,
        output_path=output_path,
        strategy=args.strategy,
        connect_line=args.connect_line,
        bin_width=args.bin_width,
        average_runs=args.average_runs,
        y_min=args.y_min,
        y_max=args.y_max,
    )

    binned = (
        bin_events_by_run_then_average(df, args.bound_col, args.bin_width)
        if args.average_runs
        else bin_events(df, args.bound_col, args.bin_width)
    )
    print(f"Wrote plot to {output_path}")
    print(
        f"runs={df['run'].nunique()} "
        f"selected_timesteps={len(df)} "
        f"covered={int((~df['miscovered_bool']).sum())} "
        f"miscovered={int(df['miscovered_bool'].sum())} "
        f"plotted_points={int(binned[args.bound_col].notna().sum())} "
        f"min_contributing_runs={int(binned['n_runs'].dropna().min()) if args.average_runs else 'NA'} "
        f"max_contributing_runs={int(binned['n_runs'].dropna().max()) if args.average_runs else 'NA'} "
        f"min_bound={df[args.bound_col].min():.6f} "
        f"max_bound={df[args.bound_col].max():.6f}"
    )


if __name__ == "__main__":
    main()
