from argparse import ArgumentParser
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/davis_eval_matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/private/tmp/davis_eval_cache")

import matplotlib.pyplot as plt


STRATEGIES = (
    ("CAP", "#1f77b4"),
    ("EXPRESS", "#d62728"),
)
DEFAULT_BIN_WIDTH = 100
DEFAULT_STARVATION_THRESHOLD = 9
DEFAULT_HORIZON = 20_000
DEFAULT_PNG_NAME = "cap_express_calibration_size_and_starvation_bin100.png"
DEFAULT_PDF_NAME = "cap_express_calibration_size_and_starvation_bin100.pdf"
DEFAULT_SUMMARY_NAME = "cap_express_calibration_size_and_starvation_bin100.csv"
RAW_COLUMNS = ("run", "t", "strategy", "n_calibration")


def load_selected_events(raw_path, chunksize=500_000):
    """Load CAP/EXPRESS selected-event rows using only the required columns."""
    strategy_names = {strategy for strategy, _ in STRATEGIES}
    frames = []
    for chunk in pd.read_csv(
        raw_path,
        usecols=list(RAW_COLUMNS),
        chunksize=chunksize,
    ):
        chunk = chunk.loc[chunk["strategy"].isin(strategy_names)].copy()
        if not chunk.empty:
            frames.append(chunk)

    if not frames:
        raise ValueError(f"No CAP or EXPRESS rows found in {raw_path}")

    events = pd.concat(frames, ignore_index=True)
    for column in ("run", "t", "n_calibration"):
        events[column] = pd.to_numeric(events[column], errors="raise")
        if not np.isfinite(events[column]).all():
            raise ValueError(f"Non-finite {column} values found in {raw_path}")

    for column in ("run", "t", "n_calibration"):
        if not np.equal(events[column], np.floor(events[column])).all():
            raise ValueError(f"Non-integer {column} values found in {raw_path}")
        events[column] = events[column].astype(np.int64)

    if (events["run"] < 0).any() or (events["t"] < 0).any():
        raise ValueError("run and t must be nonnegative")
    if (events["n_calibration"] < 0).any():
        raise ValueError("n_calibration must be nonnegative")
    if events.duplicated(["strategy", "run", "t"]).any():
        duplicates = (
            events.loc[
                events.duplicated(["strategy", "run", "t"], keep=False),
                ["strategy", "run", "t"],
            ]
            .head()
            .to_dict("records")
        )
        raise ValueError(f"Duplicate strategy/run/t rows found: {duplicates}")

    return events


def summarize_event_bins(
    events,
    bin_width=DEFAULT_BIN_WIDTH,
    starvation_threshold=DEFAULT_STARVATION_THRESHOLD,
    horizon=DEFAULT_HORIZON,
):
    if bin_width <= 0:
        raise ValueError("bin_width must be positive")
    if starvation_threshold < 0:
        raise ValueError("starvation_threshold must be nonnegative")
    if horizon <= 0 or horizon % bin_width:
        raise ValueError("horizon must be positive and divisible by bin_width")
    if (events["t"] >= horizon).any():
        observed_max = int(events["t"].max())
        raise ValueError(
            f"Observed t={observed_max} lies outside the requested horizon [0, {horizon})"
        )

    events = events.copy()
    events["bin_start"] = (events["t"] // bin_width) * bin_width
    events["starved"] = events["n_calibration"] < starvation_threshold

    grouped_events = events.groupby(["strategy", "bin_start"], sort=True)
    pooled = grouped_events.agg(
        median_n_calibration=("n_calibration", "median"),
        q10_n_calibration=("n_calibration", lambda values: values.quantile(0.10)),
        q90_n_calibration=("n_calibration", lambda values: values.quantile(0.90)),
        starved_events=("starved", "sum"),
        selected_events=("starved", "size"),
    ).reset_index()
    pooled["pooled_starvation_fraction"] = (
        pooled["starved_events"] / pooled["selected_events"]
    )

    run_bins = (
        events.groupby(["strategy", "run", "bin_start"], sort=True)
        .agg(
            run_starved_events=("starved", "sum"),
            run_selected_events=("starved", "size"),
        )
        .reset_index()
    )
    run_bins["run_starvation_fraction"] = (
        run_bins["run_starved_events"] / run_bins["run_selected_events"]
    )
    grouped_runs = run_bins.groupby(["strategy", "bin_start"], sort=True)
    run_summary = grouped_runs.agg(
        run_starvation_q10=(
            "run_starvation_fraction",
            lambda values: values.quantile(0.10),
        ),
        run_starvation_q90=(
            "run_starvation_fraction",
            lambda values: values.quantile(0.90),
        ),
        contributing_runs=("run", "nunique"),
    ).reset_index()

    strategy_names = [strategy for strategy, _ in STRATEGIES]
    all_bins = pd.MultiIndex.from_product(
        [strategy_names, range(0, horizon, bin_width)],
        names=["strategy", "bin_start"],
    ).to_frame(index=False)
    summary = (
        all_bins.merge(
            pooled,
            on=["strategy", "bin_start"],
            how="left",
            validate="one_to_one",
        )
        .merge(
            run_summary,
            on=["strategy", "bin_start"],
            how="left",
            validate="one_to_one",
        )
        .sort_values(["strategy", "bin_start"])
        .reset_index(drop=True)
    )
    for column in ("starved_events", "selected_events", "contributing_runs"):
        summary[column] = summary[column].fillna(0).astype(np.int64)
    summary["bin_end"] = summary["bin_start"] + bin_width
    summary["bin_midpoint"] = summary["bin_start"] + 0.5 * bin_width
    summary["bin_width"] = bin_width
    summary["starvation_threshold"] = starvation_threshold

    columns = [
        "strategy",
        "bin_start",
        "bin_end",
        "bin_midpoint",
        "bin_width",
        "starvation_threshold",
        "median_n_calibration",
        "q10_n_calibration",
        "q90_n_calibration",
        "starved_events",
        "selected_events",
        "pooled_starvation_fraction",
        "run_starvation_q10",
        "run_starvation_q90",
        "contributing_runs",
    ]
    return summary[columns], run_bins


def validate_summary(summary):
    if (summary["starved_events"] < 0).any():
        raise ValueError("Negative starvation numerator found")
    if (summary["selected_events"] < 0).any():
        raise ValueError("Negative selected-event count found")
    if (summary["starved_events"] > summary["selected_events"]).any():
        raise ValueError("Starvation numerator exceeds selected-event count")

    nonempty = summary["selected_events"].gt(0)
    fractions = summary.loc[nonempty, "pooled_starvation_fraction"]
    if fractions.isna().any() or not fractions.between(0.0, 1.0).all():
        raise ValueError("Pooled starvation fractions must lie in [0, 1]")
    if summary.loc[~nonempty, "pooled_starvation_fraction"].notna().any():
        raise ValueError("Empty bins must have a missing pooled starvation fraction")

    for column in ("run_starvation_q10", "run_starvation_q90"):
        values = summary[column].dropna()
        if not values.between(0.0, 1.0).all():
            raise ValueError(f"{column} must lie in [0, 1]")
    paired = summary[["run_starvation_q10", "run_starvation_q90"]].dropna()
    if (paired["run_starvation_q10"] > paired["run_starvation_q90"]).any():
        raise ValueError("Run-level q10 exceeds q90")


def plot_summary(
    summary,
    png_path,
    pdf_path,
    horizon,
    starvation_threshold,
    dpi=600,
):
    fig, axes = plt.subplots(
        1,
        2,
        figsize=(14.5, 4.8),
        sharex=True,
        sharey=True,
        constrained_layout=True,
    )

    for ax, (strategy, color) in zip(axes, STRATEGIES):
        panel = summary.loc[summary["strategy"].eq(strategy)].sort_values(
            "bin_start"
        )
        x = panel["bin_midpoint"].to_numpy(dtype=float)

        ax.fill_between(
            x,
            panel["run_starvation_q10"].to_numpy(dtype=float),
            panel["run_starvation_q90"].to_numpy(dtype=float),
            color=color,
            alpha=0.20,
            linewidth=0,
            label="Between-run 10th–90th percentile",
        )
        ax.plot(
            x,
            panel["pooled_starvation_fraction"].to_numpy(dtype=float),
            color=color,
            linewidth=2.2,
            label="Pooled event-level fraction",
        )
        ax.set_title(
            f"{strategy}: Calibration-Starvation Fraction",
            fontsize=15,
        )
        ax.set_ylim(0.0, 1.035)
        ax.set_yticks(np.linspace(0.0, 1.0, 6))
        ax.set_xlabel("Online time $t$", fontsize=13)
        ax.legend(loc="lower right", fontsize=9.5, framealpha=0.95)

    axes[0].set_ylabel(
        rf"Fraction with $|D_t| < {starvation_threshold:g}$",
        fontsize=13,
    )
    x_ticks = np.arange(0, horizon + 1, 5_000)
    for ax in axes:
        ax.set_xlim(0, horizon)
        ax.set_xticks(x_ticks)
        ax.grid(True, alpha=0.22, linewidth=0.8)
        ax.tick_params(axis="both", labelsize=11)

    png_path = Path(png_path)
    pdf_path = Path(pdf_path)
    png_path.parent.mkdir(parents=True, exist_ok=True)
    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(png_path, dpi=dpi, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)


def horizon_from_config(result_dir, fallback=DEFAULT_HORIZON):
    for name in ("config.json", "resolved_config.json"):
        path = Path(result_dir) / name
        if not path.exists():
            continue
        with path.open() as config_file:
            config = json.load(config_file)
        value = config.get("data", {}).get("n_on")
        if value is not None:
            return int(value)
    return fallback


def print_sanity_checks(events, summary, horizon, bin_width, threshold):
    print(
        f"Loaded {len(events):,} CAP/EXPRESS selected-event rows from "
        f"{events['run'].nunique()} runs; observed t={events['t'].min()}-"
        f"{events['t'].max()}."
    )
    for strategy, _ in STRATEGIES:
        strategy_events = events.loc[events["strategy"].eq(strategy)]
        starved = int(strategy_events["n_calibration"].lt(threshold).sum())
        selected = len(strategy_events)
        print(
            f"{strategy} overall pooled starvation: {starved:,}/{selected:,} "
            f"= {starved / selected:.6f}"
        )

    representative_starts = sorted(
        {
            (horizon // 2 // bin_width) * bin_width,
            (3 * horizon // 4 // bin_width) * bin_width,
            horizon - bin_width,
        }
    )
    for start in representative_starts:
        print(f"Late bin [{start}, {start + bin_width}):")
        for strategy, _ in STRATEGIES:
            row = summary.loc[
                summary["strategy"].eq(strategy)
                & summary["bin_start"].eq(start)
            ].iloc[0]
            if row["selected_events"] == 0:
                detail = "no selected events"
            else:
                detail = (
                    f"{int(row['starved_events']):,}/"
                    f"{int(row['selected_events']):,} = "
                    f"{row['pooled_starvation_fraction']:.6f}"
                )
            print(f"  {strategy}: {detail}")

    nonempty = summary["selected_events"].gt(0)
    minimum = summary.loc[nonempty, "pooled_starvation_fraction"].min()
    maximum = summary.loc[nonempty, "pooled_starvation_fraction"].max()
    print(
        "Validation passed: every starvation numerator is <= its selected-event "
        f"count; nonempty-bin pooled fractions span [{minimum:.6f}, {maximum:.6f}]."
    )


def parse_args():
    parser = ArgumentParser(
        description=(
            "Plot pooled event-level calibration-size distributions and "
            "calibration-starvation frequencies for CAP and EXPRESS."
        )
    )
    parser.add_argument(
        "--result-dir",
        type=Path,
        required=True,
        help="Result directory containing raw_selected_events.csv.",
    )
    parser.add_argument("--bin-width", type=int, default=DEFAULT_BIN_WIDTH)
    parser.add_argument(
        "--starvation-threshold",
        type=int,
        default=DEFAULT_STARVATION_THRESHOLD,
    )
    parser.add_argument(
        "--horizon",
        type=int,
        default=None,
        help="Online horizon; defaults to data.n_on in the result config.",
    )
    parser.add_argument("--chunksize", type=int, default=500_000)
    parser.add_argument("--png", type=Path, default=None)
    parser.add_argument("--pdf", type=Path, default=None)
    parser.add_argument("--summary-csv", type=Path, default=None)
    parser.add_argument("--dpi", type=int, default=600)
    return parser.parse_args()


def main():
    args = parse_args()
    result_dir = Path(args.result_dir)
    raw_path = result_dir / "raw_selected_events.csv"
    if not raw_path.exists():
        raise FileNotFoundError(f"Missing {raw_path}")
    if args.chunksize <= 0:
        raise ValueError("--chunksize must be positive")
    if args.dpi <= 0:
        raise ValueError("--dpi must be positive")

    horizon = args.horizon or horizon_from_config(result_dir)
    bin_slug = f"bin{args.bin_width}"
    png_path = args.png or result_dir / DEFAULT_PNG_NAME.replace("bin100", bin_slug)
    pdf_path = args.pdf or result_dir / DEFAULT_PDF_NAME.replace("bin100", bin_slug)
    summary_path = args.summary_csv or result_dir / DEFAULT_SUMMARY_NAME.replace(
        "bin100", bin_slug
    )

    events = load_selected_events(raw_path, chunksize=args.chunksize)
    summary, _ = summarize_event_bins(
        events,
        bin_width=args.bin_width,
        starvation_threshold=args.starvation_threshold,
        horizon=horizon,
    )
    validate_summary(summary)

    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(summary_path, index=False)
    plot_summary(
        summary,
        png_path=png_path,
        pdf_path=pdf_path,
        horizon=horizon,
        starvation_threshold=args.starvation_threshold,
        dpi=args.dpi,
    )

    print(f"Wrote PNG to {png_path}")
    print(f"Wrote PDF to {pdf_path}")
    print(f"Wrote binned summary to {summary_path}")
    print_sanity_checks(
        events,
        summary,
        horizon=horizon,
        bin_width=args.bin_width,
        threshold=args.starvation_threshold,
    )


if __name__ == "__main__":
    main()
