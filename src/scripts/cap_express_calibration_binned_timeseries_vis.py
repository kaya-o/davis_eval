from argparse import ArgumentParser
import os
from pathlib import Path

import numpy as np
import pandas as pd


os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/davis_eval_matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/private/tmp/davis_eval_cache")

import matplotlib.pyplot as plt
from matplotlib.patches import ConnectionPatch, Rectangle


STRATEGIES = (
    ("CAP", "#1f77b4"),
    ("EXPRESS", "#d62728"),
)
DEFAULT_OUTPUT_NAME = "cap_express_calibration_size_run_averaged_bin100.png"
DEFAULT_SUMMARY_NAME = "cap_express_calibration_size_run_averaged_bin100.csv"


def summarize_run_bins(raw_path, bin_width=100, chunksize=500_000):
    strategy_names = {strategy for strategy, _ in STRATEGIES}
    grouped_chunks = []

    for chunk in pd.read_csv(
        raw_path,
        usecols=["run", "t", "strategy", "n_calibration"],
        chunksize=chunksize,
    ):
        chunk = chunk[chunk["strategy"].isin(strategy_names)].copy()
        if chunk.empty:
            continue

        for column in ("run", "t", "n_calibration"):
            chunk[column] = pd.to_numeric(chunk[column], errors="coerce")
        chunk = chunk.dropna(subset=["run", "t", "n_calibration"])
        chunk["run"] = chunk["run"].astype(int)
        chunk["bin_start"] = (chunk["t"].astype(int) // bin_width) * bin_width

        grouped = (
            chunk.groupby(["strategy", "run", "bin_start"], sort=False)[
                "n_calibration"
            ]
            .agg(["sum", "count"])
            .reset_index()
        )
        grouped_chunks.append(grouped)

    if not grouped_chunks:
        raise ValueError(f"No CAP or EXPRESS rows found in {raw_path}")

    run_bins = pd.concat(grouped_chunks, ignore_index=True)
    run_bins = (
        run_bins.groupby(["strategy", "run", "bin_start"], as_index=False)
        .agg(n_calibration_sum=("sum", "sum"), selected_events=("count", "sum"))
    )
    run_bins["run_mean_n_calibration"] = (
        run_bins["n_calibration_sum"] / run_bins["selected_events"]
    )

    grouped_run_bins = run_bins.groupby(["strategy", "bin_start"])
    summary = grouped_run_bins.agg(
        mean_n_calibration=("run_mean_n_calibration", "mean"),
        contributing_runs=("run", "nunique"),
        selected_events=("selected_events", "sum"),
    ).reset_index()
    quantiles = (
        grouped_run_bins["run_mean_n_calibration"]
        .quantile([0.1, 0.9])
        .unstack()
        .rename(columns={0.1: "q10_n_calibration", 0.9: "q90_n_calibration"})
        .reset_index()
    )
    summary = summary.merge(
        quantiles,
        on=["strategy", "bin_start"],
        how="left",
        validate="one_to_one",
    ).sort_values(["strategy", "bin_start"])
    summary["bin_end"] = summary["bin_start"] + bin_width
    summary["bin_midpoint"] = summary["bin_start"] + 0.5 * bin_width
    summary["bin_width"] = bin_width
    return summary[
        [
            "strategy",
            "bin_start",
            "bin_end",
            "bin_midpoint",
            "bin_width",
            "mean_n_calibration",
            "q10_n_calibration",
            "q90_n_calibration",
            "contributing_runs",
            "selected_events",
        ]
    ]


def plot_summary(
    summary,
    output_path,
    title=None,
    t_min=None,
    finite_threshold=None,
):
    plot_data = summary
    if t_min is not None:
        plot_data = summary[summary["bin_start"] >= t_min].copy()
        if plot_data.empty:
            raise ValueError(f"No bins remain at or after t={t_min}")

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(14, 5.5),
        sharex=True,
        sharey=True,
    )

    y_max = float(plot_data["q90_n_calibration"].max())
    if finite_threshold is not None:
        y_max = max(y_max, float(finite_threshold))
    y_top = 1.05 * y_max if np.isfinite(y_max) and y_max > 0 else 1.0

    for ax, (strategy, color) in zip(axes, STRATEGIES):
        panel = plot_data[plot_data["strategy"].eq(strategy)].sort_values("bin_start")
        ax.fill_between(
            panel["bin_midpoint"],
            panel["q10_n_calibration"],
            panel["q90_n_calibration"],
            color=color,
            alpha=0.2,
            linewidth=0,
            label="10th–90th percentile",
        )
        ax.plot(
            panel["bin_midpoint"],
            panel["mean_n_calibration"],
            color=color,
            linewidth=2.2,
            label="Across-run mean",
        )
        if finite_threshold is not None:
            ax.axhline(
                finite_threshold,
                color="#555555",
                linestyle="--",
                linewidth=1.4,
                label=f"Minimum finite size ({finite_threshold:g})",
            )
        ax.set_title(strategy, fontsize=15)
        ax.set_xlabel("Online time $t$", fontsize=13)
        ax.set_ylim(0, y_top)
        if t_min is not None:
            ax.set_xlim(left=t_min)
        ax.tick_params(axis="both", labelsize=11)
        ax.grid(alpha=0.25)
        ax.legend(fontsize=10, loc="upper right")

    axes[0].set_ylabel("Mean calibration set size", fontsize=13)
    if title:
        fig.suptitle(title, fontsize=16)
    fig.tight_layout(rect=(0, 0, 1, 0.94 if title else 1))

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_full_and_zoom(
    summary,
    output_path,
    zoom_t_min,
    title=None,
    finite_threshold=None,
):
    zoom_data = summary[summary["bin_start"] >= zoom_t_min].copy()
    if zoom_data.empty:
        raise ValueError(f"No bins remain at or after t={zoom_t_min}")

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    row_specs = (
        (summary, "Full Range", 0),
        (zoom_data, f"$t \\geq {zoom_t_min}$", zoom_t_min),
    )
    x_max = int(summary["bin_end"].max())

    for row_idx, (row_data, row_label, x_min) in enumerate(row_specs):
        y_max = float(row_data["q90_n_calibration"].max())
        if finite_threshold is not None:
            y_max = max(y_max, float(finite_threshold))
        y_top = 1.05 * y_max if np.isfinite(y_max) and y_max > 0 else 1.0

        for col_idx, (strategy, color) in enumerate(STRATEGIES):
            ax = axes[row_idx, col_idx]
            panel = row_data[row_data["strategy"].eq(strategy)].sort_values(
                "bin_start"
            )
            ax.fill_between(
                panel["bin_midpoint"],
                panel["q10_n_calibration"],
                panel["q90_n_calibration"],
                color=color,
                alpha=0.2,
                linewidth=0,
                label="10th–90th percentile",
            )
            ax.plot(
                panel["bin_midpoint"],
                panel["mean_n_calibration"],
                color=color,
                linewidth=2.2,
                label="Across-run mean",
            )
            if finite_threshold is not None:
                ax.axhline(
                    finite_threshold,
                    color="#555555",
                    linestyle="--",
                    linewidth=1.4,
                    label=f"Minimum finite size ({finite_threshold:g})",
                )
            ax.set_title(f"{strategy}: {row_label}", fontsize=16)
            ax.set_xlim(x_min, x_max)
            ax.set_ylim(0, y_top)
            ax.tick_params(axis="both", labelsize=13)
            ax.grid(alpha=0.25)
            ax.legend(
                fontsize=12,
                loc="upper right" if row_idx == 0 else "upper center",
            )

            if row_idx == 0:
                ax.set_xticks(np.arange(0, x_max + 1, 5_000))
            else:
                ax.set_xticks(np.arange(zoom_t_min, x_max + 1, 2_500))
                ax.set_xlabel("Online time $t$", fontsize=13)

        axes[row_idx, 0].set_ylabel("Mean calibration set size", fontsize=13)

    if title:
        fig.suptitle(title, fontsize=16, y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.96 if title else 1))

    zoom_y_top = axes[1, 0].get_ylim()[1]
    for col_idx, (_, color) in enumerate(STRATEGIES):
        full_ax = axes[0, col_idx]
        zoom_ax = axes[1, col_idx]
        full_ax.add_patch(
            Rectangle(
                (zoom_t_min, 0),
                x_max - zoom_t_min,
                zoom_y_top,
                fill=False,
                edgecolor=color,
                linewidth=1.4,
                linestyle="--",
                alpha=0.75,
                zorder=6,
            )
        )
        for x_value in (zoom_t_min, x_max):
            fig.add_artist(
                ConnectionPatch(
                    xyA=(x_value, 0),
                    coordsA=full_ax.transData,
                    xyB=(x_value, zoom_y_top),
                    coordsB=zoom_ax.transData,
                    color=color,
                    linewidth=1.2,
                    alpha=0.5,
                    zorder=0,
                    clip_on=False,
                )
            )

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = ArgumentParser(
        description=(
            "Bin CAP and EXPRESS calibration sizes within each run, then average "
            "the run-level bin means across runs."
        )
    )
    parser.add_argument(
        "--result-dir",
        type=Path,
        required=True,
        help="Result directory containing raw_selected_events.csv.",
    )
    parser.add_argument("--bin-width", type=int, default=100)
    parser.add_argument("--chunksize", type=int, default=500_000)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--summary-csv", type=Path, default=None)
    parser.add_argument("--title", default=None)
    parser.add_argument(
        "--finite-threshold",
        type=float,
        default=None,
        help="Optional calibration-size threshold to mark with a dashed line.",
    )
    parser.add_argument(
        "--t-min",
        type=int,
        default=None,
        help="Optionally restrict the plot to bins starting at or after this t.",
    )
    parser.add_argument(
        "--combined-zoom-t-min",
        type=int,
        default=None,
        help="Write a 2x2 figure containing the full range and a zoom from this t.",
    )
    args = parser.parse_args()

    if args.bin_width <= 0:
        raise ValueError("--bin-width must be positive")

    raw_path = args.result_dir / "raw_selected_events.csv"
    if not raw_path.exists():
        raise FileNotFoundError(f"Missing {raw_path}")

    output_name = DEFAULT_OUTPUT_NAME.replace("bin100", f"bin{args.bin_width}")
    if args.combined_zoom_t_min is not None:
        output_name = output_name.replace(
            ".png",
            f"_full_and_t{args.combined_zoom_t_min}_plus.png",
        )
    elif args.t_min is not None:
        output_name = output_name.replace(".png", f"_t{args.t_min}_plus.png")
    summary_name = DEFAULT_SUMMARY_NAME.replace("bin100", f"bin{args.bin_width}")
    output_path = args.output or args.result_dir / output_name
    summary_path = args.summary_csv or args.result_dir / summary_name

    summary = summarize_run_bins(
        raw_path,
        bin_width=args.bin_width,
        chunksize=args.chunksize,
    )
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(summary_path, index=False)
    if args.combined_zoom_t_min is not None:
        plot_full_and_zoom(
            summary,
            output_path,
            zoom_t_min=args.combined_zoom_t_min,
            title=args.title,
            finite_threshold=args.finite_threshold,
        )
    else:
        plot_summary(
            summary,
            output_path,
            title=args.title,
            t_min=args.t_min,
            finite_threshold=args.finite_threshold,
        )

    print(f"Wrote plot to {output_path}")
    print(f"Wrote summary to {summary_path}")
    for strategy, _ in STRATEGIES:
        panel = summary[summary["strategy"].eq(strategy)]
        print(
            f"{strategy}: bins={len(panel)}, "
            f"contributing runs={panel['contributing_runs'].min()}-"
            f"{panel['contributing_runs'].max()}"
        )


if __name__ == "__main__":
    main()
