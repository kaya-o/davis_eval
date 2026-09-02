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
from matplotlib.ticker import MaxNLocator

DATA_PATH = PROJECT_ROOT / "data" / "davis_other_data_models.csv"
RESULTS_DIR = PROJECT_ROOT / "results"
VIS_DIR = PROJECT_ROOT / "data" / "vis"
TARGET_ALPHA = 0.4
FINITE_RELAXED_KEY = "FINITE-EXPRESS-RELAXED"
STRATEGY_ORDER = [
    "FULL",
    "S-FIX",
    "S-FULL",
    "ADA",
    "CAP",
    "EXPRESS",
    "FINITE-EXPRESS",
    FINITE_RELAXED_KEY,
    "RELAXED-EXPRESS",
    "WEIGHTED-EXPRESS",
    "WEIGHTED-NEIGHBORHOOD-EXPRESS",
    "ADAPTIVE-WEIGHTED-EXPRESS",
    "EXPRESS-M",
    "K-EXPRESS",
]
PROVABLY_CORRECT = {"S-FIX", "CAP", "EXPRESS", "EXPRESS-M", "K-EXPRESS"}
METHOD_NOTES = {
    "ADA": "(Bao et al.)",
    "CAP": "(Bao et al.)",
    "EXPRESS": "(Sale and Ramdas)",
    "FINITE-EXPRESS": "(novel)",
    "RELAXED-EXPRESS": "(novel)",
    "WEIGHTED-EXPRESS": "(novel)",
    "ADAPTIVE-WEIGHTED-EXPRESS": "(novel)",
    "WEIGHTED-NEIGHBORHOOD-EXPRESS": "(novel)",
    "EXPRESS-M": "(Sale and Ramdas)",
    "K-EXPRESS": "(Sale and Ramdas)",
}
DEFAULT_STRATEGY_LABELS = {
    "S-FIX": "S-FIX",
    "S-FULL": "S-FULL",
    "FINITE-EXPRESS": "FINITE\nEXPRESS",
    FINITE_RELAXED_KEY: "FINITE\nEXPRESS\n($b_t > 0$)",
    "RELAXED-EXPRESS": "RELAXED\nEXPRESS",
    "WEIGHTED-EXPRESS": "WEIGHTED\nEXPRESS",
    "ADAPTIVE-WEIGHTED-EXPRESS": "ADAPTIVE-\nWEIGHTED-\nEXPRESS",
    "WEIGHTED-NEIGHBORHOOD-EXPRESS": "WEIGHTED\nNBRHD\nEXPRESS",
    "EXPRESS-M": "EXPRESS-M",
    "K-EXPRESS": "K\nEXPRESS",
}


def latest_results_dir(results_dir=RESULTS_DIR):
    candidates = [
        path
        for path in Path(results_dir).glob("*_runs")
        if (path / "raw_selected_events.csv").exists()
    ]
    if not candidates:
        raise FileNotFoundError(f"No raw_selected_events.csv found under {results_dir}")
    return max(candidates, key=lambda path: path.stat().st_mtime)


def load_raw_events(result_dir=None):
    result_dir = latest_results_dir() if result_dir is None else Path(result_dir)
    raw_path = result_dir / "raw_selected_events.csv"
    if not raw_path.exists():
        raise FileNotFoundError(f"Missing {raw_path}")

    raw_df = pd.read_csv(raw_path)
    raw_df["interval_length"] = pd.to_numeric(raw_df["interval_length"], errors="coerce")
    return raw_df, result_dir


def load_result_config(result_dir):
    for filename in ("config.json", "resolved_config.json"):
        config_path = Path(result_dir) / filename
        if config_path.exists():
            with config_path.open() as f:
                return json.load(f)
    return {}


def strategy_label_overrides_from_config(config):
    labels = dict(DEFAULT_STRATEGY_LABELS)
    k_express = config.get("conformal", {}).get("k_express")
    if k_express is None:
        return labels

    if isinstance(k_express, float) and k_express.is_integer():
        k_express = int(k_express)
    labels["K-EXPRESS"] = f"{k_express}\nEXPRESS"
    return labels


def target_alpha_from_config(config, default=TARGET_ALPHA):
    value = config.get("conformal", {}).get("alpha", default)
    try:
        value = float(value)
    except (TypeError, ValueError):
        return default
    return value if 0 < value < 1 else default


def label_range_from_config(config, default_data_path=DATA_PATH):
    data_config = config.get("data", {})
    data_path = Path(data_config.get("path", default_data_path))
    if not data_path.is_absolute():
        data_path = PROJECT_ROOT / data_path

    label_column = data_config.get("label_column", "Label")
    labels = pd.read_csv(data_path, usecols=[label_column])[label_column]
    labels = pd.to_numeric(labels, errors="coerce").dropna()
    if labels.empty:
        return None

    return float(labels.min()), float(labels.max())


def summarize_events(raw_df):
    rows = []
    for strategy, strategy_df in raw_df.groupby("strategy", sort=False):
        interval_lengths = strategy_df["interval_length"].to_numpy()
        run_miscoverage = (
            strategy_df.groupby("run", sort=False)["miscovered"].mean().to_numpy()
        )
        rows.append(
            {
                "strategy": strategy,
                "miscoverage": strategy_df["miscovered"].mean(),
                "miscoverage_q10": np.quantile(run_miscoverage, 0.10),
                "miscoverage_q90": np.quantile(run_miscoverage, 0.90),
                "avg_n_calibration": strategy_df["n_calibration"].mean(),
                "median_interval_length": np.nanmedian(interval_lengths),
                "infinite_fraction": np.isinf(interval_lengths).mean(),
            }
        )
    return pd.DataFrame(rows).set_index("strategy")


def summarize_run_means(raw_df):
    run_metrics = []
    for (run_id, strategy), strategy_df in raw_df.groupby(["run", "strategy"], sort=False):
        interval_lengths = strategy_df["interval_length"].to_numpy()
        run_metrics.append(
            {
                "run": run_id,
                "strategy": strategy,
                "miscoverage": strategy_df["miscovered"].mean(),
                "avg_n_calibration": strategy_df["n_calibration"].mean(),
                "median_interval_length": np.nanmedian(interval_lengths),
                "infinite_fraction": np.isinf(interval_lengths).mean(),
            }
        )

    run_metrics_df = pd.DataFrame(run_metrics)
    return run_metrics_df.groupby("strategy", sort=False)[
        [
            "miscoverage",
            "avg_n_calibration",
            "median_interval_length",
            "infinite_fraction",
        ]
    ].mean()


def finite_relaxed_events(raw_df):
    column = "finite_express_added_nonexact"
    if column not in raw_df.columns:
        raise ValueError(f"raw_selected_events.csv is missing {column}")

    finite_df = raw_df[raw_df["strategy"].eq("FINITE-EXPRESS")].copy()
    added_nonexact = pd.to_numeric(finite_df[column], errors="coerce")
    relaxed_df = finite_df[added_nonexact.gt(0)].copy()
    if relaxed_df.empty:
        raise ValueError(
            "No FINITE-EXPRESS rows have finite_express_added_nonexact > 0"
        )

    relaxed_df["strategy"] = FINITE_RELAXED_KEY
    return relaxed_df


def limit_runs(raw_df, max_runs):
    if max_runs is None:
        return raw_df

    run_ids = sorted(raw_df["run"].unique())[:max_runs]
    return raw_df[raw_df["run"].isin(run_ids)].copy()


def ordered_metrics(metrics_df):
    present_order = [
        strategy for strategy in STRATEGY_ORDER if strategy in metrics_df.index
    ]
    remaining = [
        strategy for strategy in metrics_df.index if strategy not in present_order
    ]
    return metrics_df.loc[present_order + remaining]


def box_text(ax, x, y, text, facecolor):
    return ax.text(
        x,
        y,
        text,
        ha="center",
        va="center",
        fontsize=12,
        bbox={
            "boxstyle": "square,pad=0.22",
            "facecolor": facecolor,
            "edgecolor": "black",
            "linewidth": 0.8,
        },
        zorder=4,
    )


def stack_box_column(fig, ax, boxes, center_y):
    """Stack text boxes edge-to-edge around a shared data-coordinate center."""
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    heights = [
        box.get_bbox_patch().get_window_extent(renderer).height
        for box in boxes
    ]
    center_display = ax.transData.transform((0.0, center_y))[1]
    next_bottom = center_display - 0.5 * sum(heights)
    inverse = ax.transData.inverted()

    for box, height in zip(boxes, heights):
        box_center_display = next_bottom + 0.5 * height
        box.set_y(inverse.transform((0.0, box_center_display))[1])
        next_bottom += height


def format_interval_length(value, label_range=None):
    text = f"{value:.3f}"
    if np.isposinf(value):
        return "inf"
    if label_range is None or not np.isfinite(value):
        return text

    label_min, label_max = label_range
    label_span = label_max - label_min
    if not np.isfinite(label_span) or label_span <= 0:
        return text

    return f"{text} ({100.0 * value / label_span:.1f}%)"


def format_calibration_size(value):
    if not np.isfinite(value):
        return "-"
    return f"{value:.3f}"


def miscoverage_axis_layout(miscoverage, lower_errors, upper_errors, target_alpha):
    finite = np.concatenate(
        [
            np.asarray(miscoverage, dtype=float),
            np.asarray(lower_errors, dtype=float),
            np.asarray(upper_errors, dtype=float),
        ]
    )
    finite = finite[np.isfinite(finite)]
    values = np.concatenate([finite, np.asarray([target_alpha], dtype=float)])
    low = float(np.nanmin(values))
    high = float(np.nanmax(values))
    data_span = max(high - low, 0.04)

    data_floor = max(0.0, low - max(0.06 * data_span, 0.006))
    y_max = min(1.0, high + max(0.12 * data_span, 0.012))
    data_region_height = max(y_max - data_floor, 0.04)
    annotation_band_height = 0.18 * data_region_height
    y_min = data_floor - annotation_band_height

    row_y = {
        "n_cal": y_min + 0.18 * annotation_band_height,
        "length": y_min + 0.50 * annotation_band_height,
        "inf": y_min + 0.82 * annotation_band_height,
    }

    tick_locator = MaxNLocator(nbins=7)
    y_ticks = tick_locator.tick_values(data_floor, y_max)
    y_ticks = y_ticks[(y_ticks >= data_floor) & (y_ticks <= y_max)]
    return y_min, y_max, row_y, y_ticks


def plot_sale_ramdas_style(
    metrics_df,
    save_path,
    title,
    target_alpha=TARGET_ALPHA,
    parameter_text=None,
    strategy_labels=None,
    label_range=None,
    calibration_text_overrides=None,
    method_note_overrides=None,
    figsize=(13.5, 6.8),
    dpi=300,
):
    metrics_df = ordered_metrics(metrics_df)
    strategies = metrics_df.index.to_list()
    strategy_labels = strategy_labels or {}
    calibration_text_overrides = calibration_text_overrides or {}
    method_note_overrides = method_note_overrides or {}
    xtick_labels = [strategy_labels.get(strategy, strategy) for strategy in strategies]
    x = np.arange(len(strategies))
    miscoverage = metrics_df["miscoverage"].to_numpy(dtype=float)
    miscoverage_q10 = metrics_df["miscoverage_q10"].to_numpy(dtype=float)
    miscoverage_q90 = metrics_df["miscoverage_q90"].to_numpy(dtype=float)

    y_min, y_max, row_y, y_ticks = miscoverage_axis_layout(
        miscoverage,
        miscoverage_q10,
        miscoverage_q90,
        target_alpha,
    )

    fig, ax = plt.subplots(figsize=figsize)
    stat_box_columns = []
    ax.scatter(
        x,
        miscoverage,
        marker="x",
        s=85,
        linewidths=1.6,
        color="tab:blue",
        label="Pooled miscoverage",
        zorder=5,
    )
    error_midpoints = 0.5 * (miscoverage_q10 + miscoverage_q90)
    error_half_widths = 0.5 * (miscoverage_q90 - miscoverage_q10)
    ax.errorbar(
        x,
        error_midpoints,
        yerr=error_half_widths,
        fmt="none",
        ecolor="tab:blue",
        elinewidth=1.3,
        capsize=4,
        capthick=1.3,
        label="10th–90th percentile across runs",
        zorder=4,
    )
    ax.axhline(
        target_alpha,
        color="red",
        linestyle="--",
        linewidth=1.2,
        label=f"Target ({target_alpha:g})",
        zorder=2,
    )

    for i, strategy in enumerate(strategies):
        row = metrics_df.loc[strategy]
        infinite_box = box_text(
            ax,
            i,
            row_y["length"],
            f"{row['infinite_fraction']:.3f}",
            "#f5d6b4",
        )
        length_box = box_text(
            ax,
            i,
            row_y["length"],
            format_interval_length(row["median_interval_length"], label_range),
            "#b9d7e8",
        )
        calibration_box = box_text(
            ax,
            i,
            row_y["length"],
            calibration_text_overrides.get(
                strategy,
                format_calibration_size(row["avg_n_calibration"]),
            ),
            "#eeeeee",
        )
        stat_box_columns.append(
            (calibration_box, length_box, infinite_box)
        )

        note = method_note_overrides.get(strategy, METHOD_NOTES.get(strategy))
        if note:
            ax.text(
                i,
                -0.15,
                note,
                ha="center",
                va="top",
                fontsize=9,
                transform=ax.get_xaxis_transform(),
            )
        if strategy in PROVABLY_CORRECT:
            ax.text(
                i,
                -0.23,
                r"$\checkmark$",
                ha="center",
                va="top",
                fontsize=17,
                color="tab:green",
                transform=ax.get_xaxis_transform(),
            )

    if title:
        ax.set_title(title, fontsize=17)
    ax.set_ylabel("Miscoverage", fontsize=14)
    ax.set_ylim(y_min, y_max)
    ax.set_yticks(y_ticks)
    ax.set_xticks(x)
    ax.set_xticklabels(xtick_labels, fontsize=13)
    ax.tick_params(axis="y", labelsize=13)
    ax.grid(alpha=0.25)
    ax.legend(loc="upper right", fontsize=11)

    if parameter_text:
        fig.text(0.5, 0.018, parameter_text, ha="center", va="bottom", fontsize=11)
    fig.tight_layout(rect=(0, 0.18, 1, 1))
    for boxes in stat_box_columns:
        stack_box_column(fig, ax, boxes, row_y["length"])

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def plot_all_runs(
    raw_df,
    result_dir,
    max_runs=None,
    strategy_labels=None,
    target_alpha=TARGET_ALPHA,
    label_range=None,
):
    run_dir = VIS_DIR / "sale_ramdas_style_runs" / result_dir.name
    run_ids = sorted(raw_df["run"].unique())
    if max_runs is not None:
        run_ids = run_ids[:max_runs]

    for run_id in run_ids:
        run_df = raw_df[raw_df["run"] == run_id]
        metrics_df = summarize_events(run_df)
        plot_sale_ramdas_style(
            metrics_df,
            run_dir / f"run_{int(run_id):04d}.png",
            title=f"Sale-Ramdas Style Summary - Run {int(run_id)}",
            target_alpha=target_alpha,
            strategy_labels=strategy_labels,
            label_range=label_range,
        )

    return run_dir, len(run_ids)


def plot_signed_residuals(data_path=DATA_PATH, save_path=VIS_DIR / "muhat_1_minus_y.png"):
    data_df = pd.read_csv(data_path)
    signed_residuals = data_df["muhat_1"].to_numpy() - data_df["Label"].to_numpy()
    x = np.arange(signed_residuals.size)

    fig, ax = plt.subplots(figsize=(14, 6))
    ax.scatter(x, signed_residuals, s=10, alpha=0.35, edgecolors="none")
    ax.axhline(0, color="black", linestyle="--", linewidth=1.2)
    ax.axhline(np.mean(signed_residuals), color="tab:red", linewidth=1.6, label="Mean")
    ax.axhline(
        np.median(signed_residuals),
        color="tab:orange",
        linewidth=1.6,
        label="Median",
    )
    ax.set_title("muhat_1 - Y Across DAVIS Datapoints")
    ax.set_xlabel("Datapoint index")
    ax.set_ylabel("muhat_1 - Y")
    ax.grid(alpha=0.25)
    ax.legend()

    stats = (
        f"Mean: {np.mean(signed_residuals):.6f}    "
        f"Median: {np.median(signed_residuals):.6f}    "
        f"Std. dev.: {np.std(signed_residuals):.6f}"
    )
    fig.text(0.5, 0.02, stats, ha="center", va="bottom")
    fig.tight_layout(rect=(0, 0.07, 1, 1))

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return save_path


def parse_args():
    parser = ArgumentParser()
    parser.add_argument(
        "--result-dir",
        type=Path,
        default=None,
        help="Result directory containing raw_selected_events.csv. Defaults to latest.",
    )
    parser.add_argument(
        "--max-runs",
        type=int,
        default=None,
        help="Limit optional per-run plots when --write-run-plots is set.",
    )
    parser.add_argument(
        "--write-run-plots",
        action="store_true",
        help="Also write per-run diagnostic plots. Default writes a pooled aggregate only.",
    )
    parser.add_argument(
        "--parameter-text",
        default=None,
        help="Optional parameter text to print on the summary figure.",
    )
    parser.add_argument(
        "--hide-title",
        action="store_true",
        help="Do not print the result directory title on the summary figure.",
    )
    parser.add_argument(
        "--title",
        default=None,
        help="Optional custom title for the aggregate summary figure.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional output path for the aggregate summary figure.",
    )
    parser.add_argument(
        "--summary-max-runs",
        type=int,
        default=None,
        help="Limit the runs used for the aggregate summary plot.",
    )
    parser.add_argument(
        "--skip-signed-residuals",
        action="store_true",
        help="Do not write the signed residual diagnostic plot.",
    )
    parser.add_argument(
        "--include-finite-relaxed",
        action="store_true",
        help=(
            "Add a FINITE-EXPRESS column restricted to events where "
            "finite_express_added_nonexact is greater than zero."
        ),
    )
    return parser.parse_args()


def main():
    args = parse_args()
    raw_df, result_dir = load_raw_events(args.result_dir)
    config = load_result_config(result_dir)
    strategy_labels = strategy_label_overrides_from_config(config)
    target_alpha = target_alpha_from_config(config)
    label_range = label_range_from_config(config)
    finite_relaxed_count = 0
    if args.include_finite_relaxed:
        relaxed_df = finite_relaxed_events(raw_df)
        finite_relaxed_count = len(relaxed_df)
        raw_df = pd.concat([raw_df, relaxed_df], ignore_index=True)
    summary_df = limit_runs(raw_df, args.summary_max_runs)

    if args.output is not None:
        summary_path = args.output
    elif args.include_finite_relaxed:
        if args.summary_max_runs is None:
            summary_path = (
                result_dir / "sale_ramdas_style_summary_finite_relaxed.png"
            )
        else:
            summary_path = result_dir / (
                "sale_ramdas_style_summary_finite_relaxed_"
                f"first_{args.summary_max_runs}_runs.png"
            )
    elif args.summary_max_runs is None:
        summary_path = result_dir / "sale_ramdas_style_summary.png"
    else:
        summary_path = result_dir / f"sale_ramdas_style_summary_first_{args.summary_max_runs}_runs.png"

    plot_sale_ramdas_style(
        summarize_events(summary_df),
        summary_path,
        title=(
            None
            if args.hide_title
            else args.title or f"Sale-Ramdas Style Pooled Summary - {result_dir.name}"
        ),
        target_alpha=target_alpha,
        parameter_text=args.parameter_text,
        strategy_labels=strategy_labels,
        label_range=label_range,
    )

    run_dir = None
    n_run_plots = 0
    if args.write_run_plots:
        run_dir, n_run_plots = plot_all_runs(
            raw_df,
            result_dir,
            max_runs=args.max_runs,
            target_alpha=target_alpha,
            strategy_labels=strategy_labels,
            label_range=label_range,
        )

    print(f"Wrote summary plot to {summary_path}")
    if args.include_finite_relaxed:
        print(
            "FINITE-EXPRESS relaxed-only events: "
            f"{finite_relaxed_count:,}"
        )
    if run_dir is not None:
        print(f"Wrote {n_run_plots} per-run plots to {run_dir}")
    if not args.skip_signed_residuals:
        residual_path = plot_signed_residuals()
        print(f"Wrote signed residual plot to {residual_path}")


if __name__ == "__main__":
    main()
