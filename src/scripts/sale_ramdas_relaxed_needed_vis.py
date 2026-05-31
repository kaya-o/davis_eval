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


RESULTS_DIR = PROJECT_ROOT / "results"
TARGET_ALPHA = 0.4
RELAXED_NEEDED_KEY = "RELAXED-EXPRESS-NEEDED"
STRATEGY_ORDER = [
    "FULL",
    "S-FIX",
    "S-FULL",
    "ADA",
    "EXPRESS",
    "RELAXED-EXPRESS",
    RELAXED_NEEDED_KEY,
    "EXPRESS-M",
    "K-EXPRESS",
]
PROVABLY_CORRECT = {"S-FIX", "EXPRESS", "EXPRESS-M", "K-EXPRESS"}
METHOD_NOTES = {
    "ADA": "(Bao et al.)",
    "EXPRESS": "(Sale and Ramdas)",
    "RELAXED-EXPRESS": "(diagnostic)",
    "EXPRESS-M": "(Sale and Ramdas)",
    "K-EXPRESS": "(Sale and Ramdas)",
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


def bool_series(series):
    if series.dtype == bool:
        return series
    return series.astype(str).str.lower().isin(["true", "1", "1.0"])


def strategy_label_overrides_from_config(config):
    k_express = config.get("conformal", {}).get("k_express")
    labels = {
        RELAXED_NEEDED_KEY: "RELAXED-EXPRESS\n(Only when express is relaxed)",
    }
    if k_express is not None:
        if isinstance(k_express, float) and k_express.is_integer():
            k_express = int(k_express)
        labels["K-EXPRESS"] = f"{k_express}-EXPRESS"
    return labels


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
                "n_events": len(strategy_df),
            }
        )

    run_metrics_df = pd.DataFrame(run_metrics)
    return run_metrics_df.groupby("strategy", sort=False)[
        [
            "miscoverage",
            "avg_n_calibration",
            "median_interval_length",
            "infinite_fraction",
            "n_events",
        ]
    ].mean()


def relaxed_needed_run_means(raw_df):
    if "relaxed_express_relaxation_needed" not in raw_df.columns:
        raise ValueError("raw_selected_events.csv is missing relaxed_express_relaxation_needed")

    relaxed_df = raw_df[raw_df["strategy"] == "RELAXED-EXPRESS"].copy()
    needed_df = relaxed_df[bool_series(relaxed_df["relaxed_express_relaxation_needed"])].copy()
    if needed_df.empty:
        raise ValueError("No RELAXED-EXPRESS rows with relaxed_express_relaxation_needed=True")

    needed_df["strategy"] = RELAXED_NEEDED_KEY
    return needed_df


def build_metrics(raw_df):
    standard_metrics = summarize_run_means(raw_df)
    needed_metrics = summarize_run_means(relaxed_needed_run_means(raw_df))
    metrics_df = pd.concat([standard_metrics, needed_metrics], axis=0)
    present_order = [strategy for strategy in STRATEGY_ORDER if strategy in metrics_df.index]
    remaining = [strategy for strategy in metrics_df.index if strategy not in present_order]
    return metrics_df.loc[present_order + remaining]


def box_text(ax, x, y, text, facecolor):
    ax.text(
        x,
        y,
        text,
        ha="center",
        va="center",
        fontsize=8,
        bbox={
            "boxstyle": "round,pad=0.22",
            "facecolor": facecolor,
            "edgecolor": "black",
            "linewidth": 0.8,
        },
        zorder=4,
    )


def plot_sale_ramdas_relaxed_needed(
    metrics_df,
    save_path,
    title,
    target_alpha=TARGET_ALPHA,
    parameter_text=None,
    strategy_labels=None,
):
    strategy_labels = strategy_labels or {}
    strategies = metrics_df.index.to_list()
    xtick_labels = [strategy_labels.get(strategy, strategy) for strategy in strategies]
    x = np.arange(len(strategies))
    miscoverage = metrics_df["miscoverage"].to_numpy(dtype=float)

    y_min = min(0.25, target_alpha - 0.15, np.nanmin(miscoverage) - 0.08)
    y_max = max(0.50, target_alpha + 0.10, np.nanmax(miscoverage) + 0.06)
    span = y_max - y_min
    row_y = {
        "n_cal": y_min + 0.045 * span,
        "length": y_min + 0.095 * span,
        "inf": y_min + 0.145 * span,
    }

    fig, ax = plt.subplots(figsize=(12.8, 6.5))
    ax.scatter(
        x,
        miscoverage,
        marker="x",
        s=85,
        linewidths=1.6,
        color="tab:blue",
        label="Miscoverage",
        zorder=5,
    )
    ax.axhline(
        target_alpha,
        color="red",
        linestyle="--",
        linewidth=1.2,
        label=f"Target ({target_alpha:g})",
        zorder=2,
    )

    if RELAXED_NEEDED_KEY in strategies:
        needed_x = strategies.index(RELAXED_NEEDED_KEY)
        ax.axvspan(needed_x - 0.45, needed_x + 0.45, color="tab:orange", alpha=0.08, zorder=1)

    for i, strategy in enumerate(strategies):
        row = metrics_df.loc[strategy]
        box_text(ax, i, row_y["inf"], f"{row['infinite_fraction']:.3f}", "#f5d6b4")
        box_text(ax, i, row_y["length"], f"{row['median_interval_length']:.3f}", "#b9d7e8")
        box_text(ax, i, row_y["n_cal"], f"{row['avg_n_calibration']:.3f}", "#eeeeee")

        note = METHOD_NOTES.get(strategy)
        if note:
            ax.text(
                i,
                -0.065,
                note,
                ha="center",
                va="top",
                fontsize=7,
                transform=ax.get_xaxis_transform(),
            )
        if strategy in PROVABLY_CORRECT:
            ax.text(
                i,
                -0.11,
                r"$\checkmark$",
                ha="center",
                va="top",
                fontsize=14,
                color="tab:green",
                transform=ax.get_xaxis_transform(),
            )

    if title:
        ax.set_title(title)
    ax.set_ylabel("Miscoverage")
    ax.set_ylim(y_min, y_max)
    ax.set_xticks(x)
    ax.set_xticklabels(xtick_labels, fontsize=8)
    ax.grid(alpha=0.25)
    ax.legend(loc="upper right")

    if parameter_text:
        fig.text(0.5, 0.018, parameter_text, ha="center", va="bottom", fontsize=9)
    fig.tight_layout(rect=(0, 0.08, 1, 1))

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def parse_args():
    parser = ArgumentParser(
        description="Sale-Ramdas-style plot with a relaxation-needed RELAXED-EXPRESS column."
    )
    parser.add_argument(
        "--result-dir",
        type=Path,
        default=None,
        help="Result directory containing raw_selected_events.csv. Defaults to latest.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional output image path. Defaults under the result directory.",
    )
    parser.add_argument(
        "--hide-title",
        action="store_true",
        help="Do not print the result directory title on the summary figure.",
    )
    parser.add_argument(
        "--parameter-text",
        default=None,
        help="Optional parameter text to print on the summary figure.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    raw_df, result_dir = load_raw_events(args.result_dir)
    config = load_result_config(result_dir)
    metrics_df = build_metrics(raw_df)
    strategy_labels = strategy_label_overrides_from_config(config)
    output_path = args.output or result_dir / "sale_ramdas_relaxed_needed_summary.png"

    plot_sale_ramdas_relaxed_needed(
        metrics_df,
        output_path,
        title=None if args.hide_title else f"Sale-Ramdas Style Relaxed Summary - {result_dir.name}",
        parameter_text=args.parameter_text,
        strategy_labels=strategy_labels,
    )

    needed_events = raw_df[
        (raw_df["strategy"] == "RELAXED-EXPRESS")
        & bool_series(raw_df["relaxed_express_relaxation_needed"])
    ]
    print(f"Wrote summary plot to {output_path}")
    print(
        "RELAXED-EXPRESS relaxation-needed events: "
        f"{len(needed_events):,} across {needed_events['run'].nunique()} runs"
    )


if __name__ == "__main__":
    main()
