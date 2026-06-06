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

DATA_PATH = PROJECT_ROOT / "data" / "davis_other_data_models.csv"
RESULTS_DIR = PROJECT_ROOT / "results"
VIS_DIR = PROJECT_ROOT / "data" / "vis"
TARGET_ALPHA = 0.4
STRATEGY_ORDER = [
    "FULL",
    "S-FIX",
    "S-FULL",
    "ADA",
    "EXPRESS",
    "RELAXED-EXPRESS",
    "WEIGHTED-EXPRESS",
    "EXPRESS-M",
    "K-EXPRESS",
]
PROVABLY_CORRECT = {"S-FIX", "EXPRESS", "EXPRESS-M", "K-EXPRESS"}
METHOD_NOTES = {
    "ADA": "(Bao et al.)",
    "EXPRESS": "(Sale and Ramdas)",
    "RELAXED-EXPRESS": "(new)",
    "WEIGHTED-EXPRESS": "(new)",
    "EXPRESS-M": "(Sale and Ramdas)",
    "K-EXPRESS": "(Sale and Ramdas)",
}
DEFAULT_STRATEGY_LABELS = {
    "WEIGHTED-EXPRESS": "WEIGHTED-\nEXPRESS",
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
    labels["K-EXPRESS"] = f"{k_express}-EXPRESS"
    return labels


def summarize_events(raw_df):
    rows = []
    for strategy, strategy_df in raw_df.groupby("strategy", sort=False):
        interval_lengths = strategy_df["interval_length"].to_numpy()
        rows.append(
            {
                "strategy": strategy,
                "miscoverage": strategy_df["miscovered"].mean(),
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


def plot_sale_ramdas_style(
    metrics_df,
    save_path,
    title,
    target_alpha=TARGET_ALPHA,
    parameter_text=None,
    strategy_labels=None,
):
    metrics_df = ordered_metrics(metrics_df)
    strategies = metrics_df.index.to_list()
    strategy_labels = strategy_labels or {}
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

    fig, ax = plt.subplots(figsize=(10.5, 6.2))
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
    ax.set_xticklabels(xtick_labels, fontsize=9)
    ax.grid(alpha=0.25)
    ax.legend(loc="upper right")

    if parameter_text:
        fig.text(0.5, 0.018, parameter_text, ha="center", va="bottom", fontsize=9)
    fig.tight_layout(rect=(0, 0.08, 1, 1))

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_all_runs(raw_df, result_dir, max_runs=None, strategy_labels=None):
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
            strategy_labels=strategy_labels,
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
        help="Also write per-run diagnostic plots. Default writes aggregate means only.",
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
    return parser.parse_args()


def main():
    args = parse_args()
    raw_df, result_dir = load_raw_events(args.result_dir)
    config = load_result_config(result_dir)
    strategy_labels = strategy_label_overrides_from_config(config)
    summary_df = limit_runs(raw_df, args.summary_max_runs)

    if args.summary_max_runs is None:
        summary_path = result_dir / "sale_ramdas_style_summary.png"
    else:
        summary_path = result_dir / f"sale_ramdas_style_summary_first_{args.summary_max_runs}_runs.png"

    plot_sale_ramdas_style(
        summarize_run_means(summary_df),
        summary_path,
        title=None if args.hide_title else f"Sale-Ramdas Style Run Means - {result_dir.name}",
        parameter_text=args.parameter_text,
        strategy_labels=strategy_labels,
    )

    run_dir = None
    n_run_plots = 0
    if args.write_run_plots:
        run_dir, n_run_plots = plot_all_runs(
            raw_df,
            result_dir,
            max_runs=args.max_runs,
            strategy_labels=strategy_labels,
        )

    print(f"Wrote summary plot to {summary_path}")
    if run_dir is not None:
        print(f"Wrote {n_run_plots} per-run plots to {run_dir}")
    if not args.skip_signed_residuals:
        residual_path = plot_signed_residuals()
        print(f"Wrote signed residual plot to {residual_path}")


if __name__ == "__main__":
    main()
