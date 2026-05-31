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
DEFAULT_OUTPUT_NAME = "relaxed_express_delta_sweep_panels.png"


def load_result_config(run_dir):
    for filename in ("resolved_config.json", "config.json"):
        config_path = Path(run_dir) / filename
        if config_path.exists():
            with config_path.open() as f:
                return json.load(f)
    raise FileNotFoundError(f"Missing config.json or resolved_config.json under {run_dir}")


def bool_series(series):
    if series.dtype == bool:
        return series
    return series.astype(str).str.lower().isin(["true", "1", "1.0"])


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


def summarize_run_metrics(raw_df, subset_label):
    if raw_df.empty:
        return {
            "subset": subset_label,
            "miscoverage": np.nan,
            "interval_length": np.nan,
            "mean_n_calibration": np.nan,
            "n_events": 0,
            "n_runs": 0,
        }

    per_run = (
        raw_df.groupby("run", sort=True)
        .agg(
            miscoverage=("miscovered", "mean"),
            interval_length=("interval_length", "mean"),
            mean_n_calibration=("n_calibration", "mean"),
            n_events=("miscovered", "size"),
        )
        .reset_index()
    )
    return {
        "subset": subset_label,
        "miscoverage": per_run["miscoverage"].mean(),
        "interval_length": per_run["interval_length"].mean(),
        "mean_n_calibration": per_run["mean_n_calibration"].mean(),
        "n_events": int(per_run["n_events"].sum()),
        "n_runs": int(len(per_run)),
    }


def summarize_run_dir(run_dir):
    config = load_result_config(run_dir)
    delta = config.get("conformal", {}).get("relaxed_express_rank_delta")
    if delta is None:
        raise ValueError(f"{run_dir} has no conformal.relaxed_express_rank_delta")
    delta = float(delta)

    raw_path = Path(run_dir) / "raw_selected_events.csv"
    usecols = [
        "run",
        "strategy",
        "miscovered",
        "n_calibration",
        "interval_length",
        "relaxed_express_relaxation_needed",
    ]
    raw_df = pd.read_csv(raw_path, usecols=usecols)
    raw_df = raw_df[raw_df["strategy"] == "RELAXED-EXPRESS"].copy()
    raw_df["interval_length"] = pd.to_numeric(raw_df["interval_length"], errors="coerce")
    raw_df["n_calibration"] = pd.to_numeric(raw_df["n_calibration"], errors="coerce")

    all_summary = summarize_run_metrics(raw_df, "all")
    needed_mask = bool_series(raw_df["relaxed_express_relaxation_needed"])
    needed_summary = summarize_run_metrics(raw_df[needed_mask].copy(), "relaxation_needed")

    rows = []
    for summary in (all_summary, needed_summary):
        summary.update({
            "delta": delta,
            "run_dir": Path(run_dir).name,
        })
        rows.append(summary)
    return rows


def summarize_suite(suite_dir):
    rows = []
    for run_dir in result_dirs(suite_dir):
        rows.extend(summarize_run_dir(run_dir))
    summary_df = pd.DataFrame(rows)
    return summary_df.sort_values(["delta", "subset"]).reset_index(drop=True)


def plot_delta_sweep(summary_df, save_path):
    fig, axes = plt.subplots(2, 3, figsize=(14, 7), sharex=True)

    panels = [
        ("miscoverage", "Miscoverage rate"),
        ("interval_length", "Mean interval length"),
        ("mean_n_calibration", "Mean calibration set size"),
    ]
    subsets = [
        ("all", "All selected events"),
        ("relaxation_needed", "relaxed_express_relaxation_needed=True"),
    ]

    for row_idx, (subset, row_title) in enumerate(subsets):
        subset_df = summary_df[summary_df["subset"] == subset].sort_values("delta")
        x = subset_df["delta"].to_numpy(dtype=float)

        for col_idx, (metric, ylabel) in enumerate(panels):
            ax = axes[row_idx, col_idx]
            y = subset_df[metric].to_numpy(dtype=float)
            ax.plot(x, y, marker="o", linewidth=1.5, color="tab:blue")
            ax.set_title(row_title if col_idx == 0 else "")
            ax.set_ylabel(ylabel)
            ax.grid(alpha=0.25)

            if metric == "miscoverage":
                ax.axhline(0.4, color="red", linestyle="--", linewidth=1.0, label="alpha=0.4")
                ax.legend(loc="best", fontsize=8)

    deltas = np.sort(summary_df["delta"].unique())
    for ax in axes[-1, :]:
        ax.set_xscale("log")
        ax.set_xlabel("relaxed_express_rank_delta")
        ax.set_xticks(deltas)
        ax.set_xticklabels([f"{delta:g}" for delta in deltas], rotation=35, ha="right")

    fig.suptitle("RELAXED-EXPRESS Delta Sweep")
    fig.tight_layout(rect=(0, 0.02, 1, 0.95))

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def parse_args():
    parser = ArgumentParser(description="Plot RELAXED-EXPRESS delta-sweep metrics.")
    parser.add_argument(
        "--suite-dir",
        type=Path,
        required=True,
        help="Suite directory containing delta experiment result subdirectories.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional output image path. Defaults under the suite directory.",
    )
    parser.add_argument(
        "--summary-csv",
        type=Path,
        default=None,
        help="Optional CSV path for the computed sweep summary.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    suite_dir = Path(args.suite_dir)
    output_path = args.output or suite_dir / DEFAULT_OUTPUT_NAME
    summary_csv = args.summary_csv or suite_dir / "relaxed_express_delta_sweep_summary.csv"

    summary_df = summarize_suite(suite_dir)
    summary_df.to_csv(summary_csv, index=False)
    plot_delta_sweep(summary_df, output_path)

    print(f"Wrote plot to {output_path}")
    print(f"Wrote summary CSV to {summary_csv}")


if __name__ == "__main__":
    main()
