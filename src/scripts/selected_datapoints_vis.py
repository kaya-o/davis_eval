import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = PROJECT_ROOT / "results"


def latest_results_dir(results_dir=RESULTS_DIR):
    candidates = [
        path
        for path in Path(results_dir).glob("*_runs")
        if (path / "selected_datapoints.csv").exists()
    ]
    if not candidates:
        raise FileNotFoundError(f"No selected_datapoints.csv found under {results_dir}")
    return max(candidates, key=lambda path: path.stat().st_mtime)


def prepare_selected_datapoints(csv_path, run=None):
    df = pd.read_csv(csv_path)
    if run is not None:
        df = df[df["run"] == run].copy()
    if df.empty:
        raise ValueError(f"No selected datapoints found in {csv_path}")

    finite = np.isfinite(df["selection_upper_bound"].to_numpy())
    if not finite.all():
        finite_upper_max = df.loc[finite, "selection_upper_bound"].max()
        fallback_upper = max(finite_upper_max, df["score_t"].max())
        df.loc[~finite, "selection_upper_bound"] = fallback_upper

    return df


def draw_selected_datapoints_panel(ax, df, title, interval_alpha=0.18, point_alpha=0.42):
    t = df["t"].to_numpy()
    score = df["score_t"].to_numpy()
    lower = df["selection_lower_bound"].to_numpy()
    upper = df["selection_upper_bound"].to_numpy()

    segments = np.stack(
        [
            np.column_stack([t, lower]),
            np.column_stack([t, upper]),
        ],
        axis=1,
    )

    interval_lines = LineCollection(
        segments,
        colors="#f59e0b",
        linewidths=1.0,
        alpha=interval_alpha,
        zorder=1,
    )
    ax.add_collection(interval_lines)
    ax.scatter(t, score, s=7, alpha=point_alpha, edgecolors="none", color="#1f77b4", zorder=2)
    ax.set_title(title, fontsize=10)
    ax.grid(True, alpha=0.22)


def plot_selected_datapoints(result_dir=None, output_path=None, run=None):
    result_dir = latest_results_dir() if result_dir is None else Path(result_dir)
    csv_path = result_dir / "selected_datapoints.csv"
    if output_path is None:
        suffix = "" if run is None else f"_run_{run}"
        output_path = result_dir / f"selected_datapoints_score_time{suffix}.png"
    else:
        output_path = Path(output_path)

    df = prepare_selected_datapoints(csv_path, run=run)

    fig, ax = plt.subplots(figsize=(12, 8))
    draw_selected_datapoints_panel(ax, df, "")

    ax.set_xlabel("Time t")
    ax.set_ylabel("muhat2")
    ax.set_xlim(left=0, right=max(df["t"].max(), 1))
    ax.set_ylim(bottom=max(0.0, min(df["selection_lower_bound"].min(), df["score_t"].min()) * 0.95))

    handles = [
        plt.Line2D([0], [0], color="#f59e0b", lw=3, alpha=0.55),
        plt.Line2D([0], [0], marker="o", color="none", markerfacecolor="#1f77b4", markersize=6),
    ]
    ax.legend(handles, ["selection interval", "selected muhat2"], loc="lower right", frameon=True)

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    return output_path


def plot_selected_datapoints_panels(result_dir=None, output_path=None, n_runs=8):
    result_dir = latest_results_dir() if result_dir is None else Path(result_dir)
    csv_path = result_dir / "selected_datapoints.csv"
    if output_path is None:
        output_path = result_dir / "selected_datapoints_score_time_panels.png"
    else:
        output_path = Path(output_path)

    df = prepare_selected_datapoints(csv_path)
    runs = sorted(df["run"].unique())[:n_runs]
    if len(runs) < n_runs:
        raise ValueError(f"Requested {n_runs} runs, but only found {len(runs)} in {csv_path}")

    mean_df = (
        df.groupby("t", as_index=False)
        .agg({
            "score_t": "mean",
            "selection_lower_bound": "mean",
            "selection_upper_bound": "mean",
        })
        .sort_values("t")
    )

    panel_df = pd.concat([df[df["run"].isin(runs)], mean_df], ignore_index=True)
    y_min = max(0.0, min(panel_df["selection_lower_bound"].min(), panel_df["score_t"].min()) * 0.95)
    y_max = max(panel_df["selection_upper_bound"].max(), panel_df["score_t"].max()) * 1.05
    x_max = max(df["t"].max(), 1)

    fig, axes = plt.subplots(3, 3, figsize=(15, 12), sharex=True, sharey=True)
    axes = axes.ravel()

    for ax, run in zip(axes, runs):
        run_df = df[df["run"] == run].sort_values("t")
        draw_selected_datapoints_panel(ax, run_df, f"Run {run}")

    draw_selected_datapoints_panel(
        axes[n_runs],
        mean_df,
        "Aggregated",
        interval_alpha=0.34,
        point_alpha=0.65,
    )

    for ax in axes[: n_runs + 1]:
        ax.set_xlim(left=0, right=x_max)
        ax.set_ylim(bottom=y_min, top=y_max)

    for ax in axes[6:9]:
        ax.set_xlabel("Time t")
    for ax in axes[::3]:
        ax.set_ylabel("muhat2")

    handles = [
        plt.Line2D([0], [0], color="#f59e0b", lw=3, alpha=0.55),
        plt.Line2D([0], [0], marker="o", color="none", markerfacecolor="#1f77b4", markersize=6),
    ]
    fig.legend(handles, ["selection interval", "selected muhat2"], loc="lower center", ncol=2)
    fig.tight_layout(rect=[0, 0.04, 1, 1])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    return output_path


def main():
    parser = argparse.ArgumentParser(
        description="Plot selected datapoint scores over time with active selection intervals."
    )
    parser.add_argument("--result-dir", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--run", type=int, default=None)
    parser.add_argument("--panels", action="store_true")
    parser.add_argument("--n-runs", type=int, default=8)
    args = parser.parse_args()

    if args.panels:
        output_path = plot_selected_datapoints_panels(
            result_dir=args.result_dir,
            output_path=args.output,
            n_runs=args.n_runs,
        )
    else:
        output_path = plot_selected_datapoints(
            result_dir=args.result_dir,
            output_path=args.output,
            run=args.run,
        )
    print(f"Wrote selected datapoints plot to {output_path}")


if __name__ == "__main__":
    main()
